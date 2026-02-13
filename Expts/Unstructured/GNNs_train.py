#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GNO (GCNs and NNConv) Testing Pipeline with Simvue integration.
"""
#Imports

# %% 

import shutil
import os
import yaml 
import argparse
from timeit import default_timer
import sys
sys.path.append("..")
sys.path.append("../..")

# --- SIMVUE IMPORTS AND UTILS ---
from Utils.simvue_utils import flatten_dict
from simvue import Run, Client
# ----------------------------------

import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn 

# %% 
#Loading config
def is_notebook():
    """Check if running in Jupyter notebook"""
    try:
        get_ipython()
        return True
    except NameError:
        return False

def load_config(config_path='configs/train/Wave_GNO.yaml'):
    """Load configuration from YAML file"""
    print(f"Loading configuration from: {config_path}")
    with open(config_path, 'r') as f:
        configuration = yaml.safe_load(f)
    print(f"Configuration loaded successfully!")
    return configuration

# Determine config path based on environment
if is_notebook():
    # Jupyter notebook mode - set your config path here
    CONFIG_PATH = '/pitagora/home/userexternal/vgopakum/NOs_for_POs/Expts/configs/train/Wave_GNO.yaml'  # Change this as needed
    configuration = load_config(CONFIG_PATH)
else:
    # Command-line mode
    import argparse
    parser = argparse.ArgumentParser(description='Testing script with YAML config')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config YAML file')
    args = parser.parse_args()
    configuration = load_config(args.config)

# Flatten configuration for Simvue metadata
run_config = flatten_dict(configuration)
model_type = configuration['Model']['arch']

# %%
# --- SIMVUE RUN INITIALIZATION ---
with Run(mode='offline') as run:
    
    # Initialize the run with folder, tags, and full configuration metadata
    run.init(
        folder=configuration['Simvue']['folder'], 
        tags=['GNO', 'NOs4POS', 'Pitagora', configuration['Model']['arch'], configuration['Physics']['pde'], configuration['Train']['odesolve']['method'], 'Rebuttal'], 
        metadata=run_config
    )
    
    # Update tags with any from the config
    run.update_tags([configuration['Simvue']['tags']])
        
    if configuration['Model']['operator_splitting']:
        run.save_file(os.path.abspath('operator_splitting.py'), 'code', snapshot=True)
        run.update_tags(['OpsSplit'])

    #Setting up locations. 
    file_loc = os.getcwd()
    model_loc = file_loc + '/Weights/' + run.name
    os.mkdir(model_loc)
    plot_loc = file_loc + '/Plots'


    run.config(disable_resources_metrics=True)
    print("Run Name: " + str(run.name))
    print(yaml.dump(configuration, default_flow_style=False, indent=2))

    
    # Saving the current run file and the config
    run.save_file(os.path.abspath(__file__), 'code', snapshot=True)
    run.save_file(os.path.abspath(args.config), 'code', snapshot=True)
    
    # Saving the Git hash
    try:
        import git
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        run.update_metadata({'Git Hash': sha})
    except:
        run.update_metadata({'Git Hash': 'N/A (Git not found)'})

    # Setting up the seeds and devices
    torch.manual_seed(configuration['seed']) # Use seed from config
    np.random.seed(configuration['seed'])    # Use seed from config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_dtype(torch.float32)

    # Setting up locations. 
    file_loc = os.getcwd()
    plot_loc = file_loc + '/Plots'
    # ----------------------------------

    # %% 
    #Loading dataset
    from data_loaders import *
    pde = configuration['Physics']['pde']
    if pde == 'Incomp. Navier-Stokes':
        fields, X, Y, dt, mass, params, edge_attr, edge_index = flow_past_cylinder(configuration)
    if pde == 'Wave':
        fields, X, Y, dt, params = Wave_Spectral(configuration)

    t = torch.arange(0, configuration['Data']['t_out']*dt, dt)
    fields = fields[...,:configuration['Data']['t_out']]

    XY = torch.stack((X,Y), dim=-1)

    print("Fields shape: " + str(fields.shape))
    print("Coords shape: " + str(XY.shape))

    # %% 
    # Normalising the data -- using the same normalisations for inputs and outputs
    from Neural_PDE.Utils.processing_utils import * 
    normalizer_func = Normalisation(configuration['Data']['normalisation'])
    normalizer = normalizer_func(fields)
    fields_encoded = normalizer.encode(fields)
    
    #Normalising X,Y
    X_normalizer = normalizer_func(X)
    X = X_normalizer.encode(X)

    Y_normalizer = normalizer_func(Y)
    Y = Y_normalizer.encode(Y)

    XY = torch.stack((X,Y), dim=-1)

    #Saving Normalisation 
    saved_normalisations = model_loc + '/norms.npz'
    if normalizer_func == 'Min-Max':
        np.savez(saved_normalisations, 
                a=normalizer.a.numpy(), b=normalizer.b.numpy(), 
                )
        run.save_file(saved_normalisations, 'output')
    elif normalizer_func == 'Gaussian':
        np.savez(saved_normalisations, 
                a=normalizer.mean.numpy(), b=normalizer.std.numpy(), 
                )
        run.save_file(saved_normalisations, 'output') 

    
    from sklearn.model_selection import train_test_split
    data = fields_encoded[...,:configuration['Data']['t_out']]
    data_train, data_test = data[:int(configuration['Data']['ntrain']*(1-0.2))], data[-int(configuration['Data']['ntrain']*(0.2)):]
    params_train, params_test = params[:int(configuration['Data']['ntrain']*(1-0.2))], params[-int(configuration['Data']['ntrain']*(0.2)):]

    input_window = configuration['Train']['input_length']
    prediction_steps = configuration['Train']['rollout_length'] - 1 

    # --- 2. Initialize Dataset ---
    from data_loaders import GraphSpatioTemporalDataset # Assuming this path from GNNs_Train.py
    train_dataset = GraphSpatioTemporalDataset(
        data=data_train,
        pos=XY,
        params=params_train,
        input_window=input_window,    
        prediction_steps=prediction_steps, 
        connectivity_radius=0.1
    )

    test_dataset = GraphSpatioTemporalDataset(
        data=data_test,
        pos=XY,
        params=params_test,
        input_window=input_window,     # Look at past 10 steps
        prediction_steps=configuration['Data']['t_out']-1,  # Predict next 1 step
        connectivity_radius=0.1
    )

    #Setting up the data loaders
    from torch_geometric.loader import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=configuration['Data']['batch_size'], shuffle=False, pin_memory=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=configuration['Data']['batch_size'], shuffle=False, pin_memory=True, num_workers=4)


    # %%
    from Models.GNNs import * 
    if model_type == 'gcn':
        model = GCN(in_channels=configuration['Model']['in_vars'], hidden_channels=configuration['Model']['width'], out_channels=configuration['Model']['out_vars'], num_layers=configuration['Model']['depth'])
    if model_type == 'nnconv':
        model = NNConvNet(in_channels=configuration['Model']['in_vars'], hidden_channels=configuration['Model']['width'], out_channels=configuration['Model']['out_vars'], num_layers=configuration['Model']['depth'], edge_dim=4)
    model.to(device)

    # Log number of parameters to Simvue metadata
    run.update_metadata({'Number of Params': int(model.count_params())})
    print("Number of model params : " + str(model.count_params()))

    #Setting up the optimizer and scheduler, loss and epochs 
    optimizer = torch.optim.Adam(model.parameters(), lr=configuration['Opt']['learning_rate'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=configuration['Opt']['scheduler_step'], gamma=configuration['Opt']['scheduler_gamma'])
    loss_func = torch.nn.MSELoss()

    # %% 
    #Training
    start_time = default_timer()
    epoch_init = 0
    epochs = configuration['Opt']['epochs']
    step, train_T_out, test_T_out = configuration['Data']['step'], configuration['Train']['rollout_length']-1, configuration['Data']['t_out']-1

    def evolve(model, batch):
        return model(batch)

    for ep in tqdm(range(epoch_init, epochs+1)): 
        t1 = default_timer()
        train_loss = 0 
        for batch in train_loader:
            model.train()
            optimizer.zero_grad()
            pred = []
            for t in range(0, train_T_out, step):    
                if model_type == 'gcn':       
                    im = model(batch.x.to(device), batch.edge_index.to(device))
                elif model_type == 'nnconv':
                    im = model(batch.x.to(device), batch.edge_index.to(device), batch.edge_attr.to(device))
                # Compute loss for this step
                pred.append(im)
                # Update input for next timestep (sliding window)
                batch.x = im
            pred = torch.stack(pred, -1).flatten(-2)
            loss = loss_func(pred, batch.y.to(device))
            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2.0)
            optimizer.step()
            train_loss += loss.item()

            # # Validation Loop (omitted for brevity, keeping original GNO_test structure)
        
        train_loss = train_loss / len(train_loader)
        # test_loss = test_loss / len(test_loader)
        t2 = default_timer()

        # Log metrics to Simvue
        current_lr = optimizer.param_groups[0]['lr']
        run.log_metrics({'Train Loss': train_loss, 'Learning Rate': current_lr}, step=ep)

        print(f"Epoch {ep}, Time Taken: {round(t2-t1,3)}, Train Loss: {round(train_loss, 5)}") #, Test Loss: {round(test_loss,5)}")
        scheduler.step()

    train_time = default_timer() - start_time
    
    # Log training time to Simvue metadata
    run.update_metadata({'Training Time': float(train_time)})


    # %% 
    #Evaluation
    pred_set = []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            loss = 0 
            pred = []
            x_init = batch.x.clone() # Keep initial state for next batch
            for t in range(0, test_T_out, step):        
                if model_type == 'gcn':       
                    out = model(batch.x.to(device), batch.edge_index.to(device))
                elif model_type == 'nnconv':
                    out = model(batch.x.to(device), batch.edge_index.to(device), batch.edge_attr.to(device))
                batch.x = out
                # Collect predictions for full sequence loss
                pred.append(out)
            pred = torch.stack(pred, -1)
            batch.x = x_init # Restore initial state if you loop over test_loader again

        # Split predictions by graph
            batch_size = batch.num_graphs
            for i in range(batch_size):
                mask = batch.batch == i
                graph_preds = pred[mask]
                pred_set.append(graph_preds)
        pred_set = torch.stack(pred_set, 0)

    # %% 
    #Plottin
    from Utils.metrics import MSE, NRMSE 
    
    #Shaping back to [BS, vars, Nt, Nx, Ny]
    pred_set = pred_set.permute(0, 2, 1, 3).cpu()
    test_out = data_test[...,1:]
    
    mse_error = torch.mean((pred_set-test_out).pow(2))
    print(f"MSE: {mse_error}")

    pred_set, test_out = normalizer.decode(pred_set), normalizer.decode(test_out)

    run.update_metadata({'MSE': mse_error,
                        'NRMSE (physical)': float(NRMSE(pred_set, test_out)['average'])
                        })  

    #%%
    from Utils.plots import * # Using a simple class to mock the run object for the plotting function if needed
    #Plotting for rectangular grids
    # class Run:
    #     def __init__(self, name=None):
    #         self.name = name

    # run_mock = Run(run.name) # Use the real run.name from Simvue
    # plot_name = plot_loc + '/' + run.name + '_Field_Comparison.png'
    # plots_2d_yaml(configuration, test_out, pred_set, plot_loc=plot_loc, run=run, idx=0, save=True)

    X, Y = X_normalizer.decode(X), Y_normalizer.decode(Y)
    from Expts.Unstructured.unstructured_plot import * 
    fig, axes = create_field_comparison_plot(
    X, Y, test_out, pred_set,
    run=None,
    batch_idx=15, var_idx=0,
    time_steps=[0, 15, 30, 45],
    title="Field: u",
    obstacles=None,
    test_label='Sim.',
    pred_label='Net.'
    )
    plot_loc = os.getcwd()
    plot_name = plot_loc + '/u_'+run.name+'.png'
    plt.savefig(plot_name, dpi=300, bbox_inches='tight')
    run.save_file(plot_name, 'output')

    from Expts.Unstructured.unstructured_plot import * 
    fig, axes = create_field_comparison_plot(
    X, Y, test_out, pred_set,
    run=None,
    batch_idx=15, var_idx=1,
    time_steps=[0, 15, 30, 45],
    title="Field: v",
    obstacles=None,
    test_label='Sim.',
    pred_label='Net.'
    )
    plot_loc = os.getcwd()
    plot_name = plot_loc + '/v_'+run.name+'.png'
    plt.savefig(plot_name, dpi=300, bbox_inches='tight')
    run.save_file(plot_name, 'output')
    
    # %%
    #Saving the slurm output file. 
    if 'SLURM_JOB_ID' in os.environ:
        import time 
        time.sleep(1)
        slurm_id = os.environ['SLURM_JOB_ID']
        run.save_file(os.path.abspath('slurm-'+str(slurm_id)+'.out'), 'output', snapshot=True)

# The 'with Run(mode='offline') as run:' block automatically handles 'run.close()'
# %%