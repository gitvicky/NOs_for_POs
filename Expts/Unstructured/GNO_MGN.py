#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GNO (NeuralOp) Testing Pipeline with Simvue integration.
Testing on the cylinderflow dataset from DeepMind's MGN paper. 
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
from tqdm import tqdm 
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

#Normalisation Functions 
import torch

def min_max_normalize(tensor, min_val=None, max_val=None):
    """
    Scales the input tensor values to the range [-1, 1] along the last dimension.
    
    Formula: 2 * ((x - min) / (max - min)) - 1
    
    Args:
        tensor (torch.Tensor): Input tensor of shape (..., D)
        min_val (torch.Tensor): Optional pre-calculated min of shape (D,)
        max_val (torch.Tensor): Optional pre-calculated max of shape (D,)
    """
    # Identify all dimensions except the last one to reduce over
    # e.g., if shape is (32, 10), dims_to_reduce is (0,)
    # e.g., if shape is (32, 64, 64, 3), dims_to_reduce is (0, 1, 2)
    dims_to_reduce = tuple(range(tensor.dim() - 1))

    if min_val is None:
        # Use amin/amax to avoid returning indices tuple
        min_val = tensor.amin(dim=dims_to_reduce)
    if max_val is None:
        max_val = tensor.amax(dim=dims_to_reduce)

    # 1. Calculate the range (denominator)
    # Add epsilon to prevent division by zero
    range_val = max_val - min_val + 1e-8
    
    # 2. Normalize to [0, 1] first
    norm_0_1 = (tensor - min_val) / range_val
    
    # 3. Scale to [-1, 1]
    # Formula: result = 2 * norm_0_1 - 1
    return 2 * norm_0_1 - 1

def gaussian_normalize(tensor, mean=None, std=None):
    """
    Standardizes the input tensor to have mean 0 and std 1 along the last dimension.
    
    Formula: (x - mean) / std
    
    Args:
        tensor (torch.Tensor): Input tensor of shape (..., D)
        mean (torch.Tensor): Optional pre-calculated mean of shape (D,)
        std (torch.Tensor): Optional pre-calculated std of shape (D,)
    """
    dims_to_reduce = tuple(range(tensor.dim() - 1))

    if mean is None:
        mean = tensor.mean(dim=dims_to_reduce)
    if std is None:
        std = tensor.std(dim=dims_to_reduce)

    # Broadcasting handles the subtraction/division automatically 
    # because mean/std match the size of the last dimension of tensor.
    return (tensor - mean) / (std + 1e-8)
# %%
# --- SIMVUE RUN INITIALIZATION ---
with Run(mode='offline') as run:
    
    # Initialize the run with folder, tags, and full configuration metadata
    run.init(
        folder=configuration['Simvue']['folder'], 
        tags=['GNO', 'DeepMind', 'NOs4POS', 'Pitagora', configuration['Model']['arch'], configuration['Physics']['pde'], configuration['Train']['odesolve']['method'], 'Rebuttal'], 
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
    data_loc = '/pitagora_work/FUPB1_UKAEA_ML/vgopakum/Data/cylinderflow_deepmind'
    # ----------------------------------

    # %% 
    #Loading dataset
    pde = configuration['Physics']['pde']
    data = np.load(data_loc + '/test.npz', allow_pickle=True)
    sample = data['trajectory0'].item()
    pos_sample = sample['pos']
    velocity_sample = sample['velocity']
    # pressure_sample = sample['pressure']

    dt = configuration['Physics']['dt']
    t = torch.arange(0, configuration['Data']['t_out']*dt, dt)
    x_min, x_max = 0.00, 1.60
    y_min, y_max = 0.00, 0.41
    u_min, u_max = -1.0407,  3.1258
    v_min, v_max = -1.6986,  1.7464

    xy_min, xy_max = torch.tensor([x_min, y_min]), torch.tensor([x_max, y_max])
    uv_min, uv_max = torch.tensor([u_min, v_min]), torch.tensor([u_max, v_max])

    # %% 
    from Models.GNNs import * 
    # 2. Initialize Model
    model = GNO2DTimeSolver(
        in_channels=2,    # Scalar field (e.g. Pressure)
        out_channels=2,   # Scalar field
        coord_dim=2,      # 2D Mesh
        latent_channels=configuration['Model']['width'],
        num_layers=configuration['Model']['depth'],
        radius=0.1
    ).to(device)
    # Log number of parameters to Simvue metadata
    run.update_metadata({'Number of Params': int(model.count_params())})
    print("Number of model params : " + str(model.count_params()))

    #Setting up the optimizer and scheduler, loss and epochs 
    optimizer = torch.optim.Adam(model.parameters(), lr=configuration['Opt']['learning_rate'], weight_decay=1e-4)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=configuration['Opt']['scheduler_step'], gamma=configuration['Opt']['scheduler_gamma'])
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    # optimizer, 
    # mode='min', 
    # factor=0.5, 
    # patience=10, 
    # )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
    loss_func = torch.nn.MSELoss()

#Model Input shape" # [Batch, num_vars, N_points, 1]

# %% 
    #Training
    start_time = default_timer()
    epoch_init = 0
    epochs = configuration['Opt']['epochs']
    step, train_T_out, test_T_out = configuration['Data']['step'], configuration['Train']['rollout_length']-1, configuration['Data']['t_out']-1

    def autoreg(model, coords, xx, dt=None):
        out = model(coords, coords, xx)
        return out

    def euler(model, coords, xx, dt):
        out = xx + model(coords, coords, xx)*dt
        return out

    if configuration['Train']['odesolve']['method'] == 'AR':
        evolve = autoreg
    elif configuration['Train']['odesolve']['method'] == 'euler':
        evolve = euler
    
    for ep in tqdm(range(epoch_init, epochs+1)): 
        t1 = default_timer()
        train_loss = 0 
        for it in range(0, 80):#Batch =1 
            model.train()
            optimizer.zero_grad()

            traj = data['trajectory'+str(it)].item()
            coords = torch.tensor(traj['pos'][::configuration['Physics']['t_slice']][:configuration['Data']['t_out']])
            uv = torch.tensor(traj['velocity'][::configuration['Physics']['t_slice']][:configuration['Data']['t_out']])

            coords = min_max_normalize(coords, xy_min, xy_max).unsqueeze(0).permute(0, 3, 2, 1)
            uv = min_max_normalize(uv, uv_min, uv_max).unsqueeze(0).permute(0, 3, 2, 1)

            xx, yy = uv[...,0:1], uv[...,1:]
            xx, yy, coords= xx.to(device), yy.to(device), coords.to(device)

            pred = []
            loss = 0

            for t in range(0, configuration['Data']['t_out']-1, 1):    
                y = yy[..., t:t + step]
                im = evolve(model, coords, xx, dt)
                # Compute loss for this step
                loss+= loss_func(im, y)
                # Update input for next timestep (sliding window)
                xx = torch.cat((xx[..., step:], im), dim=-1)
                # pred.append(im)
            # pred = torch.cat(pred, -1)
            # loss = torch.clamp(loss, max=10)
            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2.0)
            optimizer.step()
            train_loss += loss.item()

        with torch.no_grad():
            test_loss = 0 
            model.eval()
            for it in range(80, 100):

                traj = data['trajectory'+str(it)].item()
                coords = torch.tensor(traj['pos'][::configuration['Physics']['t_slice']][:configuration['Data']['t_out']])
                uv = torch.tensor(traj['velocity'][::configuration['Physics']['t_slice']][:configuration['Data']['t_out']])

                coords = min_max_normalize(coords, xy_min, xy_max).unsqueeze(0).permute(0, 3, 2, 1)
                uv = min_max_normalize(uv, uv_min, uv_max).unsqueeze(0).permute(0, 3, 2, 1)

                xx, yy = uv[...,0:1], uv[...,1:]
                xx, yy, coords= xx.to(device), yy.to(device), coords.to(device)

                pred = []
                loss = 0

                for t in range(0, configuration['Data']['t_out']-1, 1):    
                    y = yy[..., t:t + step]
                    out = evolve(model, coords, xx, dt)
                    # Update input for next timestep (sliding window)
                    xx = torch.cat((xx[..., step:], out), dim=-1)
                    pred.append(out)
                pred = torch.cat(pred, -1)

                test_loss += loss.item()

        train_loss = train_loss / 80
        test_loss = test_loss / 20
        t2 = default_timer()

        # Log metrics to Simvue
        current_lr = optimizer.param_groups[0]['lr']
        run.log_metrics({'Train Loss': train_loss, 'Test Loss': test_loss,  'Learning Rate': current_lr}, step=ep)

        print(f"Epoch {ep}, Time Taken: {round(t2-t1,3)}, Train Loss: {round(train_loss, 5)}, Test Loss: {round(test_loss,5)}")
        scheduler.step()
        # scheduler.step(test_loss)

    train_time = default_timer() - start_time
    
    # Log training time to Simvue metadata
    run.update_metadata({'Training Time': float(train_time)})


# %%
    # %% 
    #Evaluation
    pred_set = []
    model.eval()

    with torch.no_grad():
        test_loss = 0 
        model.eval()
        for it in range(80, 100):

            traj = data['trajectory'+str(it)].item()
            coords = torch.tensor(traj['pos'][::configuration['Physics']['t_slice']][:configuration['Data']['t_out']])
            uv = torch.tensor(traj['velocity'][::configuration['Physics']['t_slice']][:configuration['Data']['t_out']])

            coords = min_max_normalize(coords, xy_min, xy_max).unsqueeze(0).permute(0, 3, 2, 1)
            uv = min_max_normalize(uv, uv_min, uv_max).unsqueeze(0).permute(0, 3, 2, 1)

            xx, yy = uv[...,0:1], uv[...,1:]
            xx, yy, coords= xx.to(device), yy.to(device), coords.to(device)

            pred = []

            for t in range(0, configuration['Data']['t_out']-1, 1):    
                y = yy[..., t:t + step]
                out = evolve(model, coords, xx, dt)
                # Update input for next timestep (sliding window)
                xx = torch.cat((xx[..., step:], out), dim=-1)
                pred.append(out)
            pred = torch.cat(pred, -1)
            pred_set.append(pred)
        pred_set = torch.cat(pred_set, 0)
    # %% 
    #Plottin
    from Utils.metrics import MSE, NRMSE 
    
    #Shaping back to [BS, vars, Nxy, Nt]
    pred_set = pred_set.cpu()
    test_out = data_test[...,1:]
    
    mse_error = torch.mean((pred_set-test_out).pow(2)).cpu()
    mse = float(NRMSE(pred_set, test_out)['average'])
    print(f"MSE: {mse_error}")

    pred_set, test_out = normalizer.decode(pred_set), normalizer.decode(test_out)
    mse = float(NRMSE(pred_set, test_out)['average'])
    nrmse = float(NRMSE(pred_set, test_out)['average'])

    run.update_metadata({'MSE': mse,
                         'NRMSE (physical)':nrmse})
    # run.update_metadata({'NRMSE': nrmse})


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
    test_out, pred_set = test_out.numpy(), pred_set.numpy()
    from Expts.Unstructured.unstructured_plot import * 
    fig, axes = create_field_comparison_plot(
    X, Y, test_out, pred_set,
    run=None,
    batch_idx=0, var_idx=0,
    time_steps=[0, 4, 8],
    title="Field: u",
    obstacles=None,
    test_label='Sim.',
    pred_label='Net.'
    )
    plot_name = plot_loc + '/u_'+run.name+'.png'
    plt.savefig(plot_name, dpi=300, bbox_inches='tight')
    run.save_file(plot_name, 'output')

    from Expts.Unstructured.unstructured_plot import * 
    fig, axes = create_field_comparison_plot(
    X, Y, test_out, pred_set,
    run=None,
    batch_idx=0, var_idx=1,
    time_steps=[0, 4, 8],
    title="Field: v",
    obstacles=None,
    test_label='Sim.',
    pred_label='Net.'
    )
    plot_name = plot_loc + '/v_'+run.name+'.png'
    plt.savefig(plot_name, dpi=300, bbox_inches='tight')
    run.save_file(plot_name, 'output')
    

# The 'with Run(mode='offline') as run:' block automatically handles 'run.close()'
# %%