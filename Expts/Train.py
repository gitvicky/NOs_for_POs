#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training Pipeline. 
"""
# %%

#Setting up simvue 
import os
import yaml 
import argparse

import sys
sys.path.append("..")
from Utils.simvue_utils import flatten_dict

# %%
#Config files.
def parse_args():
    parser = argparse.ArgumentParser(description='Training script with YAML config')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    return parser.parse_args()

args = parse_args()
with open(args.config, 'r') as f:
    configuration = yaml.safe_load(f)

run_config = flatten_dict(configuration)


# %% 
from simvue import Run, Client
with Run(mode='online') as run:

    run.init(folder=configuration['Simvue']['folder'], tags=['NPDE', configuration['Model']['arch'], 'POs4NOs', configuration['Physics']['pde'], configuration['Train']['odesolve']['method'], 'Tests'], metadata=run_config)

    # #if run is being disabled
    # import argparse
    # run = argparse.Namespace()
    # run.name = "test"

    #setting up the client API 
    client = Client()

    #Saving the current run file and the git hash of the repo
    run.save_file(os.path.abspath(__file__), 'code')
    run.save_file(os.path.abspath(args.config), 'code')
    
    if configuration['Model']['operator_splitting']:
        run.save_file(os.path.abspath('operator_splitting.py'), 'code')

    import git
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    run.update_metadata({'Git Hash': sha})

    #Importing the necessary packages
    import sys
    import numpy as np
    import math
    from tqdm import tqdm 
    import torch
    import torch.nn.functional as F
    from timeit import default_timer
    from tqdm import tqdm 
    from sklearn.model_selection import train_test_split

    #Setting up locations. 
    file_loc = os.getcwd()
    data_loc = os.path.dirname(os.getcwd()) + '/Data/'
    model_loc = file_loc + '/Weights/' + run.name
    os.mkdir(model_loc)
    plot_loc = file_loc + '/Plots'

    #Setting up the seeds and devices
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_dtype(torch.float32)
    # %%
    #Importing the models and utilities. 
    from model_setup import *
    from Neural_PDE.Utils.processing_utils import * 
    from Neural_PDE.Utils.training_utils import * 

    # %% 
    ####################################
    # Data Preparation.
    ####################################

    t1 = default_timer()

    from data_loaders import *
    pde = configuration['Physics']['pde']
    if pde == 'Wave':
        fields, x, y, dt = Wave_Spectral(configuration)
    if pde == 'Navier-Stokes':
        fields, x, y, dt = Navier_Stokes_Spectral(configuration)
    if pde == 'Euler-Fluid':
        fields, x, y, dt = Euler_FV(configuration)
    if pde == 'Incomp. Navier-Stokes':
        fields, force, x, y, dt = Navier_Stokes_Incomp(configuration)
    if pde == 'Comp. Navier-Stokes':
        fields, x, y, dt = Navier_Stokes_Comp(configuration)
    if pde == 'Electrostatic MHD':
        if configuration['Physics']['source'] == 'JOREK': 
            fields, x, y, dt = JOREK_electrostatic(configuration)
    if pde == 'Electromagnetic MHD':
        if configuration['Physics']['source'] == 'JOREK': 
            fields, x, y, dt = JOREK_electrostatic(configuration)
    if pde == 'FDS':
            fields, x, y, dt = FDS_Carpark(configuration)
    if pde == 'Shear Flow':
        fields, x, y, dt = Shear_Flow(configuration)
    if pde == 'Euler Quadrant':
        fields, x, y, dt = Euler_Quadrants(configuration)
            
    t = torch.arange(0, configuration['Data']['t_out']*dt, dt)
    fields = fields[...,:configuration['Data']['t_out']]

    #Making sure the data is in the correct format: [BS, N_vars, Nx, Ny, Nt]
    expected_shape = (configuration['Data']['ntrain'], configuration['Physics']['variables'], configuration['Physics']['Nx']//configuration['Physics']['x_slice'], configuration['Physics']['Ny']//configuration['Physics']['y_slice'], configuration['Data']['t_out'])
    assert fields.shape == expected_shape, \
        f"Expected fields shape to be {expected_shape}, but got {fields.shape}"
    
    print("Data shape: " + str(fields.shape))

    # %%
    # Normalising the data -- using the same normalisations for inputs and outputs
    normalizer_func = Normalisation(configuration['Data']['normalisation'])
    normalizer = normalizer_func(fields, low=-1.0, high=1.0)
    print(normalizer.a, normalizer.b)
    if configuration['Model']['ops_split_normalise']: #Normalise and Denormalise done within the Model. 
        fields_encoded = fields
    else:
        fields_encoded = normalizer.encode(fields)
    
    #Saving Normalisation 
    saved_normalisations = model_loc + '/norms.npz'
    np.savez(saved_normalisations, 
            a=normalizer.a.numpy(), b=normalizer.b.numpy(), 
            )
    run.save_file(saved_normalisations, 'output')
    
    # #Setting up the train-test pipelines. Options are for a full rollout (and then backprop) or a single step rollout (and then backprop).
    # if configuration['Train']['rollout'] == 'full':

    # #Setting up train and test
    # train_in, test_in, train_out, test_out = train_test_split(fields_encoded[...,:configuration['Data']['t_in']], fields_encoded[...,configuration['Data']['t_in']:configuration['Data']['t_out']], test_size=configuration['Data']['test_train_split'], random_state=42)

    # #Setting up the data loaders
    # train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_in, train_out), batch_size=configuration['Data']['batch_size'], shuffle=True)
    # test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_in, test_out), batch_size=configuration['Data']['batch_size'], shuffle=False)

    #If using a single step rollout, then we need to create a windowed dataset.
    train_in, test_in, train_out, test_out = train_test_split(fields_encoded[...,:configuration['Data']['t_in']], fields_encoded[...,configuration['Data']['t_in']:configuration['Data']['t_out']], test_size=configuration['Data']['test_train_split'], random_state=42)

    train_data = torch.cat((train_in, train_out), dim=-1)#Merging for creating the windowed dataset.

    input_window = configuration['Train']['input_length']
    prediction_steps = configuration['Train']['rollout_length'] - 1 
    train_dataset = SpatioTemporalDataset(train_data, input_window, prediction_steps)

    #Setting up the data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=configuration['Data']['batch_size'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_in, test_out), batch_size=configuration['Data']['batch_size'], shuffle=False)

    print("Training Input: " + str(train_in.shape))
    print("Training Output: " + str(train_out.shape))
    t2 = default_timer()
    print('preprocessing finished, time used:', t2-t1)

    # %% 
    ####################################
    # Setting up the Model and Optimizers 
    ####################################

    model = model_initialisation(configuration, normalizer, run)
    model.to(device)

    run.update_metadata({'Number of Params': int(model.count_params())})
    print("Number of model params : " + str(model.count_params()))

    #Setting up the optimizer and scheduler, loss and epochs 
    optimizer = torch.optim.Adam(model.parameters(), lr=configuration['Opt']['learning_rate'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=configuration['Opt']['scheduler_step'], gamma=configuration['Opt']['scheduler_gamma'])
    
    if configuration['Model']['arch']=='fno':
        loss_func = LpLoss(size_average=False)
    else:
        loss_func = torch.nn.MSELoss()

    epoch_init = 0
    epochs = configuration['Opt']['epochs']

    #Restarting the run from a checkpoint 
    if configuration['Train']['restart'] == True: 
        client.get_artifact_as_file(client.get_artifact_as_file(configuration['Train']['restart_run_name']))
        ckpt_path = '/tmp/checkpoint.pt'
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        epoch_init = checkpoint["epoch"]

    # Setting up the Training pipeline
    if configuration['Train']['odesolve']['source'] == 'custom':
        from Utils import explicit_time
        train = explicit_time.Train_Setup(model, train_loader, test_loader, loss_func, optimizer, scheduler, epochs,  ode_solver=configuration['Train']['odesolve']['source'], roll_out=configuration['Train']['odesolve']['method'], noise=configuration['Train']['input_noise'], batch_norm=configuration['Train']['batch_norm'])
    elif configuration['Train']['odesolve']['source'] == 'torchdiffeq':
        from Utils import torch_odesolve  
        # train = torch_odesolve.Train_Setup(model, train_loader, test_loader, loss_func, optimizer, scheduler, epochs,  configuration['Train']['odesolve']['method'],  configuration['Train']['odesolve']['adjoint'])
        train = torch_odesolve.Train_Setup(model, train_loader, test_loader, loss_func, optimizer, scheduler, epochs,  configuration['Train']['odesolve']['method'],  configuration['Train']['odesolve']['adjoint'])

    from Neural_PDE.Utils.training_utils import train_one_epoch_AR
    
    # %% 
    ####################################
    #Training
    ####################################
    start_time = default_timer()
    for ep in tqdm(range(epoch_init, epochs+1)): #Training Loop - Epochwise

        model.train()
        t1 = default_timer()
        train_loss, test_loss = train.one_epoch(configuration['Data']['step'], configuration['Train']['rollout_length']-1, configuration['Data']['t_out']-1, dt=dt)
        # train_loss, test_loss = train_one_epoch_AR(model, train_loader, test_loader, loss_func, optimizer, configuration['Data']['step'], configuration['Data']['t_out']-1)

        t2 = default_timer()

        train_loss = train_loss / len(train_loader)
        test_loss = test_loss / len(test_loader)

        print(f"Epoch {ep}, Time Taken: {round(t2-t1,3)}, Train Loss: {round(train_loss, 5)}, Test Loss: {round(test_loss,5)}")
        current_lr = optimizer.param_groups[0]['lr']
        run.log_metrics({'Train Loss': train_loss, 'Test Loss': test_loss, 'Learning Rate': current_lr}, step=ep)
        scheduler.step()

        #Alerting potential instability in training.
        run.create_metric_threshold_alert(
            name='Unstable',
            metric='Train Loss',
            threshold=1e5,
            rule='is above',  
            frequency=1,
            window=1,
            trigger_abort=True
            )
        
        #Killing the run if the training becomes unstable. 
        if math.isnan(train_loss) or math.isinf(train_loss):
            print("Training loss is NaN or Inf, stopping training.")
            run.create_user_alert(
                name='Training terminated',
                description='Training loss became NaN or Inf, stopping training.',
                notification='none',
                trigger_abort=True,
                attach_to_run=True
            )
            
        #Checkpointing. 
        if ep % configuration['Train']['checkpoint']['epochs'] == 0:
            checkpoint = {}
            checkpoint["model"] = model.state_dict()
            checkpoint["optimizer"] = optimizer.state_dict() 
            checkpoint["scheduler"] = scheduler.state_dict()
            checkpoint["epoch"] = ep
            torch.save(checkpoint, model_loc + "/checkpoint_"+str(ep)+".pt")
            run.save_file(model_loc + "/checkpoint_"+str(ep)+".pt", 'output')
            run.update_metadata({'Epochs': ep})

    train_time = default_timer() - start_time

    # %%
    # Saving the Model
    saved_model = model_loc + '/model.pth'
    torch.save(model.state_dict(), saved_model)
    run.save_file(saved_model, 'output')

    #Evaluation 
    if configuration['Train']['odesolve']['source'] == 'custom':
        eval = explicit_time.Eval_Setup(model, test_in, test_out, normalizer='False', ode_solver = configuration['Train']['odesolve']['source'], roll_out= configuration['Train']['odesolve']['method'])

    elif configuration['Train']['odesolve']['source'] == 'torchdiffeq':
        eval = torch_odesolve.Eval_Setup( model, test_in, test_out, normalizer='False', method=configuration['Train']['odesolve']['method']
)
    pred_encoded, error = eval.inference(configuration['Data']['step'], configuration['Data']['t_out']-1, dt=dt)
    # pred_encoded, error, mae = validation_AR(model, test_in, test_out, configuration['Data']['step'], configuration['Data']['t_out']-1)


    print('(MSE) Testing Error: %.3e' % (error))

    run.update_metadata({'Training Time': float(train_time),
                        'MSE Test Error': float(error)
                        })

    #Denormalising the test and predictions
    if configuration['Model']['ops_split_normalise'] == False: #Normalise/Denormalise done within the Model for OS. 
        test_out = normalizer.decode(test_out.to(device)).cpu()
        pred_set = normalizer.decode(pred_encoded.to(device)).cpu()
    else:
        test_out = test_out.cpu()
        pred_set = pred_encoded.cpu()

    #Shaping back to [BS, vars, Nt, Nx, Ny]
    test_out = test_out.permute(0,1,4,2,3)
    pred_set = pred_set.permute(0,1,4,2,3)

    from Utils.metrics import MSE, NRMSE
    run.update_metadata({'MSE (Physical)': float(MSE(pred_set, test_out)['average']),
                        'NRMSE (Physical)': float(NRMSE(pred_set, test_out)['average'])
                        })  
    # %% 
    #Plotting the results 
    from Utils.plots import plots_2d_yaml
    idx = 0 
    plots_2d_yaml(configuration, test_out, pred_set, plot_loc, run, idx, save=True)
    # %%
    run.close()
    # %%