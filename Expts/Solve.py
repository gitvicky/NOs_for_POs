#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline for solving the PDEs explicitly using hand-crafted ConvOps. 
"""
# %%
#Setting up simvue 
import os
import yaml 
import argparse

import sys
sys.path.append("..")
from Utils.simvue_utils import flatten_dict

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

    run.init(folder=configuration['Simvue']['folder'], tags=['NPDE', 'Solve', configuration['Physics']['pde'], configuration['Train']['odesolve']['method'], 'Tests'], metadata=run_config)

    #setting up the client API 
    client = Client()

    #Saving the current run file and the git hash of the repo
    run.save_file(os.path.abspath(__file__), 'code')
    run.save_file(os.path.abspath(args.config), 'code')

    import git
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    run.update_metadata({'Git Hash': sha})

    #Importing the necessary packages
    import sys
    import numpy as np
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
    if pde == 'Navier-Stokes':
        fields, x, y, dt = Navier_Stokes_Spectral(configuration)
    if pde == 'Euler-Fluid':
        fields, x, y, dt = Euler_FV(configuration)
            
    t = torch.arange(0, fields.shape[-1], dt)

    fields = fields[...,:configuration['Data']['t_out']]

    #Making sure the data is in the correct format: [BS, N_vars, Nx, Ny, Nt]
    expected_shape = (configuration['Data']['ntrain'], configuration['Physics']['variables'], configuration['Physics']['Nx']//configuration['Physics']['x_slice'], configuration['Physics']['Ny']//configuration['Physics']['y_slice'], configuration['Data']['t_out'])
    assert fields.shape == expected_shape, \
        f"Expected fields shape to be {expected_shape}, but got {fields.shape}"

    # %%
    #Normalising the data -- using the same normalisations for inputs and outputs
    normalizer_func = Normalisation(configuration['Data']['normalisation'])
    normalizer = normalizer_func(fields)
    fields_encoded = normalizer.encode(fields)
    
    #Saving Normalisation 
    saved_normalisations = model_loc + '/norms.npz'
    np.savez(saved_normalisations, 
            a=normalizer.a.numpy(), b=normalizer.b.numpy(), 
            )
    run.save_file(saved_normalisations, 'output')

    train_in, test_in, train_out, test_out = train_test_split(fields_encoded[...,:configuration['Data']['t_in']], fields_encoded[...,configuration['Data']['t_in']:configuration['Data']['t_out']], test_size=configuration['Data']['test-train-split'], random_state=42)
    print("Training Input: " + str(train_in.shape))
    print("Training Output: " + str(train_out.shape))

    # #Setting up the data loaders
    # train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_in, train_out), batch_size=configuration['Data']['batch size'], shuffle=False)
    # test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_in, test_out), batch_size=configuration['Data']['batch size'], shuffle=False)

    t2 = default_timer()
    print('preprocessing finished, time used:', t2-t1)

    # %% 
    ################################################
    # Setting up the Model with handcrafted kernels
    ################################################
    from operator_splitting_handcrafted import * 
    
    pde = configuration['Physics']['pde']

    if pde == 'Navier-Stokes':
        model = NS_spectral_OS_rhs(configuration, device)
    if pde == 'Euler-Fluid':
       model = Euler_FV_OS_rhs(configuration, device)
        

    ################################################
    # Setting up the timeintegration scheme - handcrafted for now
    ###############################################
    from Utils.explicit_time import * 
    method = configuration['Train']['odesolve']['method']
    if method == 'euler':
        timestep = euler
    elif method == 'midpoint':
        timestep = midpoint
    elif method == 'rk4':
        timestep = rk4

    # %% 
    ####################################
    #Solving the PDE using an explicit time stepping scheme    
    ####################################

    start_time = default_timer()
    u_n = train_in[...,0].to(device)
    for ii in tqdm(range(1, len(t))): 
        
        t1 = default_timer()
        u_nplus1 = timestep(model, u_n, dt)
        u_n = u_nplus1
        t2 = default_timer()

        solve_loss = (u_nplus1 - train_out[...,ii]).pow(2).mean()

        print(f"Timestep {ii}, Time Taken: {round(t2-t1,3)}, Solve Loss: {round(solve_loss, 5)}")
        run.log_metrics({'Solve Loss': solve_loss})

    solve_time = default_timer() - start_time

    # %%
    #Evaluation 
    start_time = default_timer()
    pred_set = []
    u_n = test_in[...,0:1]
    for ii in tqdm(1, range(t)): 
        
        t1 = default_timer()
        u_nplus1 = timestep(model, u_n, dt)
        pred_set.append(u_nplus1)
        u_n = u_nplus1
        t2 = default_timer()

        eval_loss = (u_nplus1 - test_out[...,ii:ii+1]).pow(2).mean()

        print(f"Timestep {ii}, Time Taken: {round(t2-t1,3)}, Solve Loss: {round(eval_loss, 5)}")
        run.log_metrics({'Eval Loss': eval_loss})

    eval_time = default_timer() - start_time
    pred_set = torch.cat(pred_set, dim=0)

    error = (pred_set - test_out).pow(2).mean()
    print('(MSE) Evaluation Error: %.3e' % (error))

    run.update_metadata({'Eval Time': float(eval_time),
                        'MSE Eval Error': float(error)
                        })
    
        #Denormalising the test and predictions
    test_out = normalizer.decode(test_out.to(device)).cpu()
    pred_set = normalizer.decode(pred_set.to(device)).cpu()

    #Shaping back to [BS, vars, Nt, Nx, Ny]
    test_out = test_out.permute(0,1,4,2,3)
    pred_set = pred_set.permute(0,1,4,2,3)

    #Plotting the results 
    from Utils.plots import plots_2d_yaml
    idx = 0 
    plots_2d_yaml(configuration, test_out, pred_set, plot_loc, run, idx, save=True)
    # %%
    run.close()
    # %%
    