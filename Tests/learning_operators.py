#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Learning the convection operator alone. 
"""
# %%
#Setting up simvue 
import os
import yaml 
import argparse

import sys
sys.path.append("..")
from Utils.simvue_utils import flatten_dict

config_file = '/home/ir-gopa2/rds/rds-ukaea-ap001/ir-gopa2/Code/NOs_for_POs/Expts/configs/NS_spectral_matrix.yaml' 

with open(config_file, 'r') as f:
    configuration = yaml.safe_load(f)

run_config = flatten_dict(configuration)

# %% 
#Importing the necessary packages
import sys
import numpy as np
from tqdm import tqdm 
import torch
import torch.nn.functional as F
from timeit import default_timer
from tqdm import tqdm 
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


from simvue import Run, Client
with Run(mode='online') as run:

    run.init(folder=configuration['Simvue']['folder'], tags=['NPDE', configuration['Model']['arch'], 'OpsLearn', configuration['Physics']['operator']], metadata=run_config)

    #Saving the current run file and the git hash of the repo
    run.save_file(os.path.abspath(__file__), 'code')
    run.save_file(os.path.abspath(config_file), 'code')

    import git
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    run.update_metadata({'Git Hash': sha})


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
    from Expts.model_setup import *
    from Neural_PDE.Utils.processing_utils import * 
    from Neural_PDE.Utils.training_utils import * 

    # from Tests.model_builder import build_model
    # model = build_model(configuration)
    # %% 
    ####################################
    # Data Preparation.
    ####################################

    t1 = default_timer()

    from Expts.data_loaders import *
    pde = configuration['Physics']['pde']
    if pde == 'Navier-Stokes':
        fields, x, y, dt = Navier_Stokes_Spectral(configuration)
    if pde == 'Euler-Fluid':
        fields, x, y, dt = Euler_FV(configuration)
            
    t = torch.arange(0, configuration['Data']['t_out']*dt, dt)

    fields = fields[...,:configuration['Data']['t_out']]

    #Making sure the data is in the correct format: [BS, N_vars, Nx, Ny, Nt]
    expected_shape = (configuration['Data']['ntrain'], configuration['Physics']['variables'], configuration['Physics']['Nx']//configuration['Physics']['x_slice'], configuration['Physics']['Ny']//configuration['Physics']['y_slice'], configuration['Data']['t_out'])
    assert fields.shape == expected_shape, \
        f"Expected fields shape to be {expected_shape}, but got {fields.shape}"

    fields = fields.permute(0, 4, 1, 2, 3).reshape(fields.shape[0]*fields.shape[4], fields.shape[1], fields.shape[2], fields.shape[3])
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

    train_data = fields_encoded[:int(0.8*fields_encoded.shape[0])]
    test_data = fields_encoded[int(0.8*fields_encoded.shape[0]):]

    print("Train Data: " + str(train_data.shape))
    print("Test Data: " + str(test_data.shape))

    #Setting up the data loaders
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_data), batch_size=configuration['Data']['batch size'], shuffle=False)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_data), batch_size=configuration['Data']['batch size'], shuffle=False)

    t2 = default_timer()
    print('preprocessing finished, time used:', t2-t1)

    # %% 
    ####################################
    # Setting up the Model and Optimizers 
    ####################################

    from Tests.model_builder import build_model
    model = build_model(configuration)
    model.to(device)
    num_params = count_parameters(model)
    run.update_metadata({'Number of Params': num_params})
    print("Number of model params : " + str(num_params))

    #Setting up the optimizer and scheduler, loss and epochs 
    optimizer = torch.optim.Adam(model.parameters(), lr=configuration['Opt']['learning rate'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=configuration['Opt']['scheduler step'], gamma=configuration['Opt']['scheduler gamma'])

    loss_func = torch.nn.MSELoss()

    epoch_init = 0
    epochs = configuration['Opt']['epochs']

    # %% 
    ####################################
    #Training
    ####################################
    from PRE.VectorConvOps_Spatial import *
    # gradient = Gradient(scale=1, taylor_order=2, boundary_cond='periodic', device=device, requires_grad=True)

    # laplace = Laplace(scale=1, taylor_order=2, boundary_cond='periodic', device=device, requires_grad=True)
    divergence = Divergence(scale=1, taylor_order=2, boundary_cond='periodic', device=device, requires_grad=True)

    # def gradient_func(uv):
    #     u = uv[:,0:1]
    #     v = uv[:,1:2]
    #     return gradient(u), gradient(v)
    
    # def laplace_func(uv):
    #     u = uv[:,0:1]
    #     v = uv[:,1:2]
    #     return laplace(u), laplace(v)
    
    def divergence_func(uv):
        with torch.no_grad():
            u = uv[:,0:1]
            v = uv[:,1:2]
            div = divergence(u, v)
        return div
    
    # def convection(uv):
    #     with torch.no_grad():
    #         u = uv[:,0:1]
    #         v = uv[:,1:2]
    #         conv = dot(uv, gradient(u)) + dot(uv, gradient(v))
    #     return conv
    
    def forward(model, uv):
        out = model(uv)
        out = out[:, 0:1] + out[:, 1:2]
        # yy = convection(uv)
        yy = divergence_func(uv)
        return out, yy 
        
    rollout_length = configuration['Train']['rollout_length']
    step = configuration['Data']['step']
    start_time = default_timer()
    for ep in tqdm(range(epoch_init, epochs+1)): #Training Loop - Epochwise

        model.train()
        t1 = default_timer()
        train_loss = 0 
        test_loss = 0 

        for xx in train_loader:
            optimizer.zero_grad()
            xx = xx[0].to(device)
            batch_size = xx.shape[0]

            im, y = forward(model, xx)

            #Recon Loss
            loss = loss_func(im.reshape(batch_size, -1), y.reshape(batch_size, -1))
            train_loss += loss.item()

            loss.backward()
            # grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()        
        t2 = default_timer()
        
        with torch.no_grad():
            for xx in test_loader:

                xx = xx[0].to(device)
                batch_size = xx.shape[0]

                im, y = forward(model, xx)

                #Recon Loss
                loss = loss_func(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

                test_loss += loss.item()
        
        t2 = default_timer()   

        train_loss = train_loss / len(train_loader)
        test_loss = test_loss / len(test_loader)

        print(f"Epoch {ep}, Time Taken: {round(t2-t1,3)}, Train Loss: {round(train_loss, 5)}, Test Loss: {round(test_loss,5)}")
        current_lr = optimizer.param_groups[0]['lr']
        run.log_metrics({'Train Loss': train_loss, 'Test Loss': test_loss, 'Learning Rate': current_lr})                
        scheduler.step()

    train_time = default_timer() - start_time

    # %%
    # Saving the Model
    saved_model = model_loc + '/model.pth'
    torch.save(model.state_dict(), saved_model)
    run.save_file(saved_model, 'output')

    test = fields_encoded[-50:]
    with torch.no_grad():

            xx = test.to(device)
            batch_size = xx.shape[0]

            im, y = forward(model, xx)

            #Recon Loss
            loss = loss_func(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

    error = (im - y).pow(2).mean()


    print('(MSE) Testing Error: %.3e' % (error))

    run.update_metadata({'Training Time': float(train_time),
                        'MSE Test Error': float(error)
                        })

    #Denormalising the test and predictions
    test_out = normalizer.decode(y.to(device)).cpu()
    pred_set = normalizer.decode(im.to(device)).cpu()


    # %% 
    #Plotting the results 
    from Utils.plots import plots_2d_yaml
    idx = 0 
    configuration['Physics']['variables']=1
    plots_2d_yaml(configuration, test_out.unsqueeze(0).permute(0, 2, 1, 3, 4), pred_set.unsqueeze(0).permute(0, 2, 1, 3, 4), plot_loc, run, idx, save=True)
    run.close()
    # %%    