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

config_file = '/home/ir-gopa2/rds/rds-ukaea-ap001/ir-gopa2/Code/NOs_for_POs/Expts/configs/NS_spectral_FNO.yaml' 

with open(config_file, 'r') as f:
    configuration = yaml.safe_load(f)

run_config = flatten_dict(configuration)

# %% 
#Time Stepping Schemes

def autoregressive(model, u_n, dt=0):
    u_new = model(u_n)
    return u_new

def euler(model, u_n, dt): 
    u_new = u_n + model(u_n)*dt 
    return u_new 

def midpoint(model, u_n, dt):#RK2
    h = dt 
    k1 = model(u_n)
    k2 = model(u_n + 0.5*h*k1)
    u_new = u_n + h*k2
    return u_new

def rk4(model, u_n, dt):
    h = dt 
    
    k1 = model(u_n)
    k2 = model(u_n + 0.5*h*k1)
    k3 = model(u_n + 0.5*h*k2)
    k4 = model(u_n + h*k3)

    u_new = u_n + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
    return u_new

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

#Setting up locations. 
file_loc = os.getcwd()
data_loc = os.path.dirname(os.getcwd()) + '/Data/'
# model_loc = file_loc + '/Weights/' + run.name
# os.mkdir(model_loc)
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
        
t = torch.arange(0, configuration['Data']['t_out']*dt, dt)

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

# #Saving Normalisation 
# saved_normalisations = model_loc + '/norms.npz'
# np.savez(saved_normalisations, 
#         a=normalizer.a.numpy(), b=normalizer.b.numpy(), 
#         )
# run.save_file(saved_normalisations, 'output')


train_in, test_in, train_out, test_out = train_test_split(fields_encoded[...,:configuration['Data']['t_in']], fields_encoded[...,configuration['Data']['t_in']:configuration['Data']['t_out']], test_size=configuration['Data']['test-train-split'], random_state=42)
print("Training Input: " + str(train_in.shape))
print("Training Output: " + str(train_out.shape))

train_data = torch.cat((train_in, train_out), dim=-1)#Merging for creating the windowed dataset.

input_window = configuration['Train']['input_length']
prediction_steps = configuration['Train']['rollout_length'] - 1 
train_dataset = SpatioTemporalDataset(train_data, input_window, prediction_steps)

#Setting up the data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=configuration['Data']['batch size'], shuffle=False)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_in, test_out), batch_size=configuration['Data']['batch size'], shuffle=False)

t2 = default_timer()
print('preprocessing finished, time used:', t2-t1)

# %% 
####################################
# Setting up the Model and Optimizers 
####################################

model = model_initialisation(configuration)
model.to(device)

# run.update_metadata({'Number of Params': int(model.count_params())})
print("Number of model params : " + str(model.count_params()))

roll_out = configuration['Train']['odesolve']['method']
if roll_out == 'AR':
    forward = autoregressive
elif roll_out == 'euler':
    forward = euler
elif roll_out == 'midpoint':
    forward = midpoint
elif roll_out == 'rk4':
    forward = rk4

#Setting up the optimizer and scheduler, loss and epochs 
optimizer = torch.optim.Adam(model.parameters(), lr=configuration['Opt']['learning rate'], weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=configuration['Opt']['scheduler step'], gamma=configuration['Opt']['scheduler gamma'])

if configuration['Model']['arch']=='fno':
    loss_func = LpLoss(size_average=False)
else:
    loss_func = torch.nn.MSELoss()

epoch_init = 0
epochs = configuration['Opt']['epochs']

#Restarting the run from a checkpoint 
if configuration['Train']['restart'] == True: 
    # client.get_artifact_as_file(client.get_artifact_as_file(configuration['Train']['restart run name']))
    ckpt_path = '/tmp/checkpoint.pt'
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    epoch_init = checkpoint["epoch"]

# %% 
####################################
#Training
####################################

rollout_length = configuration['Train']['rollout_length']
step = configuration['Data']['step']
start_time = default_timer()
for ep in tqdm(range(epoch_init, epochs+1)): #Training Loop - Epochwise

    model.train()
    t1 = default_timer()

    for xx, yy in train_loader:
        optimizer.zero_grad()
        loss = 0
        xx = xx.to(device)
        yy = yy.to(device)
        batch_size = xx.shape[0]

        for t in range(0, rollout_length, step):
            y = yy[..., t:t + step]
            im = forward(model, xx, dt)

            #Recon Loss
            loss += loss_func(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), -1)

            xx = torch.cat((xx[..., step:], im), dim=-1)

        l2_full = loss_func(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1))
        train_l2_full += l2_full.item()

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)

        optimizer.step()        
        t2 = default_timer()
    
    train_loss = train_l2_full 

    train_loss = train_loss / len(train_loader)
    test_loss = test_loss / len(test_loader)

    print(f"Epoch {ep}, Time Taken: {round(t2-t1,3)}, Train Loss: {round(train_loss, 5)}, Test Loss: {round(test_loss,5)}")
    current_lr = optimizer.param_groups[0]['lr']
    # run.log_metrics({'Train Loss': train_loss, 'Test Loss': test_loss, 'Learning Rate': current_lr})

    
    # run.create_alert(
    #     name='Unstable',
    #     source='metrics',
    #     rule='is above',
    #     metric='Train Loss',
    #     frequency=1,
    #     window=1,
    #     threshold=1e5,
    #     trigger_abort=True
    #     )
                
    scheduler.step()

    # #Checkpointing. 
    # if ep % configuration['Train']['checkpoint']['epochs'] == 0:
    #     checkpoint = {}
    #     checkpoint["model"] = model.state_dict()
    #     checkpoint["optimizer"] = optimizer.state_dict() 
    #     checkpoint["scheduler"] = scheduler.state_dict()
    #     checkpoint["epoch"] = ep
    #     # torch.save(checkpoint, model_loc + "/checkpoint_"+str(ep)+".pt")
    #     # run.save_file(model_loc + "/checkpoint_"+str(ep)+".pt", 'output')
    #     # run.update_metadata({'Epochs': ep})

train_time = default_timer() - start_time

# %%
# # Saving the Model
# saved_model = model_loc + '/model.pth'
# torch.save(model.state_dict(), saved_model)
# run.save_file(saved_model, 'output')

pred_set = []
index = 0
with torch.no_grad():
    for xx, yy in tqdm(test_loader):
        loss = 0
        xx, yy = xx.to(device), yy.to(device)
        for t in range(0, configuration['Data']['t_out']-1, step):
            y = yy[..., t:t + step]
            out = forward(model, xx, dt)

            if t == 0:
                pred = out
            else:
                pred = torch.cat((pred, out), -1)

            xx = torch.cat((xx[..., step:], out), dim=-1)

        pred_set.append(pred)
        index += 1

    pred_encoded = torch.cat(pred_set, dim=0)
    error = (pred_encoded - test_out.to(device)).pow(2).mean()


print('(MSE) Testing Error: %.3e' % (error))

# run.update_metadata({'Training Time': float(train_time),
#                     'MSE Test Error': float(error)
#                     })

#Denormalising the test and predictions
test_out = normalizer.decode(test_out.to(device)).cpu()
pred_set = normalizer.decode(pred_encoded.to(device)).cpu()

#Shaping back to [BS, vars, Nt, Nx, Ny]
test_out = test_out.permute(0,1,4,2,3)
pred_set = pred_set.permute(0,1,4,2,3)

# %% 
#Plotting the results 
from Utils.plots import plots_2d_yaml
idx = 0 
plots_2d_yaml(configuration, test_out, pred_set, plot_loc, run, idx, save=False)
# %%
# run.close()
# %%