#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Incompressible Navier-Stokes - PDEBench 
PDEBench Data subsampled in Data/data_extraction.py --> NS_incomp_velocity_Nt_Nx_Ny_N_vars
"""

# %%
configuration = {"Case": 'Incomp. Navier-Stokes',
                 "Field": 'u, v',
                 "Model": 'FNO',
                 "Epochs": 500,
                 "Batch Size": 5,
                 "Optimizer": 'Adam',
                 "Learning Rate": 0.005,
                 "Scheduler Step": 100,
                 "Scheduler Gamma": 0.5,
                 "Activation": 'GeLU',
                 "Physics Normalisation": 'No',
                 "Normalisation Strategy": 'Min-Max',
                 "Nx": 128,
                 "Ny": 128,
                 "Nt": 100, 
                 "T_in": 1,    
                 "T_out": 50,
                 "Step": 1,
                 "Width_time": 16, 
                 "Modes": 8,
                 "Variables": 2, 
                 "Loss Function": 'LP',
                 "POs": False,
                 "Rollout": 'Autoregressive'
                 }

#Setting up simvue 
import os
from simvue import Run
run = Run(mode='disabled')
run.init(folder="/Neural_PDE", tags=['NPDE', configuration['Model'], 'POs4NOs', 'NS_incomp.', configuration['Rollout'], 'Tests'], metadata=configuration)

#Saving the current run file and the git hash of the repo
run.save_file(os.path.abspath(__file__), 'code')
import git
repo = git.Repo(search_parent_directories=True)
sha = repo.head.object.hexsha
run.update_metadata({'Git Hash': sha})

# %% 
#Importing the necessary packages
import sys
import numpy as np
from tqdm import tqdm 
import h5py
import torch
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
import time 
from timeit import default_timer
from tqdm import tqdm 

# %%
#Importing the models and utilities. 
import sys
sys.path.append("..")
from Neural_PDE.Models.FNO import *
from Neural_PDE.Utils.processing_utils import * 
from Neural_PDE.Utils.training_utils import * 
from Neural_PDE.UQ.inductive_cp import * 

# %% 
#Setting up locations. 
file_loc = os.getcwd()
data_loc = os.path.dirname(os.getcwd()) + '/Data/'
model_loc = file_loc + '/Weights'
plot_loc = file_loc + '/Plots'
#Setting up the seeds and devices
torch.manual_seed(0)
np.random.seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float32)
# %% 
####################################
# Data Preparation.
####################################

t1 = default_timer()
uv = np.load(data_loc + 'NS_incomp_velocity_100_128_128.npy') #uv being the two compnoents of velocity
uv = torch.tensor(uv, torch.float)
field = ['u', 'v']
if configuration['FNO']: 
    uv = uv.permute(0, 4, 2, 3, 1)

#Normalising the data -- using the same normalisations for inputs and outputs
normalizer_func = Normalisation(configuration['Normalisation Strategy'])
normalizer = normalizer_func(uv)
uv_encoded = normalizer.encode(uv)

#Setting up train and test
from sklearn.model_selection import train_test_split
train_in, test_in, train_out, test_out = train_test_split(uv_encoded[...,:configuration['T_in']], uv_encoded[...,:configuration['T_in']:configuration['T_in']+configuration['T_out']], test_size=0.2, random_state=42)
print("Training Input: " + str(train_in.shape))
print("Training Output: " + str(train_out.shape))

#Saving Normalisation 
saved_normalisations = model_loc + '/' + configuration['Model'] + '_' + configuration['Case'] + '_' + run.name + '_' + 'norms.npz'
np.savez(saved_normalisations, 
        in_a=normalizer.a.numpy(), in_b=normalizer.b.numpy(), 
        )
run.save_file(saved_normalisations, 'output')

#Setting up the data loaders. 
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_in, train_out), batch_size=configuration['Batch Size'], shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_in, test_out), batch_size=configuration['Batch Size'], shuffle=False)

t2 = default_timer()
print('preprocessing finished, time used:', t2-t1)

# %% 
####################################
# Setting up the Model and Optimizers 
####################################

if configuration['Model'] == 'FNO':
    model = FNO_multi2d(configuration['T_in'], configuration['Step'], configuration['Modes'], configuration['Modes'], configuration['Num_vars'], configuration['Width_time'])

model.to(device)
run.update_metadata({'Number of Params': int(model.count_params())})
print("Number of model params : " + str(model.count_params()))

#Setting up the optimizer and scheduler, loss and epochs 
optimizer = torch.optim.Adam(model.parameters(), lr=configuration['Learning Rate'], weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=configuration['Scheduler Step'], gamma=configuration['Scheduler Gamma'])
loss_func = LpLoss(size_average=False)
epochs = configuration['Epochs']

# %% 
####################################
#Training
####################################
if configuration['Rollout'] == 'AR':
    train_one_epoch = train_one_epoch_AR

start_time = default_timer()
for ep in range(epochs): #Training Loop - Epochwise

    model.train()
    t1 = default_timer()
    train_loss, test_loss = train_one_epoch(model, train_loader, test_loader, loss_func, optimizer, configuration['Step'], configuration['T_out'])
    t2 = default_timer()

    train_loss = train_loss / len(train_loader)
    test_loss = test_loss / len(test_loader)

    print(f"Epoch {ep}, Time Taken: {round(t2-t1,3)}, Train Loss: {round(train_loss, 3)}, Test Loss: {round(test_loss,3)}")
    run.log_metrics({'Train Loss': train_loss, 'Test Loss': test_loss})
    
    scheduler.step()

train_time = default_timer() - start_time

# %%
#Saving the Model
saved_model = model_loc + '/' + configuration['Model'] + '_' + configuration['Case'] + '_' +run.name + '.pth'
torch.save( model.state_dict(), saved_model)
run.save_file(saved_model, 'output')

#Validation
if configuration['Rollout'] == 'Autoregressive':
    pred_encoded, mse, mae = validation_AR(model, test_in, test_out, configuration['Step'], configuration['T_out'])

print('(MSE) Testing Error: %.3e' % (mse))
print('(MAE) Testing Error: %.3e' % (mae))

run.update_metadata({'Training Time': float(train_time),
                     'MSE Test Error': float(mse),
                     'MAE Test Error': float(mae)
                    })

#Denormalising the test and predictions
test_out = normalizer.decode(test_out.to(device)).cpu()
pred_set = normalizer.decode(pred_encoded.to(device)).cpu()

if configuration['Model'] == 'FNO':
    test_out = test_out.permute(0,1,4,2,3)
    pred_set = pred_set.permute(0,1,4,2,3)

# %% 
#Plotting the results 
from utils import plots_2d
plots_2d(configuration, test_out, pred_set, plot_loc, run, idx=0)

# %%
run.close()