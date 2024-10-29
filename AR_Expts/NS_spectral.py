#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Navier-Stokes Spectral solver - NeuralPDE library 

"""

#%%
#FNO
configuration = {"Case": 'Navier-Stokes',
                 "Field": 'u, v, p',
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
                 "Nx": 100,
                 "Ny": 100,
                 "Nt": 50, 
                 "T_in": 1,    
                 "T_out": 20,
                 "Step": 1,
                 "Width": 16, 
                 "Modes": 8,
                 "Variables": 3, 
                 "Loss Function": 'LP',
                 "POs": False,
                 "Rollout": 'AR'
                 }

# #ViT
# configuration = {"Case": 'Incomp. Navier-Stokes',
#                  "Field": 'u, v',
#                  "Model": 'ViT',
#                  "Epochs": 1,
#                  "Batch Size": 5,
#                  "Optimizer": 'Adam',
#                  "Learning Rate": 0.005,
#                  "Scheduler Step": 100,
#                  "Scheduler Gamma": 0.5,
#                  "Activation": 'GeLU',
#                  "Physics Normalisation": 'No',
#                  "Normalisation Strategy": 'Min-Max',
#                  "Nx": 128,
#                  "Ny": 128,
#                  "Nt": 100, 
#                  "T_in": 1,    
#                  "T_out": 50,
#                  "Step": 1,
#                  "Patch Size": 16,
#                  "Embedded Dim": 128,
#                  "Depth": 4,
#                  "Heads": 4,
#                  "Variables": 2, 
#                  "Loss Function": 'LP',
#                  "POs": False,
#                  "Rollout": 'Autoregressive'
#                  }

# %%
#Setting up simvue 
import os
from simvue import Run
run = Run(mode='online')
run.init(folder="/Neural_PDE/tests", tags=['NPDE', configuration['Model'], 'POs4NOs', 'NS_incomp', configuration['Rollout'], 'Tests'], metadata=configuration)

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
if configuration['Model'] == 'FNO':
    from Neural_PDE.Models.FNO import *
elif configuration['Model'] == 'ViT':
    from Neural_PDE.Models.ViT import * 

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


#Testing with NS_Spectral (for now)
data_loc = '/home/ir-gopa2/rds/rds-ukaea-ap001/ir-gopa2/Code/Neural_PDE/Data'
data =  np.load(data_loc + '/NS_Spectral_combined.npz')

u = data['u'].astype(np.float32)
v = data['v'].astype(np.float32)
p = data['p'].astype(np.float32)

def stacked_fields(variables):
    stack = []
    for var in variables:
        var = torch.from_numpy(var) #Converting to Torch
        var = var.permute(0, 2, 3, 1) #Permuting to be BS, Nx, Ny, Nt
        stack.append(var)
    stack = torch.stack(stack, dim=1)
    return stack

uv = stacked_fields([u,v,p])[:400]

# %%
#Normalising the data -- using the same normalisations for inputs and outputs
normalizer_func = Normalisation(configuration['Normalisation Strategy'])
normalizer = normalizer_func(uv)
uv_encoded = normalizer.encode(uv)

#Setting up train and test
from sklearn.model_selection import train_test_split
train_in, test_in, train_out, test_out = train_test_split(uv_encoded[...,:configuration['T_in']], uv_encoded[...,configuration['T_in']:configuration['T_out']], test_size=0.5, random_state=42)
print("Training Input: " + str(train_in.shape))
print("Training Output: " + str(train_out.shape))

#Saving Normalisation 
saved_normalisations = model_loc + '/' + configuration['Model'] + '_' + configuration['Case'] + '_' + run.name + '_' + 'norms.npz'
np.savez(saved_normalisations, 
        in_a=normalizer.a.numpy(), in_b=normalizer.b.numpy(), 
        )
run.save_file(saved_normalisations, 'output')

#Setting up the data loaders
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_in, train_out), batch_size=configuration['Batch Size'], shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_in, test_out), batch_size=configuration['Batch Size'], shuffle=False)

t2 = default_timer()
print('preprocessing finished, time used:', t2-t1)

# %%
# #Using the normalisations from the previous setup 

# norm_strategy = configuration['Normalisation Strategy']

# if norm_strategy == 'Min-Max':
#     normalizer = MinMax_Normalizer
# elif norm_strategy == 'Range':
#     normalizer = RangeNormalizer
# elif norm_strategy == 'Gaussian':
#     normalizer = GaussianNormalizer

# #Setting up train and test
# ntrain = 200
# ntest = 200
# train_a = uv[:ntrain,...,:configuration['T_in']]
# train_u = uv[:ntrain,...,configuration['T_in']:configuration['T_out']+configuration['T_in']]

# test_a = uv[-ntest:,...,:configuration['T_in']]
# test_u = uv[-ntest:,...,configuration['T_in']:configuration['T_out']+configuration['T_in']]

# a_normalizer = normalizer(train_a)
# u_normalizer = normalizer(train_u)

# train_in = a_normalizer.encode(train_a)
# test_in = a_normalizer.encode(test_a)

# train_out = u_normalizer.encode(train_u)
# test_out = u_normalizer.encode(test_u)

# #Setting up the data loaders
# train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_in, train_out), batch_size=configuration['Batch Size'], shuffle=True)
# test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_in, test_out), batch_size=configuration['Batch Size'], shuffle=False)


# %% 
####################################
# Setting up the Model and Optimizers 
####################################


if configuration['Model'] == 'FNO':
    model = FNO_multi2d(configuration['T_in'], 
                        configuration['Step'], 
                        configuration['Modes'], 
                        configuration['Modes'], 
                        configuration['Variables'], 
                        configuration['Width']
                        )
    
if configuration['Model'] == 'ViT':
    model = VisionTransformer(
        img_size=(configuration['Nx'], configuration['Ny']),
        patch_size=(1, configuration['Patch Size'], configuration['Patch Size']),
        in_channels=configuration['Variables'],
        time_channels=configuration['Step'],
        out_channels=configuration['Variables'],
        embed_dim=configuration['Embedded Dim'],
        depth=configuration['Depth'],
        n_heads=configuration['Heads']
        )
      
model.to(device)
run.update_metadata({'Number of Params': int(model.count_params())})
print("Number of model params : " + str(model.count_params()))

#Setting up the optimizer and scheduler, loss and epochs 
optimizer = torch.optim.Adam(model.parameters(), lr=configuration['Learning Rate'], weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=configuration['Scheduler Step'], gamma=configuration['Scheduler Gamma'])
loss_func = LpLoss(size_average=False)
epochs = configuration['Epochs']

#Setting up the Training pipeline
from Utils import explicit_time
train = explicit_time.Train_Setup(model, train_loader, test_loader, loss_func, optimizer, scheduler, epochs,  configuration['Rollout'])

# %% 
####################################
#Training
####################################

start_time = default_timer()
for ep in range(epochs): #Training Loop - Epochwise

    model.train()
    t1 = default_timer()
    train_loss, test_loss = train.one_epoch(configuration['Step'], configuration['T_out']-1)
    t2 = default_timer()

    train_loss = train_loss / len(train_loader)
    test_loss = test_loss / len(test_loader)

    print(f"Epoch {ep}, Time Taken: {round(t2-t1,3)}, Train Loss: {round(train_loss, 3)}, Test Loss: {round(test_loss,3)}")
    run.log_metrics({'Train Loss': train_loss, 'Test Loss': test_loss})
    
    scheduler.step()

train_time = default_timer() - start_time

# %%
#Saving the Model
# saved_model = model_loc + '/' + configuration['Model'] + '_' + configuration['Case'] + '_' +run.name + '.pth'
# torch.save( model.state_dict(), saved_model)
# run.save_file(saved_model, 'output')

#Evaluation 
eval = explicit_time.Eval_Setup(model, test_in, test_out)
pred_encoded, error = eval.inference(configuration['Step'], configuration['T_out']-1)

print('(MSE) Testing Error: %.3e' % (error))

run.update_metadata({'Training Time': float(train_time),
                     'MSE Test Error': float(error)
                    })

#Denormalising the test and predictions
test_out = normalizer.decode(test_out.to(device)).cpu()
pred_set = normalizer.decode(pred_encoded.to(device)).cpu()


if configuration['Model'] == 'FNO':
    test_out = test_out.permute(0,1,4,2,3)
    pred_set = pred_set.permute(0,1,4,2,3)

if configuration['Model'] == 'ViT':
    test_out = test_out.permute(0,1,4,2,3)
    pred_set = pred_set.permute(0,1,4,2,3)


# %% 
#Plotting the results 
from Utils.plots import plots_2d
idx = 0 
plots_2d(configuration, test_out, pred_set, plot_loc, run, idx)

# %%
run.close()
# %%
