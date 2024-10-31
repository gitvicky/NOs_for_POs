#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FNO modelled over 2D MHD Equations auto-regressively

"""

# %%
configuration = {"Case": 'JOREK',
                 "Field": 'rho, Phi, T',
                 "Model": 'FNO',
                 "Epochs": 500,
                 "Batch Size": 1,
                 "Optimizer": 'Adam',
                 "Learning Rate": 0.005,
                 "Scheduler Step": 100,
                 "Scheduler Gamma": 0.5,
                 "Activation": 'GeLU',
                 "Physics Normalisation": 'No',
                 "Normalisation Strategy": 'Min-Max',
                 "Nx": 100,
                 "Ny": 100,
                 "Nt": 200, 
                 "T_in": 1,    
                 "T_out": 50,
                 "Step": 1,
                 "Width": 16, 
                 "Modes": 8,
                 "Variables": 3, 
                 "Loss Function": 'LP',
                 "POs": False,
                 "Rollout": 'AR'
                 }


# %%
#Setting up simvue 
import os
from simvue import Run
run = Run(mode='disabled')
run.init(folder="/Neural_PDE/tests", tags=['NPDE', configuration['Model'], 'POs4NOs', 'JOREK', configuration['Rollout'], 'Tests'], metadata=configuration)

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
data = data_loc + '/JOREK_filtered.npz' 
# %%
field = configuration['Field']
field = ['rho', 'Phi', 'T']
num_vars = configuration['Variables']

rho = np.load(data)['rho'].astype(np.float32)[:200] / 1e20
phi = np.load(data)['Phi'].astype(np.float32)[:200] / 1e5
T = np.load(data)['T'].astype(np.float32)[:200] / 1e6

rho = np.nan_to_num(rho)
phi = np.nan_to_num(phi)
T = np.nan_to_num(T)

def stacked_fields(variables):
    stack = []
    for var in variables:
        var = torch.from_numpy(var) #Converting to Torch
        var = var.permute(0, 2, 3, 1) #Permuting to be BS, Nx, Ny, Nt
        stack.append(var)
    stack = torch.stack(stack, dim=1)
    return stack

vars = stacked_fields([rho,phi,T])

x_grid =  np.load(data)['Rgrid'].astype(np.float32)
y_grid =  np.load(data)['Zgrid'].astype(np.float32)
t_grid =  np.load(data)['time'].astype(np.float32)

t_norm = t_grid / t_grid[-1]
dt = t_norm[1] - t_norm[0]


# %%
#Normalising the data -- using the same normalisations for inputs and outputs
normalizer_func = Normalisation(configuration['Normalisation Strategy'])
normalizer = normalizer_func(vars)
vars_encoded = normalizer.encode(vars)

#Setting up train and test
from sklearn.model_selection import train_test_split
train_in, test_in, train_out, test_out = train_test_split(vars_encoded[...,:configuration['T_in']], vars_encoded[...,configuration['T_in']:configuration['T_out']], test_size=0.2, random_state=42)
print("Training Input: " + str(train_in.shape))
print("Training Output: " + str(train_out.shape))

#Setting up the data loaders
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_in, train_out), batch_size=configuration['Batch Size'], shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_in, test_out), batch_size=configuration['Batch Size'], shuffle=False)

t2 = default_timer()
print('preprocessing finished, time used:', t2-t1)

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

#Loading the trained model:
if configuration['Rollout'] == 'AR':
    model.load_state_dict(torch.load(model_loc + '/FNO_JOREK_dull-involute.pth', map_location=device))
if configuration['Rollout'] == 'Euler':
    model.load_state_dict(torch.load(model_loc + '/FNO_JOREK_largo-shop.pth', map_location=device))
if configuration['Rollout'] == 'RK4':
    model.load_state_dict(torch.load(model_loc + '/FNO_JOREK_beige-hip.pth', map_location=device))

# %% 
from Utils import explicit_time
#Evaluation 

eval = explicit_time.Eval_Setup(model, test_in, test_out, roll_out=configuration['Rollout'])
pred_encoded, error = eval.inference(configuration['Step'], configuration['T_out']-1, eval_metric='MSE', dt=dt)

print('(MSE) Testing Error: %.3e' % (error))

#Denormalising the test and predictions
test_out_denorm = normalizer.decode(test_out.to(device)).cpu()
pred_set_denorm = normalizer.decode(pred_encoded.to(device)).cpu()

test_out_denorm = test_out_denorm.permute(0,1,4,2,3)
pred_set_denorm = pred_set_denorm.permute(0,1,4,2,3)

# %% 
#Plotting the results 
from Utils.plots import plots_2d
idx = 0
plots_2d(configuration, test_out_denorm, pred_set_denorm, plot_loc, run, idx, save=False)

# %%
run.close()
# %%
