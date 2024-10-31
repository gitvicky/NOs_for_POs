#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluating Trained Models
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
config_loc = os.getcwd() + '/configs/NS_spectral_FNO.yaml'
configuration = yaml.safe_load(open(config_loc))
run_config = flatten_dict(configuration)
# %% 
from simvue import Run, Client
run = Run(mode='disabled')

run.init(folder=configuration['Simvue']['folder'], tags=['NPDE', configuration['Model']['arch'], 'POs4NOs', configuration['Physics']['pde'], configuration['Physics']['rollout'], 'Tests'], metadata=run_config)

#setting up the client API 
client = Client()

#Saving the current run file and the git hash of the repo
run.save_file(os.path.abspath(__file__), 'code')
# run.save_file(os.path.abspath(args.config), 'code')

import git
repo = git.Repo(search_parent_directories=True)
sha = repo.head.object.hexsha
run.update_metadata({'Git Hash': sha})

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

if configuration['Model']['arch'] == 'FNO':
    from Neural_PDE.Models.FNO import *
elif configuration['Model']['arch'] == 'ViT':
    from Neural_PDE.Models.ViT import * 
# elif configuration['Model']['arch'] == 'CNO':
#     from Neural_PDE.Models.CNO import * 

from Neural_PDE.Utils.processing_utils import * 
from Neural_PDE.Utils.training_utils import * 

# %% 
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
####################################
# Data Preparation.
####################################

t1 = default_timer()

from data_loaders import *
pde = configuration['Physics']['pde']
if pde == 'Navier-Stokes':
    fields, x, y, dt = Navier_Stokes_Spectral(configuration['Data']['ntrain'])
if pde == 'Incomp. Navier-Stokes':
    fields, x, y, dt = Navier_Stokes_Incomp(configuration['Data']['ntrain'])
if pde == 'MHD':
    if configuration['Physics']['pde']['source'] == 'JOREK':
        fields, x, y, dt = JOREK(configuration['Data']['ntrain'])

fields = fields[:,:,:configuration['Data']['t_out']]
# %%
#Normalising the data -- using the same normalisations for inputs and outputs
normalizer_func = Normalisation(configuration['Data']['normalisation'])
normalizer = normalizer_func(fields)
fields_encoded = normalizer.encode(fields)

#Setting up train and test
from sklearn.model_selection import train_test_split
train_in, test_in, train_out, test_out = train_test_split(fields_encoded[...,:configuration['Data']['t_in']], fields_encoded[...,configuration['Data']['t_in']:configuration['Data']['t_out']], test_size=configuration['Data']['test-train-split'], random_state=42)
print("Training Input: " + str(train_in.shape))
print("Training Output: " + str(train_out.shape))

#Setting up the data loaders
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_in, train_out), batch_size=configuration['Data']['batch size'], shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_in, test_out), batch_size=configuration['Data']['batch size'], shuffle=False)

t2 = default_timer()
print('preprocessing finished, time used:', t2-t1)

# %% 
####################################
# Setting up the Model and Optimizers 
####################################

if configuration['Model']['arch'] == 'FNO':
    model = FNO_multi2d(configuration['Data']['t_in'], 
                        configuration['Data']['step'], 
                        configuration['Model']['modes'], 
                        configuration['Model']['modes'], 
                        configuration['Physics']['variables'], 
                        configuration['Model']['width']
                        )

model.to(device)
run.update_metadata({'Number of Params': int(model.count_params())})
print("Number of model params : " + str(model.count_params()))

#Setting up the optimizer and scheduler, loss and epochs 
optimizer = torch.optim.Adam(model.parameters(), lr=configuration['Opt']['learning rate'], weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=configuration['Opt']['scheduler step'], gamma=configuration['Opt']['scheduler gamma'])
loss_func = LpLoss(size_average=False)
epoch_init = 0
epochs = configuration['Opt']['epochs']

#Restarting the run from a checkpoint 
run_name = ''
client.get_artifacts_as_files(run_id=client.get_run_id_from_name(run_name), path='/tmp', category='output')
ckpt_path = '/tmp/checkpoint.pt'
checkpoint = torch.load(ckpt_path)
model.load_state_dict(checkpoint["model"])
optimizer.load_state_dict(checkpoint["optimizer"])
scheduler.load_state_dict(checkpoint["scheduler"])
epoch_init = checkpoint["epoch"]

#Setting up the Training pipeline
from Utils import explicit_time
train = explicit_time.Train_Setup(model, train_loader, test_loader, loss_func, optimizer, scheduler, epochs,  configuration['Physics']['rollout'])

#Evaluation 
eval = explicit_time.Eval_Setup(model, test_in, test_out, roll_out=configuration['Physics']['rollout'])
pred_encoded, error = eval.inference(configuration['Data']['step'], configuration['Data']['t_out']-1, dt=dt)

print('(MSE) Testing Error: %.3e' % (error))

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
plots_2d_yaml(configuration, test_out, pred_set, plot_loc, run, idx)
# %%
run.close()
# %%
