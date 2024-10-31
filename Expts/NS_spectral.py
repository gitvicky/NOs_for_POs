#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Navier-Stokes Spectral solver - NeuralPDE library 
"""
# %%
#Setting up simvue 
import os
import yaml 
import argparse
from omegaconf import DictConfig, OmegaConf

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
run = Run(mode='online')
run.init(folder=configuration['Simvue']['folder'], tags=['NPDE', configuration['Model']['arch'], 'POs4NOs', configuration['Physics']['pde'], configuration['Physics']['rollout'], 'Tests'], metadata=run_config)

#setting up the client API 
client = Client()

#Saving the current run file and the git hash of the repo
run.save_file(os.path.abspath(__file__), 'code')
run.save_file(os.path.abspath(args.config), 'code')

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
fields, x, y, dt = Navier_Stokes_Spectral(configuration['Data']['ntrain'])
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

#Saving Normalisation 
saved_normalisations = model_loc + '/' + configuration['Model']['arch'] + '_' + configuration['Physics']['pde'] + '_' + run.name + '_' + 'norms.npz'
np.savez(saved_normalisations, 
        in_a=normalizer.a.numpy(), in_b=normalizer.b.numpy(), 
        )
run.save_file(saved_normalisations, 'output')

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
if configuration['Train']['restart']:
    client.get_artifact_as_file(client.get_artifact_as_file(configuration['Train']['restart run name']))
    ckpt_path = '/tmp/checkpoint.pt'
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    epoch_init = checkpoint["epoch"]

#Setting up the Training pipeline
from Utils import explicit_time
train = explicit_time.Train_Setup(model, train_loader, test_loader, loss_func, optimizer, scheduler, epochs,  configuration['Physics']['rollout'])

# %% 
####################################
#Training
####################################

start_time = default_timer()
for ep in range(epoch_init, epochs): #Training Loop - Epochwise

    model.train()
    t1 = default_timer()
    train_loss, test_loss = train.one_epoch(configuration['Data']['step'], configuration['Data']['t_out']-1, dt=dt)
    t2 = default_timer()

    train_loss = train_loss / len(train_loader)
    test_loss = test_loss / len(test_loader)

    print(f"Epoch {ep}, Time Taken: {round(t2-t1,3)}, Train Loss: {round(train_loss, 3)}, Test Loss: {round(test_loss,3)}")
    run.log_metrics({'Train Loss': train_loss, 'Test Loss': test_loss})
    
    scheduler.step()

    #Checkpointing. 
    if ep % configuration['Train']['checkpoint']['epochs'] ==0:
        checkpoint = {}
        checkpoint["model"] = model.module.state_dict()
        checkpoint["optimizer"] = optimizer.state_dict() 
        checkpoint["scheduler"] = scheduler.state_dict()
        checkpoint["epoch"] = ep
        torch.save(checkpoint, model_loc + os.path.join(model_loc, "checkpoint.pt"))
        run.save_file(model_loc+ "/checkpoint.pt", 'output')
        run.update_metadata({'Epochs': ep})

train_time = default_timer() - start_time

# %%
# Saving the Model
saved_model = model_loc + '/model.pth'
torch.save( model.state_dict(), saved_model)
run.save_file(saved_model, 'output')

#Evaluation 
eval = explicit_time.Eval_Setup(model, test_in, test_out, roll_out=configuration['Rollout'])
pred_encoded, error = eval.inference(configuration['Data']['step'], configuration['Data']['t_out']-1, dt=dt)

print('(MSE) Testing Error: %.3e' % (error))

run.update_metadata({'Training Time': float(train_time),
                     'MSE Test Error': float(error)
                    })

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
