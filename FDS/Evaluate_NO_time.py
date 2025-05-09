#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluating Trained Models using simvue's client API. 
"""
# %%
#Specifying the run instance
run_name = 'prepared-music'

class Run:
    def __init__(self, name=None):
        self.name = name

run = Run(run_name)
# %% 
#Setting up simvue 
import os
import yaml 
import sys
import shutil
from pathlib import Path

sys.path.append("..")
from simvue import Client
client = Client()

#Setting up locations. 
file_loc = os.getcwd()
data_loc = os.path.dirname(os.getcwd()) + '/Data/'
model_loc = file_loc + '/Weights/' + run_name
plot_loc = file_loc + '/Plots'
tmp_loc = os.getcwd() + '/tmp'
try: 
    shutil.rmtree(tmp_loc)
    os.mkdir(tmp_loc)
except:
    pass

# %%
#Loading the yaml file to get the configuration.
run_id = client.get_run_id_from_name(run_name)
client.get_artifacts_as_files(run_id, contains='yaml', path=tmp_loc)
configuration = yaml.safe_load(open(next(Path(tmp_loc).glob('*.yaml'))))
# client.get_run(run_id)
# %% 
#Importing the necessary packages
import sys
import numpy as np
from tqdm import tqdm 
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from timeit import default_timer
from tqdm import tqdm 

#Setting up the seeds and devices
torch.manual_seed(0)
np.random.seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float32)

# %%
#Importing the models and utilities. 

from Neural_PDE.Utils.processing_utils import * 
from Neural_PDE.Utils.training_utils import * 

# %% 
# Data Preparation. - Only preparing the evaluation dataset : 20% of the data.
####################################

t1 = default_timer()
n_sims = int(configuration['Data']['ntrain']*configuration['Data']['test-train-split'])
configuration['Data']['ntrain'] = n_sims
from data_loaders import *
pde = configuration['Physics']['pde']
if pde == 'FDS':
    fields, x, y, z, t, dt, fire_loc, vent_open_time = FDS_Carpark(configuration)

# %%
# #Normalising the data -- Taking the normalisation from the trained run. 
# client.get_artifact_as_file(client.get_run_id_from_name(run_name), 'norms.npz', path=tmp_loc)
# norms = np.load(tmp_loc +'/norms.npz')
# normalizer_func = Normalisation(configuration['Data']['normalisation'])
# normalizer = normalizer_func(torch.tensor(0))
# normalizer.a, normalizer.b = torch.tensor(norms['a']), torch.tensor(norms['b'])

#Normalising the data -- using the same normalisations for inputs and outputs
normalizer_func = Normalisation(configuration['Data']['normalisation'])
normalizer = normalizer_func(fields)
fields_encoded = normalizer.encode(fields)
# fields_encoded = fields

# %% 
from sklearn.model_selection import train_test_split
train_in, test_in, train_out, test_out = train_test_split(fields_encoded[...,:configuration['Data']['t_in']], fields_encoded[...,configuration['Data']['t_in']:configuration['Data']['t_out']], test_size=0.8, random_state=42)

print("Test Input: " + str(test_in.shape))
print("Test Output: " + str(test_out.shape))

test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_in, test_out), batch_size=configuration['Data']['batch size'], shuffle=False)

t2 = default_timer()
print('preprocessing finished, time used:', t2-t1)

# %% 
# Setting up the Model and Optimizers 
####################################
from model_setup import * 
model = model_initialisation(configuration, run=None)

# #Loading the checkpoint
# client.get_artifact_as_file(client.get_run_id_from_name(run_name), 'checkpoint.pt', path=tmp_loc)
# ckpt_path = tmp_loc + '/checkpoint.pt'
# checkpoint = torch.load(ckpt_path, map_location=device)
# model.load_state_dict(checkpoint["model"])
# epoch_last = checkpoint["epoch"]

#Loading the trained model
client.get_artifact_as_file(client.get_run_id_from_name(run_name), 'model.pth', path=tmp_loc)
model_path = tmp_loc + '/model.pth'
model.load_state_dict(torch.load(model_path, map_location='cpu'))

model.to(device)
# %%
from Utils import explicit_time

#Evaluation 
eval = explicit_time.Eval_Setup(model, test_in, test_out, normalizer='False', ode_solver = configuration['Train']['odesolve']['source'], roll_out= configuration['Train']['odesolve']['method'])
pred_encoded, error = eval.inference(configuration['Data']['step'], configuration['Data']['t_out']-1, dt=dt)

print('(MSE) Testing Error: %.3e' % (error))

#Denormalising the test and predictions
test_set = normalizer.decode(test_out.to(device)).cpu()
pred_set = normalizer.decode(pred_encoded.to(device)).cpu()

# #Visualising the rollout error 
# from Utils.plots import temporal_rollout_error
# temporal_rollout_error(configuration, test_out, pred_set, tmp_loc, run, save=False)

# %% 
#Shaping back to [BS, vars, Nt, Nx, Ny]
test_set = test_set.permute(0,1,4,2,3)
pred_set = pred_set.permute(0,1,4,2,3)

# %% 
#Visualising the results
from Utils.plots import plots_2d_yaml
idx = 0
plots_2d_yaml(configuration, test_set, pred_set, tmp_loc, run, idx, save=False)

# %% 