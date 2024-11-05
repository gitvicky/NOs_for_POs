#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluating Trained Models
"""
# %%
#Setting up simvue 
import os
import yaml 

import sys
sys.path.append("..")
from simvue import Client

# %% 
#Loading the Run Config from simvue
run_name = 'tattered-strategy'

client = Client()
#Setting up locations. 
file_loc = os.getcwd()
data_loc = os.path.dirname(os.getcwd()) + '/Data/'
model_loc = file_loc + '/Weights/' + run_name
plot_loc = file_loc + '/Plots'
tmp_loc = os.getcwd() + '/tmp'
try: 
    os.mkdir(tmp_loc)
except:
    pass

client.get_artifact_as_file(client.get_run_id_from_name(run_name), 'NS_spectral_FNO.yaml', path=tmp_loc)
config_loc = tmp_loc + '/NS_spectral_FNO.yaml'
configuration = yaml.safe_load(open(config_loc))

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

if configuration['Model']['arch'] == 'FNO':
    from Neural_PDE.Models.FNO import *
elif configuration['Model']['arch'] == 'ViT':
    from Neural_PDE.Models.ViT import * 
# elif configuration['Model']['arch'] == 'CNO':
#     from Neural_PDE.Models.CNO import * 

from Neural_PDE.Utils.processing_utils import * 
from Neural_PDE.Utils.training_utils import * 

# %% 
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

fields = fields[...,:configuration['Data']['t_out']]

#Making sure the data is in the correct format: [BS, N_vars, Nx, Ny, Nt]
expected_shape = (configuration['Data']['ntrain'], configuration['Physics']['variables'], configuration['Physics']['Nx'], configuration['Physics']['Ny'], configuration['Data']['t_out'])
assert fields.shape == expected_shape, \
    f"Expected fields shape to be {expected_shape}, but got {fields.shape}"

# %%
#Normalising the data -- Taking the normalisation from the trained run. 
client.get_artifact_as_file(client.get_run_id_from_name(run_name), 'norms.npz', path=tmp_loc)
norms = np.load(tmp_loc +'/norms.npz')

normalizer_func = Normalisation(configuration['Data']['normalisation'])
normalizer = normalizer_func(torch.tensor(0))
normalizer.a, normalizer.b = torch.tensor(norms['a']), torch.tensor(norms['b'])

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
print("Number of model params : " + str(model.count_params()))

#Loading the checkpoint
client.get_artifact_as_file(client.get_run_id_from_name(run_name), 'checkpoint.pt', path=tmp_loc)
ckpt_path = tmp_loc + '/checkpoint.pt'
checkpoint = torch.load(ckpt_path, map_location=device)
model.load_state_dict(checkpoint["model"])
epoch_last = checkpoint["epoch"]

# %%
from Utils import explicit_time
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
idx = 10
plots_2d_yaml(configuration, test_out, pred_set, plot_loc, run_name, idx, save=False)
# %%
