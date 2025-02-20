#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluating Trained Models using simvue's client API. 
"""
# %%
#Specifying the run instance
run_name = 'formal-fur'

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
client.get_artifacts_as_files(client.get_run_id_from_name(run_name), contains='yaml', path=tmp_loc)
configuration = yaml.safe_load(open(next(Path(tmp_loc).glob('*.yaml'))))

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
from data_loaders import *
pde = configuration['Physics']['pde']
if pde == 'Navier-Stokes':
    fields, x, y, dt = Navier_Stokes_Spectral(n_sims)
if pde == 'Incomp. Navier-Stokes':
    fields, force, x, y, dt = Navier_Stokes_Incomp(n_sims)
if pde == 'Comp. Navier-Stokes':
    fields, x, y, dt = Navier_Stokes_Comp(n_sims, coeff=configuration['Physics']['coeff'])
if pde == 'Electrostatic MHD':
    if configuration['Physics']['source'] == 'JOREK': 
        fields, x, y, dt = JOREK_electrostatic(n_sims)
if pde == 'Electromagnetic MHD':
    if configuration['Physics']['source'] == 'JOREK': 
        fields, x, y, dt = JOREK_electrostatic(n_sims)
        
t = torch.arange(0, fields.shape[-1], dt)

fields = fields[...,:configuration['Data']['t_out']]

#Making sure the data is in the correct format: [BS, N_vars, Nx, Ny, Nt]
expected_shape = (n_sims, configuration['Physics']['variables'], configuration['Physics']['Nx'], configuration['Physics']['Ny'], configuration['Data']['t_out'])
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

# %% 
test_in = fields_encoded[...,:configuration['Data']['t_in']]
test_out = fields_encoded[...,configuration['Data']['t_in']:configuration['Data']['t_out']]

print("Test Input: " + str(test_in.shape))
print("Test Output: " + str(test_out.shape))

test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_in, test_out), batch_size=configuration['Data']['batch size'], shuffle=False)

t2 = default_timer()
print('preprocessing finished, time used:', t2-t1)

# %% 
# Setting up the Model and Optimizers 
####################################
from model_setup import * 
model = model_initialisation(configuration)


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
print("Number of model params : " + str(model.count_params()))
# %%
from Utils import explicit_time

#Evaluation 
eval = explicit_time.Eval_Setup(model, test_in, test_out, normalizer='False', ode_solver = configuration['Train']['odesolve']['source'], roll_out= configuration['Train']['odesolve']['method'])
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
