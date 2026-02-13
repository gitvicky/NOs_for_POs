#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluating Trained Models using simvue's client API. 
"""
# %%
#Specifying the run instance
run_name = 'isometric-chenille'
test_data_name = None
t_extrapolation = 50
# %%
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

# Create tmp directory if it doesn't exist, or recreate it if it does
if os.path.exists(tmp_loc):
    shutil.rmtree(tmp_loc)
os.makedirs(tmp_loc, exist_ok=True)

# %%
#Loading the yaml file to get the configuration.
run_id = client.get_run_id_from_name(run_name)
client.get_artifacts_as_files(run_id, category='code', output_dir=tmp_loc)
configuration = yaml.safe_load(open(next(Path(tmp_loc).glob('*.yaml'))))

# %% 
#Setting the experiment parameters 
if t_extrapolation != None:
    configuration['Data']['t_out'] = t_extrapolation

if test_data_name !=None:
    pde=test_data_name
else:
    pde = configuration['Physics']['pde'] 

print(configuration['Data']['t_out'])
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
# Data Preparation. - Only preparing the evaluation dataset. 
####################################

t1 = default_timer()
n_sims = int(configuration['Data']['ntrain']*configuration['Data']['test_train_split'])
configuration['Data']['ntrain'] = n_sims
from data_loaders import *

if pde == 'Wave':
    fields, x, y, dt = Wave_Spectral(configuration)
if pde == 'Navier-Stokes':
    fields, x, y, dt = Navier_Stokes_Spectral(configuration)
if pde == 'Euler-Fluid':
    fields, x, y, dt = Euler_FV(configuration)
if pde == 'Incomp. Navier-Stokes':
    fields, force, x, y, dt = Navier_Stokes_Incomp(configuration)
if pde == 'Comp. Navier-Stokes':
    fields, x, y, dt = Navier_Stokes_Comp(configuration, coeff=configuration['Physics']['coeff'])
if pde == 'Electrostatic MHD':
    if configuration['Physics']['source'] == 'JOREK': 
        fields, x, y, dt = JOREK_electrostatic(configuration)
if pde == 'Electromagnetic MHD':
    if configuration['Physics']['source'] == 'JOREK': 
        fields, x, y, dt = JOREK_electrostatic(configuration)

if pde == 'Shear Flow':
    configuration['Data']['reynolds'] =  ['1e4'] 
    configuration['Data']['schmidt'] =  ['1e-1', '1e1', '5e-1'] #train
    configuration['Data']['source'] = 'The Well' 
    configuration['Data']['ntrain'] = 96
    Nx, Ny, Nt = 256, 512, 200
    x_slice, y_slice, t_slice = 2,2,2
    configuration['Physics']['Nx'], configuration['Physics']['Ny'], configuration['Physics']['Nt'] = Nx, Ny, Nt
    configuration['Physics']['x_slice'], configuration['Physics']['y_slice'], configuration['Physics']['t_slice'] = x_slice, y_slice, t_slice
    fields, x, y, dt = Shear_Flow(configuration)

if pde == 'Euler Quadrant':
    configuration['Data']['source'] = 'The Well' 
    configuration['Data']['ntrain'] = 50
    Nx, Ny, Nt = 512, 512, 101
    x_slice, y_slice, t_slice = 4, 4, 1
    configuration['Physics']['Nx'], configuration['Physics']['Ny'], configuration['Physics']['Nt'] = Nx, Ny, Nt
    configuration['Physics']['x_slice'], configuration['Physics']['y_slice'], configuration['Physics']['t_slice'] = x_slice, y_slice, t_slice
    fields, x, y, dt = Euler_Quadrants(configuration)

t = torch.arange(0, fields.shape[-1], dt)
fields = fields[...,:configuration['Data']['t_out']]

#Making sure the data is in the correct format: [BS, N_vars, Nx, Ny, Nt]
expected_shape = (configuration['Data']['ntrain'], configuration['Physics']['variables'], configuration['Physics']['Nx']//configuration['Physics']['x_slice'], configuration['Physics']['Ny']//configuration['Physics']['y_slice'], configuration['Data']['t_out'])
assert fields.shape == expected_shape, \
    f"Expected fields shape to be {expected_shape}, but got {fields.shape}"

#Printing the current dictionary
print(yaml.dump(configuration, default_flow_style=False, indent=2))
# %%
#Normalising the data -- Taking the normalisation from the trained run. 
client.get_artifact_as_file(client.get_run_id_from_name(run_name), name='norms.npz', output_dir=tmp_loc)
norms = np.load(tmp_loc +'/norms.npz')

normalizer_func = Normalisation(configuration['Data']['normalisation'])
normalizer = normalizer_func(torch.zeros_like(fields))
normalizer.a, normalizer.b = torch.tensor(norms['a']), torch.tensor(norms['b'])
# normalizer.a, normalizer.b = torch.tensor(1.3769), torch.tensor(-0.3768)
fields_encoded = normalizer.encode(fields)
# fields_encoded = fields

# %% 

test_in = fields_encoded[...,:configuration['Data']['t_in']] # + torch.randn_like(fields_encoded[...,:configuration['Data']['t_in']])
test_out = fields_encoded[...,configuration['Data']['t_in']:configuration['Data']['t_out']]

print("Test Input: " + str(test_in.shape))
print("Test Output: " + str(test_out.shape))

test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_in, test_out), batch_size=configuration['Data']['batch_size'], shuffle=False)

t2 = default_timer()
print('preprocessing finished, time used:', t2-t1)

# %% 
# Setting up the Model and Optimizers 
####################################
from model_setup import * 
model = model_initialisation(configuration, normalizer, run=None)

# #Loading the checkpoint
# client.get_artifact_as_file(client.get_run_id_from_name(run_name), 'checkpoint.pt', path=tmp_loc)
# ckpt_path = tmp_loc + '/checkpoint.pt'
# checkpoint = torch.load(ckpt_path, map_location=device)
# model.load_state_dict(checkpoint["model"])
# epoch_last = checkpoint["epoch"]

#Loading the trained model
client.get_artifact_as_file(client.get_run_id_from_name(run_name), name='model.pth', output_dir=tmp_loc)
model_path = tmp_loc + '/model.pth'
model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=False), strict=False)

model.to(device)
print("Number of model params : " + str(model.count_params()))
# %%
from Utils import explicit_time

#Evaluation 
eval = explicit_time.Eval_Setup(model, test_in, test_out, normalizer='False', ode_solver = configuration['Train']['odesolve']['source'], roll_out= configuration['Train']['odesolve']['method'])
pred_encoded, error = eval.inference(configuration['Data']['step'], configuration['Data']['t_out']-1, dt=dt)

print(f'MSE (norm) : {float(error):.4e}')

#Denormalising the test and predictions
test_out = normalizer.decode(test_out.to(device)).cpu()
pred_set = normalizer.decode(pred_encoded.to(device)).cpu()

# #Visualising the rollout error 
# from Utils.plots import temporal_rollout_error
# temporal_rollout_error(configuration, test_out, pred_set, tmp_loc, run, save=False)


# %% 
#Shaping back to [BS, vars, Nt, Nx, Ny]
test_out = test_out.permute(0,1,4,2,3)
pred_set = pred_set.permute(0,1,4,2,3)

# %% 
#Getting the Metrics
from Utils.metrics import MSE, NRMSE
print(f'NRMSE (physical) : {float(NRMSE(pred_set, test_out)["average"]):.4e}')
# %% 
#Visualising the results
from Utils.plots import plots_2d_yaml
idx = 3
plots_2d_yaml(configuration, test_out, pred_set, tmp_loc, run, idx, save=True)
# %% 