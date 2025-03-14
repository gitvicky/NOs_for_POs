#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Learning the convection operator alone. 
"""
# %%
#Specifying the run instance
run_name = 'presto-sample'

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


#Setting up the seeds and devices
torch.manual_seed(0)
np.random.seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float32)

# %%
#Importing the models and utilities. 
from Expts.model_setup import *
from Neural_PDE.Utils.processing_utils import * 
from Neural_PDE.Utils.training_utils import * 

# from Tests.model_builder import build_model
# model = build_model(configuration)
# %% 
####################################
# Data Preparation.
####################################

t1 = default_timer()

from Expts.data_loaders import *
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

fields = fields.permute(0, 4, 1, 2, 3).reshape(fields.shape[0]*fields.shape[4], fields.shape[1], fields.shape[2], fields.shape[3])
# %%
#Normalising the data -- using the same normalisations for inputs and outputs
client.get_artifact_as_file(client.get_run_id_from_name(run_name), 'norms.npz', path=tmp_loc)
norms = np.load(tmp_loc +'/norms.npz')

normalizer_func = Normalisation(configuration['Data']['normalisation'])
normalizer = normalizer_func(torch.tensor(0))
normalizer.a, normalizer.b = torch.tensor(norms['a']), torch.tensor(norms['b'])

fields_encoded = normalizer.encode(fields)

train_data = fields_encoded[:int(0.8*fields_encoded.shape[0])]
test_data = fields_encoded[int(0.8*fields_encoded.shape[0]):]

print("Train Data: " + str(train_data.shape))
print("Test Data: " + str(test_data.shape))

#Setting up the data loaders
# train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_data), batch_size=configuration['Data']['batch size'], shuffle=False)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_data), batch_size=configuration['Data']['batch size'], shuffle=False)

t2 = default_timer()
print('preprocessing finished, time used:', t2-t1)

# %% 
####################################
# Setting up the Model and Optimizers 
####################################

from Tests.model_builder import build_model
model = build_model(configuration)
model.to(device)
#Loading the trained model
client.get_artifact_as_file(client.get_run_id_from_name(run_name), 'model.pth', path=tmp_loc)
model_path = tmp_loc + '/model.pth'
model.load_state_dict(torch.load(model_path, map_location='cpu'))
print("Number of model params : " + str(count_parameters(model)))
loss_func = torch.nn.MSELoss()

# %% 
####################################
#Testing
####################################
from PRE.VectorConvOps_Spatial import *
gradient = Gradient(scale=1, taylor_order=2, boundary_cond='periodic', device=device, requires_grad=True)

# laplace = Laplace(scale=1, taylor_order=2, boundary_cond='periodic', device=device, requires_grad=True)
# divergence = Divergence(scale=1, taylor_order=2, boundary_cond='periodic', device=device, requires_grad=True)

# def gradient_func(uv):
#     u = uv[:,0:1]
#     v = uv[:,1:2]
#     return gradient(u), gradient(v)

# def laplace_func(uv):
#     u = uv[:,0:1]
#     v = uv[:,1:2]
#     return laplace(u), laplace(v)

# def divergence_func(uv):
#     u = uv[:,0:1]
#     v = uv[:,1:2]
#     return divergence(u, v)

def convection(uv):
    with torch.no_grad():
        u = uv[:,0:1]
        v = uv[:,1:2]
        conv = dot(uv, gradient(u)) + dot(uv, gradient(v))
    return conv

def forward(model, uv):
    out = model(uv)
    out = out[:, 0:1] + out[:, 1:2]
    yy = convection(uv)
    return out, yy 


test = fields_encoded[-50:]
with torch.no_grad():

        xx = test.to(device)
        batch_size = xx.shape[0]

        im, y = forward(model, xx)

        #Recon Loss
        loss = loss_func(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

error = (im - y).pow(2).mean()


print('(MSE) Testing Error: %.3e' % (error))

#Denormalising the test and predictions
test_out = normalizer.decode(y.to(device)).cpu()
pred_set = normalizer.decode(im.to(device)).cpu()

# %% 
#Plotting the results 
from Utils.plots import plots_2d_yaml
idx = 0
configuration['Physics']['variables']=1
run = client.get_run_id_from_name(run_name)
plots_2d_yaml(configuration, test_out.unsqueeze(0).permute(0, 2, 1, 3, 4), pred_set.unsqueeze(0).permute(0, 2, 1, 3, 4), plot_loc, run, idx, save=False)
# %%    
#Testing it on the Euler Fluid Data
configuration['Data']['ntrain'] = 10
fields, x, y, dt = Euler_FV(configuration)
uv = fields[:, 1:3]
uv = uv.permute(0, 4, 1, 2, 3).reshape(uv.shape[0]*uv.shape[4], uv.shape[1], uv.shape[2], uv.shape[3])
fields_encoded = normalizer.encode(uv)
test = fields_encoded[-101:]

#Testing it on the incompressible NS PDEBench
# configuration['Data']['ntrain'] = 10
# fields, force, x, y, dt = Navier_Stokes_Incomp(configuration)
# uv = fields[:, 0:2]
# uv = uv.permute(0, 4, 1, 2, 3).reshape(uv.shape[0]*uv.shape[4], uv.shape[1], uv.shape[2], uv.shape[3])
# fields_encoded = normalizer.encode(uv)
# test = fields_encoded[-100:]
# %% 
with torch.no_grad():

        xx = test.to(device)
        batch_size = xx.shape[0]

        im, y = forward(model, xx)

        #Recon Loss
        loss = loss_func(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

error = (im - y).pow(2).mean()


print('(MSE) Testing Error: %.3e' % (error))

#Denormalising the test and predictions
test_out = normalizer.decode(y.to(device)).cpu()
pred_set = normalizer.decode(im.to(device)).cpu()

#Plotting the results 
from Utils.plots import plots_2d_yaml
idx = 0
configuration['Physics']['variables']=1
run = client.get_run_id_from_name(run_name)
plots_2d_yaml(configuration, test_out.unsqueeze(0).permute(0, 2, 1, 3, 4), pred_set.unsqueeze(0).permute(0, 2, 1, 3, 4), plot_loc, run, idx, save=False)
# %%
