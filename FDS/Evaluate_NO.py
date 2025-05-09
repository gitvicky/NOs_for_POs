#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluating Trained Models using simvue's client API. 
"""
# %%
#Specifying the run instance
run_name = 'blue-carrier'

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

pde = configuration['Physics']['pde']
from data_loaders import *
if pde == 'FDS':
    ins, conds, outs = FDS_Carpark(configuration)

if configuration['Model']['arch'] != 'FNO':
    ins = ins[...,0]
    outs = outs[...,0]

print('ins shape:', ins.shape, 'outs shape:', outs.shape)

# %%
#Normalising the data -- using the same normalisations for inputs and outputs
normalizer_func = Normalisation(configuration['Data']['normalisation'])

normalizer_in = normalizer_func(ins)
ins_encoded = normalizer_in.encode(ins)

normalizer_out = normalizer_func(outs)
outs_encoded = normalizer_out.encode(outs)

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
train_ins, test_ins, train_outs, test_outs = train_test_split(ins_encoded, outs_encoded, test_size=configuration['Data']['test-train-split'], random_state=42)

train_loader = DataLoader(TensorDataset(train_ins, train_outs), batch_size=configuration['Data']['batch size'], shuffle=True) 
test_loader = DataLoader(TensorDataset(test_ins, test_outs), batch_size=configuration['Data']['batch size'], shuffle=False)

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
#Evaluation 
def validation(model, xx, yy):
    with torch.no_grad():
        xx, yy = xx.to(device), yy.to(device)
        pred = model(xx)

        # Performance Metrics
        MSE_error = (yy - pred).pow(2).mean()
        MAE_error = torch.abs(yy - pred).mean()

    return pred, MSE_error, MAE_error

pred_encoded, mse_error, mae_error = validation(model, test_ins.to(device), test_outs.to(device))


print('(MSE) Testing Error: %.3e' % (mse_error))

#Denormalising the test and predictions
test_set = normalizer_out.decode(test_outs.to(device)).cpu()
pred_set = normalizer_out.decode(pred_encoded.to(device)).cpu()


# %% 
#Visualising the results
import matplotlib
idx = 0
names = ['targs', 'preds']
for ii, u_field in enumerate([test_set, pred_set]):

    if configuration['Model']['arch'] == 'FNO':
        u_field = u_field[idx,...,0]
    else:
        u_field = u_field[idx]
        
    v_min = torch.min(u_field)
    v_max = torch.max(u_field)

    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 5, 1)
    pcm = ax.imshow(u_field[0], cmap=matplotlib.cm.coolwarm)
    ax.title.set_text('T1')
    fig.colorbar(pcm, pad=0.05)

    ax = fig.add_subplot(1, 5, 2)
    pcm = ax.imshow(u_field[1], cmap=matplotlib.cm.coolwarm)
    ax.title.set_text('T2')
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    fig.colorbar(pcm, pad=0.05)

    ax = fig.add_subplot(1, 5, 3)
    pcm = ax.imshow(u_field[2], cmap=matplotlib.cm.coolwarm)
    ax.title.set_text('T3')
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    fig.colorbar(pcm, pad=0.05)


    ax = fig.add_subplot(1, 5, 4)
    pcm = ax.imshow(u_field[3], cmap=matplotlib.cm.coolwarm)
    ax.title.set_text('T4')
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    fig.colorbar(pcm, pad=0.05)


    ax = fig.add_subplot(1, 5, 5)
    pcm = ax.imshow(u_field[4], cmap=matplotlib.cm.coolwarm)
    ax.title.set_text('T5')
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    fig.colorbar(pcm, pad=0.05)

# %% 