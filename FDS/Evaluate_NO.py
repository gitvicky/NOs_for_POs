#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluating Trained Models using simvue's client API. 
"""
# %%
#Specifying the run instance
run_name = 'blue-carrier' #Minmax
# run_name = 'coal-panel' #Gaussian

class Run:
    def __init__(self, name=None):
        self.name = name

run = Run(run_name)
# %% 
#Setting up simvue 
import time 
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
configuration['Data']['ntrain'] = 245

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
        print(xx.shape, yy.shape)
        pred = model(xx)

        # Performance Metrics
        MSE_error = (yy - pred).pow(2).mean()
        MAE_error = torch.abs(yy - pred).mean()

    return pred, MSE_error, MAE_error

start_time = time.time()
pred_encoded, mse_error, mae_error = validation(model, test_ins.to(device), test_outs.to(device))
eval_time = (time.time() - start_time)/len(test_ins)

print('Time taken for evaluation:', eval_time)
print('(MSE) Testing Error: %.3e' % (mse_error))

#Denormalising the test and predictions
test_set = normalizer_out.decode(test_outs.to(device)).cpu()
pred_set = normalizer_out.decode(pred_encoded.to(device)).cpu()


# %%
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

idx = 38

# Set publication-quality parameters
plt.rcParams.update({
    # 'font.family': 'serif',
    # 'font.serif': ['Computer Modern Roman'],
    # 'text.usetex': True,    # Enable LaTeX rendering for text
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'figure.constrained_layout.use': True
})

names = ['Ground Truth', 'Prediction']
datasets = [test_set, pred_set]

for ii, u_field in enumerate(datasets):
    # Process data based on model architecture
    if configuration['Model']['arch'] == 'FNO':
        u_field = u_field[idx,...,0]
    else:
        u_field = u_field[idx]
    
    # Calculate global min and max for consistent colormap scaling
    v_min = 0
    v_max = 40
    
    # Create figure with appropriate aspect ratio
    fig = plt.figure(figsize=(10, 3))

    # Use GridSpec for more control over subplot layout
    gs = gridspec.GridSpec(1, 5, width_ratios=[0.5, 0.5, 0.5, 0.5, 0.5], wspace=0.005)
    
    # Create custom colormap for better visualization
    cmap = plt.cm.coolwarm
    
    # Define time points
    time_points = ['Floor 1', 'Floor 2', 'Floor 3', ' Floor 4', 'Floor 5']
    
    # Create subplots
    axes = []
    for t in range(5):
        ax = plt.subplot(gs[t])
        axes.append(ax)
        
        # Plot the data with improved settings
        im = ax.imshow(u_field[t], cmap=cmap, vmin=v_min, vmax=v_max, 
                        interpolation='nearest', aspect='equal')
        
        # Set title with LaTeX formatting
        ax.set_title(time_points[t], fontsize=12)
        
        # Only show ticks for the leftmost plot
        if t == 0:
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
        else:
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Add border to each subplot
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)
        
    # Add a single colorbar that applies to all plots
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Tempertaure')
    
    # Add main title
    fig.suptitle(f'{names[ii]}', fontsize=14, y=0)

    # Adjust spacing between subplots
    # plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    
    # save_path = os.getcwd()
    # plt.savefig(names[ii] + '_' + str(idx) + '.pdf', 
    #                 bbox_inches='tight', pad_inches=0.1, dpi=300)

    
    plt.show()
# %%
print('MSE (Physical):', torch.mean((test_set - pred_set)**2))

# %% 