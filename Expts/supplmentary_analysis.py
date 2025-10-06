#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ablation Studies: Evaluating Trained Models using simvue's client API. 
Convergence, Data Efficiency, Model Efficiency, Rollout_length 
"""
# %% 
pde = 'incompressible' #incompressible or compressible
data = 'in' #in or out
study = 'convergence' 

# %%
#Importing the necessary packages
import os
import sys
import numpy as np
from tqdm import tqdm 
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from timeit import default_timer
from tqdm import tqdm 

sys.path.append("..")
from simvue import Client
client = Client()

#Setting up the seeds and devices
torch.manual_seed(0)
np.random.seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float32)

tmp_loc = os.getcwd() + '/tmp'
plot_loc = tmp_loc
sys.path.append("..")

# %% 
def plot_losses(ar_loss, euler_loss, ops_split_loss, plot_loc, type='Train', save=False):
    """
    Plot training losses for three methods.
    
    Args:
        ar_loss: Array of shape [251] or [3, 251] - autoregressive losses
        euler_loss: Array of shape [251] or [3, 251] - euler losses  
        ops_split_loss: Array of shape [251] or [3, 251] - operator splitting losses
        plot_loc: Directory to save plots
        save: Whether to save the figure
    """
    
    epochs = np.arange(len(ar_loss))
    
    # Use LaTeX rendering for professional typography (if available)
    plt.rcParams.update({
        'font.size': 14,
        'font.serif': ['Times New Roman'],
        'axes.linewidth': 1.2,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.major.size': 7,
        'xtick.minor.size': 4,
        'ytick.major.size': 7,
        'ytick.minor.size': 4,
    })
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    # Professional color scheme (Nature/Science style)
    colors = ['#51829B', '#DA6C6C', '#78ABA8']  # Blue, Red, Teal
    linestyles = ['-', '-', '-']
    markers = ['o', 'o', 'o']
    
    data_sets = [
        (ar_loss, 'Autoregressive', colors[0], linestyles[0], markers[0]),
        (euler_loss, 'Neural ODE', colors[1], linestyles[1], markers[1]),
        (ops_split_loss, 'OpsSplit', colors[2], linestyles[2], markers[2])
    ]
    
    for data, label, color, linestyle, marker in data_sets:
        ax.plot(epochs, data,
                color=color,
                linestyle=linestyle,
                linewidth=2.5,
                marker=marker,
                markersize=5,
                markevery=max(1, len(epochs)//12),
                label=label,
                alpha=1.0)
    
    # Professional styling
    ax.set_xlabel('Epoch', fontsize=25)
    if type=='Train':
        ax.set_ylabel('Train Loss', fontsize=25)
    elif type=='Test':
        ax.set_ylabel('Test Loss', fontsize=25)

    
    # Subtle grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Professional legend
    ax.legend(fontsize=22, 
             frameon=True, 
             fancybox=False, 
             shadow=False,
             framealpha=1.0,
             edgecolor='black',
             loc='best')
    
    # Set logarithmic scale if losses span multiple orders of magnitude
    max_loss = np.max([np.max(ar_loss), np.max(euler_loss), np.max(ops_split_loss)])
    min_loss = np.min([np.min(ar_loss), np.min(euler_loss), np.min(ops_split_loss)])
    if max_loss / min_loss > 100:
        ax.set_yscale('log')
    
    plt.tight_layout()
    
    if save:
        # Multiple format saves for different publication needs
        formats = ['pdf']
        for fmt in formats:
            if type == 'Train':
                plot_name = f'{plot_loc}/train_losses_{pde}.{fmt}'
            if type == 'Test':
                plot_name = f'{plot_loc}/test_losses_{pde}.{fmt}'
            plt.savefig(plot_name, 
                    dpi=300 if fmt == 'png' else None,
                    bbox_inches='tight',
                    facecolor='none',
                    edgecolor='none',
                    transparent=True,
                    format=fmt)
        
    plt.show()
# %% 
#Defining Runs
#AR, Euler, OpsSplit
if pde == 'incompressible':
    convergence = ['simple-cinnamon', 'fat-epoch', 'gold-particle']
if pde == 'compressible':
    convergence = ['kind-ridge', 'formal-university', 'oblique-ambiance']

if study == 'convergence':
    train_loss, test_loss = [], []
    for run_name in convergence:
        model_loc = os.getcwd() + '/Weights/' + run_name
        train = np.load(model_loc+'/train_loss.npy')
        test = np.load(model_loc+'/test_loss.npy')
        train_loss.append(train)
        test_loss.append(test)
    train_loss = np.asarray(train_loss)
    test_loss = np.asarray(test_loss)

# %% 
plot_losses(train_loss[0], train_loss[1], train_loss[2], plot_loc,'Train', save=False)
plot_losses(test_loss[0], test_loss[1], test_loss[2], plot_loc, 'Test', save=False)

# %% 
