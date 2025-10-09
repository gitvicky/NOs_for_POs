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
from Neural_PDE.Utils.processing_utils import * 
from Neural_PDE.Utils.training_utils import * 

# Create tmp directory if it doesn't exist, or recreate it if it does
if os.path.exists(tmp_loc) == False:
    # shutil.rmtree(tmp_loc)
    os.makedirs(tmp_loc, exist_ok=True)

# %% 
import numpy as np
from scipy.ndimage import uniform_filter1d

def plot_losses(ar_loss, euler_loss, ops_split_loss, plot_loc, type='Train', save=False, smooth_window=5):
    """
    Plot training losses for three methods.
    
    Args:
        ar_loss: Array of shape [251] or [3, 251] - autoregressive losses
        euler_loss: Array of shape [251] or [3, 251] - euler losses  
        ops_split_loss: Array of shape [251] or [3, 251] - operator splitting losses
        plot_loc: Directory to save plots
        type: 'Train' or 'Test'
        save: Whether to save the figure
        smooth_window: Window size for smoothing (default: 10)
    """
    
    def smooth_data(data, window=10):
        """Apply uniform filter for smoothing."""
        return uniform_filter1d(data, size=window, mode='nearest')
    
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
        # Smooth the data
        data_smooth = smooth_data(data, smooth_window)
        
        # Plot raw data (lighter/more transparent)
        ax.plot(epochs, data,
                color=color,
                linestyle=linestyle,
                linewidth=1.0,
                alpha=0.3,
                zorder=1)
        
        # Plot smoothed data (darker/more visible)
        ax.plot(epochs, data_smooth,
                color=color,
                linestyle=linestyle,
                linewidth=2.5,
                marker=marker,
                markersize=5,
                markevery=max(1, len(epochs)//12),
                label=label,
                alpha=1.0,
                zorder=2)
    
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
    convergence = ['corn-bucket', 'lean-photon', 'binary-bulwark']
if pde == 'compressible':
    convergence = ['frayed-basis', 'humane-check', 'molto-designer']

if study == 'convergence':
    train_loss, test_loss = [], []
    for run_name in convergence:
        run_id = client.get_run_id_from_name(run_name)
        df = client.get_metric_values(
            run_ids=[run_id],
            metric_names=['Train Loss', 'Test Loss'],
            xaxis='step',
            output_format='dataframe'
        )
        loss = df.to_numpy()
        train, test= loss[:,1], loss[:,0]
        train_loss.append(train)
        test_loss.append(test)
    train_loss = np.asarray(train_loss)
    test_loss = np.asarray(test_loss)
    step = np.arange(0, 251, 1)

# %% 
# plot_losses(train_loss[0], train_loss[1], train_loss[2], plot_loc,'Train', save=True)
# plot_losses(test_loss[0], test_loss[1], test_loss[2], plot_loc, 'Test', save=True)

# %% 
#Finetuned 

if pde == 'incompressible':
    pt = 'happy-ray'
    ft = 'eager-mean'
    convergence = [pt, ft]
if pde == 'compressible':
    pt = 'extremal-carrier'
    ft = 'narrow-mole'
    convergence = [pt, ft]

train_loss, test_loss = [], []
for run_name in convergence:
    run_id = client.get_run_id_from_name(run_name)
    df = client.get_metric_values(
        run_ids=[run_id],
        metric_names=['Train Loss', 'Test Loss'],
        xaxis='step',
        output_format='dataframe'
    )
    loss = df.to_numpy()
    train, test= loss[:,1], loss[:,0]
    train_loss.append(train)
    test_loss.append(test)
train_loss = np.asarray(train_loss)
test_loss = np.asarray(test_loss)
step = np.arange(0, 251, 1)

# %% 
import numpy as np
from scipy.ndimage import uniform_filter1d

def plot_losses(pretrain_loss, ft_loss, plot_loc, type='Train', save=False, smooth_window=10):
    """
    Plot training losses comparing pretrained vs finetuned OpsSplit model.
    
    Args:
        pretrain_loss: Array of shape [251] - pretrained model losses
        ft_loss: Array of shape [251] - finetuned model losses
        plot_loc: Directory to save plots
        type: 'Train' or 'Test'
        save: Whether to save the figure
        smooth_window: Window size for smoothing (default: 10)
    """
    
    def smooth_data(data, window=10):
        """Apply uniform filter for smoothing."""
        return uniform_filter1d(data, size=window, mode='nearest')
    
    epochs = np.arange(len(pretrain_loss))
    
    # Use LaTeX rendering for professional typography
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
    
    # Pastel colors for pretrained and finetuned
    color_pretrain = '#5C7285'  
    color_ft = '#AF3E3E'
    # Smooth the data
    pretrain_smooth = smooth_data(pretrain_loss, smooth_window)
    ft_smooth = smooth_data(ft_loss, smooth_window)
    
    # Plot raw data (lighter/more transparent)
    ax.plot(epochs, pretrain_loss,
            color=color_pretrain,
            linestyle='--',
            linewidth=1.0,
            alpha=0.4,
            zorder=1)
    
    ax.plot(epochs, ft_loss,
            color=color_ft,
            linestyle='-',
            linewidth=1.0,
            alpha=0.4,
            zorder=1)
    
    # Plot smoothed data (darker/more visible)
    ax.plot(epochs, pretrain_smooth,
            color=color_pretrain,
            linestyle='--',
            linewidth=2.5,
            marker='s',
            markersize=5,
            markevery=max(1, len(epochs)//12),
            label='OpsSplit (pre-trained)',
            alpha=0.9,
            zorder=2)
    
    ax.plot(epochs, ft_smooth,
            color=color_ft,
            linestyle='-',
            linewidth=2.5,
            marker='o',
            markersize=5,
            markevery=max(1, len(epochs)//12),
            label='OpsSplit (fine-tuned)',
            alpha=0.9,
            zorder=2)
    
    # Professional styling
    ax.set_xlabel('Epoch', fontsize=25)
    if type == 'Train':
        ax.set_ylabel('Train Loss', fontsize=25)
    elif type == 'Test':
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
    max_loss = max(np.max(pretrain_loss), np.max(ft_loss))
    min_loss = min(np.min(pretrain_loss), np.min(ft_loss))
    if max_loss / min_loss > 100:
        ax.set_yscale('log')
    
    plt.tight_layout()
    
    if save:
        formats = ['pdf']
        for fmt in formats:
            if type == 'Train':
                plot_name = f'{plot_loc}/train_losses_transfer_learn_{pde}.{fmt}'
            if type == 'Test':
                plot_name = f'{plot_loc}/test_losses_transfer_learn_{pde}.{fmt}'
            plt.savefig(plot_name, 
                    dpi=300 if fmt == 'png' else None,
                    bbox_inches='tight',
                    facecolor='white',
                    edgecolor='none',
                    format=fmt)
        
    plt.show()

plot_losses(train_loss[0], train_loss[1], plot_loc,'Train', save=True)
plot_losses(test_loss[0], test_loss[1], plot_loc, 'Test', save=True)

# %%
