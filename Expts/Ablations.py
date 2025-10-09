#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ablation Studies: Evaluating Trained Models using simvue's client API. 
Convergence, Data Efficiency, Model Efficiency, Rollout_length 
"""
# %% 
pde = 'compressible' #incompressible or compressible
study = 'model-efficiency' 
data_dist = 'OOD'
t_exp = 100
n_sims = 100 
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
import yaml 
from pathlib import Path

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

def nRMSE(test, pred):
    return torch.sqrt(torch.mean((test - pred).pow(2), axis=(0, 1, 2, 3, 4)) / (torch.mean(test.pow(2), axis=(0, 1, 2, 3, 4)) + 1e-8)).numpy()

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
    if study == 'data-efficiency':
        ar = ['corn-bucket', 'partial-bullet', 'hard-title', 'refined-price']
        euler = ['lean-photon','humongous-tin', 'icy-wine', 'grim-urn']
        ops_split = ['binary-bulwark', 'cheesy-barrel', 'natural-elevation','briny-reflection']
        x_axis = [100,200,400,500]
        xlabel = 'Training Samples'
        train_times = np.array([6200, 12500, 24500, 31000])/3600

    if study == 'model-efficiency':
        ar = ['gentle-inventory', 'persistent-frontlist' , 'corn-bucket']
        euler = ['molto-major', 'felt-linkage', 'lean-photon']
        ops_split = ['central-bourbon', 'daring-string', 'binary-bulwark']
        x_axis = [2, 4, 6]
        xlabel = 'Number of Layers'
        train_times = np.array([2600, 4500, 6200])/3600

    if study == 'rollout-length':
        ar = ['timid-skate', 'old-sound', 'alternating-sync', 'nimble-gofer']
        euler = ['persistent-collie', 'objective-bed', 'blistering-club', 'spatial-chart']
        ops_split = ['ferocious-dish', 'counting-brass', 'radiant-template' , 'relative-wallpaper' ]
        x_axis = [1, 5, 15, 30]
        xlabel = 'Rollout Length'
        train_times = np.array([2800, 8700, 23000, 28000])/3600

if pde == 'compressible':
    ar = ['online-breakout', 'atomic-courtyard', 'amicable-sideboard', 'molto-designer']
    euler = ['spicy-stick', 'savage-concierge', 'magenta-curd', 'humane-check']
    ops_split = ['amber-theme', 'regular-plate', 'primary-cost', 'molto-designer']
    x_axis = [100,200,400,500]
    xlabel = 'Training Samples'
    train_times = np.array([15000, 30000, 68000, 80000])/3600    
# %%
#Load data 
from data_loaders_ablations import *
if pde == 'incompressible':
    fields, x, y, dt = Navier_Stokes_Spectral(n_sims, data_dist)
if pde == 'compressible':
    fields, x, y, dt = Euler_FV(n_sims, data_dist)

t = torch.arange(0, fields.shape[-1], dt)
fields = fields[...,:t_exp]

types = [ar, euler, ops_split]
for index, type in enumerate(types):
    loss = []
    for run in type:
        print('**********************')
        print(run)
        print('**********************')
        run_loc = os.getcwd() + '/Weights/' + run
        configuration = yaml.safe_load(open(next(Path(run_loc).glob('*.yaml'))))
        configuration['Data']['t_out'] = t_exp
        norms = np.load(run_loc +'/norms.npz')

        normalizer_func = Normalisation(configuration['Data']['normalisation'])
        normalizer = normalizer_func(torch.zeros_like(fields))
        normalizer.a, normalizer.b = torch.tensor(norms['a']), torch.tensor(norms['b'])

        if configuration['Model']['ops_split_normalise']: #Normalise and Denormalise done within the Model. 
            fields_encoded = fields
        else:
            fields_encoded = normalizer.encode(fields)
            
        test_in = fields_encoded[...,:configuration['Data']['t_in']] # + torch.randn_like(fields_encoded[...,:configuration['Data']['t_in']])
        test_out = fields_encoded[...,configuration['Data']['t_in']:configuration['Data']['t_out']]

        print("Test Input: " + str(test_in.shape))
        print("Test Output: " + str(test_out.shape))

        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_in, test_out), batch_size=configuration['Data']['batch_size'], shuffle=False)


        # Setting up the Model
        ####################################
        from model_setup import * 
        model = model_initialisation(configuration, normalizer, run=None)
        model_path = run_loc + '/model.pth'
        model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=False), strict=False)
        model.to(device)
        print("Number of model params : " + str(model.count_params()))

        from Utils import explicit_time

        #Evaluation 
        eval = explicit_time.Eval_Setup(model, test_in, test_out, normalizer='False', ode_solver = configuration['Train']['odesolve']['source'], roll_out= configuration['Train']['odesolve']['method'])
        pred_encoded, error = eval.inference(configuration['Data']['step'], configuration['Data']['t_out']-1, dt=dt)

        print(f'MSE (norm) : {float(error):.4e}')

        #Denormalising the test and predictions
        if configuration['Model']['ops_split_normalise'] == False: #Normalise/Denormalise done within the Model for OS. 
            test_out = normalizer.decode(test_out.to(device)).cpu()
            pred_set = normalizer.decode(pred_encoded.to(device)).cpu()
        else:
            test_out = test_out.cpu()
            pred_set = pred_encoded.cpu()

        #Shaping back to [BS, vars, Nt, Nx, Ny]
        test_out = test_out.permute(0,1,4,2,3)
        pred_set = pred_set.permute(0,1,4,2,3)

        # #Getting the Metrics
        from Utils.metrics import NRMSE
        loss.append(nRMSE(test_out, pred_set))

    if index ==0:
        ar_loss = np.asarray(loss)
    elif index ==1:
        euler_loss = np.asarray(loss)
    elif index ==2:
        ops_split_loss = np.asarray(loss)
print(ar_loss.shape)

# %% 
def temporal_rollout_error(pde, x_axis, xlabel, ar_err, euler_err, ops_split_err, plot_loc, metric='MSE', save=False):

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
    colors = ['#51829B', '#DA6C6C', '#78ABA8']  # Red, Green, Blue
    linestyles = ['-', '-', '-']
    markers = ['o', 's', 'D']
    
    data_sets = [
        (ar_err, 'Autoregressive', colors[0], linestyles[0], markers[0]),
        (euler_err, 'Neural ODE', colors[1], linestyles[1], markers[1]),
        (ops_split_err, 'OpsSplit', colors[2], linestyles[2], markers[2])
    ]
    
    for data, label, color, linestyle, marker in data_sets:
        ax.plot(x_axis, data,
                color=color,
                linestyle=linestyle,
                linewidth=2.5,
                marker=marker,
                markersize=15,
                label=label,
                alpha=1.0)


    # Professional styling
    ax.set_xlabel(xlabel, fontsize=25)
    if metric=='MSE':
        ax.set_ylabel('NRMSE', fontsize=25)
    if metric=='PRE':
        ax.set_ylabel('Physics Residual Error', fontsize=22)
        
    # ax.set_title(pde, fontsize=15, pad=15)
    
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
    
    # Set logarithmic scale if errors span multiple orders of magnitude
    if np.max([np.max(ar_err), np.max(euler_err), np.max(ops_split_err)]) / \
       np.min([np.min(ar_err), np.min(euler_err), np.min(ops_split_err)]) > 100:
        ax.set_yscale('log')
    
    plt.tight_layout()
    
    if save:
        # Multiple format saves for different publication needs
        formats = ['pdf']#, 'svg']
        for fmt in formats:
            plot_name = f'{plot_loc}/temporal_error_{pde}_{study}_{data_dist}.{fmt}'
            plt.savefig(plot_name, 
                    dpi=300 if fmt == 'png' else None,
                    bbox_inches='tight',
                    facecolor='none',  # Changed from 'white' to 'none'
                    edgecolor='none',  # Add this to make edges transparent too
                    transparent=True,  # Add this parameter
                    format=fmt)
        
    plt.show()

temporal_rollout_error(pde, x_axis, xlabel, ar_loss, euler_loss, ops_split_loss, plot_loc, metric='MSE', save=True)

# %% 
def ablation_bar_plot(x_axis, xlabel, y_values, ylabel='Training Time (hours)', 
                      plot_loc='.', title='Ablation Study', save=True, 
                      convert_to_hours=False, show_values=True):
    """
    Generic bar plot with color scale for ablation studies.
    
    Parameters:
    -----------
    x_axis : list or array
        X-axis values (e.g., [100, 200, 400, 500] for training samples)
    xlabel : str
        Label for x-axis (e.g., 'Training Samples', 'Number of Layers')
    y_values : list or array
        Y-axis values (e.g., training times in hours)
    ylabel : str
        Label for y-axis
    plot_loc : str
        Directory path for saving plots
    title : str
        Title for the saved file
    save : bool
        Whether to save the plot
    convert_to_hours : bool
        If True, divides y_values by 3600 (for seconds to hours conversion)
    show_values : bool
        Whether to show value labels on top of bars
    
    Example Usage:
    --------------
    # Training Samples
    x_axis = [100, 200, 400, 500]
    train_times = np.array([6200, 12500, 24500, 31000]) / 3600
    ablation_bar_plot(x_axis, 'Training Samples', train_times)
    
    # Number of Layers
    x_axis = [2, 4, 6]
    train_times = np.array([2600, 4500, 6200]) / 3600
    ablation_bar_plot(x_axis, 'Number of Layers', train_times)
    
    # Rollout Length
    x_axis = [1, 5, 15, 30]
    train_times = np.array([2800, 8700, 23000, 28000]) / 3600
    ablation_bar_plot(x_axis, 'Rollout Length', train_times)
    """
    
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.cm as cm
    
    # Convert to hours if needed
    if convert_to_hours:
        y_values = np.array(y_values) / 3600
    else:
        y_values = np.array(y_values)
    
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
    
    # Create a custom green colormap (light to dark based on y values)
    # colors_list = ['#FFF3E0', '#FFCC80', '#FFA726', '#FB8C00', '#E65100']
    # colors_list = ['#ECEFF1', '#CFD8DC', '#90A4AE', '#607D8B', '#37474F']
    colors_list = ['#D4E6F1', '#85C1E2', '#3498DB', '#2E86C1', '#1B4F72']

    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('custom_green', colors_list, N=n_bins)
    
    # Normalize y values to [0, 1] for color mapping
    norm = plt.Normalize(vmin=y_values.min(), vmax=y_values.max())
    
    # Adjust bar width based on number of bars
    n_bars = len(x_axis)
    if n_bars <= 4:
        bar_width = 0.6
    elif n_bars <= 6:
        bar_width = 0.7
    else:
        bar_width = 0.8
    
    # Create bars with colors from the colormap
    bars = []
    for i, (x_val, y_val) in enumerate(zip(x_axis, y_values)):
        color = cmap(norm(y_val))
        bar = ax.bar(i, y_val, 
                     width=bar_width,
                     color=color,
                     alpha=0.9,
                     edgecolor='black',
                     linewidth=1.2)
        bars.append(bar)
    
    # Add value labels on top of bars
    if show_values:
        for i, (bar, y_val) in enumerate(zip(bars, y_values)):
            height = bar[0].get_height()
            # Format based on magnitude
            if y_val < 1:
                label_text = f'{y_val:.3f}'
            elif y_val < 10:
                label_text = f'{y_val:.2f}'
            else:
                label_text = f'{y_val:.1f}'
            
            ax.text(bar[0].get_x() + bar[0].get_width()/2., height,
                    label_text,
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Professional styling
    ax.set_xlabel(xlabel, fontsize=25)
    ax.set_ylabel(ylabel, fontsize=25)
    
    # Set x-axis ticks
    ax.set_xticks(range(len(x_axis)))
    ax.set_xticklabels(x_axis)
    
    # Subtle grid (horizontal only for bar charts)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, axis='y')
    ax.set_axisbelow(True)
    
    # # Add colorbar to show the scale
    # sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    # sm.set_array([])
    # cbar = plt.colorbar(sm, ax=ax, pad=0.02, aspect=30)
    # cbar.set_label(ylabel, fontsize=18, rotation=270, labelpad=30)
    # cbar.ax.tick_params(labelsize=12)
    
    plt.tight_layout()
    
    if save:
        # Multiple format saves for different publication needs
        formats = ['pdf']
        for fmt in formats:
            plot_name = f'{plot_loc}/{title.replace(" ", "_")}.{fmt}'
            plt.savefig(plot_name, 
                    dpi=300 if fmt == 'png' else None,
                    bbox_inches='tight',
                    facecolor='none',
                    edgecolor='none',
                    transparent=True,
                    format=fmt)
        print(f"Saved: {plot_name}")
        
    plt.show()


# Training Samples
print("Example 1: Training Samples")
x_axis = [100, 200, 400, 500]
xlabel = 'Training Samples'
train_times = np.array([6200, 12500, 24500, 31000]) / 3600
ablation_bar_plot(x_axis, xlabel, train_times, plot_loc = plot_loc, title='incomp_times_training_samples')

print("Example 1.5: Training Samples")
x_axis = [100, 200, 400, 500]
xlabel = 'Training Samples'
train_times = np.array([15000, 30000, 68000, 80000])/3600    
ablation_bar_plot(x_axis, xlabel, train_times, plot_loc = plot_loc, title='comp_times_training_samples')

# Number of Layers
print("\nExample 2: Number of Layers")
x_axis = [2, 4, 6]
xlabel = 'Number of Layers'
train_times = np.array([2600, 4500, 6200]) / 3600
ablation_bar_plot(x_axis, xlabel, train_times, plot_loc = plot_loc, title='incomp_times_number_of_layers')

# Rollout Length
print("\nExample 3: Rollout Length")
x_axis = [1, 5, 15, 30]
xlabel = 'Rollout Length'
train_times = np.array([2800, 8700, 23000, 28000]) / 3600
ablation_bar_plot(x_axis, xlabel, train_times, plot_loc = plot_loc, title='incomp_times_rollout_length')

# %%

