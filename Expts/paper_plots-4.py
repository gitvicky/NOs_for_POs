 # %% 
#Temporal Rollout Error Plots and NRMSE Evaluations for the Ideal MHD Dataset

import os 
import yaml 
from pathlib import Path
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
tmp_loc = os.getcwd() + '/tmp'
plot_loc = tmp_loc
sys.path.append("..")

# Create tmp directory if it doesn't exist, or recreate it if it does
if os.path.exists(tmp_loc) == False:
    # shutil.rmtree(tmp_loc)
    os.makedirs(tmp_loc, exist_ok=True)

# %% 
from data_loaders_ablations import *
from Neural_PDE.Utils.processing_utils import * 
from Neural_PDE.Utils.training_utils import * 

#Setting up Metrics
from PRE_Eval import * 

#BS, Nvar, Nt, Nx, Ny
def MSE(test, pred):
    return torch.mean((test - pred).pow(2), axis=(0, 1, 3, 4)).numpy()

def nRMSE(test, pred):
    return torch.sqrt(torch.mean((test - pred).pow(2), axis=(0, 1, 3, 4)) / (torch.mean(test.pow(2), axis=(0, 1, 3, 4)) + 1e-8)).numpy()

# %% 
def temporal_rollout_error(pde, t_exp, ar_err, euler_err, ops_split_err, plot_loc, metric='MSE', save=False):

    time_points = torch.arange(0,t_exp-1, 1)

    # if metric=='PRE':
    #     time_points = time_points[1:-1]

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
    markers = ['o', 'o', 'o']
    
    data_sets = [
        (ar_err, 'Autoregressive', colors[0], linestyles[0], markers[0]),
        (euler_err, 'Neural ODE', colors[1], linestyles[1], markers[1]),
        (ops_split_err, 'OpsSplit', colors[2], linestyles[2], markers[2])
    ]
    
    for data, label, color, linestyle, marker in data_sets:
        ax.plot(time_points, data,
                color=color,
                linestyle=linestyle,
                linewidth=2.5,
                marker=marker,
                markersize=5,
                markevery=max(1, len(time_points)//12),
                label=label,
                alpha=1.0)

       
    # Add pastel orange shading for t_idx > 50
    if torch.max(time_points) > 50:
        # Get the y-axis limits to fill the entire vertical space
        y_min, y_max = ax.get_ylim()
        
        # Create shading from x=50 to the end of the plot
        ax.axvspan(50, torch.max(time_points), 
                   color='#FFD4B3',  # Pastel orange color
                   alpha=0.3,       # Semi-transparent
                   zorder=0,        # Put it behind the plot lines
                   label='t - extrapolate')   # Optional label for legend


    # Professional styling
    ax.set_xlabel('Time Instance', fontsize=25)
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
            plot_name = f'{plot_loc}/temporal_error_{pde}_{arch}_{metric}_{t_exp}_{data_dist}.{fmt}'
            plt.savefig(plot_name, 
                    dpi=300 if fmt == 'png' else None,
                    bbox_inches='tight',
                    facecolor='none',  # Changed from 'white' to 'none'
                    edgecolor='none',  # Add this to make edges transparent too
                    transparent=True,  # Add this parameter
                    format=fmt)
        
    plt.show()

# %% 
# Setting Run Parameters

pde = 'Ideal MHD'
arch = 'fno'
if arch == 'fno': 
    ar = 'taxonomic-nutmeg'
    euler = 'friendly-rehab'
    # ops_split = 'crunchy-tower'
    ops_split = 'zingy-milk'

# %%
t_exp = 100
data_dist = 'OOD'

models = [ar, euler, ops_split]
mses = []
pres = []

for run in models:
    print(run)
    model_loc = os.getcwd() + '/Weights/' + run
    configuration = yaml.safe_load(open(next(Path(model_loc).glob('*.yaml'))))
    configuration['Data']['t_out'] = t_exp

    n_sims = int(configuration['Data']['ntrain']*configuration['Data']['test_train_split'])

    if pde == 'Ideal MHD' or pde == 'Constrained MHD':
        fields, x, y, dt = Constrained_MHD(configuration, data_dist)

    t = torch.arange(0, fields.shape[-1], dt)
    fields = fields[...,:configuration['Data']['t_out']]
    print(yaml.dump(configuration, default_flow_style=False, indent=2))

    norms = np.load(model_loc +'/norms.npz')

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


    # Setting up the Model and Optimizers 
    ####################################
    from model_setup import * 
    model = model_initialisation(configuration, normalizer, run)
    model_path = model_loc + '/model.pth'
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=False), strict=False)
    if data_dist == 'OOD':
        model.gamma = torch.tensor(2/3, dtype=torch.float32, requires_grad=False).to(device)   
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
    mses.append(nRMSE(test_out, pred_set))

    if t_exp > 50:
        print(f'NRMSE (physical) : {float(NRMSE(pred_set[:, :, 50:], test_out[:, :, 50:])["average"]):.4f}')
    else:
        print(f'NRMSE (physical) : {float(NRMSE(pred_set, test_out)["average"]):.4f}')

# %% 
temporal_rollout_error(pde, t_exp, mses[0], mses[1], mses[2], plot_loc, metric='MSE', save=True)
# %% 
