 # %% 
#Temporal Rollout Error Plots and NRMSE Evaluations for the OpsSplit Ablations

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

# def PRE(pre, vars):
#     # return torch.mean(pre(vars, boundary=False), axis=(0, 2, 3)).numpy()
#     # return torch.mean(torch.abs(pre(vars, boundary=False)), axis=(0, 2, 3)).numpy()
#     # return torch.abs(torch.mean(pre(vars, boundary=False), axis=(0, 2, 3))).numpy()
#     return np.mean(np.abs(pre(vars, boundary=False)), axis=(0, 2, 3))

# %% 
def temporal_rollout_error(pde, t_exp, error_list, run_names, plot_loc, metric='MSE', save=False, start_index=1):
    """
    Dynamic plotting function that accepts a list of error arrays.
    
    Args:
        pde (str): Name of PDE
        t_exp (int): Time horizon
        error_list (list): List of numpy arrays containing error data
        run_names (list): List of strings for naming (optional, used for files)
        plot_loc (str): Save location
        start_index (int): The starting number for the labels (default 1). 
                           Colors/styles will shift to match this index.
    """
    
    time_points = torch.arange(0, t_exp-1, 1)

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
    
    # Generate distinct colors dynamically based on number of runs
    num_runs = len(error_list)
    
    # Use tab10 for distinct categorical colors, fallback to viridis if many runs
    if num_runs <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
    else:
        colors = plt.cm.viridis(np.linspace(0, 1, num_runs))
        
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h'] # Cycle these
    linestyles = ['-', '--', '-.', ':'] # Cycle these
    
    # Iterate through the provided error list
    all_max_err = []
    all_min_err = []
    
    for i, data in enumerate(error_list):
        # Calculate the current index based on start_index
        current_idx = i + start_index
        label = f"{current_idx}" 
        
        # Adjust style index so that Label "2" gets the 2nd color (index 1), etc.
        # This preserves color consistency across plots with different start indices.
        style_idx = current_idx - 1
        
        # Cycle styles to ensure distinctness even with many lines
        color = colors[style_idx % len(colors)]
        marker = markers[style_idx % len(markers)]
        linestyle = linestyles[style_idx % len(linestyles)]
        
        ax.plot(time_points, data,
                color=color,
                linestyle=linestyle,
                linewidth=2.5,
                marker=marker,
                markersize=5,
                markevery=max(1, len(time_points)//12),
                label=label,
                alpha=0.9)
        
        all_max_err.append(np.max(data))
        all_min_err.append(np.min(data))

       
    # Add pastel orange shading for t_idx > 50
    if torch.max(time_points) > 50:
        y_min, y_max = ax.get_ylim()
        ax.axvspan(50, torch.max(time_points), 
                   color='#FFD4B3', 
                   alpha=0.3,       
                   zorder=0,        
                   label='t - extrapolate')


    # Professional styling
    ax.set_xlabel('Time Instance', fontsize=25)
    if metric=='MSE':
        ax.set_ylabel('NRMSE', fontsize=25)
    # if metric=='PRE':
    #     ax.set_ylabel('Physics Residual Error', fontsize=22)
        
    # ax.set_title(pde, fontsize=15, pad=15)
    
    # Subtle grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Professional legend
    ax.legend(fontsize=18, 
             title="Run #",
             title_fontsize=18,
             frameon=True, 
             fancybox=False, 
             shadow=False,
             framealpha=1.0,
             edgecolor='black',
             loc='best',
             ncol=2 if num_runs > 4 else 1) # Split legend cols if many runs
    
    # Set logarithmic scale if errors span multiple orders of magnitude
    if len(all_max_err) > 0 and (np.max(all_max_err) / (np.min(all_min_err) + 1e-9) > 100):
        ax.set_yscale('log')
    
    plt.tight_layout()
    
    if save:
        formats = ['pdf']#, 'svg']
        for fmt in formats:
            # Generate a generic filename since we aren't passing specific architecture names anymore
            plot_name = f'{plot_loc}/temporal_error_{pde}_{metric}_{t_exp}_{data_dist}_opssplit_ablation_explode.{fmt}'
            plt.savefig(plot_name, 
                    dpi=300 if fmt == 'png' else None,
                    bbox_inches='tight',
                    facecolor='none', 
                    edgecolor='none', 
                    transparent=True, 
                    format=fmt)
        
    plt.show()


# %% 
# Setting Run Parameters

# pde = 'Incompressible_Navier-Stokes'
# arch = 'fno'
# runs  = [
#     'gravitational-underwriter', 
#     'sluggish-pound', 
#     'greasy-bazaar', 
#     'excited-berry',
#     'purple-midpoint',
#     'concurrent-rating']


pde = 'Compressible_Navier-Stokes'
arch = 'fno'
runs = [
    'intricate-measure',
    # 'associative-margarine',
    # 'sad-skin',
    # 'convex-leverage',
    # 'lazy-buffer',
    # 'chestnut-damask'
]
# %%
t_exp = 100
data_dist = 'ID'

mses = []
pres = []

for run in runs:
    print("*****************************************************************************************")
    print(run)
    model_loc = os.getcwd() + '/Weights/' + run
    configuration = yaml.safe_load(open(next(Path(model_loc).glob('*.yaml'))))
    configuration['Data']['t_out'] = t_exp

    n_sims = int(configuration['Data']['ntrain'])#*configuration['Data']['test_train_split'])

    if pde == 'Incompressible_Navier-Stokes':
        fields, x, y, dt = Navier_Stokes_Spectral(n_sims, data_dist)
        # pre = Incomp_NS_PRE(configuration)

    if pde == 'Compressible_Navier-Stokes':
        fields, x, y, dt = Euler_FV(n_sims, data_dist)
        # pre = Comp_NS_PRE(configuration)

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
    from model_setup_opsplit_ablations import * 
    model = model_initialisation(configuration, normalizer, run, data_dist)
    model_path = model_loc + '/model.pth'
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
    mses.append(nRMSE(test_out, pred_set))
    # pres.append(PRE(pre, pred_set))

    print(f'Run: {run}')
    if t_exp > 50:
        print(f'NRMSE (physical) : {float(NRMSE(pred_set[:, :, 50:], test_out[:, :, 50:])["average"]):.4f}')
        # print(f'PRE : {np.mean(PRE(pre, pred_set[:, :, 50:])):.4f}')
    else:
        print(f'NRMSE (physical) : {float(NRMSE(pred_set, test_out)["average"]):.4f}')
        # print(f'PRE : {np.mean(PRE(pre, pred_set)):.4f}')

# %% 
temporal_rollout_error(pde, t_exp, mses, runs, plot_loc, metric='MSE', save=True, start_index=1)
# %% 
