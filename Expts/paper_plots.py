# %% 
import os 
import numpy as np 
import torch 
import yaml 
from pathlib import Path
from matplotlib import pyplot as plt 

plot_loc = os.getcwd() + '/Plots'
# %% 
def load_data(method, data, t_extrapolate):
    test = np.load(data_loc + method+str(t_extrapolate)+data+'_test.npy')
    pred = np.load(data_loc + method+str(t_extrapolate)+data+'_pred.npy')
    return test, pred

# %% 
#Loading Data
pde = 'Incompressible Navier-Stokes'
data_loc = os.getcwd() + '/NS_incomp_preds/'

ar = 'wide-timer'
euler = 'happy-walk'
ops_split = 'symmetric-chocolate'

model_loc = os.getcwd() + '/Weights/'+ ar
configuration = yaml.safe_load(open(next(Path(model_loc).glob('*.yaml'))))


t_exp = 50

test, ar_pred = load_data(ar, 'ID', t_exp)
_, euler_pred = load_data(euler, 'ID', t_exp)
_, ops_split_pred = load_data(ops_split, 'ID', t_exp)


# %% 
#Setting up Metrics
from PRE_Eval import * 

pre = Incomp_NS_PRE(configuration)

def MSE(aa, bb):
    return np.mean((aa - bb)**2, axis=(0, 1, 2, 3))

def PRE(aa, bb):
    vars = torch.tensor(bb, dtype=torch.float32)
    return np.abs(np.mean(pre(vars.permute(0, 1, 4, 2, 3), boundary=False).numpy(), axis=(0, 2, 3)))

metric = MSE

# %% 

def temporal_rollout_error(pde, t_exp, test, pred_ar, pred_euler, pred_ops_split, plot_loc, metric=metric, save=False):
    
    ar_err = metric(test, pred_ar)
    euler_err = metric(test, pred_euler)
    ops_split_err = metric(test, pred_ops_split)


    time_points = torch.arange(0, t_exp-1, 1)
    if metric==PRE:
        time_points = time_points[1:-1]

    # Use LaTeX rendering for professional typography (if available)
    plt.rcParams.update({
        'font.size': 12,
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
        (euler_err, 'Euler', colors[1], linestyles[1], markers[1]),
        (ops_split_err, 'Euler - OpsSplit', colors[2], linestyles[2], markers[2])
    ]
    
    for data, label, color, linestyle, marker in data_sets:
        ax.plot(time_points, data,
                color=color,
                linestyle=linestyle,
                linewidth=2.0,
                marker=marker,
                markersize=5,
                markevery=max(1, len(time_points)//12),
                label=label,
                alpha=1.0)
    
    # Professional styling
    ax.set_xlabel('Time Instance', fontsize=14)
    if metric==MSE:
        ax.set_ylabel('Mean Squared Error', fontsize=14)
    if metric==PRE:
        ax.set_ylabel('Physics Residual Error', fontsize=14)
        
    ax.set_title(f'Temporal Rollout Error: {pde}', fontsize=15, pad=15)
    
    # Subtle grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Professional legend
    ax.legend(fontsize=11, 
             frameon=True, 
             fancybox=False, 
             shadow=False,
             framealpha=1.0,
             edgecolor='black',
             loc='best')
    
    # Set logarithmic scale if errors span multiple orders of magnitude
    # if np.max([np.max(ar_err), np.max(euler_err), np.max(ops_split_error)]) / \
    #    np.min([np.min(ar_err), np.min(euler_err), np.min(ops_split_error)]) > 100:
        # ax.set_yscale('log')
    
    plt.tight_layout()
    
    if save:
        # Multiple format saves for different publication needs
        formats = ['png', 'pdf', 'svg']
        for fmt in formats:
            plot_name = f'{plot_loc}/temporal_error_{pde}.{fmt}'
            plt.savefig(plot_name, 
                       dpi=300 if fmt == 'png' else None,
                       bbox_inches='tight',
                       facecolor='white',
                       format=fmt)
    
    plt.show()
# %% 
temporal_rollout_error(pde, t_exp, test, ar_pred, euler_pred, ops_split, plot_loc)
# %% 
