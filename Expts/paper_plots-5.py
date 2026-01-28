# %% 
#Temporal Rollout Error Plots and NRMSE Evaluations for the seed ablations

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

#BS, Nvar, Nt, Nx, Ny
def MSE(test, pred):
    return torch.mean((test - pred).pow(2), axis=(0, 1, 3, 4)).numpy()

def nRMSE(test, pred):
    return torch.sqrt(torch.mean((test - pred).pow(2), axis=(0, 1, 3, 4)) / (torch.mean(test.pow(2), axis=(0, 1, 3, 4)) + 1e-8)).numpy()

# %% 
def temporal_rollout_error_with_std(pde, arch, t_exp, data_dist, error_stats_list, method_names, plot_loc, save=False):
    """
    Dynamic plotting function that accepts a list of error statistics (mean and std).
    
    Args:
        pde (str): Name of PDE
        arch (str): Architecture name
        t_exp (int): Time horizon
        data_dist (str): Data distribution (ID/OOD)
        error_stats_list (list): List of dicts containing 'mean' and 'std' arrays
        method_names (list): List of method names (e.g., ['AR', 'NODE', 'OpsSplit'])
        plot_loc (str): Save location
        save (bool): Whether to save the plot
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
    
    # Professional color scheme (Nature/Science style) - matching the original
    colors = ['#51829B', '#DA6C6C', '#78ABA8']  # Blue, Red, Teal
    linestyles = ['-', '-', '-']
    markers = ['o', 'o', 'o']
    
    # Map method names to display labels
    label_map = {
        'AR': 'Autoregressive',
        'NODE': 'Neural ODE',
        'OpsSplit': 'OpsSplit'
    }
    
    # Iterate through the provided error stats list
    all_max_err = []
    all_min_err = []
    
    for i, stats in enumerate(error_stats_list):
        mean_err = stats['mean']
        std_err = stats['std']
        
        # Get display label
        method_name = method_names[i] if i < len(method_names) else f"Method {i+1}"
        label = label_map.get(method_name, method_name)
        
        # Use the predefined colors/styles
        color = colors[i % len(colors)]
        linestyle = linestyles[i % len(linestyles)]
        marker = markers[i % len(markers)]
        
        # Plot mean line
        ax.plot(time_points, mean_err,
                color=color,
                linestyle=linestyle,
                linewidth=2.5,
                marker=marker,
                markersize=5,
                markevery=max(1, len(time_points)//12),
                label=label,
                alpha=1.0)
        
        # Add shaded region for std
        ax.fill_between(time_points, 
                        mean_err - std_err, 
                        mean_err + std_err,
                        color=color,
                        alpha=0.2)
        
        all_max_err.append(np.max(mean_err + std_err))
        all_min_err.append(np.min(mean_err - std_err))

       
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
    ax.set_ylabel('NRMSE', fontsize=25)
        
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
    if len(all_max_err) > 0 and (np.max(all_max_err) / (np.min(all_min_err) + 1e-9) > 100):
        ax.set_yscale('log')
    
    plt.tight_layout()
    
    if save:
        formats = ['pdf']#, 'svg']
        for fmt in formats:
            plot_name = f'{plot_loc}/temporal_error_{pde}_{arch}_{t_exp}_{data_dist}.{fmt}'
            plt.savefig(plot_name, 
                    dpi=300 if fmt == 'png' else None,
                    bbox_inches='tight',
                    facecolor='none', 
                    edgecolor='none', 
                    transparent=True, 
                    format=fmt)
        
    plt.show()


def load_test_data(pde, configuration):
    """
    Load test data once for all runs.
    
    Returns:
        fields: The raw field data
        dt: Time step
    """
    # n_sims = int(configuration['Data']['ntrain']*configuration['Data']['test_train_split'])
    n_sims = 100 

    if pde == 'Incompressible_Navier-Stokes':
        fields, x, y, dt = Navier_Stokes_Spectral(n_sims, data_dist)

        if data_dist == 'ID' and configuration['Data']['t_out'] == 100:
            mask = ~torch.isnan(fields).any(dim=(1,2,3,4))
            fields = fields[mask]
            
        # fields, x, y, dt = Navier_Stokes_Spectral(configuration)


    if pde == 'Compressible_Navier-Stokes':
        fields, x, y, dt = Euler_FV(n_sims, data_dist)

    fields = fields[...,:configuration['Data']['t_out']]
    
    return fields, dt

# from model_setup_opsplit_ablations import * 
from model_setup import * 
def compute_errors_for_run(run, pde, arch, configuration, fields, dt, model_loc):
    """
    Compute NRMSE errors for a single run using pre-loaded data.
    
    Returns:
        nrmse_errors: numpy array of NRMSE values per timestep
    """
    norms = np.load(model_loc +'/norms.npz')

    normalizer_func = Normalisation(configuration['Data']['normalisation'])
    normalizer = normalizer_func(torch.zeros_like(fields))
    normalizer.a, normalizer.b = torch.tensor(norms['a']), torch.tensor(norms['b'])

    fields_encoded = normalizer.encode(fields)
        
    test_in = fields_encoded[...,:configuration['Data']['t_in']]
    test_out = fields_encoded[...,configuration['Data']['t_in']:configuration['Data']['t_out']]


    # Setting up the Model
    model = model_initialisation(configuration, normalizer, run)#, data_dist)
    model_path = model_loc + '/model.pth'
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=False), strict=False)
    model.to(device)

    from Utils import explicit_time

    # Evaluation 
    eval = explicit_time.Eval_Setup(
        model, test_in, test_out, 
        normalizer='False', 
        ode_solver=configuration['Train']['odesolve']['source'], 
        roll_out=configuration['Train']['odesolve']['method']
    )
    pred_encoded, error = eval.inference(
        configuration['Data']['step'], 
        configuration['Data']['t_out']-1, 
        dt=dt
    )

    print(f'MSE (norm) : {float(error):.4e}')

    # Denormalising the test and predictions
    test_out = normalizer.decode(test_out.to(device)).cpu()
    pred_set = normalizer.decode(pred_encoded.to(device)).cpu()


    # Reshaping back to [BS, vars, Nt, Nx, Ny]
    test_out = test_out.permute(0,1,4,2,3)
    pred_set = pred_set.permute(0,1,4,2,3)

    # Getting the NRMSE per timestep
    nrmse_errors = nRMSE(test_out, pred_set)
    print(nrmse_errors.shape)
    
    return nrmse_errors


# %% 
# ============================================================================
# MAIN EXECUTION LOOP
# ============================================================================

# Define your experimental setup
experimental_configs = [
    {
        'pde': 'Incompressible_Navier-Stokes',
        't_exp': 50,
        'data_dist': 'ID',
        'arch': 'fno',
        'methods': {
            'AR': [
                'wide-timer', 'counting-pepper', 'pounded-minute', 'senile-loft', 'shabby-keyword'
                # Add more AR seed runs here
            ],
            'NODE': [
                'happy-walk', 'radiant-jamb', 'stringy-curve', 'juicy-refraction'#, 'hot-billet'
                # Add more NODE seed runs here
            ],
            'OpsSplit': [
                'reduced-roundel','exponential-sap', 'cheesy-molecule', 'exact-burn', 'tart-oasis' 
                # Add more OpsSplit seed runs here
            ]
        }
    },
    #     {
    #     'pde': 'Compressible_Navier-Stokes',
    #     't_exp': 100,
    #     'data_dist': 'ID',
    #     'arch': 'fno',
    #     'methods': {
    #         'AR': [
    #             'obnoxious-yard', 'bone-flow', 'drab-product', 'wan-take'
    #         ],
    #         'NODE': [
    #             'beige-bocaccio', 'threadbare-station', 'freezing-tangerine', 'formal-cloud'
    #         ],
    #         'OpsSplit': [
    #             'terminal-rehab', 'legato-map', 'primary-antiquity', 'constant-borzoi'
    #         ]
    #     }
    # }
]

# Process each configuration
for config in experimental_configs:
    pde = config['pde']
    t_exp = config['t_exp']
    data_dist = config['data_dist']
    arch = config['arch']
    methods = config['methods']
    
    print(f"\n{'='*80}")
    print(f"Processing: {pde} | arch={arch} | t_exp={t_exp} | data_dist={data_dist}")
    print(f"{'='*80}\n")
    
    # Load test data once for this configuration
    # Get a sample configuration to determine data parameters
    first_run = None
    for method_runs in methods.values():
        if method_runs:
            first_run = method_runs[0]
            break
    
    if first_run is None:
        print("No runs defined for this configuration. Skipping...")
        continue
    
    model_loc = os.getcwd() + '/Weights/' + first_run
    configuration = yaml.safe_load(open(next(Path(model_loc).glob('*.yaml'))))
    configuration['Data']['t_out'] = t_exp
    
    print("Loading test data...")
    fields, dt = load_test_data(pde, configuration)
    print(f"Test data loaded: {fields.shape}")
    print(yaml.dump(configuration, default_flow_style=False, indent=2))
    
    error_stats_list = []
    method_names = []
    
    # Process each method (AR, NODE, OpsSplit)
    for method_name, runs in methods.items():
        if not runs:  # Skip if no runs defined
            continue
            
        print(f"\n{'-'*60}")
        print(f"Method: {method_name}")
        print(f"{'-'*60}")
        
        method_errors = []  # Store errors from all seeds for this method
        
        # Process each seed/run
        for run in runs:
            print(f"\nProcessing run: {run}")
            
            model_loc = os.getcwd() + '/Weights/' + run
            configuration = yaml.safe_load(open(next(Path(model_loc).glob('*.yaml'))))
            configuration['Data']['t_out'] = t_exp
            
            # Compute errors for this run using pre-loaded data
            nrmse_errors = compute_errors_for_run(
                run, pde, arch, configuration, fields, dt, model_loc
            )
            
            method_errors.append(nrmse_errors)
            
            # Print metrics
            from Utils.metrics import NRMSE
            if t_exp > 50:
                print(f'NRMSE (extrapolation, t>50): {np.mean(nrmse_errors[50:]):.4f}')
            else:
                print(f'NRMSE (full): {np.mean(nrmse_errors):.4f}')
        
        # Compute mean and std across all seeds for this method
        method_errors = np.array(method_errors)  # Shape: [num_seeds, num_timesteps]
        mean_err = np.mean(method_errors, axis=0)
        std_err = np.std(method_errors, axis=0)
        
        error_stats_list.append({
            'mean': mean_err,
            'std': std_err
        })
        
        method_names.append(method_name)
        
        print(f"\n{method_name} Summary:")
        print(f"  Mean NRMSE: {np.mean(mean_err):.4f} ± {np.mean(std_err):.4f}")
    
# Generate plot for this configuration
    if error_stats_list:
        print(f"\nGenerating NRMSE plot...")
        temporal_rollout_error_with_std(
            pde, arch, t_exp, data_dist, error_stats_list, method_names, 
            plot_loc, save=True
        )
        
        # Print final summary table
        print(f"\n{'='*80}")
        print(f"FINAL SUMMARY TABLE")
        print(f"PDE: {pde} | Architecture: {arch} | Data Distribution: {data_dist} | t_out: {t_exp}")
        print(f"{'='*80}")
        print(f"{'Method':<15} {'Overall NRMSE (Mean ± Std)':<30}")
        print(f"{'-'*80}")
        
        for method_name, stats in zip(method_names, error_stats_list):
            mean_err = stats['mean']
            std_err = stats['std']
            
            overall = f"{np.mean(mean_err):.4f} ± {np.mean(std_err):.4f}"
            
            print(f"{method_name:<15} {overall:<30}")
        
        print(f"{'='*80}\n")

print("\n" + "="*80)
print("All configurations processed!")
print("="*80)

# %%