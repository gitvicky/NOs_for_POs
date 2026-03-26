
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

if os.path.exists(tmp_loc) == False:
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
    time_points = torch.arange(0, t_exp-1, 1)

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
    
    colors = ['#51829B', '#DA6C6C', '#78ABA8']
    linestyles = ['-', '-', '-']
    markers = ['o', 'o', 'o']
    
    label_map = {
        'AR': 'Autoregressive',
        'NODE': 'Neural ODE',
        'OpsSplit': 'OpsSplit'
    }
    
    all_max_err = []
    all_min_err = []
    
    for i, stats in enumerate(error_stats_list):
        mean_err = stats['mean']
        std_err = stats['std']
        
        method_name = method_names[i] if i < len(method_names) else f"Method {i+1}"
        label = label_map.get(method_name, method_name)
        
        color = colors[i % len(colors)]
        linestyle = linestyles[i % len(linestyles)]
        marker = markers[i % len(markers)]
        
        ax.plot(time_points, mean_err,
                color=color, linestyle=linestyle, linewidth=2.5,
                marker=marker, markersize=5,
                markevery=max(1, len(time_points)//12),
                label=label, alpha=1.0)
        
        ax.fill_between(time_points, 
                        mean_err - std_err, 
                        mean_err + std_err,
                        color=color, alpha=0.2)
        
        all_max_err.append(np.max(mean_err + std_err))
        all_min_err.append(np.min(mean_err - std_err))

    if torch.max(time_points) > 50:
        y_min, y_max = ax.get_ylim()
        ax.axvspan(50, torch.max(time_points), 
                   color='#FFD4B3', alpha=0.3, zorder=0, label='t - extrapolate')

    ax.set_xlabel('Time Instance', fontsize=25)
    ax.set_ylabel('NRMSE', fontsize=25)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=22, frameon=True, fancybox=False, shadow=False,
             framealpha=1.0, edgecolor='black', loc='best')
    
    if len(all_max_err) > 0 and (np.max(all_max_err) / (np.min(all_min_err) + 1e-9) > 100):
        ax.set_yscale('log')
    
    plt.tight_layout()
    
    if save:
        for fmt in ['pdf']:
            plot_name = f'{plot_loc}/temporal_error_{pde}_{arch}_{t_exp}_{data_dist}.{fmt}'
            plt.savefig(plot_name, dpi=300 if fmt == 'png' else None,
                    bbox_inches='tight', facecolor='none', edgecolor='none', 
                    transparent=True, format=fmt)
        
    plt.show()


def load_test_data(pde, data_dist, configuration):
    n_sims = 100 

    if pde == 'Incompressible_Navier-Stokes':
        fields, x, y, dt = Navier_Stokes_Spectral(n_sims, data_dist)
        if data_dist == 'ID' and configuration['Data']['t_out'] == 100:
            mask = ~torch.isnan(fields).any(dim=(1,2,3,4))
            fields = fields[mask]

    if pde == 'Compressible_Navier-Stokes':
        fields, x, y, dt = Euler_FV(n_sims, data_dist)

    fields = fields[...,:configuration['Data']['t_out']]
    
    return fields, dt

from model_setup import * 

def compute_errors_for_run(run, pde, arch, configuration, fields, dt, model_loc):
    norms = np.load(model_loc + '/norms.npz')

    normalizer_func = Normalisation(configuration['Data']['normalisation'])
    normalizer = normalizer_func(torch.zeros_like(fields))
    normalizer.a, normalizer.b = torch.tensor(norms['a']), torch.tensor(norms['b'])

    fields_encoded = normalizer.encode(fields)
    test_in  = fields_encoded[..., :configuration['Data']['t_in']]
    test_out = fields_encoded[..., configuration['Data']['t_in']:configuration['Data']['t_out']]

    model = model_initialisation(configuration, normalizer, run)
    model_path = model_loc + '/model.pth'
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=False), strict=False)
    model.to(device)

    from Utils import explicit_time

    eval = explicit_time.Eval_Setup(
        model, test_in, test_out, 
        normalizer='False', 
        ode_solver=configuration['Train']['odesolve']['source'], 
        roll_out=configuration['Train']['odesolve']['method']
    )
    pred_encoded, error = eval.inference(
        configuration['Data']['step'], 
        configuration['Data']['t_out'] - 1, 
        dt=dt
    )

    print(f'MSE (norm) : {float(error):.4e}')

    test_out = normalizer.decode(test_out.to(device)).cpu()
    pred_set  = normalizer.decode(pred_encoded.to(device)).cpu()

    test_out = test_out.permute(0, 1, 4, 2, 3)
    pred_set  = pred_set.permute(0, 1, 4, 2, 3)

    nrmse_errors = nRMSE(test_out, pred_set)
    print(nrmse_errors.shape)
    
    return nrmse_errors


# %% 
# ============================================================================
# EXPERIMENTAL CONFIGURATION
# Structured as a list of groups, each group sharing the same pde/t_exp/data_dist
# but containing multiple arch configs.
# ============================================================================
# experimental_configs = [
#     {
#         'pde': 'Compressible_Navier-Stokes',
#         't_exp': 50,
#         'data_dist': 'ID',
#         'archs': [
#             {
#                 'arch': 'fno',
#                 'methods': {
#                     'AR': [
#                         'inverse-bollard', 'metal-redoubt', 'muted-rotor'
#                     ],
#                     'NODE': [
#                         'old-polygon', 'polite-trie', 'deterministic-edging'
#                     ],
#                     'OpsSplit': [
#                         'vivid-pigeon', 'blended-plaid', 'ripe-hound'
#                     ]
#                 }
#             },
#             {
#                 'arch': 'unet',
#                 'methods': {
#                     'AR': [
#                         'humid-sky', 'internal-cycle', 'diagonal-havarti'
#                     ],
#                     'NODE': [
#                         'light-frequency', 'sensitive-dynamic', 'rosy-block'
#                     ],
#                     'OpsSplit': [
#                         'reduced-centerline', 'interior-nut', 'chief-vehicle'
#                     ]
#                 }
#             },
#             {
#                 'arch': 'vit',
#                 'methods': {
#                     'AR': [
#                         'vain-wake', 'regular-basin', 'visible-taking'
#                     ],
#                     'NODE': [
#                         'independent-heat', 'smoggy-reflection', 'medium-vinegar'
#                     ],
#                     'OpsSplit': [
#                         'obvious-pier', 'lead-athlete', 'immediate-accelerometer'
#                     ]
#                 }
#             },
#             {
#                 'arch': 'uno',
#                 'methods': {
#                     'AR': [
#                         'magenta-league', 'lively-headline', 'kind-purse'
#                     ],
#                     'NODE': [
#                         'little-gravity', 'planar-circuit', 'ferocious-game'
#                     ],
#                     'OpsSplit': [
#                         'greasy-area', 'silent-front', 'worried-facade'
#                     ]
#                 }
#             }
#         ]
#     }
# ]


experimental_configs = [
    {
        'pde': 'Incompressible_Navier-Stokes',
        't_exp': 50,
        'data_dist': 'ID',
        'archs': [
            {
                'arch': 'fno',
                'methods': {
                    'AR': [
                        'lazy-badger', 'rounded-flounder', 'briny-turret'
                    ],
                    'NODE': [
                        'brownian-infomediary', 'silver-costume', 'customer-pug'
                    ],
                    'OpsSplit': [
                        'fast-herring', 'parallel-spire', 'staccato-leverage'
                    ]
                }
            },
            {
                'arch': 'unet',
                'methods': {
                    'AR': [
                        'dark-exit', 'pink-hardball', 'visible-magazine'
                    ],
                    'NODE': [
                        'trite-drone', 'icy-alligator', 'relaxed-period'
                    ],
                    'OpsSplit': [
                        'lower-jog', 'frosty-barbette', 'bitter-opacity'
                    ]
                }
            },
            {
                'arch': 'vit',
                'methods': {
                    'AR': [
                        'medium-volume', 'iron-retail', 'denim-cognac'
                    ],
                    'NODE': [
                        'cool-hospital', 'arctic-functionality', 'lime-setter'
                    ],
                    'OpsSplit': [
                        'lenient-style', 'indulgent-landscape', 'quiet-talent'
                    ]
                }
            },
            {
                'arch': 'cno',
                'methods': {
                    'AR': [
                        'muffled-mansion', 'direct-outlet', 'perfect-fish'
                    ],
                    'NODE': [
                        'equidistant-wealth', 'immediate-cornice', 'crazy-pulsar'
                    ],
                    'OpsSplit': [
                        'matte-instance', 'resultant-gizmo', 'intense-buyer'
                    ]
                }
            },
            {
                'arch': 'uno',
                'methods': {
                    'AR': [
                        'rectilinear-granite', 'spicy-designer', 'extended-carriage'
                    ],
                    'NODE': [
                        'khaki-depth', 'another-goo', 'extremal-electron'
                    ],
                    'OpsSplit': [
                        'boolean-angel', 'teal-subscriber', 'acyclic-map'
                    ]
                }
            }
        ]
    }
]

# ============================================================================
# MAIN EXECUTION LOOP
# ============================================================================

print(f"Found {len(experimental_configs)} config group(s) to process.")

for group_idx, group in enumerate(experimental_configs):
    pde       = group['pde']
    t_exp     = group['t_exp']
    data_dist = group['data_dist']
    archs     = group['archs']

    print(f"\n{'='*80}")
    print(f"Group {group_idx+1}: PDE={pde} | t_exp={t_exp} | data_dist={data_dist} | archs={[a['arch'] for a in archs]}")
    print(f"{'='*80}\n")

    # Find first available run for sample config
    first_run = next(
        (run
         for arch_cfg in archs
         for runs in arch_cfg['methods'].values()
         for run in runs),
        None  # default if nothing found
    )

    if first_run is None:
        print("ERROR: No runs found in any arch/method. Check your experimental_configs.")
        continue

    print(f"Using first run for sample config: {first_run}")

    sample_model_loc = os.getcwd() + '/Weights/' + first_run
    yaml_files = list(Path(sample_model_loc).glob('*.yaml'))

    if not yaml_files:
        print(f"ERROR: No yaml file found in {sample_model_loc}")
        continue

    sample_config = yaml.safe_load(open(yaml_files[0]))
    sample_config['Data']['t_out'] = t_exp

    print("Loading test data...")
    fields, dt = load_test_data(pde, data_dist, sample_config)
    print(f"Test data loaded: fields.shape={fields.shape}, dt={dt}\n")

    # results[arch][method] = {'mean': ..., 'std': ...}
    results = {}

    for arch_cfg in archs:
        arch    = arch_cfg['arch']
        methods = arch_cfg['methods']

        print(f"\n{'-'*60}")
        print(f"Architecture: {arch}")
        print(f"{'-'*60}")

        results[arch] = {}
        arch_error_stats  = []
        arch_method_names = []

        for method_name, runs in methods.items():
            if not runs:
                print(f"  [{method_name}] No runs defined — skipping.")
                continue

            print(f"\n  Method: {method_name} ({len(runs)} run(s))")
            method_errors = []

            for run in runs:
                print(f"    Processing run: {run}")

                model_loc = os.getcwd() + '/Weights/' + run

                if not os.path.exists(model_loc):
                    raise FileNotFoundError(
                        f"Run directory missing! The script expected to find the run at: {model_loc}, "
                        f"but the directory does not exist."
                    )

                yaml_files = list(Path(model_loc).glob('*.yaml'))
                if not yaml_files:
                    raise FileNotFoundError(
                        f"Configuration missing! The directory {model_loc} exists, "
                        f"but no '.yaml' file was found inside it."
                    )

                configuration = yaml.safe_load(open(yaml_files[0]))
                configuration['Data']['t_out'] = t_exp

                try:
                    nrmse_errors = compute_errors_for_run(
                        run, pde, arch, configuration, fields, dt, model_loc
                    )
                    method_errors.append(nrmse_errors)

                    if t_exp > 50:
                        print(f'      NRMSE (extrap, t>50): {np.mean(nrmse_errors[50:]):.4f}')
                    else:
                        print(f'      NRMSE (full): {np.mean(nrmse_errors):.4f}')

                except Exception as e:
                    print(f"    ERROR on run {run}: {e}")
                    import traceback; traceback.print_exc()
                    continue

            if not method_errors:
                print(f"  [{method_name}] All runs failed or were skipped.")
                continue

            method_errors = np.array(method_errors)
            mean_err = np.mean(method_errors, axis=0)
            std_err  = np.std(method_errors,  axis=0)

            results[arch][method_name] = {'mean': mean_err, 'std': std_err}
            arch_error_stats.append({'mean': mean_err, 'std': std_err})
            arch_method_names.append(method_name)

            print(f"\n  {method_name} Summary: Mean NRMSE = {np.mean(mean_err):.4f} ± {np.mean(std_err):.4f}")

        # Per-arch temporal rollout plot
        if arch_error_stats:
            print(f"\n  Generating temporal NRMSE plot for {arch}...")
            temporal_rollout_error_with_std(
                pde, arch, t_exp, data_dist,
                arch_error_stats, arch_method_names,
                plot_loc, save=True
            )
        else:
            print(f"\n  No results collected for arch={arch}, skipping plot.")

    # -------------------------------------------------------------------------
    # SUMMARY TABLE — arch × method
    # -------------------------------------------------------------------------
    all_methods = sorted({m for arch_cfg in archs for m in arch_cfg['methods']})
    all_archs   = [cfg['arch'] for cfg in archs]

    col_w  = 28
    arch_w = 10

    print(f"\n{'='*80}")
    print(f"SUMMARY TABLE  |  PDE: {pde}  |  data_dist: {data_dist}  |  t_out: {t_exp}")
    print(f"{'='*80}")
    header = f"{'Arch':<{arch_w}}" + "".join(f"{'Method: ' + m:^{col_w}}" for m in all_methods)
    print(header)
    print(f"{'':.<{arch_w}}" + "".join(f"{'Mean NRMSE ± Std':^{col_w}}" for _ in all_methods))
    print("-" * (arch_w + col_w * len(all_methods)))

    for arch in all_archs:
        row = f"{arch:<{arch_w}}"
        for method in all_methods:
            if arch in results and method in results[arch]:
                stats = results[arch][method]
                cell  = f"{np.mean(stats['mean']):.4f} ± {np.mean(stats['std']):.4f}"
            else:
                cell = "N/A"
            row += f"{cell:^{col_w}}"
        print(row)

    print(f"{'='*80}\n")

print("\n" + "="*80)
print("All configurations processed!")
print("="*80)
# %%