 # %% 
#Importing the necessary packages
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
from data_loaders import *
from Neural_PDE.Utils.processing_utils import * 
from Neural_PDE.Utils.training_utils import * 

#Setting up Metrics
from PRE_Eval import * 

#BS, Nvar, Nt, Nx, Ny
def MSE(test, pred):
    return torch.mean((test - pred).pow(2), axis=(0, 1, 3, 4)).numpy()

def nRMSE(test, pred):
    return torch.sqrt(torch.mean((test - pred).pow(2), axis=(0, 1, 3, 4)) / (torch.mean(test.pow(2), axis=(0, 1, 3, 4)) + 1e-8)).numpy()

def PRE(pre, vars):
    # return torch.mean(pre(vars, boundary=False), axis=(0, 2, 3)).numpy()
    return torch.mean(torch.abs(pre(vars, boundary=False)), axis=(0, 2, 3)).numpy()
    # return torch.abs(torch.mean(pre(vars, boundary=False), axis=(0, 2, 3))).numpy()
    # return np.mean(np.abs(pre(vars, boundary=False)), axis=(0, 2, 3))

# %% 
#Setting Run Parameters
pde = 'Incompressible_Navier-Stokes'
arch = 'FNO'

if arch == 'FNO':
    ar = 'wide-timer'
    euler = 'happy-walk'
    # ops_split = 'symmetric-chocolate'
    ops_split = 'reduced-roundel' #NO + FD

if arch == 'UNet':
    ar = 'trite-accelerator'
    euler = 'caramelized-commit'
    # ops_split = 'creative-assurance' 
    ops_split = 'scared-assistant' #NO + FD
  
if arch == 'CNO':
    ar = 'bold-canal'
    euler = 'similar-river'
    # ops_split = 'complicated-ideation'
    ops_split = 'cheerful-mercury' #NO + FD Diff Seed

if arch == 'ViT':
    ar = 'icy-methodology'
    euler = 'bright-novella'
    # ops_split = 'intricate-factor'   
    ops_split = 'sticky-chimpanzee' #NO + FD

if arch == 'UNO':
    ar = 'concave-falls'
    euler = 'icy-attache'
    ops_split = 'adventurous-buck'

# %%
t_exp = 100
data_dist = 'ID'

models = [ops_split]
mses = []
pres = []

for model in models:
    model_loc = os.getcwd() + '/Weights/' + model
    configuration = yaml.safe_load(open(next(Path(model_loc).glob('*.yaml'))))
    configuration['Data']['t_out'] = t_exp

    n_sims = int(configuration['Data']['ntrain']*configuration['Data']['test_train_split'])

    if pde == 'Incompressible_Navier-Stokes':
        fields, x, y, dt = Navier_Stokes_Spectral(configuration)
        pre = Incomp_NS_PRE(configuration)

    if pde == 'Compressible_Navier-Stokes':
        fields, x, y, dt = Euler_FV(configuration)
        pre = Comp_NS_PRE(configuration)

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
    model = model_initialisation(configuration, normalizer, run=None)
    model_path = model_loc + '/model.pth'
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=False), strict=False)

    model.to(device)
    print("Number of model params : " + str(model.count_params()))

    from Utils import explicit_time

    #Evaluation 
    eval = explicit_time.Eval_Setup(model, test_in, test_out, normalizer='False', ode_solver = configuration['Train']['odesolve']['source'], roll_out= configuration['Train']['odesolve']['method'])
    pred_encoded, error = eval.inference(configuration['Data']['step'], configuration['Data']['t_out']-1, dt=dt)

    print(f'MSE (norm) : {float(error):.4e}')

    # #Denormalising the test and predictions
    # if configuration['Model']['ops_split_normalise'] == False: #Normalise/Denormalise done within the Model for OS. 
    #     test_out = normalizer.decode(test_out.to(device)).cpu()
    #     pred_set = normalizer.decode(pred_encoded.to(device)).cpu()
    # else:
    #     test_out = test_out.cpu()
    #     pred_set = pred_encoded.cpu()

    #Without Denormalisations 
    test_out = test_out.cpu()
    pred_set = pred_encoded.cpu()

    #Shaping back to [BS, vars, Nt, Nx, Ny]
    test_out = test_out.permute(0,1,4,2,3)
    pred_set = pred_set.permute(0,1,4,2,3)

    # #Getting the Metrics
    from Utils.metrics import NRMSE
    mses.append(nRMSE(test_out, pred_set))
    # pres.append(PRE(pre, pred_set))

    if t_exp > 50:
        print(f'NRMSE (physical) : {float(NRMSE(pred_set[:, :, 50:], test_out[:, :, 50:])["average"]):.4f}')
        # print(f'PRE : {np.mean(PRE(pre, pred_set[:, :, 50:])):.4f}')
    else:
        print(f'NRMSE (physical) : {float(NRMSE(pred_set, test_out)["average"]):.4f}')
        # print(f'PRE : {np.mean(PRE(pre, pred_set)):.4f}')

# %% 
# temporal_rollout_error(pde, t_exp, mses[0], mses[1], mses[2], plot_loc, metric='MSE', save=True)
# temporal_rollout_error(pde, t_exp, pres[0], pres[1], pres[2], plot_loc, metric='PRE', save=True)
import cmocean as cmo

def imshow_plot(data_matrix, plot_title, plot_loc, xlabel='X', ylabel='Y', 
                cbar_label='Value', vmin=None, vmax=None, cmap=cmo.cm.thermal, 
                save=False, filename='imshow_plot'):
    """
    
    Args:
        data_matrix: 2D array/tensor for heatmap
        plot_title: Title for the plot
        plot_loc: Directory to save plots
        xlabel: X-axis label
        ylabel: Y-axis label
        cbar_label: Colorbar label
        vmin, vmax: Color scale limits (optional)
        cmap: Colormap name (default: 'viridis')
        save: Whether to save the figure
        filename: Base filename for saving (without extension)
    """
    
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
    
    # Convert to numpy if tensor
    if torch.is_tensor(data_matrix):
        data_matrix = data_matrix.cpu().numpy()
    
    # Create the imshow plot
    im = ax.imshow(data_matrix, 
                   aspect='auto',
                   cmap=cmap,
                   interpolation='nearest',
                   vmin=vmin,
                   vmax=vmax,
                   origin='lower')
    
    # Add colorbar with professional styling
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(cbar_label, fontsize=16, rotation=270, labelpad=25)
    cbar.ax.tick_params(labelsize=14)
    cbar.outline.set_linewidth(1.2)
    
    # Professional styling
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    # ax.set_title(plot_title, fontsize=15, pad=15)
    
    plt.tight_layout()
    
    if save:
        # Multiple format saves for different publication needs
        formats = ['pdf']
        for fmt in formats:
            plot_name = f'{plot_loc}/{filename}.{fmt}'
            plt.savefig(plot_name, 
                       dpi=300 if fmt == 'png' else None,
                       bbox_inches='tight',
                       facecolor='none',
                       edgecolor='none',
                       transparent=True,
                       format=fmt)
    
    plt.show()

# %% 
#Interprtability Experiments 
#Convection Operator - Spectral taken from the solver 
from Neural_PDE.Numerical_Solvers.Navier_Stokes.NS_2D_spectral import * 
def convection_operator(uv):
    """
    Compute the convection operator -(v·∇)v
    
    Parameters:
    -----------
    vx, vy : ndarray
        Velocity field components
    kx, ky : ndarray
        Wavenumber grids
    dealias : ndarray
        Dealiasing mask
    
    Returns:
    --------
    conv_x, conv_y : ndarray
        Convection operator applied to velocity field
    """
    #Setting upt he code
    N = 400 #Number of grid points
    tStart = 0.0 #Starting time of the simulation
    tEnd = 1.0 #Simulation ending time
    dt = 0.001 #dt
    nu = 0.001#kinematic viscosity
    L = 1 #Domain Length
    aa = 0.5#parametrisation of initial Vx 
    bb = 0.5#parametrisation of initial Vx 
    solver= Navier_Stokes_2d(N, tStart, tEnd, dt, nu, L, aa, bb)
    kx, ky, dealias = solver.kx[::4, ::4], solver.ky[::4, ::4], solver.dealias[::4, ::4]

    vx, vy = uv[0], uv[1]

    # Compute gradients of velocity components
    dvx_x, dvx_y = grad(vx, kx, ky)
    dvy_x, dvy_y = grad(vy, kx, ky)
    
    # Compute convection term: -(v·∇)v
    conv_x = -(vx * dvx_x + vy * dvx_y)
    conv_y = -(vx * dvy_x + vy * dvy_y)
    
    # Apply dealiasing (2/3 rule)
    conv_x = apply_dealias(conv_x, dealias)
    conv_y = apply_dealias(conv_y, dealias)
    
    return -conv_x, -conv_y


# u, v, p, w, x, t, err = solver.solve()
#Convection Operator - Finite Difference
from PRE.ConvOps_2d import * 
def convection_operator_FD(uv):
    u, v = uv[:, 0], uv[:, 1]
    D_x = ConvOperator(domain='x', order=1) 
    D_y = ConvOperator(domain='y', order=1)

    conv_FD = u*D_x(u) + v*D_y(u), u*D_x(v) + v*D_y(v)
    return conv_FD[0][..., 1:-1, 1:-1].numpy(), conv_FD[1][..., 1:-1, 1:-1].numpy()

# # from PRE.VectorConvOps_Spatial import *
# def divergence_ops_FD(uv, rho):
#     divergence_operator = Divergence(scale=1, taylor_order=4, boundary_cond='periodic', device=device, requires_grad=False)
#     gradient_operator = Gradient(scale=1, taylor_order=4, boundary_cond='periodic', device=device, requires_grad=False)
#     div_FD =   rho*divergence_operator(uv) + dot(uv, gradient_operator(rho))
#     return div_FD

def normalize_array(array, target_min=-1, target_max=1):
    """Normalize array to [target_min, target_max] range"""
    a_min = array.min()
    a_max = array.max()
    normalized = (array - a_min) / (a_max - a_min)  # Scale to [0, 1]
    normalized = normalized * (target_max - target_min) + target_min  # Scale to target range
    return normalized

# %% 
numerical = 'Spectral' #or FD
idx = 10
for t_idx in [5, 25, 50, 75]:
    test = test_out.to(device)
    if numerical == 'FD':
        conv_num_x, conv_num_y = convection_operator_FD(test.cpu())
        title = 'Convection Operator: FD'
        name = 'conv_ops_FD'
    elif numerical == 'Spectral':
        conv_num_x, conv_num_y = convection_operator(test[idx, :, t_idx].cpu().numpy())
        title = 'Convection Operator: Spectral'
        name = 'conv_ops_Spectral'
    
    conv_no = model.convection_operator(test[:,:,t_idx:t_idx+1].permute(0,1,3,4,2)).permute(0,1,4,2,3).cpu().detach().numpy()
    
    if numerical == 'FD':
        conv_num_idx = conv_num_x[idx, t_idx]+conv_num_y[idx, t_idx]
    elif numerical == 'Spectral':
        conv_num_idx = conv_num_x + conv_num_y
   
    conv_num_idx = normalize_array(conv_num_idx)

    imshow_plot(conv_num_idx, 
                plot_title=title,
                plot_loc=plot_loc,
                xlabel='X',
                ylabel='Y',
                cbar_label='',
                save=True,
                filename= f'{name}_{idx}_{data_dist}_{t_idx}')

    conv_no_idx = conv_no[idx, 0, 0] + conv_no[idx, 1, 0]
    conv_no_idx = normalize_array(conv_no_idx)

    imshow_plot(conv_no_idx, 
                plot_title=f'Convection Operator: {arch}',
                plot_loc=plot_loc,
                xlabel='X',
                ylabel='Y',
                cbar_label='',
                save=True,
                filename=f'conv_ops_{arch}_{idx}_{data_dist}_{t_idx}')
