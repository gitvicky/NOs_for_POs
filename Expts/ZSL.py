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


import matplotlib
def plots_2d_yaml(configuration, test_out, pred_set, plot_loc, name, idx=0, save=True):
    field = configuration['Physics']['field'].split(',')
    field = [letter.strip() for letter in field]
    for var in range(configuration['Physics']['variables']):
        u_field = test_out[idx][var]
            
        v_min_1 = torch.min(u_field[0])
        v_max_1 = torch.max(u_field[0])

        v_min_2 = torch.min(u_field[configuration['Data']['t_out'] // 2])
        v_max_2 = torch.max(u_field[configuration['Data']['t_out'] // 2])

        v_min_3 = torch.min(u_field[-1])
        v_max_3 = torch.max(u_field[-1])

        fig = plt.figure(figsize=plt.figaspect(0.5))
        ax = fig.add_subplot(2, 3, 1)
        pcm = ax.imshow(u_field[0], cmap=matplotlib.cm.coolwarm, vmin=v_min_1, vmax=v_max_1)
        # ax.title.set_text('Initial')
        ax.title.set_text('t=' + str(configuration['Data']['t_in']))
        ax.set_ylabel('Solution -  ' + field[var])
        fig.colorbar(pcm, pad=0.05)

        ax = fig.add_subplot(2, 3, 2)
        pcm = ax.imshow(u_field[configuration['Data']['t_out'] // 2], cmap=matplotlib.cm.coolwarm, vmin=v_min_2,
                        vmax=v_max_2)
        # ax.title.set_text('Middle')
        ax.title.set_text('t=' + str((configuration['Data']['t_out']+ configuration['Data']['t_in']) // 2))
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
        fig.colorbar(pcm, pad=0.05)

        ax = fig.add_subplot(2, 3, 3)
        pcm = ax.imshow(u_field[ -1], cmap=matplotlib.cm.coolwarm, vmin=v_min_3, vmax=v_max_3)
        # ax.title.set_text('Final')
        ax.title.set_text('t=' + str(configuration['Data']['t_out']))
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
        fig.colorbar(pcm, pad=0.05)

        u_field = pred_set[idx][var]

        ax = fig.add_subplot(2, 3, 4)
        pcm = ax.imshow(u_field[0], cmap=matplotlib.cm.coolwarm, vmin=v_min_1, vmax=v_max_1)
        ax.set_ylabel('Prediction')

        fig.colorbar(pcm, pad=0.05)

        ax = fig.add_subplot(2, 3, 5)
        pcm = ax.imshow(u_field[int(configuration['Data']['t_out']/ 2)], cmap=matplotlib.cm.coolwarm, vmin=v_min_2,
                        vmax=v_max_2)
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
        fig.colorbar(pcm, pad=0.05)

        ax = fig.add_subplot(2, 3, 6)
        pcm = ax.imshow(u_field[-1], cmap=matplotlib.cm.coolwarm, vmin=v_min_3, vmax=v_max_3)
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
        fig.colorbar(pcm, pad=0.05)

        if save == True:
            plot_name = plot_loc + '/' + field[var] + '_' + name + '.png'
            plt.savefig(plot_name)

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

runs = [ops_split]
mses = []
pres = []

for run in runs:
    run_loc = os.getcwd() + '/Weights/' + run
    configuration = yaml.safe_load(open(next(Path(run_loc).glob('*.yaml'))))
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


    # Setting up the Model and Optimizers 
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

plots_2d_yaml(configuration, test_out, pred_set, plot_loc, 'OG', idx=0, save=True)

# %% 
#Using the convection operator from the Compressible case for Incompressible ZSL

from model_setup import * 
run_name = 'terminal-rehab'
model_loc = os.getcwd() + '/Weights/' + run_name
configuration = yaml.safe_load(open(next(Path(model_loc).glob('*.yaml'))))
model_comp = model_initialisation(configuration, normalizer, run=None)
model_path = model_loc + '/model.pth'
model_comp.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=False), strict=False)
model_comp.to(device)

#Setting the convection operator
# model.convection_operator.load_state_dict(model_comp.convection_operator.state_dict())
model.convection_operator = model_comp.convection_operator
configuration = yaml.safe_load(open(next(Path(run_loc).glob('*.yaml'))))


#Evaluation 
test_out = normalizer.encode(test_out.permute(0, 1, 3, 4, 2))
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

# # #Getting the Metrics
# from Utils.metrics import NRMSE
# print(nRMSE(test_out, pred_set))

plots_2d_yaml(configuration, test_out, pred_set, plot_loc, 'ZSL', idx=0, save=True)

# %%
