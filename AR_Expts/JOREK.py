#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FNO modelled over 2D MHD Equations auto-regressively

"""

# %%
configuration = {"Case": 'MHD',
                 "Field": 'rho, Phi, T',
                 "Model": 'FNO',
                 "Epochs": 500,
                 "Batch Size": 10,
                 "Optimizer": 'Adam',
                 "Learning Rate": 0.005,
                 "Scheduler Step": 100,
                 "Scheduler Gamma": 0.5,
                 "Activation": 'GeLU',
                 "Physics Normalisation": 'No',
                 "Normalisation Strategy": 'Min-Max',
                 "T_in": 10,    
                 "T_out": 40,
                 "Step": 5,
                 "Width_time": 32, 
                 "Width_vars": 0,  
                 "Modes": 16,
                 "Variables":3, 
                 "Loss Function": 'LP',
                 "UQ": 'None', #None, Dropout
                 "Ntrain": 1750
                 }

# %%
import os
from simvue import Run
run = Run(mode='online')
run.init(folder="/Neural_PDE", tags=['NPDE', 'FNO', 'MHD', 'JOREK', 'AR'], metadata=configuration)

#Saving the current run file and the git hash of the repo
run.save(os.path.abspath(__file__), 'code')
import git
repo = git.Repo(search_parent_directories=True)
sha = repo.head.object.hexsha
run.update_metadata({'Git Hash': sha})

# %% 
#Importing the necessary packages
import sys
import numpy as np
from tqdm import tqdm 
import torch
import matplotlib
import matplotlib.pyplot as plt
import time 
from timeit import default_timer
from tqdm import tqdm 

#Adding the NPDE package to the system python path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
# %%
#Importing the models and utilities. 
from Neural_PDE.Models.FNO import *
from Neural_PDE.Utils.processing_utils import * 
from Neural_PDE.Utils.training_utils import * 

# %% 
#Settung up locations. 
file_loc = os.getcwd()
# data_loc = os.path.dirname(os.getcwd()) + '/Data'
data_loc = '/home/ir-gopa2/rds/rds-ukaea-ap001/ir-gopa2/Data'
model_loc = file_loc + '/Weights'
plot_loc = file_loc + '/Plots'

#Setting up the seeds and devices
torch.manual_seed(0)
np.random.seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
################################################################
# Loading Data 
################################################################
t1 = default_timer()
data = data_loc + '/FNO_MHD_data_multi_blob_2000_T50.npz' #2000 simulation dataset
# %%
field = configuration['Field']
field = ['rho', 'Phi', 'T']
num_vars = configuration['Variables']

rho = np.load(data)['rho'].astype(np.float32) / 1e20
phi = np.load(data)['Phi'].astype(np.float32) / 1e5
T = np.load(data)['T'].astype(np.float32) / 1e6

rho = np.nan_to_num(rho)
phi = np.nan_to_num(phi)
T = np.nan_to_num(T)

def stacked_fields(variables):
    stack = []
    for var in variables:
        var = torch.from_numpy(var) #Converting to Torch
        var = var.permute(0, 2, 3, 1) #Permuting to be BS, Nx, Ny, Nt
        stack.append(var)
    stack = torch.stack(stack, dim=1)
    return stack

vars = stacked_fields([rho, phi, T])
vars = np.delete(vars, (11, 160, 222, 273, 303, 357, 620, 797, 983, 1275, 1391, 1458, 1554, 1600, 1613, 1888, 1937, 1946, 1959), axis=0) #2000 dataset

x_grid = np.load(data)['Rgrid'][0, :].astype(np.float32)
y_grid = np.load(data)['Zgrid'][:, 0].astype(np.float32)
t_grid = np.load(data)['time'].astype(np.float32)

# %% 
ntrain = configuration['Ntrain']
ntest = 85

#Extracting configuration files
T_in = configuration['T_in']
T_out = configuration['T_out']
step = configuration['Step']
modes = configuration['Modes']
width_vars = configuration['Width_vars']
width_time = configuration['Width_time']
output_size = configuration['Step']
num_vars = configuration['Variables']
batch_size = configuration['Batch Size']

#Setting up train and test
train_a = vars[:ntrain,...,:T_in]
train_u = vars[:ntrain,...,T_in:T_out+T_in]

test_a = vars[-ntest:,...,:T_in]
test_u = vars[-ntest:,...,T_in:T_out+T_in]

print("Training Input: " + str(train_a.shape))
print("Training Output: " + str(train_u.shape))

# %%
#Normalising the train and test datasets with the preferred normalisation. 

norm_strategy = configuration['Normalisation Strategy']

if norm_strategy == 'Min-Max':
    normalizer = MinMax_Normalizer
elif norm_strategy == 'Range':
    normalizer = RangeNormalizer
elif norm_strategy == 'Gaussian':
    normalizer = GaussianNormalizer

a_normalizer = normalizer(train_a)
u_normalizer = normalizer(train_u)

train_a = a_normalizer.encode(train_a)
test_a = a_normalizer.encode(test_a)

train_u = u_normalizer.encode(train_u)
test_u_encoded = u_normalizer.encode(test_u)

#Saving Normalisation 
saved_normalisations = model_loc + '/' + configuration['Model'] + '_' + configuration['Case'] + '_' + run.name + '_' + 'norms.npz'

np.savez(saved_normalisations, 
        in_a=a_normalizer.a.numpy(), in_b=a_normalizer.b.numpy(), 
        out_a=u_normalizer.a.numpy(), out_b=u_normalizer.b.numpy()
        )

run.save(saved_normalisations, 'output')
# %%
#Setting up the data loaders. 
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u_encoded), batch_size=batch_size, shuffle=False)

t2 = default_timer()
print('preprocessing finished, time used:', t2-t1)

# %%
################################################################
# training and evaluation
################################################################

model = FNO_multi2d(T_in, step, modes, modes, num_vars, width_time)#, grid=[x_grid, y_grid])
model.to(device)

run.update_metadata({'Number of Params': int(model.count_params())})
print("Number of model params : " + str(model.count_params()))

#Setting up the optimizer and scheduler, loss and epochs 
optimizer = torch.optim.Adam(model.parameters(), lr=configuration['Learning Rate'], weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=configuration['Scheduler Step'], gamma=configuration['Scheduler Gamma'])
loss_func = LpLoss(size_average=False)
epochs = configuration['Epochs']

# %%
####################################
#Training Loop 
####################################
start_time = default_timer()
for ep in range(epochs): #Training Loop - Epochwise

    model.train()
    t1 = default_timer()
    train_loss, test_loss = train_one_epoch_AR(model, train_loader, test_loader, loss_func, optimizer, step, T_out)
    t2 = default_timer()

    train_loss = train_loss / ntrain / num_vars
    test_loss = test_loss / ntest / num_vars

    print(f"Epoch {ep}, Time Taken: {round(t2-t1,3)}, Train Loss: {round(train_loss, 3)}, Test Loss: {round(test_loss,3)}")
    run.log_metrics({'Train Loss': train_loss, 'Test Loss': test_loss})
    
    scheduler.step()

train_time = default_timer() - start_time


# %%
#Saving the Model
saved_model = model_loc + '/' + configuration['Model'] + '_' + configuration['Case'] + '_' +run.name + '.pth'
torch.save( model.state_dict(), saved_model)
run.save(saved_model, 'output')
# %%
#Validation
pred_set_encoded, mse, mae = validation_AR(model, test_a, test_u_encoded, step, T_out)
# %%
print('(MSE) Testing Error: %.3e' % (mse))
print('(MAE) Testing Error: %.3e' % (mae))

run.update_metadata({'Training Time': float(train_time),
                     'MSE Test Error': float(mse),
                     'MAE Test Error': float(mae)
                    })

#%%
#Denormalising the predictions
pred_set = u_normalizer.decode(pred_set_encoded.to(device)).cpu()

# %% 
#Plotting performance

idx = np.random.randint(0,ntest) 
idx = 0

# %%
for var in range(num_vars):
    u_field = test_u[idx][var]
        
    v_min_1 = torch.min(u_field[0,...,0])
    v_max_1 = torch.max(u_field[0,..., 0])

    v_min_2 = torch.min(u_field[0, ..., int(T_out/ 2)])
    v_max_2 = torch.max(u_field[0, ..., int(T_out/ 2)])

    v_min_3 = torch.min(u_field[0, ..., -1])
    v_max_3 = torch.max(u_field[0, ..., -1])

    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(2, 3, 1)
    pcm = ax.imshow(u_field[..., 0], cmap=matplotlib.cm.coolwarm, extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_1, vmax=v_max_1)
    # ax.title.set_text('Initial')
    ax.title.set_text('t=' + str(T_in))
    ax.set_ylabel('Solution -  ' + field[var])
    fig.colorbar(pcm, pad=0.05)

    ax = fig.add_subplot(2, 3, 2)
    pcm = ax.imshow(u_field[..., int(T_out/ 2)], cmap=matplotlib.cm.coolwarm, extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_2,
                    vmax=v_max_2)
    # ax.title.set_text('Middle')
    ax.title.set_text('t=' + str(int((T_out+ T_in) / 2)))
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    fig.colorbar(pcm, pad=0.05)

    ax = fig.add_subplot(2, 3, 3)
    pcm = ax.imshow(u_field[..., -1], cmap=matplotlib.cm.coolwarm, extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_3, vmax=v_max_3)
    # ax.title.set_text('Final')
    ax.title.set_text('t=' + str(T_out+ T_in))
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    fig.colorbar(pcm, pad=0.05)

    u_field = pred_set[idx][var]

    ax = fig.add_subplot(2, 3, 4)
    pcm = ax.imshow(u_field[..., 0], cmap=matplotlib.cm.coolwarm, extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_1, vmax=v_max_1)
    ax.set_ylabel('FNO' )

    fig.colorbar(pcm, pad=0.05)

    ax = fig.add_subplot(2, 3, 5)
    pcm = ax.imshow(u_field[..., int(T_out/ 2)], cmap=matplotlib.cm.coolwarm, extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_2,
                    vmax=v_max_2)
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    fig.colorbar(pcm, pad=0.05)

    ax = fig.add_subplot(2, 3, 6)
    pcm = ax.imshow(u_field[..., -1], cmap=matplotlib.cm.coolwarm, extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_3, vmax=v_max_3)
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    fig.colorbar(pcm, pad=0.05)

    plot_name = plot_loc + '/' + field[var] + '_' + run.name + '.png'
    plt.savefig(plot_name)
    run.save(plot_name, 'output')

run.close()
# %%
