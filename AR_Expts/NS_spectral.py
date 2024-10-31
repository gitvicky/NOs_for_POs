#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Navier-Stokes Spectral solver - NeuralPDE library 
"""
# %%
#Setting up simvue 
import os
import yaml 
import argparse
from omegaconf import DictConfig, OmegaConf

import sys
sys.path.append("..")
from Utils.simvue_utils import flatten_dict

#Config files.
def parse_args():
    parser = argparse.ArgumentParser(description='Training script with YAML config')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    return parser.parse_args()

args = parse_args()
with open(args.config, 'r') as f:
    configuration = yaml.safe_load(f)

run_config = flatten_dict(configuration)
# %% 
from simvue import Run, Client
run = Run(mode='online')
run.init(folder=configuration['Simvue']['folder'], tags=['NPDE', configuration['Model']['arch'], 'POs4NOs', configuration['Physics']['pde'], configuration['Physics']['rollout'], 'Tests'], metadata=run_config)

#setting up the client API 
client = Client()

#Saving the current run file and the git hash of the repo
run.save_file(os.path.abspath(__file__), 'code')
run.save_file(os.path.abspath(args.config), 'code')

import git
repo = git.Repo(search_parent_directories=True)
sha = repo.head.object.hexsha
run.update_metadata({'Git Hash': sha})

# %% 
#Importing the necessary packages
import sys
import numpy as np
from tqdm import tqdm 
import h5py
import torch
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
import time 
from timeit import default_timer
from tqdm import tqdm 

# %%
#Importing the models and utilities. 

if configuration['Model']['arch'] == 'FNO':
    from Neural_PDE.Models.FNO import *
elif configuration['Model']['arch'] == 'ViT':
    from Neural_PDE.Models.ViT import * 
elif configuration['Model']['arch'] == 'CNO':
    from Neural_PDE.Models.CNO import * 

from Neural_PDE.Utils.processing_utils import * 
from Neural_PDE.Utils.training_utils import * 

# %% 
#Setting up locations. 
file_loc = os.getcwd()
data_loc = os.path.dirname(os.getcwd()) + '/Data/'
model_loc = file_loc + '/Weights'
plot_loc = file_loc + '/Plots'

#Setting up the seeds and devices
torch.manual_seed(0)
np.random.seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float32)
# %% 
####################################
# Data Preparation.
####################################

t1 = default_timer()


#Testing with NS_Spectral (for now)
data_loc = '/home/ir-gopa2/rds/rds-ukaea-ap001/ir-gopa2/Code/Neural_PDE/Data'
data =  np.load(data_loc + '/NS_Spectral_combined.npz')

u = data['u'].astype(np.float32)
v = data['v'].astype(np.float32)
p = data['p'].astype(np.float32)

def stacked_fields(variables):
    stack = []
    for var in variables:
        var = torch.from_numpy(var) #Converting to Torch
        var = var.permute(0, 2, 3, 1) #Permuting to be BS, Nx, Ny, Nt
        stack.append(var)
    stack = torch.stack(stack, dim=1)
    return stack

uv = stacked_fields([u,v,p])[:configuration['Data']['ntrain']]

# %%
#Normalising the data -- using the same normalisations for inputs and outputs
normalizer_func = Normalisation(configuration['Data']['normalisation'])
normalizer = normalizer_func(uv)
uv_encoded = normalizer.encode(uv)

#Setting up train and test
from sklearn.model_selection import train_test_split
train_in, test_in, train_out, test_out = train_test_split(uv_encoded[...,:configuration['Data']['t_in']], uv_encoded[...,configuration['Data']['t_in']:configuration['Data']['t_out']], test_size=0.2, random_state=42)
print("Training Input: " + str(train_in.shape))
print("Training Output: " + str(train_out.shape))

#Saving Normalisation 
saved_normalisations = model_loc + '/' + configuration['Model']['arch'] + '_' + configuration['Physics']['pde'] + '_' + run.name + '_' + 'norms.npz'
np.savez(saved_normalisations, 
        in_a=normalizer.a.numpy(), in_b=normalizer.b.numpy(), 
        )
run.save_file(saved_normalisations, 'output')

#Setting up the data loaders
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_in, train_out), batch_size=configuration['Data']['batch size'], shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_in, test_out), batch_size=configuration['Data']['batch size'], shuffle=False)

t2 = default_timer()
print('preprocessing finished, time used:', t2-t1)


# %% 
####################################
# Setting up the Model and Optimizers 
####################################

if configuration['Model']['arch'] == 'FNO':
    if configuration['Model']['operator splitting']: 
        #Currently only including the momentum equation 
        class NO_PO(nn.Module):
            def __init__(self):
                super(NO_PO, self).__init__()
                self.NO_convection = FNO_multi2d(configuration['Data']['t_in'], configuration['Data']['step'], configuration['Model']['modes'], configuration['Model']['modes'], 2, configuration['Model']['width']) 
                self.NO_diffusion = FNO_multi2d(configuration['Data']['t_in'], configuration['Data']['step'], configuration['Model']['modes'], 2, configuration['Model']['width']) 
                self.NO_p_grad =  FNO_multi2d(configuration['Data']['t_in'], configuration['Data']['step'], configuration['Model']['modes'], configuration['Model']['modes'], 1, configuration['Model']['width']) 

            def forward(self, vars):
                uv = vars[:, 0:2]
                p = vars[:, 2:3]
                nu = 0.001
                rhs = - self.NO_convection(uv) + nu*self.NO_diffusion - self.NO_p_grad(p)

                return rhs

    else:
        model = FNO_multi2d(configuration['Data']['t_in'], 
                            configuration['Data']['step'], 
                            configuration['Model']['modes'], 
                            configuration['Model']['modes'], 
                            configuration['Physics']['variables'], 
                            configuration['Model']['width']
                            )
    

#Restarting the run.   
if configuration['Train']['restart']:
    ckpt = client.get_artifact_as_file(configuration['Train']['restart run name'])

    
    os.path.exists(ckpt_path):
    if is_main_process():
        print(f'Loading checkpoint: {ckpt_path}')
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    epoch_init = checkpoint["epoch"]

model.to(device)
run.update_metadata({'Number of Params': int(model.count_params())})
print("Number of model params : " + str(model.count_params()))

#Setting up the optimizer and scheduler, loss and epochs 
optimizer = torch.optim.Adam(model.parameters(), lr=configuration['Opt']['learning rate'], weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=configuration['Opt']['scheduler step'], gamma=configuration['Opt']['scheduler gamma'])
loss_func = LpLoss(size_average=False)
epochs = configuration['Opt']['epochs']

#Setting up the Training pipeline
from Utils import explicit_time
train = explicit_time.Train_Setup(model, train_loader, test_loader, loss_func, optimizer, scheduler, epochs,  configuration['Physics']['rollout'])

# %% 
####################################
#Training
####################################

start_time = default_timer()
for ep in range(epochs): #Training Loop - Epochwise

    model.train()
    t1 = default_timer()
    train_loss, test_loss = train.one_epoch(configuration['Data']['step'], configuration['Data']['t_out']-1)
    t2 = default_timer()

    train_loss = train_loss / len(train_loader)
    test_loss = test_loss / len(test_loader)

    print(f"Epoch {ep}, Time Taken: {round(t2-t1,3)}, Train Loss: {round(train_loss, 3)}, Test Loss: {round(test_loss,3)}")
    run.log_metrics({'Train Loss': train_loss, 'Test Loss': test_loss})
    
    scheduler.step()

    if ep % configuration['Train']['checkpoint']['epochs'] ==0:

        checkpoint = {}
        checkpoint["model"] = model.module.state_dict()
        checkpoint["optimizer"] = optimizer.state_dict() 
        checkpoint["scheduler"] = scheduler.state_dict()
        checkpoint["epoch"] = ep
        torch.save(checkpoint, model_loc + os.path.join(log_dir, "checkpoint.pt"))
        print(f'Epoch: {epoch:<{3}} {blank:<{5}}' + 'Checkpoint saved')

        #Saving the output directory
        if run_simvue:
            try:
                run.save_file( os.getcwd() + '/' + log_dir +"/checkpoint.pt", 'output')
                run.save_file( os.getcwd() + '/' + log_dir + "/logs.txt", 'output')
            except:
                pass




train_time = default_timer() - start_time

# %%
# Saving the Model
saved_model = model_loc + '/' + configuration['Model']['arch'] + '_' + configuration['Physics']['pde'] + '_' +run.name + '.pth'
torch.save( model.state_dict(), saved_model)
run.save_file(saved_model, 'output')

#Evaluation 
eval = explicit_time.Eval_Setup(model, test_in, test_out)
pred_encoded, error = eval.inference(configuration['Data']['step'], configuration['Data']['t_out']-1)

print('(MSE) Testing Error: %.3e' % (error))

run.update_metadata({'Training Time': float(train_time),
                     'MSE Test Error': float(error)
                    })

#Denormalising the test and predictions
test_out = normalizer.decode(test_out.to(device)).cpu()
pred_set = normalizer.decode(pred_encoded.to(device)).cpu()

#Shaping back to [BS, vars, Nt, Nx, Ny]
test_out = test_out.permute(0,1,4,2,3)
pred_set = pred_set.permute(0,1,4,2,3)

# %% 
#Plotting the results 
from Utils.plots import plots_2d_yaml
idx = 0 
plots_2d_yaml(configuration, test_out, pred_set, plot_loc, run, idx)

# %%
run.close()
# %%
