#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training Pipeline. 
"""
# %%
#Setting up simvue 
import os
import yaml 
import argparse

import sys
sys.path.append("..")
from Utils.simvue_utils import flatten_dict
import matplotlib
from matplotlib import pyplot as plt

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
with Run(mode='online') as run:

    run.init(folder=configuration['Simvue']['folder'], tags=[configuration['Physics']['pde'], configuration['Model']['arch'], configuration['Physics']['variable']], metadata=run_config)

    #setting up the client API 
    client = Client()

    #Saving the current run file and the git hash of the repo
    run.save_file(os.path.abspath(__file__), 'code')
    run.save_file(os.path.abspath(args.config), 'code')

    import git
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    run.update_metadata({'Git Hash': sha})

    #Importing the necessary packages
    import sys
    import numpy as np
    from tqdm import tqdm 
    import torch
    import torch.nn.functional as F
    from timeit import default_timer
    from tqdm import tqdm 
    from sklearn.model_selection import train_test_split

    #Setting up locations. 
    file_loc = os.getcwd()
    data_loc = os.path.dirname(os.getcwd()) + '/Data/'
    model_loc = file_loc + '/Weights/' + run.name
    os.mkdir(model_loc)
    plot_loc = file_loc + '/Plots'

    #Setting up the seeds and devices
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_dtype(torch.float32)
    # %%
    #Importing the models and utilities. 
    from model_setup import *
    from Neural_PDE.Utils.processing_utils import * 
    from Neural_PDE.Utils.training_utils import * 

    # %% 
    ####################################
    # Data Preparation.
    ####################################

    t1 = default_timer()

    from data_loaders import *
    pde = configuration['Physics']['pde']
    if pde == 'FDS':
        ins, conds, outs = FDS_Carpark(configuration)
        ins = ins[...,0]
        outs = outs[...,0]
    # %%
    #Normalising the data -- using the same normalisations for inputs and outputs
    normalizer_func = Normalisation(configuration['Data']['normalisation'])
    normalizer_in = normalizer_func(ins)
    ins_encoded = normalizer_in.encode(ins)
    conds_encoded = normalizer_in.encode(conds)

    normalizer_out = normalizer_func(outs)
    outs_encoded = normalizer_out.encode(outs)

    from torch.utils.data import DataLoader, TensorDataset
    train_conds, test_conds, train_outs, test_outs = train_test_split(conds_encoded, outs_encoded, test_size=configuration['Data']['test-train-split'], random_state=42)

    train_loader = DataLoader(TensorDataset(train_conds, train_outs), batch_size=configuration['Data']['batch size'], shuffle=True) 
    test_loader = DataLoader(TensorDataset(test_conds, test_outs), batch_size=configuration['Data']['batch size'], shuffle=False)

    t2 = default_timer()
    print('preprocessing finished, time used:', t2-t1)

    # %% 
    ####################################
    # Setting up the Model and Optimizers 
    ####################################

    model = model_initialisation(configuration, run)
    model.to(device)

    run.update_metadata({'Number of Params': count_parameters(model)})
    print("Number of model params : " + str(count_parameters(model)))

    #Setting up the optimizer and scheduler, loss and epochs 
    optimizer = torch.optim.Adam(model.parameters(), lr=configuration['Opt']['learning rate'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=configuration['Opt']['scheduler step'], gamma=configuration['Opt']['scheduler gamma'])
    
    if configuration['Model']['arch']=='fno':
        loss_func = LpLoss(size_average=False)
    elif configuration['Model']['arch']=='AE':
        loss_func = torch.nn.MSELoss()

    epoch_init = 0
    epochs = configuration['Opt']['epochs']


    # %% 
    ####################################
    #Training
    ####################################
    start_time = default_timer()

    def train_one_epoch():
        model.train()
        train_loss = 0
        for xx, yy in train_loader:
            xx, yy = xx.to(device), yy.to(device)
            grid = ins_encoded.unsqueeze(0).repeat(xx.shape[0], 1, 1, 1, 1).to(device)
            optimizer.zero_grad()
            print('xx shape:', xx.shape, 'yy shape:', yy.shape, 'grid shape:', grid.shape)
            im = model(grid, xx)
            loss = loss_func(im, yy) 
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        
        # Validation Loop
        test_loss = 0
        with torch.no_grad():
            for xx, yy in test_loader:
                xx, yy = xx.to(device), yy.to(device)
                grid = ins_encoded.unsqueeze(0).repeat(xx.shape[0], 1, 1, 1, 1).to(device)
                im = model(grid, xx)
                loss = loss_func(im, yy)
                test_loss += loss.item()

            train_loss =  train_loss / len(train_loader)
            test_loss =  test_loss / len(test_loader)

        return train_loss, test_loss #remember to divide the ntrain/ntest and num_vars at the other end before logging.

    def validation(model, xx, yy):
        with torch.no_grad():
            xx, yy = xx.to(device), yy.to(device)
            grid = ins_encoded.unsqueeze(0).repeat(xx.shape[0], 1, 1, 1, 1).to(device)
            pred = model(grid, xx)

            # Performance Metrics
            MSE_error = (yy - pred).pow(2).mean()
            MAE_error = torch.abs(yy - pred).mean()

        return pred, MSE_error, MAE_error

    for ep in tqdm(range(epoch_init, epochs+1)): #Training Loop - Epochwise

        model.train()
        t1 = default_timer()
        train_loss, test_loss = train_one_epoch()
        t2 = default_timer()

        train_loss = train_loss / len(train_loader)
        test_loss = test_loss / len(test_loader)

        print(f"Epoch {ep}, Time Taken: {round(t2-t1,3)}, Train Loss: {round(train_loss, 5)}, Test Loss: {round(test_loss,5)}")
        current_lr = optimizer.param_groups[0]['lr']
        run.log_metrics({'Train Loss': train_loss, 'Test Loss': test_loss, 'Learning Rate': current_lr}, step=ep)

        scheduler.step()

        #Checkpointing. 
        if ep % configuration['Train']['checkpoint']['epochs'] == 0:
            checkpoint = {}
            checkpoint["model"] = model.state_dict()
            checkpoint["optimizer"] = optimizer.state_dict() 
            checkpoint["scheduler"] = scheduler.state_dict()
            checkpoint["epoch"] = ep
            torch.save(checkpoint, model_loc + "/checkpoint_"+str(ep)+".pt")
            run.save_file(model_loc + "/checkpoint_"+str(ep)+".pt", 'output')
            run.update_metadata({'Epochs': ep})

    train_time = default_timer() - start_time

    # %%
    # Saving the Model
    saved_model = model_loc + '/model.pth'
    torch.save(model.state_dict(), saved_model)
    run.save_file(saved_model, 'output')

    #Evaluation 
    pred_set_encoded, mse, mae = validation(model, test_conds, test_outs)

    print('(MSE) Testing Error: %.3e' % (mse))

    run.update_metadata({'Training Time': float(train_time),
                        'MSE Test Error': float(mse)
                        })
    
    #Denormalising and reshaping the target and the predictions
    pred_set = normalizer_out.decode(pred_set_encoded.to(device)).cpu()
    test_u = normalizer_out.decode(test_outs.to(device)).cpu()
        
    #Plotting performance

    idx = 0
    names = ['targs', 'preds']
    for ii, u_field in enumerate([test_u, pred_set]):

        u_field = u_field[idx][0]
            
        v_min = torch.min(u_field)
        v_max = torch.max(u_field)

        fig = plt.figure(figsize=plt.figaspect(0.5))
        ax = fig.add_subplot(1, 5, 1)
        pcm = ax.imshow(u_field[...,0], cmap=matplotlib.cm.coolwarm)
        ax.title.set_text('T1')
        fig.colorbar(pcm, pad=0.05)

        ax = fig.add_subplot(1, 5, 2)
        pcm = ax.imshow(u_field[...,1], cmap=matplotlib.cm.coolwarm)
        ax.title.set_text('T2')
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
        fig.colorbar(pcm, pad=0.05)

        ax = fig.add_subplot(1, 5, 3)
        pcm = ax.imshow(u_field[...,2], cmap=matplotlib.cm.coolwarm)
        ax.title.set_text('T3')
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
        fig.colorbar(pcm, pad=0.05)


        ax = fig.add_subplot(1, 5, 4)
        pcm = ax.imshow(u_field[...,3], cmap=matplotlib.cm.coolwarm)
        ax.title.set_text('T4')
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
        fig.colorbar(pcm, pad=0.05)


        ax = fig.add_subplot(1, 5, 5)
        pcm = ax.imshow(u_field[...,4], cmap=matplotlib.cm.coolwarm)
        ax.title.set_text('T5')
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
        fig.colorbar(pcm, pad=0.05)

        plot_name = plot_loc + '/AE_T_'+ names[ii] + '_' + run.name + '.png'
        plt.savefig(plot_name)
        run.save_file(plot_name, 'output')

    run.close()
    # %%