# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 25 Oct 2024 
@author: @vgopakum

Training and Inference pipelines for Neural-PDE solvers using torchdiffeq odeint. Data shape - [Batch, variables, Nx, Ny, Nt]

"""

import numpy as np 
import torch 
import torch.nn as nn 
from tqdm import tqdm 
from timeit import default_timer
from torchdiffeq import odeint, odeint_adjoint

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
max_grad_clip_norm = 2.0   

# New torchdiffeq integration
class ODEFunc(nn.Module):
    def __init__(self, model):
        super(ODEFunc, self).__init__()
        self.model = model
        
    def forward(self, t, x):
        return self.model(x)

#Setting up the training pipeline. 
class Train_Setup():
    def __init__(self, model, train_loader, test_loader, loss_func, optimizer, scheduler, epochs, method='euler', adjoint=False): #roll_out = AR, Euler, RK4
        super(Train_Setup, self).__init__()

        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = epochs
        self.method = method
        self.device = device

        if adjoint:
            self.odesolve = odeint_adjoint
        else:
            self.odesolve = odeint

        self.ode_func = ODEFunc(model)
        self.ode_func.to(device)
            
        self.grad_clip = 2.0
        self.ode_func.train()

    def one_epoch(self, step, T_out, dt=0):
        t = torch.arange(0, T_out +1, dt).to(device)
        train_l2_step = 0
        train_l2_full = 0

        for xx, yy in self.train_loader:
            self.optimizer.zero_grad()
            loss = 0
            xx = xx.to(self.device)
            yy = yy.to(self.device)
            batch_size = xx.shape[0]

            pred = self.odesolve(self.ode_func, xx, t, method=self.method)

            #Recon Loss
            loss += self.loss_func(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1))

            train_l2_step += loss.item()
            l2_full = self.loss_func(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1))
            train_l2_full += l2_full.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=max_grad_clip_norm, norm_type=2.0)
            self.optimizer.step()

        train_loss = train_l2_full 

        # Validation Loop
        test_loss = 0
        with torch.no_grad():
            for xx, yy in self.test_loader:
                xx, yy = xx.to(self.device), yy.to(self.device)
                batch_size = xx.shape[0]

                pred = odeint(self.ode_func, xx, t, method=self.method)

                test_loss += self.loss_func(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1)).item()


        return train_loss, test_loss #remember to divide the ntrain/ntest and num_vars at the other end before logging.

    def train(self, run,  step, T_out, dt=0):
        for ep in self.epochs():
            self.model.train()
            t1 = default_timer()
            train_loss, test_loss = self.train_one_epoch(step, T_out, dt)
            t2 = default_timer()

            train_loss = train_loss / len(self.train_loader)
            test_loss = test_loss / len(self.test_loader)

            print(f"Epoch {ep}, Time Taken: {round(t2-t1,3)}, Train Loss: {round(train_loss, 3)}, Test Loss: {round(test_loss,3)}")
            run.log_metrics({'Train Loss': train_loss, 'Test Loss': test_loss})
                
            self.scheduler.step()


# %%
class Eval_Setup():
    def __init__(self, model, test_in, test_out, normalizer = 'False', method='euler'): #roll_out = AR, Euler, RK4
        super(Eval_Setup, self).__init__()

        self.model = model
        self.test_in = test_in
        self.test_out = test_out
        self.device = device

        self.test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(self.test_in, self.test_out), batch_size=1, shuffle=False)

        self.ode_func = ODEFunc(model)
        self.ode_func.to(device)
        self.ode_func.eval()

    def inference(self, step, T_out, eval_metric = 'MSE', dt=0):
        t = torch.arange(0, T_out+1, dt).to(device)
        pred_set = torch.zeros(self.test_out.shape)
        index = 0
        with torch.no_grad():
            for xx, yy in tqdm(self.test_loader):
                loss = 0
                xx, yy = xx.to(device), yy.to(device)

                pred_set = odeint(self.ode_func, xx, t, method=self.method)

            # Performance Metrics
            if eval_metric == 'MSE':
                error = (pred_set - self.test_out).pow(2).mean()
            if eval_metric == 'MAE':
                error = torch.abs(pred_set - self.test_out).mean()

        return pred_set, error
