# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 25 Oct 2024 
@author: @vgopakum

Training and Inference pipelines for Neural-PDE solvers with explicit temporal rollouts. Data shape - [Batch, variables, Nx, Ny, Nt]

Currently supports: 
    1. Autoregressive rollouts 
    2. Euler Timestep 
    3. MidPoint
    4. Runge-Kutta 4 

"""

import numpy as np 
import torch 
import torch.nn as nn 
from tqdm import tqdm 
from timeit import default_timer
from torchdiffeq import odeint

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
max_grad_clip_norm = 2.0   
# %% 
#Options for temporal propagation. 

def autoregressive(model, u_n, dt=0):
    u_new = model(u_n)
    return u_new

def euler(model, u_n, dt): 
    u_new = u_n + model(u_n)*dt 
    return u_new 

def midpoint(model, u_n, dt):#RK2
    h = dt 
    k1 = model(u_n)
    k2 = model(u_n + 0.5*h*k1)
    u_new = u_n + h*k2
    return u_new

def rk4(model, u_n, dt):
    h = dt 
    
    k1 = model(u_n)
    k2 = model(u_n + 0.5*h*k1)
    k3 = model(u_n + 0.5*h*k2)
    k4 = model(u_n + h*k3)

    u_new = u_n + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
    return u_new

# New torchdiffeq integration
class ODEFunc(nn.Module):
    def __init__(self, model, method='euler'):
        super(ODEFunc, self).__init__()
        self.model = model
        self.method= method
        
    def forward(self, t, x):
        return self.model(x)

def neural_ode(ode_func, u_n, dt):
    # ode_func = ODEFunc(model)
    # ode_func = model
    t = torch.tensor([0, dt]).to(device)
    u_new = odeint(ode_func, u_n, t, method=ode_func.method)[-1]
    return u_new

#Setting up the training pipeline. 
class Train_Setup():
    def __init__(self, model, train_loader, test_loader, loss_func, optimizer, scheduler, epochs, ode_solver='custom', roll_out='AR', noise=False, batch_norm=False): #roll_out = AR, Euler, RK4
        super(Train_Setup, self).__init__()

        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = epochs
        self.device = device

        if ode_solver == 'torchdiffeq':
            self.model = ODEFunc(model, method=roll_out)
            self.forward = neural_ode
        else: 
            if roll_out == 'AR':
                self.forward = autoregressive
            elif roll_out == 'euler':
                self.forward = euler
            elif roll_out == 'midpoint':
                self.forward = midpoint
            elif roll_out == 'rk4':
                self.forward = rk4
            
        self.grad_clip = max_grad_clip_norm
        
        self.noisy_factor = torch.tensor(noise, dtype=torch.float32)

        if batch_norm:
            self.bn = nn.BatchNorm3d(num_features=4, device=device)
        else:
            self.bn = nn.Identity()

        model.to(device)
        self.model.train()


    def one_epoch(self, step, train_T_out, test_T_out, dt=0):
        
        # train_l2_step = 0
        train_l2_full = 0

        for xx, yy in self.train_loader:
            self.optimizer.zero_grad()
            loss = 0
            xx = xx.to(self.device)
            yy = yy.to(self.device)
            batch_size = xx.shape[0]

            for t in range(0, train_T_out, step):
                y = yy[..., t:t + step]
                xx = xx + self.noisy_factor*torch.randn_like(xx) #Adding noise to the input.
                xx = self.bn(xx) #Batch Normalisation
                im = self.forward(self.model, xx, dt)

                #Recon Loss
                loss += self.loss_func(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), -1)

                xx = torch.cat((xx[..., step:], im), dim=-1)

            # train_l2_step += loss.item()
            l2_full = self.loss_func(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1))
            train_l2_full += l2_full.item()

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            if grad_norm > self.grad_clip:
                print(f"Warning: Gradient norm {grad_norm:.2f} exceeded clip threshold")
            self.optimizer.step()

        train_loss = train_l2_full 

        # Validation Loop
        test_loss = 0
        with torch.no_grad():
            for xx, yy in self.test_loader:
                xx, yy = xx.to(self.device), yy.to(self.device)
                batch_size = xx.shape[0]

                for t in range(0, test_T_out, step):
                    y = yy[..., t:t + step]
                    out = self.forward(self.model, xx, dt)

                    if t == 0:
                        pred = out
                    else:
                        pred = torch.cat((pred, out), -1)
    
                    xx = torch.cat((xx[..., step:], out), dim=-1)
                test_loss += self.loss_func(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1)).item()


        return train_loss, test_loss #remember to divide the ntrain/ntest and num_vars at the other end before logging.

    def train(self, run,  step, train_T_out, test_T_out, dt=0):
        for ep in self.epochs():
            self.model.train()
            t1 = default_timer()
            train_loss, test_loss = self.one_epoch(step, train_T_out, test_T_out, dt)
            t2 = default_timer()

            train_loss = train_loss / len(self.train_loader)
            test_loss = test_loss / len(self.test_loader)

            print(f"Epoch {ep}, Time Taken: {round(t2-t1,3)}, Train Loss: {round(train_loss, 3)}, Test Loss: {round(test_loss,3)}")
            run.log_metrics({'Train Loss': train_loss, 'Test Loss': test_loss})
                
            self.scheduler.step()


# %%
class Eval_Setup():
    def __init__(self, model, test_in, test_out, normalizer = 'False', ode_solver = 'custom', roll_out='AR', batch_size=3): #roll_out = AR, Euler, RK4
        super(Eval_Setup, self).__init__()

        self.model = model
        self.test_in = test_in
        self.test_out = test_out
        self.device = device
        
        self.test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(self.test_in, self.test_out), batch_size=batch_size, shuffle=False)

        if ode_solver == 'torchdiffeq':
            self.model = ODEFunc(model, method=roll_out)
            self.forward = neural_ode
        else: 
            if roll_out == 'AR':
                self.forward = autoregressive
            elif roll_out == 'euler':
                self.forward = euler
            elif roll_out == 'midpoint':
                self.forward = midpoint
            elif roll_out == 'rk4':
                self.forward = rk4
                
        model.to(device)
        self.model.eval()

    def inference(self, step, T_out, eval_metric = 'MSE', dt=0):
        pred_set = []
        index = 0
        with torch.no_grad():
            for xx, yy in tqdm(self.test_loader):
                loss = 0
                xx, yy = xx.to(device), yy.to(device)
                for t in range(0, T_out, step):
                    y = yy[..., t:t + step]
                    out = self.forward(self.model, xx, dt)

                    if t == 0:
                        pred = out
                    else:
                        pred = torch.cat((pred, out), -1)

                    xx = torch.cat((xx[..., step:], out), dim=-1)

                pred_set.append(pred)
                index += 1

            pred_set = torch.cat(pred_set, dim=0)

            # Performance Metrics
            if eval_metric == 'MSE':
                error = (pred_set - self.test_out.to(device)).pow(2).mean()
            if eval_metric == 'MAE':
                error = torch.abs(pred_set - self.test_out.to(device)).mean()

        return pred_set, error
