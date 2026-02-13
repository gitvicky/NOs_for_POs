# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 25 Oct 2024 
@author: @vgopakum

Training and Inference pipelines for Neural-PDE solvers with explicit temporal rollouts. 
Data shape - [Batch, variables, Nx, Ny, Nt]

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
import warnings

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
max_grad_clip_norm = 1.0

# %% 
# Options for temporal propagation. 

def autoregressive(model, u_n, dt=0):
    """Simple autoregressive prediction."""
    u_new = model(u_n)
    return u_new

def euler(model, u_n, dt): 
    """Euler method for temporal integration."""
    if dt == 0:
        raise ValueError("dt must be non-zero for Euler method")
    u_new = u_n + model(u_n) * dt 
    # print(torch.mean(model(u_n)))
    return u_new 

def midpoint(model, u_n, dt):
    """Midpoint method (RK2) for temporal integration."""
    if dt == 0:
        raise ValueError("dt must be non-zero for midpoint method")
    h = dt 
    k1 = model(u_n)
    k2 = model(u_n + 0.5 * h * k1)
    u_new = u_n + h * k2
    return u_new

def rk4(model, u_n, dt):
    """Runge-Kutta 4th order method for temporal integration."""
    if dt == 0:
        raise ValueError("dt must be non-zero for RK4 method")
    h = dt 
    
    k1 = model(u_n)
    k2 = model(u_n + 0.5 * h * k1)
    k3 = model(u_n + 0.5 * h * k2)
    k4 = model(u_n + h * k3)

    u_new = u_n + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
    return u_new

# New torchdiffeq integration
class ODEFunc(nn.Module):
    def __init__(self, model, method='euler'):
        super(ODEFunc, self).__init__()
        self.model = model
        self.method = method
        
    def forward(self, t, x):
        return self.model(x)

def neural_ode(ode_func, u_n, dt):
    """Neural ODE integration using torchdiffeq."""
    if dt == 0:
        raise ValueError("dt must be non-zero for neural ODE")
    t = torch.tensor([0.0, dt], device=u_n.device, dtype=u_n.dtype)
    u_new = odeint(ode_func, u_n, t, method=ode_func.method)[-1]
    return u_new

# Setting up the training pipeline. 
class Train_Setup():
    def __init__(self, model, train_loader, test_loader, loss_func, optimizer, scheduler, epochs, 
                 ode_solver='custom', roll_out='AR', noise=False, batch_norm=False, 
                 grad_clip_norm=None, debug=False):
        """
        Initialize training setup.
        
        Args:
            model: Neural network model
            train_loader: Training data loader
            test_loader: Test data loader
            loss_func: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            epochs: Number of training epochs
            ode_solver: 'custom' or 'torchdiffeq'
            roll_out: 'AR', 'euler', 'midpoint', 'rk4'
            noise: Noise factor for input (False or float)
            batch_norm: Whether to use batch normalization
            grad_clip_norm: Gradient clipping norm (None uses default)
            debug: Enable debug prints
        """
        super(Train_Setup, self).__init__()

        # Validate inputs
        if roll_out not in ['AR', 'euler', 'midpoint', 'rk4']:
            raise ValueError(f"Invalid roll_out method: {roll_out}")
        if ode_solver not in ['custom', 'torchdiffeq']:
            raise ValueError(f"Invalid ode_solver: {ode_solver}")

        self.original_model = model  # Keep reference to original model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = epochs
        self.device = device
        self.debug = debug
        self.roll_out_method = roll_out

        if ode_solver == 'torchdiffeq':
            self.model = ODEFunc(model, method=roll_out)
            self.forward = neural_ode
        else: 
            self.model = model
            if roll_out == 'AR':
                self.forward = autoregressive
            elif roll_out == 'euler':
                self.forward = euler
            elif roll_out == 'midpoint':
                self.forward = midpoint
            elif roll_out == 'rk4':
                self.forward = rk4
            
        # Gradient clipping
        self.grad_clip = grad_clip_norm if grad_clip_norm is not None else max_grad_clip_norm
        
        # Noise setup
        if isinstance(noise, bool):
            self.noisy_factor = 0.0 if not noise else 0.01  # Default small noise
        else:
            self.noisy_factor = float(noise)

        if batch_norm:
            # Infer number of features from a sample batch
            sample_batch = next(iter(train_loader))[0]
            num_features = sample_batch.shape[1]  # Assuming [B, C, H, W, T] format
            self.bn = nn.BatchNorm3d(num_features=num_features).to(device)
        else:
            self.bn = nn.Identity()

        # Move model to device
        self.model.to(device)
        
    def one_epoch(self, step, train_T_out, test_T_out, dt=0):
        """Run one epoch of training and validation matching the original train_one_epoch_AR."""
        
        train_l2_step = 0
        train_l2_full = 0
        
        # Training loop
        self.model.train()
        for xx, yy in self.train_loader:
            self.optimizer.zero_grad()
            loss = 0
            xx = xx.to(self.device, non_blocking=True)
            yy = yy.to(self.device, non_blocking=True)
            batch_size = xx.shape[0]
            
            
            for t in range(0, train_T_out, step):
                y = yy[..., t:t + step]
                
                im = self.forward(self.model, xx, dt)# Ensure output has time dimension

                # Compute loss for this step
                loss += self.loss_func(im.reshape(batch_size, -1), y.reshape(batch_size, -1))
                
                # Collect predictions for full sequence loss
                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), -1)
                
                # Update input for next timestep (sliding window)
                xx = torch.cat((xx[..., step:], im), dim=-1)
            
            train_l2_step += loss.item()
            
            # Compute full sequence loss
            l2_full = self.loss_func(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1))
            train_l2_full += l2_full.item()
            
            # Backward pass
            loss.backward(retain_graph=True)
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip, norm_type=2.0)
            self.optimizer.step()
        
        # Return the step loss (following original convention)
        train_loss = train_l2_full
        
        # Validation Loop
        test_loss = 0
        self.model.eval()
        with torch.no_grad():
            for xx, yy in self.test_loader:
                xx, yy = xx.to(self.device, non_blocking=True), yy.to(self.device, non_blocking=True)
                batch_size = xx.shape[0]
                
                for t in range(0, test_T_out, step):
                    y = yy[..., t:t + step]
                    
                    out = self.forward(self.model, xx, dt)
                    
                    if t == 0:
                        pred = out
                    else:
                        pred = torch.cat((pred, out), -1)
                    
                    # Update input for next timestep
                    xx = torch.cat((xx[..., step:], out), dim=-1)
                
                test_loss += self.loss_func(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1)).item()
        
        # Note: The division by dataset size should be done outside this function
        # to match the original implementation
        return train_loss, test_loss


    # def one_epoch(self, step, train_T_out, test_T_out, dt=0):
    #     """Run one epoch of training and validation."""
        
    #     # Validate time integration requirements
    #     if self.roll_out_method != 'AR' and dt == 0:
    #         raise ValueError(f"dt must be non-zero for {self.roll_out_method} method")
        
    #     train_l2_full = 0
    #     num_train_batches = 0

    #     # Training loop
    #     self.model.train()
    #     for xx, yy in self.train_loader:
    #         self.optimizer.zero_grad()
    #         loss = 0
    #         xx = xx.to(self.device)
    #         yy = yy.to(self.device)
    #         batch_size = xx.shape[0]

    #         # # Apply noise if specified
    #         # if self.noisy_factor > 0:
    #         #     xx = xx + self.noisy_factor * torch.randn_like(xx)
            
    #         # # Apply batch normalization
    #         # xx = self.bn(xx)

    #         pred_list = []
            
    #         for t in range(0, train_T_out, step):
    #             y_target = yy[..., t:t + step]
                
    #             # Forward pass - use first timestep for input
    #             im = self.forward(self.model, xx, dt)
                
    #             # # Ensure output has correct shape
    #             # if im.dim() == xx.dim() - 1:  # Missing time dimension
    #             #     im = im.unsqueeze(-1)


    #             # Compute step loss
    #             step_loss = self.loss_func(
    #                 im.reshape(batch_size, -1), 
    #                 y_target.reshape(batch_size, -1)
    #             )
    #             loss += step_loss
                
    #             # Store prediction
    #             pred_list.append(im.detach())
                
    #             # Update input for next timestep (sliding window)
    #             xx = torch.cat((xx[..., step:], im), dim=-1)
            
    #         # Backward pass
    #         loss.backward()
            
    #         # Gradient clipping with proper checking
    #         grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
    #         if self.debug and grad_norm > self.grad_clip:
    #             print(f"Warning: Gradient norm {grad_norm:.4f} was clipped to {self.grad_clip}")
            
    #         self.optimizer.step()

    #         # Compute full sequence loss for logging
    #         if pred_list:
    #             pred_full = torch.cat(pred_list, dim=-1)
    #             l2_full = self.loss_func(
    #                 pred_full.reshape(batch_size, -1), 
    #                 yy[..., :train_T_out].reshape(batch_size, -1).detach()
    #             ).item()
    #             train_l2_full += l2_full
    #             num_train_batches += 1

    #     # Validation loop
    #     test_l2_full = 0
    #     num_test_batches = 0
        
    #     self.model.eval()
    #     with torch.no_grad():
    #         for xx, yy in self.test_loader:
    #             xx, yy = xx.to(self.device), yy.to(self.device)
    #             batch_size = xx.shape[0]

    #             pred_list = []
                
    #             for t in range(0, test_T_out, step):
    #                 out = self.forward(self.model, xx, dt)
                    
    #                 # # Ensure output has correct shape
    #                 # if out.dim() == xx.dim() - 1:
    #                 #     out = out.unsqueeze(-1)
                    
    #                 pred_list.append(out)
    #                 xx = torch.cat((xx[..., step:], out), dim=-1)
                
    #             # Compute full sequence loss
    #             if pred_list:
    #                 pred_full = torch.cat(pred_list, dim=-1)
    #                 test_loss = self.loss_func(
    #                     pred_full.reshape(batch_size, -1), 
    #                     yy[..., :test_T_out].reshape(batch_size, -1)
    #                 ).item()
    #                 test_l2_full += test_loss
    #                 num_test_batches += 1

    #     # Return average losses
    #     avg_train_loss = train_l2_full / max(num_train_batches, 1)
    #     avg_test_loss = test_l2_full / max(num_test_batches, 1)
        
    #     return avg_train_loss, avg_test_loss

    def train(self, run, step, train_T_out, test_T_out, dt=0):
        """Main training loop."""
        
        if self.debug:
            print(f"Starting training with:")
            print(f"  Method: {self.roll_out_method}")
            print(f"  Step size: {step}")
            print(f"  Train T_out: {train_T_out}")
            print(f"  Test T_out: {test_T_out}")
            print(f"  dt: {dt}")
        
        for ep in range(self.epochs):
            t1 = default_timer()
            
            try:
                train_loss, test_loss = self.one_epoch(step, train_T_out, test_T_out, dt)
                t2 = default_timer()

                print(f"Epoch {ep+1}/{self.epochs}, Time: {t2-t1:.3f}s, "
                      f"Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")
                
                # Log metrics
                if hasattr(run, 'log_metrics'):
                    run.log_metrics({
                        'epoch': ep,
                        'train_loss': train_loss, 
                        'test_loss': test_loss,
                        'epoch_time': t2-t1
                    })
                
                # Update learning rate
                if self.scheduler is not None:
                    self.scheduler.step()
                    
            except Exception as e:
                print(f"Error in epoch {ep}: {str(e)}")
                if self.debug:
                    raise e
                continue


# %%
class Eval_Setup():
    def __init__(self, model, test_in, test_out, normalizer=None, ode_solver='custom', 
                 roll_out='AR', batch_size=3, debug=False):
        """
        Initialize evaluation setup.
        
        Args:
            model: Trained neural network model
            test_in: Test input data
            test_out: Test output data  
            normalizer: Data normalizer (not currently used)
            ode_solver: 'custom' or 'torchdiffeq'
            roll_out: 'AR', 'euler', 'midpoint', 'rk4'
            batch_size: Batch size for evaluation
            debug: Enable debug prints
        """
        super(Eval_Setup, self).__init__()

        # Validate inputs
        if roll_out not in ['AR', 'euler', 'midpoint', 'rk4']:
            raise ValueError(f"Invalid roll_out method: {roll_out}")

        self.original_model = model
        self.test_in = test_in
        self.test_out = test_out
        self.device = device
        self.debug = debug
        self.roll_out_method = roll_out
        
        self.test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(self.test_in, self.test_out), 
            batch_size=batch_size, 
            shuffle=False
        )

        # Setup forward function and model
        if ode_solver == 'torchdiffeq':
            self.model = ODEFunc(model, method=roll_out)
            self.forward = neural_ode
        else: 
            self.model = model
            if roll_out == 'AR':
                self.forward = autoregressive
            elif roll_out == 'euler':
                self.forward = euler
            elif roll_out == 'midpoint':
                self.forward = midpoint
            elif roll_out == 'rk4':
                self.forward = rk4
                
        self.model.to(device)
        self.model.eval()

    def inference(self, step, T_out, eval_metric='MSE', dt=0):
        """
        Run inference on test data.
        
        Args:
            step: Time step size
            T_out: Total time steps to predict
            eval_metric: 'MSE' or 'MAE'
            dt: Time increment for integration methods
            
        Returns:
            pred_set: Predictions
            error: Evaluation metric value
        """
        
        # Validate inputs
        if self.roll_out_method != 'AR' and dt == 0:
            raise ValueError(f"dt must be non-zero for {self.roll_out_method} method")
        if eval_metric not in ['MSE', 'MAE']:
            raise ValueError(f"Invalid eval_metric: {eval_metric}")
        
        pred_set = []
        
        self.model.eval()
        with torch.no_grad():
            for xx, yy in tqdm(self.test_loader, desc="Running inference"):
                xx, yy = xx.to(device), yy.to(device)
                batch_size = xx.shape[0]
                
                pred_list = []
                
                for t in range(0, T_out, step):
                    out = self.forward(self.model, xx, dt)
                    
                    # # Ensure output has correct shape
                    # if out.dim() == xx.dim() - 1:
                    #     out = out.unsqueeze(-1)
                    
                    pred_list.append(out)
                    xx = torch.cat((xx[..., step:], out), dim=-1)

                # Concatenate predictions
                if pred_list:
                    pred = torch.cat(pred_list, dim=-1)
                    pred_set.append(pred)

            if pred_set:
                pred_set = torch.cat(pred_set, dim=0)

                # Compute performance metrics
                test_out_device = self.test_out.to(device)
                
                # Ensure shapes match
                min_time_steps = min(pred_set.shape[-1], test_out_device.shape[-1])
                pred_set_trimmed = pred_set[..., :min_time_steps]
                test_out_trimmed = test_out_device[..., :min_time_steps]
                
                if eval_metric == 'MSE':
                    error = (pred_set_trimmed - test_out_trimmed).pow(2).mean()
                elif eval_metric == 'MAE':
                    error = torch.abs(pred_set_trimmed - test_out_trimmed).mean()
                
                if self.debug:
                    print(f"Prediction shape: {pred_set.shape}")
                    print(f"Ground truth shape: {self.test_out.shape}")
                    print(f"{eval_metric} Error: {error.item():.6f}")
                    
            else:
                pred_set = torch.empty(0)
                error = float('inf')
                warnings.warn("No predictions generated")

        return pred_set, error