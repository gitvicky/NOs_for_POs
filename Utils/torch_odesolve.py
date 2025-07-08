# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Neural ODE Training Pipeline with torchdiffeq integration

# """

# import numpy as np 
# import torch 
# import torch.nn as nn 
# from tqdm import tqdm 
# from timeit import default_timer
# from torchdiffeq import odeint, odeint_adjoint
# import warnings

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# max_grad_clip_norm = 2.0   

# # %% 
# # ODEFunc to be used with torchdiffeq
# class ODEFunc(nn.Module):
#     def __init__(self, model):
#         super(ODEFunc, self).__init__()
#         self.model = model
        
#     def forward(self, t, x):
#         """
#         Forward pass for ODE function.
        
#         Args:
#             t: Time tensor (scalar or 1D)
#             x: State tensor with shape [batch, channels, height, width]
            
#         Returns:
#             Time derivative of x
#         """
#         return self.model(x)

# class Train_Setup():
#     def __init__(self, model, train_loader, test_loader, loss_func, optimizer, scheduler, epochs, 
#                  method='rk4', adjoint=False, rtol=1e-3, atol=1e-4, grad_clip_norm=None, debug=False):
#         """
#         Initialize Neural ODE training setup.
        
#         Args:
#             model: Neural network model
#             train_loader: Training data loader
#             test_loader: Test data loader
#             loss_func: Loss function
#             optimizer: Optimizer
#             scheduler: Learning rate scheduler
#             epochs: Number of training epochs (int)
#             method: ODE solver method ('dopri5', 'euler', 'rk4', etc.)
#             adjoint: Whether to use adjoint method for memory efficiency
#             rtol: Relative tolerance for ODE solver
#             atol: Absolute tolerance for ODE solver
#             grad_clip_norm: Gradient clipping norm (None uses default)
#             debug: Enable debug prints
#         """
#         super(Train_Setup, self).__init__()

#         # Validate inputs
#         if not isinstance(epochs, int) or epochs <= 0:
#             raise ValueError("epochs must be a positive integer")
        
#         valid_methods = ['dopri5', 'dopri8', 'euler', 'midpoint', 'rk4', 'explicit_adams', 'implicit_adams']
#         if method not in valid_methods:
#             warnings.warn(f"Method '{method}' may not be supported. Valid methods: {valid_methods}")

#         # self.original_model = model  # Keep reference to original model
#         self.train_loader = train_loader
#         self.test_loader = test_loader
#         self.loss_func = loss_func
#         self.optimizer = optimizer
#         self.scheduler = scheduler
#         self.epochs = epochs
#         self.device = device
#         self.method = method
#         self.adjoint = adjoint
#         self.rtol = rtol
#         self.atol = atol
#         self.debug = debug
        
#         # Wrap model in ODEFunc
#         self.odefunc = ODEFunc(model).to(device)
#         self.model = model  # Keep reference for compatibility
        
#         # Gradient clipping
#         self.grad_clip = grad_clip_norm if grad_clip_norm is not None else max_grad_clip_norm
        
#         # Move model to device
#         self.model.to(device)
        
#         if self.debug:
#             print(f"Initialized Neural ODE training with:")
#             print(f"  Method: {method}")
#             print(f"  Adjoint: {adjoint}")
#             print(f"  Tolerances: rtol={rtol}, atol={atol}")
#             print(f"  Device: {device}")

#     def _validate_tensor_shapes(self, xx, yy, expected_time_dim):
#         """Validate input tensor shapes and provide helpful error messages."""
#         if xx.dim() != 5:
#             raise ValueError(f"Input tensor xx should have 5 dimensions [B,C,H,W,T], got shape {xx.shape}")
#         if yy.dim() != 5:
#             raise ValueError(f"Target tensor yy should have 5 dimensions [B,C,H,W,T], got shape {yy.shape}")
        
#         if xx.shape[0] != yy.shape[0]:
#             raise ValueError(f"Batch size mismatch: xx has {xx.shape[0]}, yy has {yy.shape[0]}")
        
#         if yy.shape[-1] != expected_time_dim:
#             warnings.warn(f"Expected {expected_time_dim} time steps, but yy has {yy.shape[-1]}")

#     def one_epoch(self, step, train_T_out, test_T_out, dt=0.01):
#         """
#         Run one epoch of training and validation.
        
#         Args:
#             step: Step size (not used in this ODE implementation, kept for compatibility)
#             train_T_out: Number of time steps to predict during training
#             test_T_out: Number of time steps to predict during testing
#             dt: Time step size
            
#         Returns:
#             Average training loss, average test loss
#         """
        
#         if dt <= 0:
#             raise ValueError("dt must be positive for ODE integration")
        
#         if train_T_out <= 0 or test_T_out <= 0:
#             raise ValueError("T_out values must be positive")
        
#         train_l2_full = 0
#         num_train_batches = 0

#         # Training loop
#         self.model.train()
#         self.odefunc.train()
        
#         for batch_idx, (xx, yy) in enumerate(self.train_loader):
#             try:
#                 self.optimizer.zero_grad()
#                 xx = xx.to(self.device)
#                 yy = yy.to(self.device)
#                 batch_size = xx.shape[0]

#                 # Validate shapes
#                 # self._validate_tensor_shapes(xx, yy, train_T_out)

#                 # Extract initial condition (first time step)
#                 x0 = xx[..., 0]  # Shape: [batch, vars, nx, ny]
                
#                 # Create time points for entire trajectory
#                 t_span = torch.linspace(0, dt * train_T_out, train_T_out + 1, 
#                                       device=self.device, dtype=xx.dtype)
                
#                 # Use adjoint method for memory-efficient backprop if enabled
#                 try:
#                     if self.adjoint:
#                         pred = odeint_adjoint(
#                             self.odefunc, x0, t_span, 
#                             method=self.method, rtol=self.rtol, atol=self.atol
#                         )
#                     else:
#                         pred = odeint(
#                             self.odefunc, x0, t_span, 
#                             method=self.method, rtol=self.rtol, atol=self.atol
#                         )
#                 except Exception as e:
#                     print(f"ODE integration failed in training batch {batch_idx}: {str(e)}")
#                     if self.debug:
#                         print(f"  Input shape: {x0.shape}")
#                         print(f"  Time span: {t_span}")
#                         raise e
#                     continue
                
#                 # Remove initial condition and reshape
#                 # pred shape: [time_steps+1, batch, vars, nx, ny]
#                 pred = pred[1:]  # Remove t=0, shape: [time_steps, batch, vars, nx, ny]
#                 pred = pred.permute(1, 2, 3, 4, 0)  # Reshape to [batch, vars, nx, ny, time_steps]
                
#                 # Ensure we have the right number of time steps
#                 actual_time_steps = min(pred.shape[-1], yy.shape[-1])
#                 pred_trimmed = pred[..., :actual_time_steps]
#                 yy_trimmed = yy[..., :actual_time_steps]

#                 # Compute loss on the full trajectory
#                 l2_full = self.loss_func(
#                     pred_trimmed.reshape(batch_size, -1), 
#                     yy_trimmed.reshape(batch_size, -1)
#                 )
#                 train_l2_full += l2_full.item()
#                 num_train_batches += 1
                
#                 # Backward pass and optimization
#                 l2_full.backward()
                
#                 # Gradient clipping with proper warning
#                 grad_norm = torch.nn.utils.clip_grad_norm_(
#                     list(self.model.parameters()) + list(self.odefunc.parameters()), 
#                     self.grad_clip
#                 )
#                 if self.debug and grad_norm > self.grad_clip:
#                     print(f"Batch {batch_idx}: Gradient norm {grad_norm:.4f} was clipped to {self.grad_clip}")
                
#                 self.optimizer.step()
                
#             except RuntimeError as e:
#                 print(f"Runtime error in training batch {batch_idx}: {str(e)}")
#                 if self.debug:
#                     raise e
#                 continue

#         # Validation Loop
#         test_l2_full = 0
#         num_test_batches = 0
        
#         self.model.eval()
#         self.odefunc.eval()
        
#         with torch.no_grad():
#             for batch_idx, (xx, yy) in enumerate(self.test_loader):
#                 try:
#                     xx, yy = xx.to(self.device), yy.to(self.device)
#                     batch_size = xx.shape[0]
                    
#                     # Validate shapes
#                     # self._validate_tensor_shapes(xx, yy, test_T_out)
                    
#                     # Extract initial condition
#                     x0 = xx[..., 0]
                    
#                     # Create time points for entire trajectory
#                     t_span = torch.linspace(0, dt * test_T_out, test_T_out + 1, 
#                                           device=self.device, dtype=xx.dtype)
                    
#                     # Forward pass with odeint (no need for adjoint during validation)
#                     try:
#                         pred = odeint(
#                             self.odefunc, x0, t_span, 
#                             method=self.method, rtol=self.rtol, atol=self.atol
#                         )
#                     except Exception as e:
#                         print(f"ODE integration failed in validation batch {batch_idx}: {str(e)}")
#                         if self.debug:
#                             raise e
#                         continue
                    
#                     # Remove initial condition and reshape
#                     pred = pred[1:]  # Remove t=0
#                     pred = pred.permute(1, 2, 3, 4, 0)  # Reshape to [batch, vars, nx, ny, time_steps]
                    
#                     # Ensure we have the right number of time steps
#                     actual_time_steps = min(pred.shape[-1], yy.shape[-1])
#                     pred_trimmed = pred[..., :actual_time_steps]
#                     yy_trimmed = yy[..., :actual_time_steps]

#                     # Compute validation loss
#                     test_loss = self.loss_func(
#                         pred_trimmed.reshape(batch_size, -1), 
#                         yy_trimmed.reshape(batch_size, -1)
#                     )
#                     test_l2_full += test_loss.item()
#                     num_test_batches += 1
                    
#                 except RuntimeError as e:
#                     print(f"Runtime error in validation batch {batch_idx}: {str(e)}")
#                     if self.debug:
#                         raise e
#                     continue

#         # Return average losses
#         avg_train_loss = train_l2_full / max(num_train_batches, 1)
#         avg_test_loss = test_l2_full / max(num_test_batches, 1)
        
#         return avg_train_loss, avg_test_loss

#     def train(self, run, step, train_T_out, test_T_out, dt=0.01):
#         """
#         Main training loop.
        
#         Args:
#             run: Experiment tracking object (e.g., wandb run)
#             step: Step size (kept for compatibility)
#             train_T_out: Training time steps
#             test_T_out: Testing time steps  
#             dt: Time increment
#         """
        
#         if self.debug:
#             print(f"Starting Neural ODE training:")
#             print(f"  Epochs: {self.epochs}")
#             print(f"  Train T_out: {train_T_out}")
#             print(f"  Test T_out: {test_T_out}")
#             print(f"  dt: {dt}")
        
#         for ep in range(self.epochs):  # Fixed: was self.epochs()
#             t1 = default_timer()
#             train_loss, test_loss = self.one_epoch(step, train_T_out, test_T_out, dt)
#             t2 = default_timer()

#             print(f"Epoch {ep+1}/{self.epochs}, Time: {t2-t1:.3f}s, "
#                     f"Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")
            
#             # Log metrics
#             if hasattr(run, 'log_metrics'):
#                 run.log_metrics({
#                     'epoch': ep,
#                     'train_loss': train_loss, 
#                     'test_loss': test_loss,
#                     'epoch_time': t2-t1
#                 })
            
#             # Update learning rate
#             if self.scheduler is not None:
#                 self.scheduler.step()
                
#     # except Exception as e:
#     #     print(f"Error in epoch {ep}: {str(e)}")
#     #     if self.debug:
#     #         raise e
#     #     continue

# # %%
# class Eval_Setup():
#     def __init__(self, model, test_in, test_out, normalizer=None, method='dopri5', 
#                  batch_size=3, rtol=1e-3, atol=1e-4, debug=False):
#         """
#         Initialize Neural ODE evaluation setup.
        
#         Args:
#             model: Trained neural network model
#             test_in: Test input data
#             test_out: Test output data
#             normalizer: Data normalizer (not currently used)
#             method: ODE solver method
#             batch_size: Batch size for evaluation
#             rtol: Relative tolerance for ODE solver
#             atol: Absolute tolerance for ODE solver
#             debug: Enable debug prints
#         """
#         super(Eval_Setup, self).__init__()

#         # Validate inputs
#         valid_methods = ['dopri5', 'dopri8', 'euler', 'midpoint', 'rk4', 'explicit_adams', 'implicit_adams']
#         if method not in valid_methods:
#             warnings.warn(f"Method '{method}' may not be supported. Valid methods: {valid_methods}")

#         self.original_model = model
#         self.test_in = test_in
#         self.test_out = test_out
#         self.device = device
#         self.method = method
#         self.rtol = rtol
#         self.atol = atol
#         self.debug = debug
        
#         # Validate input data shapes
#         if test_in.dim() != 5:
#             raise ValueError(f"test_in should have 5 dimensions [B,C,H,W,T], got shape {test_in.shape}")
#         if test_out.dim() != 5:
#             raise ValueError(f"test_out should have 5 dimensions [B,C,H,W,T], got shape {test_out.shape}")
        
#         self.test_loader = torch.utils.data.DataLoader(
#             torch.utils.data.TensorDataset(self.test_in, self.test_out), 
#             batch_size=batch_size, 
#             shuffle=False
#         )
        
#         # Wrap model in ODEFunc
#         self.odefunc = ODEFunc(model).to(device)
#         self.model = model
                
#         self.model.to(device)
#         self.model.eval()
#         self.odefunc.eval()

#     def inference(self, step, T_out, eval_metric='MSE', dt=0.01):
#         """
#         Run inference on test data.
        
#         Args:
#             step: Step size (kept for compatibility)
#             T_out: Number of time steps to predict
#             eval_metric: 'MSE' or 'MAE'
#             dt: Time step size
            
#         Returns:
#             pred_set: Predictions tensor
#             error: Evaluation metric value
#         """
        
#         # Validate inputs
#         if dt <= 0:
#             raise ValueError("dt must be positive for ODE integration")
#         if T_out <= 0:
#             raise ValueError("T_out must be positive")
#         if eval_metric not in ['MSE', 'MAE']:
#             raise ValueError(f"Invalid eval_metric: {eval_metric}. Use 'MSE' or 'MAE'")
        
#         pred_set = []
#         successful_batches = 0
        
#         self.model.eval()
#         self.odefunc.eval()
        
#         with torch.no_grad():
#             for batch_idx, (xx, yy) in enumerate(tqdm(self.test_loader, desc="Running inference")):
#                 try:
#                     xx, yy = xx.to(device), yy.to(device)
#                     batch_size = xx.shape[0]
                    
#                     # Extract initial condition
#                     x0 = xx[..., 0]  # Shape: [batch, vars, nx, ny]
                    
#                     # Create time points for entire trajectory
#                     t_span = torch.linspace(0, dt * T_out, T_out + 1, 
#                                           device=self.device, dtype=xx.dtype)
                    
#                     # Forward pass with odeint
#                     try:
#                         pred = odeint(
#                             self.odefunc, x0, t_span, 
#                             method=self.method, rtol=self.rtol, atol=self.atol
#                         )
#                     except Exception as e:
#                         print(f"ODE integration failed in inference batch {batch_idx}: {str(e)}")
#                         if self.debug:
#                             print(f"  Input shape: {x0.shape}")
#                             print(f"  Time span shape: {t_span.shape}")
#                             raise e
#                         continue
                    
#                     # Remove initial condition and reshape
#                     pred = pred[1:]  # Remove t=0, shape: [time_steps, batch, vars, nx, ny]
#                     pred = pred.permute(1, 2, 3, 4, 0)  # Reshape to [batch, vars, nx, ny, time_steps]
                    
#                     pred_set.append(pred)
#                     successful_batches += 1
                    
#                 except RuntimeError as e:
#                     print(f"Runtime error in inference batch {batch_idx}: {str(e)}")
#                     if self.debug:
#                         raise e
#                     continue

#             if pred_set:
#                 pred_set = torch.cat(pred_set, dim=0)
                
#                 # Ensure shapes match for error computation
#                 test_out_device = self.test_out.to(device)
#                 min_samples = min(pred_set.shape[0], test_out_device.shape[0])
#                 min_time_steps = min(pred_set.shape[-1], test_out_device.shape[-1])
                
#                 pred_set_trimmed = pred_set[:min_samples, ..., :min_time_steps]
#                 test_out_trimmed = test_out_device[:min_samples, ..., :min_time_steps]

#                 # Performance Metrics
#                 if eval_metric == 'MSE':
#                     error = (pred_set_trimmed - test_out_trimmed).pow(2).mean()
#                 elif eval_metric == 'MAE':
#                     error = torch.abs(pred_set_trimmed - test_out_trimmed).mean()
                
#                 if self.debug:
#                     print(f"Successful batches: {successful_batches}/{len(self.test_loader)}")
#                     print(f"Prediction shape: {pred_set.shape}")
#                     print(f"Ground truth shape: {self.test_out.shape}")
#                     print(f"{eval_metric} Error: {error.item():.6f}")
                    
#             else:
#                 pred_set = torch.empty(0)
#                 error = float('inf')
#                 warnings.warn("No successful predictions generated during inference")

#         return pred_set, error

# %% 

import numpy as np 
import torch 
import torch.nn as nn 
from tqdm import tqdm 
from timeit import default_timer
from torchdiffeq import odeint, odeint_adjoint

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
max_grad_clip_norm = 2.0   

# %% 
# ODEFunc to be used with torchdiffeq
class ODEFunc(nn.Module):
    def __init__(self, model):
        super(ODEFunc, self).__init__()
        self.model = model
        
    def forward(self, t, x):
        return self.model(x)


class Train_Setup():
    def __init__(self, model, train_loader, test_loader, loss_func, optimizer, scheduler, epochs, 
                 method='rk4', adjoint=False, rtol=1e-3, atol=1e-4, max_grad_clip_norm=max_grad_clip_norm):
        super(Train_Setup, self).__init__()

        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = epochs
        self.device = device
        self.method = method
        self.adjoint = adjoint
        self.rtol = rtol
        self.atol = atol
        
        # Wrap model in ODEFunc
        self.odefunc = ODEFunc(model)
        
        self.grad_clip = max_grad_clip_norm
        
        model.to(device)
        self.model.train()

    def one_epoch(self, step, train_T_out, test_T_out, dt=0.01):
        
        train_l2_step = 0
        train_l2_full = 0

        for xx, yy in self.train_loader:
            self.optimizer.zero_grad()
            loss = 0
            xx = xx.to(self.device)
            yy = yy.to(self.device)
            batch_size = xx.shape[0]

            # Create time points for entire trajectory
            t_span = torch.linspace(0, dt * train_T_out, train_T_out + 1).to(self.device)
            
            # Use adjoint method for memory-efficient backprop if enabled
            if self.adjoint:
                pred = odeint_adjoint(self.odefunc, xx, t_span, method=self.method, rtol=self.rtol, atol=self.atol)
            else:
                pred = odeint(self.odefunc, xx, t_span, method=self.method, rtol=self.rtol, atol=self.atol)

            pred = pred[1:,...,0].permute(1, 2, 3, 4, 0)  # Reshape to [batch, vars, nx, ny, nt]

            # Compute loss on the full trajectory
            l2_full = self.loss_func(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1))
            train_l2_full += l2_full.item()
            
            # Backward pass and optimization
            l2_full.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

        train_loss = train_l2_full 

        # Validation Loop
        test_loss = 0
        with torch.no_grad():
            for xx, yy in self.test_loader:
                xx, yy = xx.to(self.device), yy.to(self.device)
                batch_size = xx.shape[0]
                
                # Create time points for entire trajectory
                t_span = torch.linspace(0, dt * test_T_out, test_T_out + 1).to(self.device)
                
                # Forward pass with odeint (no need for adjoint during validation)
                pred = odeint(self.odefunc, xx, t_span, method=self.method, rtol=self.rtol, atol=self.atol)
                
                pred = pred[1:,...,0].permute(1, 2, 3, 4, 0)  # Reshape to [batch, vars, nx, ny, nt]

                # Compute validation loss
                test_loss += self.loss_func(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1)).item()

        return train_loss, test_loss

    def train(self, run, step, train_T_out, test_T_out, dt=0.01):
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
    def __init__(self, model, test_in, test_out, normalizer='False', method='dopri5', batch_size=3):
        super(Eval_Setup, self).__init__()

        self.model = model
        self.test_in = test_in
        self.test_out = test_out
        self.device = device
        self.method = method
        
        self.test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(self.test_in, self.test_out), batch_size=batch_size, shuffle=False)
        
        # Wrap model in ODEFunc
        self.odefunc = ODEFunc(model)
                
        model.to(device)
        self.model.eval()

    def inference(self, step, T_out, eval_metric='MSE', dt=0.01):
        pred_set = []
        with torch.no_grad():
            for xx, yy in tqdm(self.test_loader):
                xx, yy = xx.to(device), yy.to(device)
                
                # Create time points for entire trajectory
                t_span = torch.linspace(0, dt * T_out, T_out + 1).to(self.device)
                
                # Forward pass with odeint
                pred = odeint(self.odefunc, xx, t_span, method=self.method)
                
                # Slice out the initial condition
                pred = pred[1:,...,0].permute(1, 2, 3, 4, 0)  # Reshape to [batch, vars, nx, ny, nt]
                
                pred_set.append(pred)

            pred_set = torch.cat(pred_set, dim=0)

            # Performance Metrics
            if eval_metric == 'MSE':
                error = (pred_set - self.test_out.to(device)).pow(2).mean()
            if eval_metric == 'MAE':
                error = torch.abs(pred_set - self.test_out.to(device)).mean()

        return pred_set, error