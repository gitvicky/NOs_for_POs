import numpy as np 
import torch 
import torch.nn as nn 
from tqdm import tqdm 
from timeit import default_timer
from torchdiffeq import odeint, odeint_adjoint

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
max_grad_clip_norm = 10.0   

# %% 
# ODEFunc to be used with torchdiffeq
class ODEFunc(nn.Module):
    def __init__(self, model):
        super(ODEFunc, self).__init__()
        self.model = model
        
    def forward(self, t, x):
        return self.model(x)

class Train_Setup():
    def __init__(self, model, train_loader, test_loader, loss_func, optimizer, scheduler, epochs, method='dopri5', adjoint=True):
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
                pred = odeint_adjoint(self.odefunc, xx, t_span, method=self.method)
            else:
                pred = odeint(self.odefunc, xx, t_span, method=self.method)
            
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
                pred = odeint(self.odefunc, xx, t_span, method=self.method)
                
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