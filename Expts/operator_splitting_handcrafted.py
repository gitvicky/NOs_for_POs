
# %%
#Operator splitting with handcrafted kernels
import torch 
import torch.nn as nn

import sys
sys.path.append("..")
from PRE.VectorConvOps_Spatial import *

# %%

class NS_spectral_OS_rhs(nn.Module):#Navier-Stokes Operator-Splitting right-hand-side. 
    def __init__(self, configuration, device):
        super(NS_spectral_OS_rhs, self).__init__()

        self.nu = torch.tensor(0.001, dtype=torch.float32, device=device, requires_grad=True)
        self.dx = torch.tensor(configuration['Physics']['dx'], dtype=torch.float32, device=device, requires_grad=True)
        self.dy = torch.tensor(configuration['Physics']['dy'], dtype=torch.float32, device=device, requires_grad=True)

        self.gradient = Gradient(scale=1/(self.dx), taylor_order=2, boundary_cond='periodic', device=device, requires_grad=True)
        self.laplace = Laplace(scale=1/(self.dx**2), taylor_order=2, boundary_cond='periodic', device=device, requires_grad=True)
        self.pressure_poisson = Vector_Gradient(scale=1/(self.dx**2), taylor_order=2, boundary_cond='periodic', device=device, requires_grad=True)

    def forward(self, vars):
        #vars is for a single time instance

        u  = vars[:, 0:1]
        v  = vars[:, 1:2]
        uv = vars[:, 0:2]
        # p  = vars[:, 2:3]

        p_laplace = self.pressure_poisson(u,v)
        # p = 
        rhs =  -dot(uv, self.gradient(u)) - dot(uv, self.gradient(v)) + self.nu*self.laplace(u, v) #+ self.gradient(p)               

        return rhs 

    def count_params(self):
        nparams = 0

        for param in self.parameters():
            nparams += param.numel()
        return nparams 

# %% 
class Euler_FV_OS_rhs(nn.Module):#Compressible Navier-Stokes Finite Volume Operator-Splitting right-hand-side.
    def __init__(self, configuration, device):
        super(Euler_FV_OS_rhs, self).__init__()

        self.dx = torch.tensor(configuration['Physics']['dx'], dtype=torch.float32, requires_grad=True).to(device)
        self.dy = torch.tensor(configuration['Physics']['dy'], dtype=torch.float32, requires_grad=True).to(device)
        self.gamma = torch.tensor(5/3, dtype=torch.float32, requires_grad=True).to(device)

        self.gradient = Gradient(scale=1/(self.dx), taylor_order=2, boundary_cond='periodic', device=device, requires_grad=True)
        self.laplace = Laplace(scale=1/(self.dx**2), taylor_order=2, boundary_cond='periodic', device=device, requires_grad=True)
        self.divergence = Divergence(scale = 1/(self.dx), taylor_order=2, boundary_cond='periodic', device=device, requires_grad=True)

    def forward(self, vars):
        #vars is for a single time instance
        rho = vars[:, 0:1]
        u   = vars[:, 1:2]
        v   = vars[:, 2:3]
        uv  = vars[:, 1:3]
        p   = vars[:, 3:4]
        
        rhs_mass = - rho*self.divergence(u,v) - dot(uv, self.gradient(rho))
        rhs_mom = -dot(uv, self.gradient(u)) - dot(uv, self.gradient(v)) + self.laplace(u, v) + (1/rho)*self.gradient(p)            
        rhs_energy = -self.gamma*p*self.divergence(u,v) - dot(uv, self.gradient(rho))        
        
        rhs = torch.cat((rhs_mass, rhs_mom[:, 0:1], rhs_mom[:, 1:2], rhs_energy), dim=1)
        return rhs

    def count_params(self):
        nparams = 0

        for param in self.parameters():
            nparams += param.numel()
        return nparams 

# %% 
# n_sims = 10

# import numpy as np 
# import torch 
# from torch.utils.data import Dataset
# import h5py 
# import glob 
# from tqdm import tqdm

# def stacked_fields(variables):
#     stack = []
#     for var in variables:
#         var = torch.from_numpy(var) #Converting to Torch
#         var = var.permute(0, 2, 3, 1) #Permuting to be BS, Nx, Ny, Nt
#         stack.append(var)
#     stack = torch.stack(stack, dim=1)
#     return stack


# data_loc = '/home/ir-gopa2/rds/rds-ukaea-ap001/ir-gopa2/Code/Neural_PDE/Data'
# data =  np.load(data_loc + '/NS_FV_combined.npz')
# u = data['u'].astype(np.float32)[:n_sims]
# v = data['v'].astype(np.float32)[:n_sims]
# p = data['p'].astype(np.float32)[:n_sims] 
# rho = data['rho'].astype(np.float32)[:n_sims]
# dx = data['dx']
# x = np.linspace(0, 1, 128)

# dt = data['dt']
# dt = torch.tensor(dt, dtype=torch.float)

# fields = stacked_fields([rho,u,v,p])
# # %% 
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# dx = torch.tensor(0.0078125, dtype=torch.float32, requires_grad=True).to(device)
# dy = torch.tensor(0.0078125, dtype=torch.float32, requires_grad=True).to(device)
# dt = torch.tensor(0.02, dtype=torch.float32, requires_grad=True).to(device)
# gamma = torch.tensor(5/3, dtype=torch.float32, requires_grad=True).to(device)

# gradient = Gradient(scale=1/(dx), taylor_order=2, boundary_cond='periodic', device=device, requires_grad=True)
# laplace = Laplace(scale=1/(dx**2), taylor_order=2, boundary_cond='periodic', device=device, requires_grad=True)
# divergence = Divergence(scale = 1/(dx), taylor_order=2, boundary_cond='periodic', device=device, requires_grad=True)

# # %%
# vars = fields 
# rho = vars[:, 0:1][...,20]
# u   = vars[:, 1:2][...,20]
# v   = vars[:, 2:3][...,20]
# uv  = vars[:, 1:3][...,20]
# p   = vars[:, 3:4][...,20]

# rhs_mass = - rho*divergence(u,v) - dot(uv, gradient(rho))
# rhs_mom = -dot(uv, gradient(u)) - dot(uv, gradient(v)) + laplace(u, v) + (1/rho)*gradient(p)            
# rhs_energy = -gamma*p*divergence(u,v) - dot(uv, gradient(rho))       

# # %%

# sys.path.append("..")
# from Neural_PDE.Models.FNO import FNO_multi2d
# gradient = FNO_multi2d(in_vars=1, out_vars=2, modes1=4, modes2=4, width=8, n_layers=2)  
# laplace = FNO_multi2d(in_vars=2, out_vars=2, modes1=4, modes2=4, width=8, n_layers=2)  
# divergence = FNO_multi2d(in_vars=2, out_vars=1, modes1=4, modes2=4, width=8, n_layers=2)  

# gamma = torch.tensor(5/3, dtype=torch.float32, requires_grad=True).to(device)

# rho = vars[:, 0:1]
# u   = vars[:, 1:2]
# v   = vars[:, 2:3]
# uv  = vars[:, 1:3]
# p   = vars[:, 3:4]

# div_uv = divergence(uv)
# grad_rho = gradient(rho)

# rhs_mass = - rho*div_uv - dot(uv, grad_rho)
# rhs_mom = -dot(uv, gradient(u)) - dot(uv, gradient(v)) + laplace(uv) + (1/rho)*gradient(p)            
# rhs_energy = -gamma*p*div_uv - dot(uv, grad_rho)        

# rhs = torch.cat((rhs_mass, rhs_mom[:, 0:1], rhs_mom[:, 1:2], rhs_energy), dim=1)

# %%
