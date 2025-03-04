
# %%
#Operator splitting with handcrafted kernels
import torch 
import torch.nn as nn

import sys
sys.path.append("..")
from Neural_PDE.Models.FNO import FNO_multi2d
from PRE.VectorConvOps_Spatial import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# %%

class NS_spectral_OS_rhs(nn.Module):#Navier-Stokes Operator-Splitting right-hand-side. 
    def __init__(self, configuration):
        super(NS_spectral_OS_rhs, self).__init__()
        self.nu = torch.tensor(0.001, dtype=torch.float32, requires_grad=True).to(device)
        self.dx = torch.tensor(configuration['Physics']['dx'], dtype=torch.float32, requires_grad=True).to(device)
        self.dy = torch.tensor(configuration['Physics']['dy'], dtype=torch.float32, requires_grad=True).to(device)

        self.gradient = Gradient(scale=1/(2*self.dx), device=device, taylor_order=2, boundary_cond='periodic', device=device, requires_grad=True)
        self.laplace = Laplace(scale=1/(self.dx**2), device=device, taylor_order=2, boundary_cond='periodic', device=device, requires_grad=True)

    def forward(self, vars):
        u = vars[:, 0:1][0]
        v = vars[:, 1:2][0]
        uv = vars[:, 0:2][0]
        p = vars[:, 2:3][0]
        rhs =  -dot(uv, self.gradient(u)) - dot(uv, self.gradient(v)) + self.nu*self.laplace(u, v) + self.gradient(p)                   

        return rhs 

    def count_params(self):
        nparams = 0

        for param in self.parameters():
            nparams += param.numel()
        return nparams 

# %% 
class Euler_FV_OS_rhs(nn.Module):#Compressible Navier-Stokes Finite Volume Operator-Splitting right-hand-side.
    def __init__(self, configuration):
        super(Euler_FV_OS_rhs, self).__init__()

        self.dx = torch.tensor(configuration['Physics']['dx'], dtype=torch.float32, requires_grad=True).to(device)
        self.dy = torch.tensor(configuration['Physics']['dy'], dtype=torch.float32, requires_grad=True).to(device)

        self.gamma = torch.tensor(5/3, dtype=torch.float32, requires_grad=True).to(device)

        self.gradient = Gradient(scale=1/(2*self.dx), taylor_order=2, boundary_cond='periodic', device=device, requires_grad=True)
        self.laplace = Laplace(scale=1/(self.dx**2), taylor_order=2, boundary_cond='periodic', device=device, requires_grad=True)
        self.divergence = Divergence(scale = 1/(2*self.dx), taylor_order=2, boundary_cond='periodic', device=device, requires_grad=True)

    def forward(self, vars):
        #[0] index for the single time instance
        u = vars[:, 0:1][0]
        v = vars[:, 1:2][0]
        uv = vars[:, 0:2][0]
        p = vars[:, 2:3][0]
        rho = vars[:, 3:4][0]
        
        rhs_mass = - rho*self.divergence(uv) - dot(uv, self.gradient(rho))
        rhs_mom = -dot(uv, self.gradient(u)) - dot(uv, self.gradient(v)) + self.laplace(u, v) + (1/self.rho)*self.gradient(p)            
        rhs_energy = -self.gamma*p*self.divergence(uv) - dot(uv, self.gradient(rho))        
        
        rhs = torch.cat((rhs_mass, rhs_mom[:, 0:1], rhs_mom[:, 1:2], rhs_energy), dim=1)
        return rhs

    def count_params(self):
        nparams = 0

        for param in self.parameters():
            nparams += param.numel()
        return nparams 

# # %% 
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
# data =  np.load(data_loc + '/NS_Spectral_combined.npz')

# u = data['u'].astype(np.float32)[:n_sims]
# v = data['v'].astype(np.float32)[:n_sims]
# p = data['p'].astype(np.float32)[:n_sims]
# rho = np.ones_like(u) #Taking rho to be 1. 
# x = data['x']
# dt = data['dt']
# dt = torch.tensor(dt, dtype=torch.float)

# fields = stacked_fields([u,v,p])

# # %% 
# nu = torch.tensor(0.001, dtype=torch.float32, requires_grad=True).to(device)
# dx = torch.tensor(0.01, dtype=torch.float32, requires_grad=True).to(device)
# dy = torch.tensor(0.01, dtype=torch.float32, requires_grad=True).to(device)

# gradient = Gradient(scale=1/(2*dx), device=device, taylor_order=2, boundary_cond ='periodic', requires_grad=True)
# laplace = Laplace(scale=1/(dx**2), scalar=False, device=device, taylor_order=2, boundary_cond ='periodic', requires_grad=True)
# divergence = Divergence(scale = 1/(2*dx), device=device, taylor_order=2, boundary_cond ='periodic', requires_grad=True)

# # %%
# vars = fields[...,20]#Taking the 20th time instance 
# u = vars[:, 0:1]
# v = vars[:, 1:2]
# uv = vars[:, 0:2]
# p = vars[:, 2:3]
# rho = vars[:, 2:3]


# rhs_mom =  -dot(uv, gradient(u)) - dot(uv, gradient(v)) + nu*laplace(uv[:,0:1], uv[:, 1:2]) + gradient(p)                   
# rhs_mass = - rho*divergence(u,v) - dot(uv, gradient(rho))
# rhs_energy = (5/3)*p*divergence(u,v) - dot(uv, gradient(rho))        

# %%
