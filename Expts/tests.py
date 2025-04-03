# %% 
from matplotlib import pyplot as plt 
import numpy as np 
import torch 
from torch.utils.data import Dataset
import h5py 
import glob 
from tqdm import tqdm
import os 
import shutil 

tmp_loc = os.getcwd()
# try: 
#     shutil.rmtree(tmp_loc)
#     os.mkdir(tmp_loc)
# except:
#     pass

import sys
sys.path.append("..")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# %% 
# #Loading data 
# from data_loaders import * 

# n_sims = 10
# data_loc = '/home/ir-gopa2/rds/rds-ukaea-ap001/ir-gopa2/Code/Neural_PDE/Data'
# data =  np.load(data_loc + '/NS_Spectral_combined.npz')
# u = data['u'].astype(np.float32)[:n_sims]
# v = data['v'].astype(np.float32)[:n_sims]
# p = data['p'].astype(np.float32)[:n_sims]
# x = data['x']
# dt = data['dt']
# dt = torch.tensor(dt, dtype=torch.float)

# fields = stacked_fields([u,v])

# # %% 
# #Normalisaiton
# from Neural_PDE.Utils.processing_utils import MinMax_Normalizer_variable
# #Normalising the data -- using the same normalisations for inputs and outputs
# normalizer_func = MinMax_Normalizer_variable
# normalizer = normalizer_func(fields)
# # fields = normalizer.encode(fields)
# # %% 
# import yaml
# with open('/home/ir-gopa2/rds/rds-ukaea-ap001/ir-gopa2/Code/NOs_for_POs/Expts/configs/NS_spectral_FNO.yaml', 'r') as file:
#     config = yaml.safe_load(file)

# from operator_splitting_handcrafted import * 
# RHS = NS_spectral_OS_rhs(config, device)
# # %%
# rhs = RHS(fields[...,1])
# # %%
# plt.figure()
# plt.hist(fields[...,1].flatten())
# plt.title('Fields')
# plt.figure()
# plt.hist(rhs.detach().flatten())
# plt.title('RHS')
# %% 

#Loading data 
from data_loaders import * 

n_sims = 10
data_loc = '/home/ir-gopa2/rds/rds-ukaea-ap001/ir-gopa2/Code/Neural_PDE/Data'
data =  np.load(data_loc + '/NS_FV_combined.npz')
rho = data['rho'].astype(np.float32)[:n_sims]
u = data['u'].astype(np.float32)[:n_sims]
v = data['v'].astype(np.float32)[:n_sims]
p = data['p'].astype(np.float32)[:n_sims] 


fields = stacked_fields([rho,u,v,p])

# %% 
#Normalisaiton
from Neural_PDE.Utils.processing_utils import MinMax_Normalizer_variable
#Normalising the data -- using the same normalisations for inputs and outputs
normalizer_func = MinMax_Normalizer_variable
normalizer = normalizer_func(fields)
# fields = normalizer.encode(fields)
# %% 
import yaml
with open('/home/ir-gopa2/rds/rds-ukaea-ap001/ir-gopa2/Code/NOs_for_POs/Expts/configs/Euler_FV_FNO.yaml', 'r') as file:
    config = yaml.safe_load(file)

from operator_splitting_handcrafted import * 
RHS = Euler_FV_OS_rhs(config, device)
# %%
rhs = RHS(fields[...,1])
# %%
plt.figure()
plt.hist(fields[:,3,...,1].flatten())
plt.title('Fields-p')
plt.figure()
plt.hist(rhs[:,3].detach().flatten())
plt.title('RHS-p')

# %% 
#RHS Explorations 
gamma = torch.tensor(5/3, dtype=torch.float32, requires_grad=True).to(device)
dx=1
dy=1
gradient = Gradient(scale=1/(dx), taylor_order=2, boundary_cond='periodic', device=device, requires_grad=True)
laplace = Laplace(scale=1/(dx**2), taylor_order=2, boundary_cond='periodic', device=device, requires_grad=True)
divergence = Divergence(scale = 1/(dx), taylor_order=2, boundary_cond='periodic', device=device, requires_grad=True)

#fields is for a single time instance
rho = fields[:, 0:1, ..., 1]
u   = fields[:, 1:2, ..., 1]
v   = fields[:, 2:3, ..., 1]
uv  = fields[:, 1:3, ..., 1]
p   = fields[:, 3:4, ..., 1]

rhs_mass = - rho*divergence(u,v) - dot(uv, gradient(rho))
rhs_mom = -dot(uv, gradient(u)) - dot(uv, gradient(v)) + laplace(u, v) + (1/rho)*gradient(p)            
rhs_energy = -gamma*p*divergence(u,v) - dot(uv, gradient(rho))        

rhs = torch.cat((rhs_mass, rhs_mom[:, 0:1], rhs_mom[:, 1:2], rhs_energy), dim=1)

# %%
rhs = gradient(rho)
plt.figure()
plt.hist(fields[:,0,...,1].flatten())
plt.title('Fields-rho')
plt.figure()
plt.hist(rhs[:,0].detach().flatten())
plt.title('RHS-rho')
# %%
