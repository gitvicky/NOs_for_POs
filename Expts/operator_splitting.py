
# %%
import torch 
import torch.nn as nn

import sys
sys.path.append("..")
from Neural_PDE.Models.FNO import FNO_multi2d
from PRE.VectorConvOps import Gradient

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

##Will need to modify the time stepping as well to take in rhs and p and then adjust for the variables. Currently modelling for u and v. 

class NS_OS_rhs(nn.Module):#Navier-Stokes Operator-Splitting right-hand-side. 
    def __init__(self, configuration):
        super(NS_OS_rhs, self).__init__()
        self.NO_convection = FNO_multi2d(in_vars=2, out_vars=2, modes1=configuration['Model']['modes'], modes2=configuration['Model']['modes'], width=configuration['Model']['width'],n_layers=configuration['Model']['n_layers']) 
        self.NO_diffusion = FNO_multi2d(in_vars=2, out_vars=2, modes1=configuration['Model']['modes'], modes2=configuration['Model']['modes'], width=configuration['Model']['width'],n_layers=configuration['Model']['n_layers']) 
        self.NO_pressure_poisson = FNO_multi2d(in_vars=2, out_vars=1, modes1=configuration['Model']['modes'], modes2=configuration['Model']['modes'], width=configuration['Model']['width'],n_layers=configuration['Model']['n_layers']) 
        self.gradient = Gradient(scale=1, device=device, requires_grad=True)
    
    def forward(self, vars):
        uv = vars[:, 0:2]
        # p = vars[:, 2:3]
        nu = 0.001
        convection = self.NO_convection(uv)
        diffusion = self.NO_diffusion(uv) 
        pressure = self.NO_pressure_poisson(vars) #Poisson Solve
        pressure_grad = self.gradient(pressure.permute(0, 4, 2, 3, 1)[...,-1])
        # pressure_grad = pressure_grad[0] + pressure_grad[1] #Adding dp/dx + dp/dy 
        pressure_grad = pressure_grad.unsqueeze(0).unsqueeze(-1) #Adding the batch and time channels which are removed within the gradient. 
        rhs = - convection + nu*diffusion - pressure_grad
        return rhs #, pressure #Only modelling for u and v for the time being. 

    def count_params(self):
        nparams = 0

        for param in self.parameters():
            nparams += param.numel()
        return nparams 

# %% 
# #Example Usage
# import yaml

# config_loc = '/home/ir-gopa2/rds/rds-ukaea-ap001/ir-gopa2/Code/NOs_for_POs/Expts/configs/NS_spectral_FNO.yaml'
# with open(config_loc, 'r') as f:
#     configuration = yaml.safe_load(f)

# model = NS_OS_rhs(configuration)
# # X = torch.ones(32, 2, 64, 64, 1)
# X = torch.ones(1, 2, 100, 100, 1)
# out = model(X)

# %%
