
# %%
import torch 
import torch.nn as nn

import sys
sys.path.append("..")
from Neural_PDE.Models.FNO import FNO_multi2d

#Currently only including the momentum equation 
class NS_OS_rhs(nn.Module):#Navier-Stokes Operator-Splitting right-hand-side. 
    def __init__(self, configuration):
        super(NS_OS_rhs, self).__init__()
        self.NO_convection = FNO_multi2d(in_vars=2, out_vars=2, modes1=configuration['Model']['modes'], modes2=configuration['Model']['modes'], width=configuration['Model']['width'],n_layers=configuration['Model']['n_layers']) 
        self.NO_diffusion = FNO_multi2d(in_vars=2, out_vars=2, modes1=configuration['Model']['modes'], modes2=configuration['Model']['modes'], width=configuration['Model']['width'],n_layers=configuration['Model']['n_layers']) 
        self.NO_p = FNO_multi2d(in_vars=3, out_vars=1, modes1=configuration['Model']['modes'], modes2=configuration['Model']['modes'], width=configuration['Model']['width'],n_layers=configuration['Model']['n_layers']) 

    def forward(self, vars):
        uv = vars[:, 0:2]
        # p = vars[:, 2:3]
        nu = 0.001
        convection = self.NO_convection(uv)
        diffusion = self.NO_diffusion(uv) 
        pressure = self.NO_p(vars) #Poisson Solve
        # press_grad = ...
        rhs = - (convection[:,0:1] + convection[:,1:2]) + nu*(diffusion[:,0:1]+diffusion[:,1:2]) - pressure_grad
        rhs = rhs.repeat(1, 3, 1, 1, 1)
        return rhs

    def count_params(self):
        nparams = 0

        for param in self.parameters():
            nparams += param.numel()
        return nparams 

# %% 
#Example Usage
import yaml

config_loc = '/home/ir-gopa2/rds/rds-ukaea-ap001/ir-gopa2/Code/NOs_for_POs/Expts/configs/NS_spectral_FNO.yaml'
with open(config_loc, 'r') as f:
    configuration = yaml.safe_load(f)

# %% 
model = NS_OS_rhs(configuration)
X = torch.ones(32, 3, 64, 64, 1)
out = model(X)
# %%
