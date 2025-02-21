
# %%
import torch 
import torch.nn as nn

import sys
sys.path.append("..")
from Neural_PDE.Models.FNO import FNO_multi2d
from PRE.VectorConvOps import Gradient

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

##Will need to modify the time stepping as well to take in rhs and p and then adjust for the variables. Currently modelling for u and v. 

class NS_spectral_OS_rhs(nn.Module):#Navier-Stokes Operator-Splitting right-hand-side. 
    def __init__(self, configuration):
        super(NS_spectral_OS_rhs, self).__init__()
        self.NO_convection = FNO_multi2d(in_vars=2, out_vars=2, modes1=configuration['Model']['modes'], modes2=configuration['Model']['modes'], width=configuration['Model']['width'],n_layers=configuration['Model']['n_layers']) 
        self.NO_diffusion = FNO_multi2d(in_vars=2, out_vars=2, modes1=configuration['Model']['modes'], modes2=configuration['Model']['modes'], width=configuration['Model']['width'],n_layers=configuration['Model']['n_layers']) 
        self.NO_pressure_poisson = FNO_multi2d(in_vars=2, out_vars=1, modes1=configuration['Model']['modes'], modes2=configuration['Model']['modes'], width=configuration['Model']['width'],n_layers=configuration['Model']['n_layers']) 
        self.gradient = Gradient(scale=1, device=device, requires_grad=True)#stacks the p_x and p_y along the first dimension. 
        self.nu = torch.tensor(0.001, dtype=torch.float32).to(device)

    def forward(self, vars):
        uv = vars[:, 0:2]
        # p = vars[:, 2:3]
        convection = self.NO_convection(uv)
        diffusion = self.NO_diffusion(uv) 
        pressure = self.NO_pressure_poisson(vars) #Poisson Solve
        pressure_grad = self.gradient(pressure[:,0].permute(0, 3, 1, 2)).permute(1, 0, 2, 3).unsqueeze(-1) #PRE for gradient. First permute for getting it as [bs, t, x, y], second to bring it back to bs, vars, x, y and then adding the time at the end. 
        rhs = - convection + self.nu*diffusion - pressure_grad
        return rhs #, pressure #Only modelling for u and v for the time being. 

    def count_params(self):
        nparams = 0

        for param in self.parameters():
            nparams += param.numel()
        return nparams 


class Euler_FV_OS_rhs(nn.Module):#Compressible Navier-Stokes Finite Volume Operator-Splitting right-hand-side.
    def __init__(self, configuration):
        super(Euler_FV_OS_rhs, self).__init__()
        self.continuity = FNO_multi2d(in_vars=3, out_vars=1, modes1=configuration['Model']['modes'], modes2=configuration['Model']['modes'], width=configuration['Model']['width'],n_layers=configuration['Model']['n_layers'])
        self.convection = FNO_multi2d(in_vars=2, out_vars=2, modes1=configuration['Model']['modes'], modes2=configuration['Model']['modes'], width=configuration['Model']['width'],n_layers=configuration['Model']['n_layers'])  
        self.gradient = FNO_multi2d(in_vars=1, out_vars=2, modes1=configuration['Model']['modes'], modes2=configuration['Model']['modes'], width=configuration['Model']['width'],n_layers=configuration['Model']['n_layers'])
        self.divergence = FNO_multi2d(in_vars=2, out_vars=1, modes1=configuration['Model']['modes'], modes2=configuration['Model']['modes'], width=configuration['Model']['width'],n_layers=configuration['Model']['n_layers'])
        self.gamma = torch.tensor(5/3, dtype=torch.float32).to(device)

    def forward(self, vars):
        uv = vars[:, 0:2]
        p = vars[:, 2:3]
        rho = vars[:, 3:4]
        
        grad_p = self.gradient(p)
        rhs_mass = -self.continuity(torch.cat((uv,rho), dim=1))
        rhs_mom = -self.convection(uv) - grad_p / rho
        rhs_energy = -self.gamma*p*self.divergence(uv) - uv[:,0:1]*grad_p[:,0:1] - uv[:,1:2]*grad_p[:,1:2]        
        
        rhs = torch.cat((rhs_mass, rhs_mom[:, 0:1], rhs_mom[:, 1:2], rhs_energy), dim=1)
        return rhs

    def count_params(self):
        nparams = 0

        for param in self.parameters():
            nparams += param.numel()
        return nparams 


class Incomp_PDEB_NS_OS_rhs(nn.Module):#PDEBench incompressible Navier-Stokes Operator-Splitting right-hand-side. 
    def __init__(self, configuration):
        super(Incomp_PDEB_NS_OS_rhs, self).__init__()
        self.NO_convection = FNO_multi2d(in_vars=2, out_vars=2, modes1=configuration['Model']['modes'], modes2=configuration['Model']['modes'], width=configuration['Model']['width'],n_layers=configuration['Model']['n_layers']) 
        self.NO_diffusion = FNO_multi2d(in_vars=2, out_vars=2, modes1=configuration['Model']['modes'], modes2=configuration['Model']['modes'], width=configuration['Model']['width'],n_layers=configuration['Model']['n_layers']) 
        self.NO_pressure_poisson = FNO_multi2d(in_vars=2, out_vars=1, modes1=configuration['Model']['modes'], modes2=configuration['Model']['modes'], width=configuration['Model']['width'],n_layers=configuration['Model']['n_layers']) 
        self.gradient = Gradient(scale=1, device=device, requires_grad=True)#stacks the p_x and p_y along the first dimension. 
    
    def forward(self, vars):
        uv = vars[:, 0:2]
        # p = vars[:, 2:3]
        eta = 0.001
        rho = 0.4
        convection = self.NO_convection(uv)
        diffusion = self.NO_diffusion(uv) 
        pressure = self.NO_pressure_poisson(vars) #Poisson Solve
        pressure_grad = self.gradient(pressure[:,0].permute(0, 3, 1, 2)).permute(1, 0, 2, 3).unsqueeze(-1) #PRE for gradient. First permute for getting it as [bs, t, x, y], second to bring it back to bs, vars, x, y and then adding the time at the end. 
        rhs = - convection + (eta*diffusion - pressure_grad) / rho
        return rhs #, pressure #Only modelling for u and v for the time being. 

    def count_params(self):
        nparams = 0

        for param in self.parameters():
            nparams += param.numel()
        return nparams 
    
class Comp_NS_PDEB_OS_rhs(nn.Module):#PDE Bench Compressible Navier-Stokes Operator-Splitting right-hand-side. #Momentum equation only at the moment. 
    def __init__(self, configuration):
        super(Comp_NS_PDEB_OS_rhs, self).__init__()
        self.NO_convection = FNO_multi2d(in_vars=2, out_vars=2, modes1=configuration['Model']['modes'], modes2=configuration['Model']['modes'], width=configuration['Model']['width'],n_layers=configuration['Model']['n_layers']) 
        self.NO_diffusion = FNO_multi2d(in_vars=2, out_vars=2, modes1=configuration['Model']['modes'], modes2=configuration['Model']['modes'], width=configuration['Model']['width'],n_layers=configuration['Model']['n_layers']) 
        self.NO_pressure_poisson = FNO_multi2d(in_vars=2, out_vars=1, modes1=configuration['Model']['modes'], modes2=configuration['Model']['modes'], width=configuration['Model']['width'],n_layers=configuration['Model']['n_layers']) 
        self.gradient = Gradient(scale=1, device=device, requires_grad=True)
        self.NO_compression = FNO_multi2d(in_vars=2, out_vars=2, modes1=configuration['Model']['modes'], modes2=configuration['Model']['modes'], width=configuration['Model']['width'],n_layers=configuration['Model']['n_layers']) 
    
    def forward(self, vars):
        uv = vars[:, 0:2]
        p = vars[:, 2:3]
        rho = vars[:, 3:4]

        eta, zeta = 0.001, 0.001

        convection = self.NO_convection(uv)
        diffusion = self.NO_diffusion(uv) 
        pressure = self.NO_pressure_poisson(vars) #Poisson Solve
        pressure_grad = self.gradient(pressure[:,0].permute(0, 3, 1, 2)).permute(1, 0, 2, 3).unsqueeze(-1) #PRE for gradient. First permute for getting it as [bs, t, x, y], second to bring it back to bs, vars, x, y and then adding the time at the end. 
        compression = self.NO_compression(uv)
        rhs =  - convection + (eta*diffusion - pressure_grad + (eta+zeta/3)*compression)/rho #rho might need to be duplicated to match the dimensions. 
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
# X = torch.ones(32, 2, 64, 64, 1)
# out = model(X)

# %%
#Example usage Euler equations 
# import yaml 
# config_loc = '/home/ir-gopa2/rds/rds-ukaea-ap001/ir-gopa2/Code/NOs_for_POs/Expts/configs/Euler_FV_FNO.yaml'
# with open(config_loc, 'r') as f:
#     configuration = yaml.safe_load(f)

# model = Euler_FV_OS_rhs(configuration)
# X = torch.ones(32, 4, 64, 64, 1)
# out = model(X)

# %%
