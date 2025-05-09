
# %%
import torch 
import torch.nn as nn

import sys
sys.path.append("..")
from Neural_PDE.Models.FNO import FNO_multi2d
from Neural_PDE.Models.ConvOperator import ConvolutionalModel
from Neural_PDE.Models.UNet import UNet2d
from PRE.VectorConvOps_Spatial import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

##Will need to modify the time stepping as well to take in rhs and p and then adjust for the variables. Currently modelling for u and v. 

class NS_spectral_OS_rhs(nn.Module):#Navier-Stokes Operator-Splitting right-hand-side. 
    def __init__(self, configuration, normalizer, run):
        super(NS_spectral_OS_rhs, self).__init__()

        self.normalizer = normalizer
        if device == 'cuda':
            self.normalizer.cuda()
        else:
            self.normalizer.cpu()
        self.run = run
        
        # #FNO
        self.pressure_poisson = FNO_multi2d(in_vars=2, out_vars=2, modes1=configuration['Model']['modes'], modes2=configuration['Model']['modes'], width=configuration['Model']['width'],n_layers=configuration['Model']['n_layers']) 
        self.convection_operator = FNO_multi2d(in_vars=2, out_vars=2, modes1=configuration['Model']['modes'], modes2=configuration['Model']['modes'], width=configuration['Model']['width'],n_layers=configuration['Model']['n_layers']) 
        self.diffusion_operator = FNO_multi2d(in_vars=2, out_vars=2, modes1=configuration['Model']['modes'], modes2=configuration['Model']['modes'], width=configuration['Model']['width'],n_layers=configuration['Model']['n_layers']) 
        
        # # self.gradient = Gradient(scale=1, taylor_order=2, boundary_cond='periodic', device=device, requires_grad=True)
        self.nu = torch.tensor(0.001, dtype=torch.float32, requires_grad=True).to(device)

        # #Vector physical operators
        # self.gradient = FNO_multi2d(in_vars=1, out_vars=2, modes1=configuration['Model']['modes'], modes2=configuration['Model']['modes'], width=configuration['Model']['width'],n_layers=configuration['Model']['n_layers'])  
        # self.laplace = FNO_multi2d(in_vars=2, out_vars=2, modes1=configuration['Model']['modes'], modes2=configuration['Model']['modes'], width=configuration['Model']['width'],n_layers=configuration['Model']['n_layers'])  

        # #Convolutions with BCs
        # self.pressure_poisson = ConvolutionalModel(
        #         in_features=2,
        #         out_features=2,
        #         hidden_features=configuration['Model']['hidden_vars'],
        #         num_layers=configuration['Model']['n_layers'],
        #         activation=configuration['Model']['act'],
        #         final_activation='none',
        #         init_type='random')
        
        # self.convection_operator = ConvolutionalModel(
        #         in_features=2,
        #         out_features=2,
        #         hidden_features=configuration['Model']['hidden_vars'],
        #         num_layers=configuration['Model']['n_layers'],
        #         activation=configuration['Model']['act'],
        #         final_activation='none',
        #         init_type='random')

        # self.diffusion_operator = ConvolutionalModel(
        #         in_features=2,
        #         out_features=2,
        #         hidden_features=configuration['Model']['hidden_vars'],
        #         num_layers=configuration['Model']['n_layers'],
        #         activation=configuration['Model']['act'],
        #         final_activation='none',
                # init_type='random')

#   #UNets 
#         self.pressure_poisson = UNet2d(in_channels=configuration['Data']['t_in'], 
#                         out_channels=configuration['Data']['step'], 
#                         init_features=configuration['Model']['width'], 
#                         in_vars=configuration['Model']['in_vars'],
#                         out_vars=configuration['Model']['out_vars']
#                         )
#         self.convection_operator = UNet2d(in_channels=configuration['Data']['t_in'], 
#                         out_channels=configuration['Data']['step'], 
#                         init_features=configuration['Model']['width'], 
#                         in_vars=configuration['Model']['in_vars'],
#                         out_vars=configuration['Model']['out_vars']
#                         )

#         self.diffusion_operator = UNet2d(in_channels=configuration['Data']['t_in'], 
#                         out_channels=configuration['Data']['step'], 
#                         init_features=configuration['Model']['width'], 
#                         in_vars=configuration['Model']['in_vars'],
#                         out_vars=configuration['Model']['out_vars']
#                         )
        
    def forward(self, vars):

        # u = vars[:, 0:1]
        # v = vars[:, 1:2]
        uv = vars[:, 0:2]
        # uv = self.normalizer.encode(uv)

        # p = vars[:, 2:3]

        # #Mixed Operators
        # convection = self.NO_convection(uv)
        # diffusion = self.NO_diffusion(uv) 
        # pressure = self.NO_pressure_poisson(vars) #Poisson Solve
        # pressure_grad = self.gradient(pressure[...,0]).unsqueeze(-1) #PRE for gradient. 
        # rhs = - convection + self.nu*diffusion + pressure_grad

        # # Vector physical operators
        # p = self.NO_pressure_poisson(uv) #Poisson Solve
        # rhs = -dot(uv, self.gradient(u)) - dot(uv, self.gradient(v)) + self.nu * self.laplace(uv) + self.gradient(p)            

        # pressure_grad = self.normalizer.decode(self.pressure_poisson(uv))
        # convection = self.normalizer.decode(self.convection_operator(uv))
        # diffusion = self.normalizer.decode(self.diffusion_operator(uv))

        pressure_grad = self.pressure_poisson(uv)
        convection = self.convection_operator(uv)
        diffusion = self.diffusion_operator(uv)


        rhs = - convection + self.nu*diffusion + pressure_grad

        self.run.log_metrics({"rhs_momx": rhs[:, 0].detach().mean(),
                              "rhs_momy": rhs[:, 1].detach().mean(),
                             })

        return rhs #, pressure #Only modelling for u and v for the time being. 

    def count_params(self):
        nparams = 0

        for param in self.parameters():
            nparams += param.numel()
        return nparams 

class Euler_FV_OS_rhs(nn.Module):#Compressible Navier-Stokes Finite Volume Operator-Splitting right-hand-side.
    def __init__(self, configuration, normalizer, run):
        super(Euler_FV_OS_rhs, self).__init__()

        self.run = run
        self.normalizer = normalizer
        if device == 'cuda':
            self.normalizer.cuda()
        else:
            self.normalizer.cpu()

        self.gamma = torch.tensor(5/3, dtype=torch.float32, requires_grad=True).to(device)
        self.eps = torch.tensor(1e-6, dtype=torch.float32, requires_grad=True).to(device)

        #FNO
        self.divergence_operator = FNO_multi2d(in_vars=2, out_vars=1, modes1=configuration['Model']['modes'], modes2=configuration['Model']['modes'], width=configuration['Model']['width'],n_layers=configuration['Model']['n_layers'])  
        self.convection_operator = FNO_multi2d(in_vars=2, out_vars=2, modes1=configuration['Model']['modes'], modes2=configuration['Model']['modes'], width=configuration['Model']['width'],n_layers=configuration['Model']['n_layers']) 
        self.gradient_operator = FNO_multi2d(in_vars=1, out_vars=2, modes1=configuration['Model']['modes'], modes2=configuration['Model']['modes'], width=configuration['Model']['width'],n_layers=configuration['Model']['n_layers'])  

        #  #Convolutions with BCs
        # self.divergence_operator = ConvolutionalModel(
        #         in_features=2,
        #         out_features=1,
        #         hidden_features=4,
        #         num_layers=2,
        #         activation='tanh',
        #         final_activation='none',
        #         init_type='random')
        
        # self.convection_operator = ConvolutionalModel(
        #         in_features=2,
        #         out_features=2,
        #         hidden_features=4,
        #         num_layers=2,
        #         activation='tanh',
        #         final_activation='none',
        #         init_type='random')

        # self.gradient_operator = ConvolutionalModel(
        #         in_features=1,
        #         out_features=2,
        #         hidden_features=4,
        #         num_layers=2,
        #         activation='tanh',
        #         final_activation='none',
        #         init_type='random')

    def forward(self, vars):

        rho = vars[:, 0:1]
        uv  = vars[:, 1:3]
        p   = vars[:, 3:4]
        
        # vars_enc = self.normalizer.encode(vars)
        # rho_enc = vars_enc[:, 0:1]
        # uv_enc  = vars_enc[:, 1:3]
        # p_enc   = vars_enc[:, 3:4]

        # div_uv = self.normalizer.decode(self.divergence_operator(uv_enc), var_idx = [1])
        # grad_rho = self.normalizer.decode(self.gradient_operator(rho_enc), var_idx = [0,0])
        # grad_p = self.normalizer.decode(self.gradient_operator(p_enc), var_idx = [3,3])
        # convection = self.normalizer.decode(self.convection_operator(uv_enc), var_idx = [1,2])
        
        div_uv = self.divergence_operator(uv)
        grad_rho = self.gradient_operator(rho)
        grad_p = self.gradient_operator(p)
        convection = self.convection_operator(uv)

        rhs_mass = - rho*div_uv - dot(uv, grad_rho)
        
        # rhs_mom = -convection - (1/rho)*grad_p         
        rhs_mom =  -rho*convection - grad_p #Reformulated to avoid division by rho
        
        rhs_energy = -self.gamma*p*div_uv - dot(uv, grad_p)
        

        self.run.log_metrics({"rhs_mass": rhs_mass.detach().mean(),
                              "rhs_mom.": rhs_mom.detach().mean(),
                              "rhs_energy": rhs_energy.detach().mean()
                             })


        
        rhs = torch.cat((rhs_mass, rhs_mom, rhs_energy), dim=1)
        return rhs

    def count_params(self):
        nparams = 0

        for param in self.parameters():
            nparams += param.numel()
        return nparams 


class Incomp_PDEB_NS_OS_rhs(nn.Module):#PDEBench incompressible Navier-Stokes Operator-Splitting right-hand-side. 
    def __init__(self, configuration, normalizer, run):
        super(Incomp_PDEB_NS_OS_rhs, self).__init__()

        self.normalizer = normalizer
        if device == 'cuda':
            self.normalizer.cuda()
        else:
            self.normalizer.cpu()
        self.run = run

        self.pressure_poisson = FNO_multi2d(in_vars=2, out_vars=2, modes1=configuration['Model']['modes'], modes2=configuration['Model']['modes'], width=configuration['Model']['width'],n_layers=configuration['Model']['n_layers']) 
        self.convection_operator = FNO_multi2d(in_vars=2, out_vars=2, modes1=configuration['Model']['modes'], modes2=configuration['Model']['modes'], width=configuration['Model']['width'],n_layers=configuration['Model']['n_layers']) 
        self.diffusion_operator = FNO_multi2d(in_vars=2, out_vars=2, modes1=configuration['Model']['modes'], modes2=configuration['Model']['modes'], width=configuration['Model']['width'],n_layers=configuration['Model']['n_layers']) 

        self.eta = torch.tensor(0.001, dtype=torch.float32, requires_grad=True).to(device)
        self.rho = torch.tensor(0.4, dtype=torch.float32, requires_grad=True).to(device)

    def forward(self, vars):
        uv = vars[:, 0:2]

        uv = self.normalizer.encode(uv)
        pressure_grad = self.normalizer.decode(self.pressure_poisson(uv))
        convection = self.normalizer.decode(self.convection_operator(uv))
        diffusion = self.normalizer.decode(self.diffusion_operator(uv))

        # pressure_grad = self.pressure_poisson(uv)
        # convection = self.convection_operator(uv)
        # diffusion = self.diffusion_operator(uv)

        rhs = - convection + (self.eta*diffusion - pressure_grad) / self.rho
        
        self.run.log_metrics({"rhs_momx": rhs[:, 0].detach().mean(),
                              "rhs_momy": rhs[:, 1].detach().mean(),
                             })
        
        return rhs #, pressure #Only modelling for u and v for the time being. 


    def count_params(self):
        nparams = 0

        for param in self.parameters():
            nparams += param.numel()
        return nparams 
    
class Comp_NS_PDEB_OS_rhs(nn.Module):#PDE Bench Compressible Navier-Stokes Operator-Splitting right-hand-side. #Momentum equation only at the moment. 
    def __init__(self, configuration):
        super(Comp_NS_PDEB_OS_rhs, self).__init__()

        self.gradient = FNO_multi2d(in_vars=1, out_vars=2, modes1=configuration['Model']['modes'], modes2=configuration['Model']['modes'], width=configuration['Model']['width'],n_layers=configuration['Model']['n_layers'])  
        self.laplace = FNO_multi2d(in_vars=2, out_vars=2, modes1=configuration['Model']['modes'], modes2=configuration['Model']['modes'], width=configuration['Model']['width'],n_layers=configuration['Model']['n_layers'])  
        self.divergence = FNO_multi2d(in_vars=2, out_vars=1, modes1=configuration['Model']['modes'], modes2=configuration['Model']['modes'], width=configuration['Model']['width'],n_layers=configuration['Model']['n_layers'])  
        self.NO_pressure = FNO_multi2d(in_vars=3, out_vars=1, modes1=configuration['Model']['modes'], modes2=configuration['Model']['modes'], width=configuration['Model']['width'],n_layers=configuration['Model']['n_layers']) 

        # self.NO_convection = FNO_multi2d(in_vars=2, out_vars=2, modes1=configuration['Model']['modes'], modes2=configuration['Model']['modes'], width=configuration['Model']['width'],n_layers=configuration['Model']['n_layers']) 
        # self.NO_diffusion = FNO_multi2d(in_vars=2, out_vars=2, modes1=configuration['Model']['modes'], modes2=configuration['Model']['modes'], width=configuration['Model']['width'],n_layers=configuration['Model']['n_layers']) 
        # self.NO_pressure_poisson = FNO_multi2d(in_vars=2, out_vars=1, modes1=configuration['Model']['modes'], modes2=configuration['Model']['modes'], width=configuration['Model']['width'],n_layers=configuration['Model']['n_layers']) 
        # self.gradient = Gradient(scale=1, device=device, requires_grad=True)
        # self.NO_compression = FNO_multi2d(in_vars=2, out_vars=2, modes1=configuration['Model']['modes'], modes2=configuration['Model']['modes'], width=configuration['Model']['width'],n_layers=configuration['Model']['n_layers']) 

        self.eta = torch.tensor(0.1, dtype=torch.float32, requires_grad=True).to(device)
        self.zeta = torch.tensor(0.1, dtype=torch.float32, requires_grad=True).to(device)

    def forward(self, vars):
        rho = vars[:, 0:1]
        u = vars[:, 1:2]
        uv = vars[:, 1:3]
        v = vars[:, 2:3]
        # p = vars[:, 3:4]

        # convection = self.NO_convection(uv)
        # diffusion = self.NO_diffusion(uv) 
        # pressure = self.NO_pressure_poisson(vars) #Poisson Solve
        # pressure_grad = self.gradient(pressure[...,0]).unsqueeze(-1)
        # compression = self.NO_compression(uv)
        # rhs =  - convection + (eta*diffusion - pressure_grad + (eta+zeta/3)*compression)/rho #rho might need to be duplicated to match the dimensions. 
        # return rhs #, pressure #Only modelling for u and v for the time being. 


        div_uv = self.divergence(uv)
        grad_rho = self.gradient(rho)
        p = self.NO_pressure(vars)

        rhs_mass = - rho*div_uv - dot(uv, grad_rho)
        rhs_mom = -dot(uv, self.gradient(u)) - dot(uv, self.gradient(v)) + 1/rho * (self.eta*self.laplace(uv) - self.gradient(p) + (self.zeta + self.eta/3)*self.gradient(div_uv))
        # rhs_energy = -self.gamma*p*div_uv - dot(uv, grad_rho)        
        
        rhs = torch.cat((rhs_mass, rhs_mom[:, 0:1], rhs_mom[:, 1:2]), dim=1)

        return rhs


    def count_params(self):
        nparams = 0

        for param in self.parameters():
            nparams += param.numel()
        return nparams 


# %% 
# #Example Usage
# import yaml

# config_loc = '/home/ir-gopa2/rds/rds-ukaea-ap001/ir-gopa2/Code/NOs_for_POs/Expts/configs/NS_spectral_matrix.yaml'
# with open(config_loc, 'r') as f:
#     configuration = yaml.safe_load(f)

# model = NS_spectral_OS_rhs(configuration)
# X = torch.ones(32, 2, 64, 64, 1)
# out = model(X)

# %%
# # Example usage Euler equations 
# import yaml 
# config_loc = '/home/ir-gopa2/rds/rds-ukaea-ap001/ir-gopa2/Code/NOs_for_POs/Expts/configs/Euler_FV_FNO.yaml'
# with open(config_loc, 'r') as f:
#     configuration = yaml.safe_load(f)

# model = Euler_FV_OS_rhs(configuration)
# X = torch.ones(32, 4, 64, 64, 1)
# out = model(X)

# %%
