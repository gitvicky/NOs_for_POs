
# %%
import torch 
import torch.nn as nn

import sys
sys.path.append("..")
from model_setup import * 
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
        
        #Discretisation
        dx, dy = configuration['Physics']['dx'] * configuration['Physics']['x_slice'], configuration['Physics']['dy'] * configuration['Physics']['y_slice']
        Nx, Ny = configuration['Physics']['Nx'] / configuration['Physics']['x_slice'], configuration['Physics']['Ny'] / configuration['Physics']['y_slice']

        gridx = torch.tensor(np.linspace(0, int(Nx*dx), int(Nx)), dtype=torch.float)
        gridy = torch.tensor(np.linspace(0, int(Ny*dy), int(Ny)), dtype=torch.float)

        config = configuration
        config['Model']['in_vars'], config['Model']['out_vars'] = 2, 2
        self.pressure_poisson = model_selection(config)
        self.convection_operator = model_selection(config)

        self.laplace = Laplace(scale=1, taylor_order=2, boundary_cond='periodic', device=device, requires_grad=False, scalar=False)
        self.nu = torch.tensor(0.001, dtype=torch.float32, requires_grad=False).to(device)

    def forward(self, vars):

        uv = vars[:, 0:2]

        pressure_grad = self.pressure_poisson(uv)
        convection = self.convection_operator(uv)
        diffusion = self.laplace(uv[:,0:1,...,0], uv[:,1:2,...,0]).unsqueeze(-1) #Assuming uv is a 2D vector field with shape (batch_size, 2, height, width, 1).


        rhs = - convection + self.nu*diffusion - pressure_grad

        try:
            self.run.log_metrics({"rhs_momx": rhs[:, 0].detach().mean(),
                              "rhs_momy": rhs[:, 1].detach().mean(),
                             })

        except:
            pass

        return rhs #, pressure #Only modelling for u and v for the time being. 

    def count_params(self):
        nparams = 0

        for param in self.parameters():
            nparams += param.numel()
        return nparams 
    

class NS_shearflow_OS_rhs(nn.Module):#Navier-Stokes ShearFlow Operator-Splitting right-hand-side. 
    def __init__(self, configuration, normalizer, run):
        super(NS_shearflow_OS_rhs, self).__init__()

        self.normalizer = normalizer
        if device == 'cuda':
            self.normalizer.cuda()
        else:
            self.normalizer.cpu()
        self.run = run
        
        #Discretisation
        dx, dy = configuration['Physics']['dx'] * configuration['Physics']['x_slice'], configuration['Physics']['dy'] * configuration['Physics']['y_slice']
        Nx, Ny = configuration['Physics']['Nx'] / configuration['Physics']['x_slice'], configuration['Physics']['Ny'] / configuration['Physics']['y_slice']

        gridx = torch.tensor(np.linspace(0, int(Nx*dx), int(Nx)), dtype=torch.float)
        gridy = torch.tensor(np.linspace(0, int(Ny*dy), int(Ny)), dtype=torch.float)

        # #FNO
        self.pressure_poisson = FNO_multi2d(in_vars=2, out_vars=2, modes1=configuration['Model']['modes'], modes2=configuration['Model']['modes'], width=configuration['Model']['width'],n_layers=configuration['Model']['n_layers'], grid=[gridx, gridy]) 
        self.convection_operator = FNO_multi2d(in_vars=2, out_vars=2, modes1=configuration['Model']['modes'], modes2=configuration['Model']['modes'], width=configuration['Model']['width'],n_layers=configuration['Model']['n_layers'], grid=[gridx, gridy]) 
        self.laplace = Laplace(scale=1, taylor_order=2, boundary_cond='periodic', device=device, requires_grad=True, scalar=False)
        nu = 1/float(configuration['Data']['reynolds'][0])
        self.nu = torch.tensor(nu, dtype=torch.float32, requires_grad=True).to(device)

    def forward(self, vars):

        uv = vars[:, 0:2]

        pressure_grad = self.pressure_poisson(uv)
        convection = self.convection_operator(uv)
        diffusion = self.laplace(uv[:,0:1,...,0], uv[:,1:2,...,0]).unsqueeze(-1) #Assuming uv is a 2D vector field with shape (batch_size, 2, height, width, 1).


        rhs = - convection + self.nu*diffusion - pressure_grad

        try:
            self.run.log_metrics({"rhs_momx": rhs[:, 0].detach().mean(),
                                  "rhs_momy": rhs[:, 1].detach().mean(),
                                })
        except:
            pass

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

        #Model Selection 
        config = configuration
        config['Model']['in_vars'], config['Model']['out_vars'] = 2, 2
        self.convection_operator = model_selection(config)
        # config['Model']['in_vars'], config['Model']['out_vars'] = 2, 1
        # self.divergence_operator = model_selection(config)
        # config['Model']['in_vars'], config['Model']['out_vars'] = 1, 2
        # self.gradient_operator = model_selection(config)

        # #Using FNO
        # self.convection_operator = FNO_multi2d(in_vars=2, out_vars=2, modes1=configuration['Model']['modes'], modes2=configuration['Model']['modes'], width=configuration['Model']['width'],n_layers=configuration['Model']['n_layers']) 
        # self.divergence_operator = FNO_multi2d(in_vars=2, out_vars=1, modes1=configuration['Model']['modes'], modes2=configuration['Model']['modes'], width=configuration['Model']['width'],n_layers=configuration['Model']['n_layers'])  
        # self.gradient_operator = FNO_multi2d(in_vars=1, out_vars=2, modes1=configuration['Model']['modes'], modes2=configuration['Model']['modes'], width=configuration['Model']['width'],n_layers=configuration['Model']['n_layers'])  

        #Using predetermined operators.
        self.divergence_operator = Divergence(scale=1, taylor_order=2, boundary_cond='periodic', device=device, requires_grad=False)
        self.gradient_operator = Gradient(scale=1, taylor_order=2, boundary_cond='periodic', device=device, requires_grad=False)

        # #FNO - conservative variables.
        # self.grad_x = FNO_multi2d(in_vars=4, out_vars=4, modes1=configuration['Model']['modes'], modes2=configuration['Model']['modes'], width=configuration['Model']['width'],n_layers=configuration['Model']['n_layers'])  
        # self.grad_y = FNO_multi2d(in_vars=4, out_vars=4, modes1=configuration['Model']['modes'], modes2=configuration['Model']['modes'], width=configuration['Model']['width'],n_layers=configuration['Model']['n_layers'])
        
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

#Using primitive variables.
    def forward(self, vars): 

        rho = vars[:, 0:1]
        # u, v = vars[:, 1:2], vars[:, 2:3]
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
        

        # div_uv = self.divergence_operator(uv[:,0:1,...,0], uv[:,1:2,...,0]).unsqueeze(-1) #Assuming uv is a 2D vector field with shape (batch_size, 2, height, width, 1).
        # grad_rho = self.gradient_operator(rho[...,0]).unsqueeze(-1) #Assuming rho is a scalar field with shape (batch_size, 1, height, width, 1).
        # grad_p = self.gradient_operator(p[...,0]).unsqueeze(-1) #Assuming p is a scalar field with shape (batch_size, 1, height, width, 1).
        
        div_uv = self.divergence_operator(uv[:,0:1,...,0], uv[:,1:2,...,0]).unsqueeze(-1)
        grad_rho = self.gradient_operator(rho[...,0]).unsqueeze(-1)
        grad_p = self.gradient_operator(p[...,0]).unsqueeze(-1)

        convection = self.convection_operator(uv)

        rhs_mass = - rho*div_uv - dot(uv, grad_rho)
        
        rhs_mom = -convection - (1/(rho+1e-6))*grad_p   #regularisation to avoid division by zero.     
        # rhs_mom =  -rho*convection - grad_p #Reformulated to avoid division by rho
        
        rhs_energy = -self.gamma*p*div_uv - dot(uv, grad_p)
        

        self.run.log_metrics({"rhs_mass": rhs_mass.detach().mean(),
                              "rhs_mom.": rhs_mom.detach().mean(),
                              "rhs_energy": rhs_energy.detach().mean()
                             })

        rhs = torch.cat((rhs_mass, rhs_mom, rhs_energy), dim=1)
        return rhs


# #Using conservative variables.
#     def forward(self, vars): 

#         M = vars[:, 0:1]  # Mass density
#         Mx = vars[:, 1:2]  # x-momentum density
#         My = vars[:, 2:3]  # y-momentum density
#         E = vars[:, 3:4]  # Energy density
#         P = (self.gamma - 1) * (E - 0.5 * (Mx**2 + My**2) / M)  # Pressure from energy density

#         F_rhs = torch.cat((Mx, Mx**2/M + P,  (Mx*My)/M, (E+P)*(Mx/M)), dim=1)
#         G_rhs = torch.cat((My, (Mx*My)/M, My**2/M + P, (E+P)*(My/M)), dim=1)
#         F_x = self.grad_x(F_rhs)
#         G_y = self.grad_y(G_rhs)
#         rhs = -  F_x - G_y # Divergence of F

#         self.run.log_metrics({"rhs_mass": rhs[:,0:1].detach().mean(),
#                               "rhs_mom.": rhs[:,1:3].detach().mean(),
#                               "rhs_energy": rhs[:,3:4].detach().mean()
#                              })
#         return rhs

    def count_params(self):
        nparams = 0

        for param in self.parameters():
            nparams += param.numel()
        return nparams 


class Euler_Quadrant_OS_rhs(nn.Module):#Compressible Navier-Stokes Finite Volume Operator-Splitting right-hand-side.
    def __init__(self, configuration, normalizer, run):
        super(Euler_Quadrant_OS_rhs, self).__init__()

        self.run = run
        self.normalizer = normalizer
        if device == 'cuda':
            self.normalizer.cuda()
        else:
            self.normalizer.cpu()

        self.gamma = torch.tensor(float(configuration['Data']['gamma']), dtype=torch.float32, requires_grad=True).to(device)
        self.eps = torch.tensor(1e-6, dtype=torch.float32, requires_grad=True).to(device)

        #FNO - primitive variables.
        self.F_x = FNO_multi2d(in_vars=4, out_vars=4, modes1=configuration['Model']['modes'], modes2=configuration['Model']['modes'], width=configuration['Model']['width'],n_layers=configuration['Model']['n_layers'])
        self.G_y = FNO_multi2d(in_vars=4, out_vars=4, modes1=configuration['Model']['modes'], modes2=configuration['Model']['modes'], width=configuration['Model']['width'],n_layers=configuration['Model']['n_layers'])

#Using conservative variables.
    def forward(self, vars): 

        M = vars[:, 0:1]  # Mass density
        Mx = vars[:, 1:2]  # x-momentum density
        My = vars[:, 2:3]  # y-momentum density
        E = vars[:, 3:4]  # Energy density
        P = (self.gamma - 1) * (E - 0.5 * (Mx**2 + My**2) / M)  # Pressure from energy density
        P = torch.maximum(P, 1e-10)  # Pressure floor

        F_rhs = torch.cat((Mx, Mx**2/M + P,  (Mx*My)/M, (E+P)*(Mx/M)), dim=1)
        G_rhs = torch.cat((My, (Mx*My)/M, My**2/M + P, (E+P)*(My/M)), dim=1)

        F_x_vals = self.F_x(F_rhs)
        G_y_vals = self.G_y(G_rhs)
        rhs = - F_x_vals - G_y_vals # Divergence of F

        self.run.log_metrics({"rhs_mass": rhs[:,0:1].detach().mean(),
                              "rhs_mom.": rhs[:,1:3].detach().mean(),
                              "rhs_energy": rhs[:,3:4].detach().mean()
                             })
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


class JOREK_ES_OS_RHS(nn.Module):
    def __init__(self, configuration):
        super(JOREK_ES_OS_RHS, self).__init__()
        #R taken as the x-axis and Z as the y-axis.

        self.grad_R = ConvOperator(domain=('x'), order=1, taylor_order=2 , boundary_cond='periodic', device=device, requires_grad=True)
        self.grad_Z = ConvOperator(domain=('y'), order=1, taylor_order=2 , boundary_cond='periodic', device=device, requires_grad=True)
        self.grad_RR = ConvOperator(domain=('x'), order=2, taylor_order=2 , boundary_cond='periodic', device=device, requires_grad=True)
        self.grad_ZZ = ConvOperator(domain=('y'), order=2, taylor_order=2 , boundary_cond='periodic', device=device, requires_grad=True)
        self.NO = FNO_multi2d(in_vars=2, out_vars=2, modes1=configuration['Model']['modes'], modes2=configuration['Model']['modes'], width=configuration['Model']['width'],n_layers=configuration['Model']['n_layers'])

        R = torch.tensor(np.linspace(9.5, 10.5, 100), dtype=torch.float32, requires_grad=True).to(device)
        Z = torch.tensor(np.linspace(-0.5, 0.5, 100), dtype=torch.float32, requires_grad=True).to(device)

        self.R, self.Z = torch.meshgrid(R, Z, indexing='ij')  # Create a meshgrid for R and Z
        self.D = torch.tensor(1e-5, dtype=torch.float32, requires_grad=True).to(device)
        self.mu = torch.tensor(1e-4, dtype=torch.float32, requires_grad=True).to(device)
        self.T = torch.tensor(1e-3, dtype=torch.float32, requires_grad=True).to(device)
        
    def forward(self, vars):
        rho = vars[:, 0:1]
        phi = vars[:, 1:2]

        rho_rhs = self.R*self.NO(vars) + 2*rho*self.grad_Z(phi) + self.D*(self.grad_RR(rho) + (1/self.R)*self.grad_R(rho) + self.grad_ZZ(rho))

        return rho_rhs                  
                                                              
                                        
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
