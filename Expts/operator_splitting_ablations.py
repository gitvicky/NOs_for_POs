
# %%
import torch 
import torch.nn as nn

import sys
sys.path.append("..")
from model_setup import * 
from PRE.VectorConvOps_Spatial import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# data_distr = 'OOD'

def dist(data_dist_param):
    global data_distr
    data_distr = data_dist_param


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
        self.convection_operator = model_selection(config)

        self.laplace = Laplace(scale=1, taylor_order=4, boundary_cond='periodic', device=device, requires_grad=False, scalar=False)
        if data_distr == 'ID':
            self.nu = torch.tensor(0.001, dtype=torch.float32, requires_grad=False).to(device)
        elif data_distr == 'OOD':
            self.nu = torch.tensor(0.01, dtype=torch.float32, requires_grad=False).to(device)
         
        self.nu = self.normalizer.encode(self.nu.unsqueeze(-1)).squeeze()
        print(self.nu)

    def forward(self, vars):

        uv = vars[:, 0:2]

        convection = self.convection_operator(uv)
        diffusion = self.laplace(uv[:,0:1,...,0], uv[:,1:2,...,0]).unsqueeze(-1) #Assuming uv is a 2D vector field with shape (batch_size, 2, height, width, 1).


        rhs = - convection + self.nu*diffusion #- pressure_grad

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

        if data_distr == 'ID':
            self.gamma = torch.tensor(5/3, dtype=torch.float32, requires_grad=False).to(device)
        if data_distr == 'OOD':
            self.gamma = torch.tensor(2/3, dtype=torch.float32, requires_grad=False).to(device)

        self.gamma = self.normalizer.encode(self.gamma.repeat(1,4))
        print(self.gamma)
        self.gamma = self.gamma[0, -1]#Taking the normalisation from pressure. 
        # self.gamma = self.gamma[0, -2]#Taking the normalisation from velocity. 

        self.eps = torch.tensor(1e-6, dtype=torch.float32, requires_grad=True).to(device)

    #Primitive Variables
        #Model Selection 
        config = configuration
        config['Model']['in_vars'], config['Model']['out_vars'] = 2, 2
        self.convection_operator = model_selection(config)

        config['Model']['in_vars'], config['Model']['out_vars'] = 3, 1
        self.div_cons = model_selection(config)   #Divergence of a conservative variable


        #Using predetermined operators.
        # self.divergence_operator = Divergence(scale=1, taylor_order=4, boundary_cond='periodic', device=device, requires_grad=False)
        self.gradient_operator = Gradient(scale=1, taylor_order=4, boundary_cond='periodic', device=device, requires_grad=False)


    
#Using primitive variables.
    def forward(self, vars): 

        rho = vars[:, 0:1]
        uv  = vars[:, 1:3]
        p   = vars[:, 3:4]

        grad_p = self.gradient_operator(p[...,0]).unsqueeze(-1)

        convection = self.convection_operator(uv)

        rhs_mass = - self.div_cons(vars[:, 0:3])        
        rhs_mom = - convection - torch.log(torch.abs(rho+self.eps))*grad_p #reformulated to avoid division by zero 
        rhs_energy = -self.gamma * self.div_cons(vars[:, 1:])

        try: 
        
            self.run.log_metrics({"rhs_mass": rhs_mass.detach().mean(),
                                "rhs_mom.": rhs_mom.detach().mean(),
                                "rhs_energy": rhs_energy.detach().mean()
                                })
        except: 
            pass
        
        rhs = torch.cat((rhs_mass, rhs_mom, rhs_energy), dim=1)
        return rhs



    def count_params(self):
        nparams = 0

        for param in self.parameters():
            nparams += param.numel()
        return nparams 
