
# %%
import torch 
import torch.nn as nn

import sys
sys.path.append("..")
from model_setup import * 
from PRE.VectorConvOps_Spatial import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class NS_incompressible_rhs(nn.Module):#Navier-Stokes Operator-Splitting right-hand-side. 
    def __init__(self, configuration, normalizer, run, x, y):
        super(NS_incompressible_rhs, self).__init__()

        # self.normalizer = normalizer
        # if device == 'cuda':
        #     self.normalizer.cuda()
        # else:
        #     self.normalizer.cpu()
        self.run = run
        

        config = configuration
        config['Model']['in_vars'], config['Model']['out_vars'] = 2, 2
        self.convection_operator = model_selection(config, x, y)
        self.diffusion_operator = model_selection(config, x, y)

        self.nu = torch.tensor(0.001, dtype=torch.float32, requires_grad=False).to(device)
        # self.nu = self.normalizer.encode(self.nu.unsqueeze(-1)).squeeze()

    def forward(self, coords_in, coords_out, vars):

        # uv, nu = vars[0], vars[1] #Nvidia
        uv = vars #GDM

        convection = self.convection_operator(coords_in, coords_out, vars)
        diffusion = self.diffusion_operator(coords_in, coords_out, vars)

        # self.nu = self.normalizer.encode(nu).view(uv.shape[0], 1, 1, 1)  # Shape: [50, 1, 1, 1] #Nvidia

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
    