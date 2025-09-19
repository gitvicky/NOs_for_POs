import torch 
import torch.nn as nn

import sys
sys.path.append("..")
from model_setup import * 
from PRE.VectorConvOps import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %% 

class Incomp_Navier_Stokes():
    def __init__(self):
        super(Incomp_Navier_Stokes, self).__init__()

        self.divergence_operator = Divergence(scale=1, taylor_order=2, boundary_cond='periodic', device=device, requires_grad=False)

    def forward(self, uv):
        u,v = uv[:, 0:1], uv[:, 1:2]
        div_uv = self.divergence_operator(u, v).unsqueeze(-1)
        return div_uv

class Euler_fluid():
    def __init__(self):
        super(Euler_fluid, self).__init__()

        self.


