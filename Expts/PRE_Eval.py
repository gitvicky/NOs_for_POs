import torch 
import torch.nn as nn

import sys
sys.path.append("..")
from model_setup import * 
from PRE.ConvOps_2d import * 
# from PRE.VectorConvOps_Spatial import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# %% 
class Incomp_NS_PRE(nn.Module):
    def __init__(self, configuration):
        super(Incomp_NS_PRE, self).__init__()
    
        self.dx = configuration['Physics']['dx'] * configuration['Physics']['x_slice']
        self.dy = configuration['Physics']['dy'] * configuration['Physics']['y_slice']
        self.dt = configuration['Physics']['dt'] * configuration['Physics']['t_slice']

        self.D_t = ConvOperator(domain='t', order=1)#, scale=alpha)
        self.D_x = ConvOperator(domain='x', order=1)#, scale=beta) 
        self.D_y = ConvOperator(domain='y', order=1)#, scale=beta)
        self.D_x_y = ConvOperator(domain=('x', 'y'), order=1)#, scale=beta)
        self.D_xx_yy = ConvOperator(domain=('x','y'), order=2)#, scale=gamma)

        self.nu = torch.tensor(0.001, dtype=torch.float32, requires_grad=False).to(device)

    def forward(self, vars, boundary=False):
        u, v = vars[:,0], vars[:, 1]
        res = self.D_x(u) + self.D_y(v)
        if boundary:
            return res
        else: 
            return res[...,1:-1,1:-1,1:-1]


    # def forward(self, vars, boundary=False):
    #     u, v, p = vars[:,0], vars[:, 1], vars[:, 2]

    #     res_x = self.D_t(u)*self.dx*self.dy + self.u*self.D_x(u)*self.dt*self.dy + v*self.D_y(u)*self.dt*self.dx - self.nu*self.D_xx_yy(u)*self.dt + self.D_x(p)*self.dt*self.dy
    #     res_y = self.D_t(v)*self.dx*self.dy + u*self.D_x(v)*self.dt*self.dx + v*self.D_y(v)*self.dt*self.dy - self.nu*self.D_xx_yy(v)*self.dt + self.D_y(p)*self.dt*self.dx

    #     if boundary:
    #         return res_x + res_y
    #     else: 
    #         return res_x[...,1:-1,1:-1,1:-1] + res_y[...,1:-1,1:-1,1:-1]
        

class Comp_NS_PRE(nn.Module):
    def __init__(self, configuration):
        super(Comp_NS_PRE, self).__init__()
    
        self.dx = configuration['Physics']['dx'] * configuration['Physics']['x_slice']
        self.dy = configuration['Physics']['dy'] * configuration['Physics']['y_slice']
        self.dt = configuration['Physics']['dt'] * configuration['Physics']['t_slice']

        self.D_t = ConvOperator(domain='t', order=1)#, scale=alpha)
        self.D_x = ConvOperator(domain='x', order=1)#, scale=beta) 
        self.D_y = ConvOperator(domain='y', order=1)#, scale=beta)
        self.D_x_y = ConvOperator(domain=('x', 'y'), order=1)#, scale=beta)
        self.D_xx_yy = ConvOperator(domain=('x','y'), order=2)#, scale=gamma)

        self.gamma = torch.tensor(5/3, dtype=torch.float32, requires_grad=False)

    def forward(self, vars, boundary=False):
        rho, u, v, p = vars[:,0], vars[:, 1], vars[:, 2], vars[:, 3]
        cont = self.D_t(rho) + rho*self.D_x(u) + u*self.D_x(rho) + rho*self.D_y(v) + v*self.D_y(rho)
        mom_x = self.D_t(u) + u*self.D_x(u) + v*self.D_y(u) + (1/rho)* self.D_x(p)
        mom_y = self.D_t(v) + u*self.D_x(v) + v*self.D_y(v) + (1/rho)* self.D_y(p)
        energy = self.D_t(p) + u*self.D_x(p) + v*self.D_y(p) + self.gamma*p*(self.D_x(u)+self.D_y(v))

        res = cont + mom_x + mom_y + energy 

        if boundary:
            return res
        else: 
            return res[...,1:-1,1:-1,1:-1]

# %% 



