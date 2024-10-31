
import torch 
import torch.nn as nn

from Neural_PDE.Models.FNO import FNO_multi2d

#Currently only including the momentum equation 
class NS_OS_rhs(nn.Module):#Navier-Stokes Operator-Splitting right-hand-side. 
    def __init__(self, configuration):
        super(NS_OS_rhs, self).__init__()
        self.NO_convection = FNO_multi2d(configuration['Data']['t_in'], configuration['Data']['step'], configuration['Model']['modes'], configuration['Model']['modes'], 2, configuration['Model']['width']) 
        self.NO_diffusion = FNO_multi2d(configuration['Data']['t_in'], configuration['Data']['step'], configuration['Model']['modes'], 2, configuration['Model']['width']) 
        self.NO_p_grad =  FNO_multi2d(configuration['Data']['t_in'], configuration['Data']['step'], configuration['Model']['modes'], configuration['Model']['modes'], 1, configuration['Model']['width']) 

    def forward(self, vars):
        uv = vars[:, 0:2]
        p = vars[:, 2:3]
        nu = 0.001
        rhs = - self.NO_convection(uv) + nu*self.NO_diffusion - self.NO_p_grad(p)

        return rhs