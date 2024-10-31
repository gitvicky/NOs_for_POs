if configuration['Model']['operator splitting']: 
    #Currently only including the momentum equation 
    class NO_PO(nn.Module):
        def __init__(self):
            super(NO_PO, self).__init__()
            self.NO_convection = FNO_multi2d(configuration['Data']['t_in'], configuration['Data']['step'], configuration['Model']['modes'], configuration['Model']['modes'], 2, configuration['Model']['width']) 
            self.NO_diffusion = FNO_multi2d(configuration['Data']['t_in'], configuration['Data']['step'], configuration['Model']['modes'], 2, configuration['Model']['width']) 
            self.NO_p_grad =  FNO_multi2d(configuration['Data']['t_in'], configuration['Data']['step'], configuration['Model']['modes'], configuration['Model']['modes'], 1, configuration['Model']['width']) 

        def forward(self, vars):
            uv = vars[:, 0:2]
            p = vars[:, 2:3]
            nu = 0.001
            rhs = - self.NO_convection(uv) + nu*self.NO_diffusion - self.NO_p_grad(p)

            return rhs