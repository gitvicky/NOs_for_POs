'''
Model Setup

'''

import sys
sys.path.append("..")

from Neural_PDE.Models.FNO import *
from Neural_PDE.Models.ViT_new import * 
from Neural_PDE.Models.UNet import * 
from Neural_PDE.Models.CNO import * 
from Neural_PDE.Models.gMLP_Vision import * 


def model_initialisation(configuration):
    pde = configuration['Physics']['pde']

    if configuration['Model']['operator splitting'] == True: 
    
    #With Operator Splitting.
        if pde == 'Navier-Stokes':
            from operator_splitting import NS_spectral_OS_rhs
            model = NS_spectral_OS_rhs(configuration)
        if pde == 'Euler':
            from operator_splitting import Euler_FV_OS_rhs
            model = Euler_FV_OS_rhs(configuration)
        if pde == 'Incomp. Navier-Stokes':
            from operator_splitting import Incomp_NS_OS_rhs
            model = Incomp_NS_OS_rhs(configuration)
        if pde == 'Comp. Navier-Stokes':
            from operator_splitting import Comp_NS_OS_rhs
            model = Comp_NS_OS_rhs(configuration)
    
    else:
            
        if configuration['Model']['arch'] == 'FNO':
            if configuration['Model']['operator splitting'] == False:
                model = FNO_multi2d(configuration['Model']['in_vars'], 
                                    configuration['Model']['out_vars'], 
                                    configuration['Model']['modes'], 
                                    configuration['Model']['modes'],
                                    configuration['Model']['width'],
                                    configuration['Model']['n_layers']
                                    )
            
        if configuration['Model']['arch'] == 'U-Net':
            model = UNet2d(configuration['Data']['t_in'], 
                        configuration['Data']['step'], 
                        configuration['Model']['width'], 
                        configuration['Physics']['variables']
                        )
        
        if configuration['Model']['arch'] == 'ViT':
            model = ViT(
                image_size=(configuration['Physics']['Nx'], configuration['Physics']['Ny']),
                patch_size=(configuration['Model']['patch size'], configuration['Model']['patch size']),
                embed_dim=configuration['Model']['embed dim'],
                depth=configuration['Model']['depth'],
                n_heads=configuration['Model']['num heads'],
                channels=configuration['Physics']['variables'],
                mlp_dim = 256,
                dim_head = 32
                )
        
        if configuration['Model']['arch'] == 'CNO':
            model = CNO2d(in_dim = configuration['Model']['in channels'],             
                        out_dim = configuration['Model']['out channels'],
                        size = configuration['Model']['Nx'],
                        N_layers = configuration['Model']['N_layers'],
                        N_res = configuration['Model']['N_res'],
                        N_res_neck = configuration['Model']['N_res_neck'],
                        channel_multiplier = configuration['Model']['channel multiplier'],
                        use_bn = True
                        )      
            

        if configuration['Model']['arch'] == 'gMLP':
            model = gMLP(n_blocks = configuration['Model']['n_blocks'],
                        d_in = configuration['Model']['d_in'],
                        d_ffn = configuration['Model']['d_ffn'],
                        Nx = configuration['Model']['Nx'],
                        Ny = configuration['Model']['Ny'])

    return model

#Function to count_params
count_parameters = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)