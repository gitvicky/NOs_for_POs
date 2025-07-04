'''
Model Setup

'''

import sys
sys.path.append("..")

from Neural_PDE.Models.FNO_classic import *
from Neural_PDE.Models.ViT import * 
from Neural_PDE.Models.UNet import * 
# from Neural_PDE.Models.PDEUnet import *
from Neural_PDE.Models.CNO import * 
from Neural_PDE.Models.gMLP_Vision import * 
# from Neural_PDE.Models.ConvOperator import *
# from neuralop.models import FNO2d
from Neural_PDE.Models.LNO import * 

#Function to count_params
count_parameters = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def model_initialisation(configuration, normalizer, run):
    pde = configuration['Physics']['pde']

    if configuration['Model']['operator_splitting'] == True: 
    
    #With operator_splitting.
        if pde == 'Navier-Stokes':
            from operator_splitting import NS_spectral_OS_rhs
            model = NS_spectral_OS_rhs(configuration, normalizer, run)
        if pde == 'Shear Flow':
            from operator_splitting import NS_shearflow_OS_rhs
            model = NS_shearflow_OS_rhs(configuration, normalizer, run)
        if pde == 'Euler-Fluid':
            from operator_splitting import Euler_FV_OS_rhs
            model = Euler_FV_OS_rhs(configuration, normalizer, run)
        if pde == 'Incomp. Navier-Stokes':
            from operator_splitting import Incomp_PDEB_NS_OS_rhs
            model = Incomp_PDEB_NS_OS_rhs(configuration, normalizer, run)
        if pde == 'Comp. Navier-Stokes':
            from operator_splitting import Comp_NS_PDEB_OS_rhs
            model = Comp_NS_PDEB_OS_rhs(configuration)

    else:
            #FNO Classic
        if configuration['Model']['arch'] == 'FNO':
            if configuration['Model']['operator_splitting'] == False:
                model = FNO_multi2d(T_in = configuration['Data']['t_in'],
                                    step = configuration['Data']['step'],
                                    modes1 = configuration['Model']['modes'],       # Number of Fourier modes to keep along height dimension
                                    modes2 = configuration['Model']['modes'],        # Number of Fourier modes to keep along width dimension
                                    num_vars = configuration['Model']['in_vars'],     #
                                    width_time = configuration['Model']['width'],    # Width of the FNO (number of channels)
                                    )

        # IF USING THE NEURALOP LIBRARY:
        # if configuration['Model']['arch'] == 'FNO':
        #     if configuration['Model']['operator_splitting'] == False:
        #         model = FNO2d(
        #                 n_modes_height=configuration['Model']['modes'],       # Number of Fourier modes to keep along height dimension
        #                 n_modes_width=configuration['Model']['modes'],        # Number of Fourier modes to keep along width dimension
        #                 hidden_channels=configuration['Model']['width'],      # Width of the FNO (number of channels)
        #                 in_channels=configuration['Model']['in_vars'],           # Number of input channels
        #                 out_channels=configuration['Model']['in_vars'],          # Number of output channels
        #                 lifting_channels=configuration['Model']['width'],    # Channels in the lifting block (lifting_channel_ratio * hidden_channels)
        #                 projection_channels=256, # Channels in the projection block
        #                 n_layers=4,              # Number of Fourier layers
        #                 # factorization=None,      # No tensor factorization
        #                 # stabilizer=None,         # No stabilizer
        #                 # fno_block_precision="full", # Precision mode for spectral convolution
        #                 # domain_padding=0.1,      # Pad the domain by 10%
        #                 # domain_padding_mode="symmetric" # Symmetric padding
        #         )
            
        elif configuration['Model']['arch'] == 'U-Net':
            model = UNet2d(in_channels=configuration['Data']['t_in'], 
                        out_channels=configuration['Data']['step'], 
                        init_features=configuration['Model']['width'], 
                        in_vars=configuration['Model']['in_vars'],
                        out_vars=configuration['Model']['out_vars']
                        )
            
        # elif configuration['Model']['arch'] == 'U-Net':
        #     model = PDEUNet(
        #             spatial_channels=2,
        #             field_channels=configuration['Data']['t_in'],
        #             output_channels=configuration['Data']['step'],
        #             len_x=configuration['Physics']['Nx'],
        #             len_y=configuration['Physics']['Ny'],
        #             base_channels=configuration['Model']['width'],
        #             include_residual=True,
        #             device = device
        #             )
        
        elif configuration['Model']['arch'] == 'ViT':
            model = ViT(
                image_size=(configuration['Physics']['Nx'], configuration['Physics']['Ny']),
                patch_size=(configuration['Model']['patch_size'], configuration['Model']['patch_size']),
                embed_dim=configuration['Model']['embed_dim'],
                depth=configuration['Model']['depth'],
                n_heads=configuration['Model']['num_heads'],
                channels=configuration['Physics']['variables'],
                mlp_dim = 256,
                dim_head = 32
                )
        
        elif configuration['Model']['arch'] == 'CNO':
            model = CNO2d(in_dim = configuration['Model']['in_channels'],             
                        out_dim = configuration['Model']['out_channels'],
                        size = configuration['Model']['Nx'],
                        N_layers = configuration['Model']['N_layers'],
                        N_res = configuration['Model']['N_res'],
                        N_res_neck = configuration['Model']['N_res_neck'],
                        channel_multiplier = configuration['Model']['channel_multiplier'],
                        use_bn = True
                        )      
            

        elif configuration['Model']['arch'] == 'gMLP':
            model = gMLP(n_blocks = configuration['Model']['n_blocks'],
                        d_in = configuration['Model']['d_in'],
                        d_ffn = configuration['Model']['d_ffn'],
                        Nx = configuration['Model']['Nx'],
                        Ny = configuration['Model']['Ny'])
            
            
        elif configuration['Model']['arch'] == 'LNO':
                model = LNO_multi2d(T_in = configuration['Data']['t_in'],
                                    step = configuration['Data']['step'],
                                    modes1 = configuration['Model']['modes'],       # Number of Fourier modes to keep along height dimension
                                    modes2 = configuration['Model']['modes'],        # Number of Fourier modes to keep along width dimension
                                    num_vars = configuration['Model']['in_vars'],     #
                                    width_time = configuration['Model']['width'],    # Width of the FNO (number of channels)
                                    )

    return model