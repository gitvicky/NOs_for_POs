'''
Model Setup

'''

import sys
sys.path.append("..")

# from Neural_PDE.Models.FNO_classic import *
from Neural_PDE.Models.ViT import * 
from Neural_PDE.Models.UNet import * 
from Neural_PDE.Models.UNet_Classic import *
from Neural_PDE.Models.CNO import * 
from Neural_PDE.Models.gMLP_Vision import * 
from Neural_PDE.Models.ConvOperator import *
from Neural_PDE.Models.LNO import * 
from Neural_PDE.Models.Neural_Ops_lib import *
from Neural_PDE.Models.GNO import * 

#Function to count_params
count_parameters = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def model_selection(configuration):
    """    Selects the model based on the configuration provided."""

            
    #Discretisation
    dx, dy = configuration['Physics']['dx'] * configuration['Physics']['x_slice'], configuration['Physics']['dy'] * configuration['Physics']['y_slice']
    Nx, Ny = configuration['Physics']['Nx'] / configuration['Physics']['x_slice'], configuration['Physics']['Ny'] / configuration['Physics']['y_slice']

    gridx = torch.tensor(np.linspace(0, int(Nx*dx), int(Nx)), dtype=torch.float)
    gridy = torch.tensor(np.linspace(0, int(Ny*dy), int(Ny)), dtype=torch.float)


    if configuration['Model']['arch'] == 'FNO':
        #FNO Classic
            # model = FNO_multi2d(T_in = configuration['Data']['t_in'],
            #                     step = configuration['Data']['step'],
            #                     modes1 = configuration['Model']['modes'],       # Number of Fourier modes to keep along height dimension
            #                     modes2 = configuration['Model']['modes'],        # Number of Fourier modes to keep along width dimension
            #                     num_vars = configuration['Model']['in_vars'],     #
            #                     width_time = configuration['Model']['width'],    # Width of the FNO (number of channels)
            #                     )
        #Using Neural-Ops library
        model = FNO_multi2d(in_channels=configuration['Model']['in_vars'],           # Number of input channels
                        out_channels=configuration['Model']['out_vars'],          # Number of output channels,
                        width=configuration['Model']['width'],      # Width of the FNO (number of channels)
                        n_modes_height=configuration['Model']['modes'],       # Number of Fourier modes to keep along height dimension
                        n_modes_width=configuration['Model']['modes'],        # Number of Fourier modes to keep along width dimension
                        n_layers=configuration['Model']['n_layers'] ,             # Number of Fourier layers

        )
    elif configuration['Model']['arch'] == 'TFNO':
        model = TFNO_multi2d(in_channels=configuration['Model']['in_vars'], 
                            out_channels=configuration['Model']['out_vars'], 
                            hidden_channels=configuration['Model']['width'], 
                            n_modes_height=configuration['Model']['modes'], 
                            n_modes_width=configuration['Model']['modes'],
                            rank=configuration['Model']['rank']
                            )

    
    elif configuration['Model']['arch'] == 'UNO':
        model = UNO_multi2d(in_channels=configuration['Model']['in_vars'], 
                            out_channels=configuration['Model']['out_vars'], 
                            hidden_channels=configuration['Model']['width']
                            )
        
    # elif configuration['Model']['arch'] == 'U-Net':
    #     model = UNet2d(in_channels=configuration['Data']['t_in'], 
    #                 out_channels=configuration['Data']['step'], 
    #                 init_features=configuration['Model']['width'], 
    #                 in_vars=configuration['Model']['in_vars'],
    #                 out_vars=configuration['Model']['out_vars']
    #                 )
        
    elif configuration['Model']['arch'] == 'U-Net':
        model = UNetClassic(
                dim_in=configuration['Model']['in_vars'],
                dim_out=configuration['Model']['out_vars'],
                n_spatial_dims=2,
                spatial_resolution=[configuration['Physics']['Nx'], configuration['Physics']['Ny']],
                init_features=configuration['Model']['width'],
                )

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
            in_channels=configuration['Model']['in_vars'],
            out_channels=configuration['Model']['out_vars'],
            mlp_dim = 256,
            dim_head = 32
            )
        
    
    elif configuration['Model']['arch'] == 'CNO':
        model = CNO2d(in_dim = configuration['Model']['in_vars'],             
                    out_dim = configuration['Model']['out_vars'],
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
        
    elif configuration['Model']['arch'] == 'ConvOperator':  
            model = ConvolutionalModel(
                        in_features=configuration['Model']['in_vars'], 
                        out_features=configuration['Model']['out_vars'], 
                        hidden_features=configuration['Model']['width'],
                        num_layers=configuration['Model']['n_layers'],
                        kernel_size=configuration['Model']['kernel_size'],
                        boundary_type=configuration['Model']['boundary_type'],
                        activation=configuration['Model']['activation'],
                        init_type=configuration['Model']['init_type'],
        )
    
    elif configuration['Model']['arch'] == 'GNO':
            x = torch.linspace(0, configuration['Physics']['Nx']*configuration['Physics']['dt'] , configuration['Physics']['Nx'])
            y = torch.linspace(0, configuration['Physics']['Ny']*configuration['Physics']['dt'] , configuration['Physics']['Ny'])
            model = GNO(
                    in_channels=configuration['Model']['in_vars'], 
                    out_channels=configuration['Model']['out_vars'], 
                    hidden_channels=configuration['Model']['width'], 
                    r=configuration['Model']['r'], 
                    # k_neighbours = 20,
                    n_layers=configuration['Model']['depth'],
                    x_in=x,
                    y_in=y
        )  
        
        
    # elif configuration['Model']['arch'] == 'LNO':
    #         model = LNO_multi2d(T_in = configuration['Data']['t_in'],
    #                             step = configuration['Data']['step'],
    #                             modes1 = configuration['Model']['modes'],       # Number of Fourier modes to keep along height dimension
    #                             modes2 = configuration['Model']['modes'],        # Number of Fourier modes to keep along width dimension
    #                             num_vars = configuration['Model']['in_vars'],     #
    #                             width_time = configuration['Model']['width'],    # Width of the FNO (number of channels)
    #                             )

    else:
        raise ValueError(f"Unknown architecture: {configuration['Model']['arch']}. ")
    
    return model

def model_initialisation(configuration, normalizer, run):
    pde = configuration['Physics']['pde']
    if configuration['Model']['operator_splitting'] == True: 
    #With operator_splitting.
        if pde == 'ConvDiff':
            from operator_splitting import Conv_Diff_OS_rhs
            model = Conv_Diff_OS_rhs(configuration, normalizer, run)
        elif pde == 'Navier-Stokes':
            from operator_splitting import NS_spectral_OS_rhs
            model = NS_spectral_OS_rhs(configuration, normalizer, run)
        elif pde == 'Shear Flow':
            from operator_splitting import NS_shearflow_OS_rhs
            model = NS_shearflow_OS_rhs(configuration, normalizer, run)
        elif pde == 'Euler-Fluid':
            from operator_splitting import Euler_FV_OS_rhs
            model = Euler_FV_OS_rhs(configuration, normalizer, run)
        elif pde == 'Incomp. Navier-Stokes':
            from operator_splitting import Incomp_PDEB_NS_OS_rhs
            model = Incomp_PDEB_NS_OS_rhs(configuration, normalizer, run)
        elif pde == 'Comp. Navier-Stokes':
            from operator_splitting import Comp_NS_PDEB_OS_rhs
            model = Comp_NS_PDEB_OS_rhs(configuration, normalizer, run)
        elif pde == 'Constrained MHD':
            from operator_splitting import Ideal_MHD_OS_rhs
            model = Ideal_MHD_OS_rhs(configuration, normalizer, run)

        else:
            raise ValueError(f"Unknown PDE: {pde} in operator splitting")
        
        return model
    
    else:

        return model_selection(configuration)
