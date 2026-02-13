'''
Model Setup

'''

import sys
sys.path.append("..")


# from Neural_PDE.Models.Neural_Ops_lib import *
from Models.GNNs import * 
from Neural_PDE.Models.INR_NOs4POs import *
from Models.DeepONet import * 

#Function to count_params
count_parameters = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def model_selection(configuration, x, y):
    """    Selects the model based on the configuration provided."""

    if configuration['Model']['arch'] == 'GNO':


            model = GNO2DTimeSolver(
            in_channels=configuration['Model']['in_vars'],    # Scalar field (e.g. Pressure)
            out_channels=configuration['Model']['in_vars'],   # Scalar field
            coord_dim=2,      # 2D Mesh
            latent_channels=configuration['Model']['width'],
            num_layers=configuration['Model']['depth'],
            radius=0.1
        )


    elif configuration['Model']['arch'] == 'GINO':

            model = GINO2DTimeSolver(
                    in_channels=2, 
                    out_channels=2,
                    coord_dim=2, 
                    fno_modes=(configuration['Model']['modes'], configuration['Model']['modes']),
                    fno_hidden_channels=configuration['Model']['hidden_channels'], 
                    latent_resolution=(32, 32), # Grid size for FNO
                    radius=0.2
                )  
            

    elif configuration['Model']['arch'] == 'SIREN':
 
        model = SIREN(
            in_channels=configuration['Model']['in_vars'],
            out_channels=configuration['Model']['out_vars'],
            hidden_features=configuration['Model']['width'],      # Network width
            hidden_layers=configuration['Model']['depth'],        # Network depth
            x_in=x,
            y_in=y,
            first_omega_0=configuration['Model'].get('first_omega', 30),    # First layer frequency
            hidden_omega_0=configuration['Model'].get('hidden_omega', 30),  # Hidden layer frequency
            outermost_linear=True  # Use linear output layer
        )


    elif configuration['Model']['arch'] == 'FourierNet':

        model = FourierNet(
            in_channels=configuration['Model']['in_vars'],
            out_channels=configuration['Model']['out_vars'],
            hidden_size=configuration['Model']['width'],          # Network width
            n_layers=configuration['Model']['depth'],             # Network depth
            x_in=x,
            y_in=y,
            input_scale=configuration['Model'].get('input_scale', 256.0),  # Fourier feature scale
            weight_scale=configuration['Model'].get('weight_scale', 1.0),
            bias=True,
            output_act=False
        )

    elif configuration['Model']['arch'] == 'DeepONet':
        model = MIONet(
            in_channels=configuration['Model']['in_vars'],
            out_channels=configuration['Model']['out_vars'],
            branch_width = configuration['Model']['branch_width'],
            branch_depth = configuration['Model']['branch_depth'],
            trunk_width = configuration['Model']['trunk_width'],
            trunk_depth = configuration['Model']['trunk_depth'],
            x_in=x,
            y_in=y
        )
        model.coords = model.coords.to(device)
    
    else:
        raise ValueError(f"Unknown architecture: {configuration['Model']['arch']}. ")
    
    return model

def model_initialisation(configuration, normalizer, run, x, y):
    pde = configuration['Physics']['pde']
    if configuration['Model']['operator_splitting'] == True: 
    #With operator_splitting.
        if pde == 'Incomp. Navier-Stokes':
            from operator_splitting import NS_incompressible_rhs
            model = NS_incompressible_rhs(configuration, normalizer, run, x, y)
        else:
            raise ValueError(f"Unknown PDE: {pde} in operator splitting")
        return model
    else:
        return model_selection(configuration, x, y)

