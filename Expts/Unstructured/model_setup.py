'''
Model Setup

'''

import sys
sys.path.append("..")


from Neural_PDE.Models.Neural_Ops_lib import *
from Neural_PDE.Models.GNO_neuralop import * 
from Neural_PDE.Models.GINO_neuralop import * 
from Neural_PDE.Models.INR_NOs4POs import *
from Models.DeepONet import * 

#Function to count_params
count_parameters = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def model_selection(configuration, x, y):
    """    Selects the model based on the configuration provided."""

    if configuration['Model']['arch'] == 'GNO':

            model = GNO(
                    in_channels=configuration['Model']['in_vars'], 
                    out_channels=configuration['Model']['out_vars'], 
                    hidden_channels=configuration['Model']['width'], 
                    r=configuration['Model']['r'], 
                    n_layers=configuration['Model']['depth'],
                    x_in=x,
                    y_in=y,
                    grid_type='unstructured'
        )  
        
    
    elif configuration['Model']['arch'] == 'GINO':

            model = GINO(
                    in_channels=configuration['Model']['in_vars'], 
                    out_channels=configuration['Model']['out_vars'], 
                    hidden_channels=configuration['Model']['width'], 
                    fno_n_modes=(12, 12),
                    fno_n_layers=configuration['Model']['depth'],
                    in_gno_radius=configuration['Model']['r'], 
                    n_gno_layers=configuration['Model']['depth'],
                    x_in=x,
                    y_in=y,
                    latent_grid_size=32
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

