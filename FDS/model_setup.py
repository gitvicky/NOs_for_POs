'''
Model Setup

'''

import sys
sys.path.append("..")
from FNO import * 
from Cond_AE import *
from Cond_VAE import *
from UNet import *
def model_initialisation(configuration, run):
    pde = configuration['Physics']['pde']

    if configuration['Model']['arch'] == 'FNO':
        model = FNO_multi2d(configuration['Model']['in_vars'], 
                            configuration['Model']['out_vars'], 
                            configuration['Model']['modes_x'], 
                            configuration['Model']['modes_y'],
                            configuration['Model']['width'],
                            configuration['Model']['n_layers']
                            )
    elif configuration['Model']['arch'] == 'AE':
        model = Conv3DAutoencoder(
            in_channels=3,
            out_channels=1, 
            conditional_features=3
        )
    elif configuration['Model']['arch'] == 'UNO':
        model = UNOModel(
            in_channels=configuration['Model']['in_vars'],
            out_channels=configuration['Model']['in_vars'],
            n_layers=configuration['Model']['n_layers']
        )

    elif configuration['Model']['arch'] == 'VAE':
        model = Conv3DVarConditionalAutoencoder(
            in_channels=3,
            out_channels=1, 
            conditional_features=3,
            latent_dim=configuration['Model']['latent_dim']
        )

    elif configuration['Model']['arch'] == 'UNet':
        model = UNet(
            n_channels=configuration['Model']['in_vars'],
            n_classes=configuration['Model']['out_vars'],
        )
    else:
        raise ValueError("Invalid architecture specified in the configuration.")
    
    return model
    

#Function to count_params
count_parameters = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)