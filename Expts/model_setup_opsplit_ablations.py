'''
Model Setup for the OpsSplit Ablations so that the operator_splitting used is called from the run itself.

'''

import sys
sys.path.append("..")
import importlib.util

from Neural_PDE.Models.Neural_Ops_lib import *

#Function to count_params
count_parameters = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def model_initialisation(configuration, normalizer, run, data_dist):
    pde = configuration['Physics']['pde']
    run_loc = '/pitagora/home/userexternal/vgopakum/NOs_for_POs/Expts/Weights/'

    # Dynamically import operator_splitting from the run directory
    operator_splitting_path = f"{run_loc}{run}/operator_splitting.py"
    spec = importlib.util.spec_from_file_location("operator_splitting", operator_splitting_path)
    operator_splitting = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(operator_splitting)

    if pde == 'Navier-Stokes':
        model = operator_splitting.NS_spectral_OS_rhs(configuration, normalizer, run)
        if data_dist=='OOD':
            model.nu = torch.tensor(0.01, dtype=torch.float32, requires_grad=False).to(device)
    elif pde == 'Euler-Fluid':
        model = operator_splitting.Euler_FV_OS_rhs(configuration, normalizer, run)
        if data_dist=='OOD':
            model.gamma = torch.tensor(2/3, dtype=torch.float32, requires_grad=False).to(device)   
    else:
        raise ValueError(f"Unknown PDE: {pde} in operator splitting")
    
    return model
