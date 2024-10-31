import numpy as np 
import torch 

def stacked_fields(variables):
    stack = []
    for var in variables:
        var = torch.from_numpy(var) #Converting to Torch
        var = var.permute(0, 2, 3, 1) #Permuting to be BS, Nx, Ny, Nt
        stack.append(var)
    stack = torch.stack(stack, dim=1)
    return stack


def Navier_Stokes_Spectral(n_sims):
    #Testing with NS_Spectral (for now)
    data_loc = '/home/ir-gopa2/rds/rds-ukaea-ap001/ir-gopa2/Code/Neural_PDE/Data'
    data =  np.load(data_loc + '/NS_Spectral_combined.npz')

    u = data['u'].astype(np.float32)
    v = data['v'].astype(np.float32)
    p = data['p'].astype(np.float32)
    x = data['x']
    dt = data['dt']


    uv = stacked_fields([u,v,p])[:n_sims]

    return uv, x, x, dt
