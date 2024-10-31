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
    
    dt = torch.tensor(dt, dtype=torch.float)

    uv = stacked_fields([u,v,p])[:n_sims]

    return uv, x, x, dt


def Navier_Stokes_Incomp(n_sims):
    #PDEBench data
    data_loc = '/home/ir-gopa2/rds/rds-ukaea-ap001/ir-gopa2/Code/NOs_for_POs/Data'
    uv = np.load(data_loc + '/NS_incomp_velocity_100_128_128.npy') #uv being the two compnoents of velocity
    uv = torch.tensor(uv, dtype=torch.float) #Converting to tensor
    uv = uv.permute(0, 4, 2, 3, 1)[:n_sims]
    x, y = np.arange(0, 1, 128), np.arange(0, 1, 128) 
    t = np.arange(0, 5.0, 0.005)
    dt = 0.005
    dt = torch.tensor(dt, dtype=torch.float)

    return uv, x, y, dt


def JOREK(n_sims):
    #JOREK MultiBlob Data 
    #https://iopscience.iop.org/article/10.1088/1741-4326/ad313a/meta
    data_loc = '/home/ir-gopa2/rds/rds-ukaea-ap001/ir-gopa2/Code/NOs_for_POs/Data'
    data = data_loc + '/JOREK_filtered.npz' 

    rho = np.load(data)['rho'].astype(np.float32)[:n_sims] / 1e20
    phi = np.load(data)['Phi'].astype(np.float32)[:n_sims] / 1e5
    T = np.load(data)['T'].astype(np.float32)[:200][:n_sims] / 1e6

    rho = np.nan_to_num(rho)
    phi = np.nan_to_num(phi)
    T = np.nan_to_num(T)

    fields = stacked_fields([rho,phi,T])

    x_grid =  np.load(data)['Rgrid'].astype(np.float32)
    y_grid =  np.load(data)['Zgrid'].astype(np.float32)
    t_grid =  np.load(data)['time'].astype(np.float32)

    t_norm = t_grid / t_grid[-1]
    dt = t_norm[1] - t_norm[0]
    dt = torch.tensor(dt, dtype=torch.float)

    return fields, x_grid, y_grid, dt
