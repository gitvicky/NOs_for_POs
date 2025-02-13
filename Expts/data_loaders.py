import numpy as np 
import torch 
import h5py 

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

    uvp = stacked_fields([u,v,p])[:n_sims]

    return uvp, x, x, dt


def Navier_Stokes_Incomp(n_sims=100):
    #PDEBench data
    data_loc = '/home/ir-gopa2/rds/rds-ukaea-ap001/ir-gopa2/Code/NOs_for_POs/Data'
    data = np.load(data_loc + '/NS_incomp_velocity_100_128_128.npz') 
    u = data['velocity'][...,0]
    v = data['velocity'][...,1]
    p = data['pressure'][...,0]
    force = data['force']

    uvp = stacked_fields([u,v,p])[:n_sims]
    x, y = np.arange(0, 1, 128), np.arange(0, 1, 128) 
    t = np.arange(0, 5.0, 0.005) * 10 
    dt = 0.005 * 10 
    dt = torch.tensor(dt, dtype=torch.float)

    return uvp, force, x, y, dt


incompressible_files = {'M0.1_Eta0.01_Zeta0.01': '2D_CFD_Rand_M0.1_Eta0.01_Zeta0.01_periodic_128_Train.hdf5',
                        'M0.1_Eta0.1_Zeta0.1': '2D_CFD_Rand_M0.1_Eta0.1_Zeta0.1_periodic_128_Train.hdf5',
                        'M1.0_Eta0.01_Zeta0.01': '2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5',
                        'M1.0_Eta0.1_Zeta0.1': '2D_CFD_Rand_M1.0_Eta0.1_Zeta0.1_periodic_128_Train.hdf5'}

def Navier_Stokes_Comp(n_sims=100, coeffs = 'M0.1_Eta0.01_Zeta0.01'):
    data_loc = '/home/ir-gopa2/rds/rds-ukaea-ap001/ir-gopa2/Data/PDEBench/pdebench/2D/CFD/2D_Train_Rand/'
    with h5py.File(data_loc + incompressible_files[coeffs], 'r') as f:
        keys = list(f.keys())
        vx = f['Vx'][:n_sims]
        vy = f['Vy'][:n_sims]
        # vz = f['Vz']
        density = f['density'][:n_sims]
        pressure = f['pressure'][:n_sims]
        t = f['t-coordinate']
        x = f['x-coordinate']
        y = f['y-coordinate']
        # z = f['z-coordinate']

        vx = np.array(vx, dtype=np.float32)
        vy = np.array(vy, dtype=np.float32)
        # vz = np.array(vz, dtype=np.float32)
        density = np.array(density, dtype=np.float32)
        pressure = np.array(pressure, dtype=np.float32)

        t = np.array(t, dtype=np.float32)    ###, t, x are equispaced
        x = np.array(x, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        # z = np.array(z, dtype=np.float32)
        dt = t[1] - t[0]

        fields = stacked_fields([vx, vy, pressure, density])
        dt, x, y = torch.tensor(dt), torch.tensor(x), torch.tensor(y)

    return fields, x, y, dt


# def Navier_Stokes_Incom(n_sims):
#     #PDEBench data
#     data_loc = '/home/ir-gopa2/rds/rds-ukaea-ap001/ir-gopa2/Data/PDEBench/pdebench/2D/NS_incom'

#     # Need to write a h5 data loader here for all the pdebench files. ÷

#     # return uv, x, y, dt


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
