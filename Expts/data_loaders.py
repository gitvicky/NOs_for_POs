# %%
import numpy as np 
import torch 
from torch.utils.data import Dataset
import h5py 
import glob 
from tqdm import tqdm

def stacked_fields(variables):
    stack = []
    for var in variables:
        var = torch.from_numpy(var) #Converting to Torch
        var = var.permute(0, 2, 3, 1) #Permuting to be BS, Nx, Ny, Nt
        stack.append(var)
    stack = torch.stack(stack, dim=1)
    return stack

class SpatioTemporalDataset(Dataset):
    def __init__(self, data, input_window=64, prediction_steps=1):
        """
        Initialize the dataset for spatiotemporal sequence prediction.
        
        Args:
            data (numpy.ndarray): Input data of shape (batch_size, variables, x_dim, y_dim, time_steps)
            input_window (int): Number of time steps to use as input
            prediction_steps (int): Number of steps to predict ahead
        """
        self.data = torch.FloatTensor(data)
        self.input_window = input_window
        self.prediction_steps = prediction_steps
        
        # Store data dimensions
        self.batch_size, self.variables, self.x_dim, self.y_dim, self.time_steps = self.data.shape
        
        # Calculate valid start indices for sliding windows
        self.indices = self._create_indices()
    
    def _create_indices(self):
        """Create valid start indices for the sliding windows."""
        total_required_length = self.input_window + self.prediction_steps
        valid_start_indices = []
        
        # For each sample in the batch
        for sample_idx in range(self.batch_size):
            # Create sliding windows along temporal dimension
            max_start_idx = self.time_steps - total_required_length + 1
            sample_indices = [(sample_idx, i) for i in range(max_start_idx)]
            valid_start_indices.extend(sample_indices)
            
        return valid_start_indices
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.indices)
    
    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        Returns:
            tuple: (input_sequence, target_sequence)
                - input_sequence: Tensor of shape (variables, x_dim, y_dim, input_window)
                - target_sequence: Tensor of shape (variables, x_dim, y_dim, prediction_steps)
        """
        sample_idx, start_idx = self.indices[idx]
        
        # Get input sequence
        end_idx = start_idx + self.input_window
        input_sequence = self.data[sample_idx, :, :, :, start_idx:end_idx]
        
        # Get target sequence
        target_start = end_idx
        target_end = target_start + self.prediction_steps
        target_sequence = self.data[sample_idx, :, :, :, target_start:target_end]
        
        return input_sequence, target_sequence


# %% 
def Navier_Stokes_Spectral(n_sims):
    #Testing with NS_Spectral (for now)
    data_loc = '/home/ir-gopa2/rds/rds-ukaea-ap001/ir-gopa2/Code/Neural_PDE/Data'
    data =  np.load(data_loc + '/NS_Spectral_combined.npz')

    u = data['u'].astype(np.float32)
    v = data['v'].astype(np.float32)
    p = data['p'].astype(np.float32)
    rho = np.ones_like(u) #Taking rho to be 1. 
    x = data['x']
    dt = data['dt']
    
    dt = torch.tensor(dt, dtype=torch.float)

    uvp = stacked_fields([u,v])[:n_sims]

    return uvp, x, x, dt

def Euler_FV(n_sims):
    #Finite Volume Simulation Data from Philip Mocz for Compressible Navier-Stokes 
    data_loc = '/home/ir-gopa2/rds/rds-ukaea-ap001/ir-gopa2/Code/Neural_PDE/Data'
    data =  np.load(data_loc + '/NS_FV_combined.npz')
    u = data['u'].astype(np.float32)
    v = data['v'].astype(np.float32)
    p = data['p'].astype(np.float32)
    rho = data['rho'].astype(np.float32)
    dx = data['dx']
    x = np.linspace(0, 1, 128)

    dt = data['dt']
    dt = torch.tensor(dt, dtype=torch.float)

    fields = stacked_fields([u,v,p,rho])[:n_sims]

    return fields, x, x, dt

def Navier_Stokes_Incomp(n_sims=100):
    #PDEBench data
    data_loc = '/home/ir-gopa2/rds/rds-ukaea-ap001/ir-gopa2/Code/NOs_for_POs/Data'
    data = np.load(data_loc + '/NS_incomp_velocity_100_128_128.npz') 
    u = data['velocity'][...,0]
    v = data['velocity'][...,1]
    p = data['pressure'][...,0] / 3.0
    force = data['force']

    uvp = stacked_fields([u,v,p])[:n_sims]
    x, y = np.arange(0, 1, 128), np.arange(0, 1, 128) 
    t = np.arange(0, 5.0, 0.005) * 10 
    dt = 0.005 * 10 
    dt = torch.tensor(dt, dtype=torch.float)

    return uvp, force, x, y, dt


# def Navier_Stokes_Incom(n_sims):
#     #PDEBench data
#     data_loc = '/home/ir-gopa2/rds/rds-ukaea-ap001/ir-gopa2/Data/PDEBench/pdebench/2D/NS_incom'

#     # Need to write a h5 data loader here for all the pdebench files. ÷

#     # return uv, x, y, dt


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



def JOREK_electrostatic(n_sims):
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


def JOREK_electrostatic_naomi(n_sims):
    data_loc = '/home/ir-gopa2/rds/rds-ukaea-shared-bLH38hg3nf0/JOREK_tblob_dataset/JOREK_tblob/'
    folders = ['0001-0100', '0100-0300', '0300-0500', '0500-1000', '1000-1500', '1500-1800', 'valid']
    folders = ['0100-0300']
    for ii in range(len(folders)):
        rho_array, phi_array, T_array = [], [], []
        files = glob.glob(data_loc + folders[ii] + '/*.h5')
        for jj in tqdm(range(len(files))):
            with h5py.File(files[jj], 'r') as f:
                keys = list(f.keys())
                rho = f['rho(nR,nZ,n_times)']
                phi = f['Phi(nR,nZ,n_times)']
                T = f['T(nR,nZ,n_times)']
                Rgrid = f['Rgrid(nR,nZ)']
                Zgrid = f['Zgrid(nR,nZ)']

                rho_array.append(np.asarray(rho, dtype=np.float32))
                phi_array.append(np.asarray(phi, dtype=np.float32))
                T_array.append(np.asarray(T, dtype=np.float32))
                x = np.asarray(Rgrid, dtype=np.float32)
                y = np.asarray(Zgrid, dtype=np.float32)

        rho = np.asarray(rho_array)
        phi = np.asarray(phi_array)
        T = np.asarray(T_array)

        fields = stacked_fields([rho, phi, T])
        dt = 1.5e-6 #1.5 microseconds
        dt, x, y = torch.tensor(dt), torch.tensor(x), torch.tensor(y)

        return fields, dt, x, y
    
def JOREK_electromagnetic(n_sims=20):
    # all 6: Psi (poloidal magnetic flux), T (temperature), omega (vorticity), rho (particle density), u (electric potential), zj (toroidal plasma current density)
    data_loc = '/home/ir-gopa2/rds/rds-ukaea-shared-bLH38hg3nf0/JOREK_tblob_dataset_MHD/JOREK_V2/longer_traj'
    # file = 'jorek_run001.h5'
    files = glob.glob(data_loc + '/*.h5')
    rho_array, psi_array, T_array, u_array, omega_array, zj_array = [], [], [], [], [], []
    for file in tqdm(files):
        with h5py.File(file, 'r') as f:
            keys = list(f.keys())

            rho = f['rho']
            psi = f['Psi']
            T = f['T']
            u = f['u']
            omega = f['omega']
            zj = f['zj']
            
            Rgrid = f['R_mesh(nR,nZ)']
            Zgrid = f['Z_mesh(nR,nZ)']

            rho_array.append(np.asarray(rho, dtype=np.float32))
            psi_array.append(np.asarray(psi, dtype=np.float32))
            T_array.append(np.asarray(T, dtype=np.float32))
            u_array.append(np.asarray(u, dtype=np.float32))
            omega_array.append(np.asarray(omega, dtype=np.float32))
            zj_array.append(np.asarray(zj, dtype=np.float32))

            x = np.asarray(Rgrid, dtype=np.float32)
            y = np.asarray(Zgrid, dtype=np.float32)

    rho = np.asarray(rho_array)
    psi = np.asarray(psi_array)
    T = np.asarray(T_array)
    u = np.asarray(u_array)
    omega = np.asarray(omega_array)
    zj = np.asarray(zj_array)

    fields = stacked_fields([rho, psi, T, u, omega, zj])
    dt = 1.5e-6 #1.5 microseconds
    dt, x, y = torch.tensor(dt), torch.tensor(x), torch.tensor(y)

    return fields, dt, x, y

# %%
