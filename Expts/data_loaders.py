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
            data (numpy.ndarray or torch.Tensor): Input data of shape (batch_size, variables, x_dim, y_dim, time_steps)
            input_window (int): Number of time steps to use as input
            prediction_steps (int): Number of steps to predict ahead
        """
        if isinstance(data, np.ndarray):
            self.data = torch.tensor(data, dtype=torch.float32)
        else:
            self.data = data.float()
        
        self.input_window = input_window
        self.prediction_steps = prediction_steps
        
        # Store data dimensions
        self.batch_size, self.variables, self.x_dim, self.y_dim, self.time_steps = self.data.shape
        
        # Validate parameters
        total_required_length = self.input_window + self.prediction_steps
        if total_required_length > self.time_steps:
            raise ValueError(f"input_window ({input_window}) + prediction_steps ({prediction_steps}) = "
                           f"{total_required_length} exceeds time_steps ({self.time_steps})")
        
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

def Wave_Spectral(configuration):
    #Testing with NS_Spectral (for now)
    n_sims = configuration['Data']['ntrain']
    data_loc = '/home/ir-gopa2/rds/rds-ukaea-ap001/ir-gopa2/Code/Neural_PDE/Data'
    data =  np.load(data_loc + '/Spectral_Wave_data_LHS.npz')

    u = data['u'].astype(np.float32)[:n_sims]
    x = data['x']
    y = data['y']
    t = data['t']
    dt = torch.tensor(t[1] - t[0], dtype=torch.float)

    fields = stacked_fields([u])

    #Slicing the data to reduce the size.
    fields = fields[:,:,::configuration['Physics']['x_slice'],::configuration['Physics']['y_slice'],::configuration['Physics']['t_slice']]
    x = x[::configuration['Physics']['x_slice']]
    y = x[::configuration['Physics']['y_slice']]
    dt = dt*configuration['Physics']['t_slice']

    return fields, x, y, dt


def Conv_Diff_Jax(configuration):

    n_sims = configuration['Data']['ntrain']
    data_loc = '/pitagora_work/FUPB1_UKAEA_ML/vgopakum/Data/ConvDiff'
    data =  np.load(data_loc + '/ConvDiff_D_0.1_cx_1.0_cy_0.5.npz')
    # data =  np.load(data_loc + '/ConvDiff_D_0.5_cx_0.5_cy_1.0.npz')

    u = data['u'].astype(np.float32)[:n_sims]
    x = data['x']
    y = data['y']
    t = data['t']
    dt = torch.tensor(t[1] - t[0], dtype=torch.float)

    fields = stacked_fields([u])

    #Slicing the data to reduce the size.
    fields = fields[:,:,::configuration['Physics']['x_slice'],::configuration['Physics']['y_slice'],::configuration['Physics']['t_slice']]
    x = x[::configuration['Physics']['x_slice']]
    y = x[::configuration['Physics']['y_slice']]
    dt = dt*configuration['Physics']['t_slice']

    return fields, x, y, dt


def Navier_Stokes_Spectral(configuration):
    #Testing with NS_Spectral (for now)
    n_sims = configuration['Data']['ntrain']
    data_loc = '/pitagora_work/FUPB1_UKAEA_ML/vgopakum/Data/PMocz'
    data =  np.load(data_loc + '/NS_Spectral_combined_pitagora.npz')
    # data =  np.load(data_loc + '/NS_Spectral_combined_pitagora_OOD_nu_1e-2.npz')

    u = data['u'].astype(np.float32)[:n_sims]
    v = data['v'].astype(np.float32)[:n_sims]
    p = data['p'].astype(np.float32)[:n_sims]
    rho = np.ones_like(u) #Taking rho to be 1. 
    x = data['x']
    dt = data['dt']
    dt = torch.tensor(dt, dtype=torch.float)

    fields = stacked_fields([u,v])

    #Slicing the data to reduce the size.
    fields = fields[:,:,::configuration['Physics']['x_slice'],::configuration['Physics']['y_slice'],::configuration['Physics']['t_slice']]
    x = x[::configuration['Physics']['x_slice']]
    y = x[::configuration['Physics']['y_slice']]
    dt = dt*configuration['Physics']['t_slice']

    # mask = ~torch.isnan(fields).any(dim=(1,2,3,4))
    # fields = fields[mask]

    return fields, x, y, dt

def Euler_FV(configuration):
    #Finite Volume Simulation Data from Philip Mocz for Compressible Navier-Stokes 
    n_sims = configuration['Data']['ntrain']
    data_loc = '/pitagora_work/FUPB1_UKAEA_ML/vgopakum/Data/PMocz'
    # data =  np.load(data_loc + '/NS_FV_combined_pitagora.npz')
    data =  np.load(data_loc + '/NS_FV_combined_pitagora_gamma_2by3.npz')

    rho = data['rho'].astype(np.float32)[:n_sims]
    u = data['u'].astype(np.float32)[:n_sims]
    v = data['v'].astype(np.float32)[:n_sims]
    p = data['p'].astype(np.float32)[:n_sims] 

    dx = data['dx'].astype(np.float32)
    x = np.linspace(0, 1, 128)

    dt = data['dt']
    dt = torch.tensor(dt, dtype=torch.float)

    if configuration['Physics']['conservative']:
        # Converting from primitive variables to conservative variables
        vol = dx**2
        gamma = 5/3
        Mass   = rho * vol
        Momx   = rho * u * vol
        Momy   = rho * v * vol
        Energy = (p/(gamma-1) + 0.5*rho*(u**2+v**2))*vol
        
        fields =  stacked_fields([Mass, Momx, Momy, Energy]) #Reformulating to avoid division by rho
    else:
        fields = stacked_fields([rho,u,v,p])
        # fields = stacked_fields([rho,rho*u,rho*v,p])

    #Slicing the data to reduce the size.
    fields = fields[:,:,::configuration['Physics']['x_slice'],::configuration['Physics']['y_slice'],::configuration['Physics']['t_slice']]
    x = x[::configuration['Physics']['x_slice']]
    y = x[::configuration['Physics']['y_slice']]
    dt = dt*configuration['Physics']['t_slice']

    return fields, x, y, dt

def Navier_Stokes_Incomp(configuration):
    n_sims = configuration['Data']['ntrain']
    #PDEBench data
    data_loc = '/home/ir-gopa2/rds/rds-ukaea-ap001/ir-gopa2/Code/NOs_for_POs/Data'
    data = np.load(data_loc + '/NS_incomp_velocity_100_128_128.npz') 
    u = data['velocity'][...,0][:n_sims]
    v = data['velocity'][...,1][:n_sims]
    p = data['pressure'][...,0][:n_sims] #/ 3.0
    force = data['force']   

    fields = stacked_fields([u,v])
    x, y = np.arange(0, 1, 128), np.arange(0, 1, 128) 
    t = np.arange(0, 5.0, 0.005) * 10 
    dt = 0.005 * 10 
    dt = torch.tensor(dt, dtype=torch.float)

    #Slicing the data to reduce the size.
    fields = fields[:,:,::configuration['Physics']['x_slice'],::configuration['Physics']['y_slice'],::configuration['Physics']['t_slice']]
    x = x[::configuration['Physics']['x_slice']]
    y = y[::configuration['Physics']['y_slice']]
    dt = dt*configuration['Physics']['t_slice']

    return fields, force, x, y, dt


# def Navier_Stokes_Incom(configuration):
#     #PDEBench data
#     data_loc = '/home/ir-gopa2/rds/rds-ukaea-ap001/ir-gopa2/Data/PDEBench/pdebench/2D/NS_incom'

#     # Need to write a h5 data loader here for all the pdebench files. ÷

#     # return uv, x, y, dt


compressible_files = {'M0.1_Eta0.01_Zeta0.01': '2D_CFD_Rand_M0.1_Eta0.01_Zeta0.01_periodic_128_Train.hdf5',
                        'M0.1_Eta0.1_Zeta0.1': '2D_CFD_Rand_M0.1_Eta0.1_Zeta0.1_periodic_128_Train.hdf5',
                        'M1.0_Eta0.01_Zeta0.01': '2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5',
                        'M1.0_Eta0.1_Zeta0.1': '2D_CFD_Rand_M1.0_Eta0.1_Zeta0.1_periodic_128_Train.hdf5'}

def Navier_Stokes_Comp(configuration):
    n_sims = configuration['Data']['ntrain']
    coeffs = configuration['Physics']['coeff']
    data_loc = '/home/ir-gopa2/rds/rds-ukaea-ap001/ir-gopa2/Data/PDEBench/pdebench/2D/CFD/2D_Train_Rand/'

    with h5py.File(data_loc + compressible_files[coeffs], 'r') as f:
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

        fields = stacked_fields([density, vx, vy, pressure])
        dt, x, y = torch.tensor(dt), torch.tensor(x), torch.tensor(y)

    #Slicing the data to reduce the size.
    fields = fields[:,:,::configuration['Physics']['x_slice'],::configuration['Physics']['y_slice'],::configuration['Physics']['t_slice']]
    x = x[::configuration['Physics']['x_slice']]
    y = y[::configuration['Physics']['y_slice']]
    dt = dt*configuration['Physics']['t_slice']

    return fields, x, y, dt

def Constrained_MHD(configuration):
    data_loc = '/pitagora_work/FUPB1_UKAEA_ML/vgopakum/Data'
    n_sims = configuration['Data']['ntrain']
    data =  np.load(data_loc + '/Constrained_MHD_combined.npz')

    rho = data['rho'].astype(np.float32)[:n_sims]
    u = data['u'].astype(np.float32)[:n_sims]
    v = data['v'].astype(np.float32)[:n_sims]
    p = data['p'].astype(np.float32)[:n_sims]
    Bx = data['Bx'].astype(np.float32)[:n_sims]
    By  = data['By'].astype(np.float32)[:n_sims]

    x = data['x'].astype(np.float32)
    y = data['x'].astype(np.float32)
    dt = data['dt'].astype(np.float32)[0]
    dt = torch.tensor(dt, dtype=torch.float)

    fields = stacked_fields([rho, u, v, p, Bx, By])

    #Slicing the data to reduce the size.
    fields = fields[:,:,::configuration['Physics']['x_slice'],::configuration['Physics']['y_slice'],::configuration['Physics']['t_slice']]
    x = x[::configuration['Physics']['x_slice']]
    y = y[::configuration['Physics']['y_slice']]
    dt = dt*configuration['Physics']['t_slice']

    return fields, x, y, dt

def JOREK_electrostatic(configuration):
    n_sims = configuration['Data']['ntrain']
    #JOREK MultiBlob Data 
    #https://iopscience.iop.org/article/10.1088/1741-4326/ad313a/meta
    data_loc = '/home/ir-gopa2/rds/rds-ukaea-ap001/ir-gopa2/Code/NOs_for_POs/Data'
    data = data_loc + '/JOREK_filtered.npz' 

    rho = np.load(data)['rho'].astype(np.float32)[:n_sims] / 1e20
    phi = np.load(data)['Phi'].astype(np.float32)[:n_sims] / 1e5
    T = np.load(data)['T'].astype(np.float32)[:n_sims] / 1e6

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
    print(dt)

    #Slicing the data to reduce the size.
    fields = fields[:,:,::configuration['Physics']['x_slice'],::configuration['Physics']['y_slice'],::configuration['Physics']['t_slice']]
    x = x_grid[::configuration['Physics']['x_slice']]
    y = y_grid[::configuration['Physics']['y_slice']]
    dt = dt*configuration['Physics']['t_slice']

    return fields, x, y, dt


def JOREK_electrostatic_naomi(configuration):
    n_sims = configuration['Data']['ntrain']
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

    #Slicing the data to reduce the size.
    fields = fields[:,:,::configuration['Physics']['x_slice'],::configuration['Physics']['y_slice'],::configuration['Physics']['t_slice']]
    x = x[::configuration['Physics']['x_slice']]
    y = y[::configuration['Physics']['y_slice']]
    dt = dt*configuration['Physics']['t_slice']


    return fields, x, y, dt
    
def JOREK_electromagnetic(configuration):
    n_sims = configuration['Data']['ntrain']
    #JOREK MultiBlob Data
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

    #Slicing the data to reduce the size.
    fields = fields[:,:,::configuration['Physics']['x_slice'],::configuration['Physics']['y_slice'],::configuration['Physics']['t_slice']]
    x = x[::configuration['Physics']['x_slice']]
    y = y[::configuration['Physics']['y_slice']]
    dt = dt*configuration['Physics']['t_slice']

    return fields, x, y, dt

def FDS_Carpark(configuration):
    ntrain = configuration['ntrain']
    data_loc = '/home/ir-gopa2/rds/rds-ukaea-ap001/ir-gopa2/Code/NOs_for_POs/Data/FDS'
    data = np.load(data_loc + '/FDS_Carpark_temp_time_average.npz')
    fire_loc = data['fire_locations']
    temp = data['temperature']
    temp = np.nan_to_num(temp)
    temp = temp.astype(np.float32)
    
    T = torch.tensor(temp, dtype=torch.float32)
    x = np.arange(0, 100, 1.0)
    y = np.arange(0, 30, 1.0)
    z = np.array((2, 6, 10, 14, 18))
    t = np.arange(0, 1800, 15)
    vent_open = 120 #Vent Opening time

    dt = t[1] - t[0]
    dt = torch.tensor(dt, dtype=torch.float)

    # fields = stacked_fields([T])
    fields = T 
    fields = fields.permute(2, 0, 1, 3).unsqueeze(0)

    return fields, x, y, z, t, dt, fire_loc, vent_open


# The well datasets.
def Shear_Flow(configuration, reynolds = '1e4', schmidt='1e0'):
    #https://polymathic-ai.org/the_well/datasets/shear_flow/
    # configuration['Data']['reynolds'][0] = '5e5'
    reynolds = configuration['Data']['reynolds'][0]
    schmidt = configuration['Data']['schmidt']
    data_loc = '/pitagora_work/FUPB1_UKAEA_ML/vgopakum/Data/shear_flow/data/train/train/'
    # data_loc = '/pitagora_work/FUPB1_UKAEA_ML/vgopakum/Data/shear_flow/data/test'
    # data_loc = configuration['Data']['loc']
    u_list = []
    v_list = []
    p_list = []
    s_list = []

    for ii in tqdm(range(len(schmidt))):
        file = f'shear_flow_Reynolds_{reynolds}_Schmidt_{schmidt[ii]}.hdf5'
        print(file)
        data_vars = {}
        
        with h5py.File(data_loc + '/' + file, 'r') as f:
            def extract_data(name, obj):
                """Recursively extract data from HDF5 file"""
                if isinstance(obj, h5py.Dataset):
                    # Convert dataset to numpy array and store with key as variable name
                    var_name = name.replace('/', '_')  # Replace '/' with '_' for valid variable names
                    data_vars[var_name] = np.asarray(obj)
                    print(f"Extracted {var_name}: shape {data_vars[var_name].shape}")

            f.visititems(extract_data)

        x = data_vars.get('dimensions_x')
        y = data_vars.get('dimensions_y') 
        t = data_vars.get('dimensions_time')
        dt = t[1] - t[0]

        u = data_vars.get('t1_fields_velocity')[..., 0]
        v = data_vars.get('t1_fields_velocity')[..., 1]
        p = data_vars.get('t0_fields_pressure')
        s = data_vars.get('t0_fields_tracer')

        reynolds_scalar = data_vars.get('scalars_Reynolds')
        schmidt_scalar = data_vars.get('scalars_Schmidt')

        u_list.append(u)
        v_list.append(v)       
        p_list.append(p)
        s_list.append(s)
        
    u = np.concatenate(u_list, axis=0)
    v = np.concatenate(v_list, axis=0)
    p = np.concatenate(p_list, axis=0)
    s = np.concatenate(s_list, axis=0)

    fields = stacked_fields([u,v,s])

    #Slicing the data to reduce the size.
    fields = fields[:configuration['Data']['ntrain'],:,::configuration['Physics']['x_slice'],::configuration['Physics']['y_slice'],::configuration['Physics']['t_slice']]
    x = x[::configuration['Physics']['x_slice']]
    y = x[::configuration['Physics']['y_slice']]
    dt = dt*configuration['Physics']['t_slice']

    return fields, x, y, dt


def Euler_Quadrants(configuration, gamma= ['1.365'], gas = ['Dry_air_1000']):
    #https://polymathic-ai.org/the_well/datasets/euler_multi_quadrants_periodicBC/
    # data_loc = '/pitagora_work/FUPB1_UKAEA_ML/vgopakum/Data/euler_multi_quadrants_periodicBC/data/train'
    data_loc = configuration['Data']['loc']
    rho_list = []
    E_list = []
    px_list = []
    py_list = []
    P_list = []
    
    gamma = gamma[0]  # Assuming gamma is a list with one element
    for ii in tqdm(range(len(gas))):
        file = f'euler_multi_quadrants_periodicBC_gamma_{gamma}_{gas[ii]}.hdf5'
        print(file)
        data_vars = {}
        
        with h5py.File(data_loc + '/' + file, 'r') as f:
            def extract_data(name, obj):
                """Recursively extract data from HDF5 file"""
                if isinstance(obj, h5py.Dataset):
                    # Convert dataset to numpy array and store with key as variable name
                    var_name = name.replace('/', '_')  # Replace '/' with '_' for valid variable names
                    data_vars[var_name] = np.asarray(obj)
                    print(f"Extracted {var_name}: shape {data_vars[var_name].shape}")

            f.visititems(extract_data)

        # Now process the extracted data (outside the with block is fine)
        x = data_vars.get('dimensions_x')
        y = data_vars.get('dimensions_y')  # Note: this looks like a typo - should this be 'dimensions_y'?
        t = data_vars.get('dimensions_time')
        dt = t[1] - t[0]

        rho = data_vars.get('t0_fields_density')
        E = data_vars.get('t0_fields_energy')
        px = data_vars.get('t1_fields_momentum')[..., 0]
        py = data_vars.get('t1_fields_momentum')[..., 1]
        P = data_vars.get('t0_fields_pressure')
        gamma_scalar = data_vars.get('scalars_gamma')

        rho_list.append(rho)
        E_list.append(E)
        px_list.append(px)
        py_list.append(py)
        P_list.append(P)
            
        del rho, E, px, py, P


    rho = np.concatenate(rho_list, axis=0)
    E = np.concatenate(E_list, axis=0)
    px = np.concatenate(px_list, axis=0)
    py = np.concatenate(py_list, axis=0)
    P = np.concatenate(P_list, axis=0)

    del rho_list, E_list, px_list, py_list, P_list


    fields = stacked_fields([rho, E, px, py, P])
    print(fields.shape)

    del rho, E, px, py, P


    #Slicing the data to reduce the size.
    fields = fields[:configuration['Data']['ntrain'],:,::configuration['Physics']['x_slice'],::configuration['Physics']['y_slice'],::configuration['Physics']['t_slice']]
    x = x[::configuration['Physics']['x_slice']]
    y = x[::configuration['Physics']['y_slice']]
    dt = dt*configuration['Physics']['t_slice']


    return fields, x, y, dt

# %%

# import torch.nn as nn
# import sys
# sys.path.append('..')
# from PRE.ConvOps_2d import ConvOperator
# class CNS_residuals(nn.Module):
#     def __init__(self, device='cpu'):
#         super(CNS_residuals, self).__init__()

#         self.dx = torch.tensor(0.0078, dtype=torch.float32, requires_grad=True).to(device)
#         self.dy = torch.tensor(0.0078, dtype=torch.float32, requires_grad=True).to(device)  
#         self.dt = torch.tensor(0.05, dtype=torch.float32, requires_grad=True).to(device)
        
#         #Defining the required Convolutional Operations. 
#         self.D_t = ConvOperator(domain='t', order=1)
#         self.D_x = ConvOperator(domain='x', order=1)
#         self.D_y = ConvOperator(domain='y', order=1)
#         self.D_xx_yy = ConvOperator(domain=('x', 'y'), order=2)
#         self.eta = torch.tensor(0.01, dtype=torch.float32, requires_grad=True).to(device) #Dynamic viscosity
#         self.zeta = torch.tensor(0.01, dtype=torch.float32, requires_grad=True).to(device) #Bulk viscosity

#     def mass(self, vars, boundary=False):
#         rho = vars[:, 0]
#         u   = vars[:, 1]
#         v   = vars[:, 2]

#         print(u.shape, v.shape)
        
#         mass_residual = self.D_t(rho)*self.dx*self.dy + rho*(self.D_x(u) + self.D_y(v))*self.dt*self.dx + u*self.D_x(rho)*self.dx*self.dt + v*self.D_y(rho)*self.dy*self.dt

#         if boundary: 
#             return mass_residual
#         else:
#             return mass_residual[...,1:-1,1:-1,1:-1]
        
#     def momentum(self, vars, boundary=False):
#         rho = vars[:, 0]
#         u   = vars[:, 1]
#         v   = vars[:, 2]
#         p   = vars[:, 3]

#         mom_res_x = rho*(self.D_t(u)*2*self.dx**2 + u*self.D_x(u)*2*self.dt*self.dx + v*self.D_y(u)*2*self.dt*self.dx) + self.D_x(p)*2*self.dt*self.dx - self.eta*self.D_xx_yy(u)*4*self.dt - (self.zeta+self.eta/3)*(self.D_x(self.D_x(u) + self.D_y(v)))*self.dt
#         mom_res_y = rho*(self.D_t(v)*2*self.dx**2 + u*self.D_x(v)*2*self.dt*self.dx + v*self.D_y(v)*2*self.dt*self.dx) + self.D_y(p)*2*self.dt*self.dx - self.eta*self.D_xx_yy(v)*4*self.dt - (self.zeta+self.eta/3)*(self.D_y(self.D_x(u) + self.D_y(v)))*self.dt

#         mom_residuals = mom_res_x + mom_res_y
#         if boundary: 
#             return mom_residuals
#         else:
#             return mom_residuals[...,1:-1,1:-1,1:-1]
    
# %%
# res = CNS_residuals()
# mass_residual = res.mass(fields.permute(0, 1, 4, 2, 3)) #BS, Nvars, Nt, Nx, Ny
# mom_residual = res.momentum(fields.permute(0, 1, 4, 2, 3)) #BS, Nvars, Nt, Nx, Ny

# # %%
# from PRE.VectorConvOps_Spatial import * 

# class Euler_FV_OS_rhs(nn.Module):#Compressible Navier-Stokes Finite Volume Operator-Splitting right-hand-side.
#     def __init__(self, device='cpu'):
#         super(Euler_FV_OS_rhs, self).__init__()

#         self.dx = torch.tensor(0.0078, dtype=torch.float32, requires_grad=True).to(device)
#         self.dy = torch.tensor(0.0078, dtype=torch.float32, requires_grad=True).to(device)  
#         self.dt = torch.tensor(0.05, dtype=torch.float32, requires_grad=True).to(device)

#         self.gradient = Gradient(scale=1/(self.dx), taylor_order=2, boundary_cond='periodic', device=device, requires_grad=True)
#         self.laplace = Laplace(scale=1/(self.dx**2), taylor_order=2, boundary_cond='periodic', device=device, requires_grad=True)
#         self.divergence = Divergence(scale = 1/(self.dx), taylor_order=2, boundary_cond='periodic', device=device, requires_grad=True)

#     def forward(self, vars):
#         #vars is for a single time instance
#         rho = vars[:, 0]
#         u   = vars[:, 1]
#         v   = vars[:, 2]
                
#         rhs_mass = - rho*self.divergence(u,v) - dot(vectorize(u,v), self.gradient(rho)) 

#         return rhs_mass

# res = Euler_FV_OS_rhs()
# residual = res(fields.permute(0, 1, 4, 2, 3))

# # %%

