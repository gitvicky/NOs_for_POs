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

def Navier_Stokes_Spectral(n_sims, data_dist):
    data_loc = '/pitagora_work/FUPB1_UKAEA_ML/vgopakum/Data/PMocz'
    if data_dist == 'ID':
        data =  np.load(data_loc + '/NS_Spectral_combined_pitagora.npz')
    elif data_dist == 'OOD':
        data =  np.load(data_loc + '/NS_Spectral_combined_pitagora_OOD_nu_1e-2.npz')

    u = data['u'].astype(np.float32)[:n_sims]
    v = data['v'].astype(np.float32)[:n_sims]
    p = data['p'].astype(np.float32)[:n_sims]
    rho = np.ones_like(u) #Taking rho to be 1. 
    x = data['x']
    y = x
    dt = data['dt']
    dt = torch.tensor(dt, dtype=torch.float)

    fields = stacked_fields([u,v])

    mask = ~torch.isnan(fields).any(dim=(1,2,3,4))
    fields = fields[mask]

    return fields, x, y, dt

def Euler_FV(n_sims, data_dist):
    #Finite Volume Simulation Data from Philip Mocz for Compressible Navier-Stokes 
    data_loc = '/pitagora_work/FUPB1_UKAEA_ML/vgopakum/Data/PMocz'
    if data_dist == 'ID':
        data =  np.load(data_loc + '/NS_FV_combined_pitagora.npz')
    if data_dist == 'OOD':
        data =  np.load(data_loc + '/NS_FV_combined_pitagora_gamma_2by3.npz')

    rho = data['rho'].astype(np.float32)[:n_sims]
    u = data['u'].astype(np.float32)[:n_sims]
    v = data['v'].astype(np.float32)[:n_sims]
    p = data['p'].astype(np.float32)[:n_sims] 

    dx = data['dx'].astype(np.float32)
    x = np.linspace(0, 1, 128)
    y = x 

    dt = data['dt']
    dt = torch.tensor(dt, dtype=torch.float)

    fields = stacked_fields([rho,u,v,p])

    #Slicing the data to reduce the size.

    return fields, x, y, dt
