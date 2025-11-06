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
        var = var.permute(0, 2, 1) #Permuting to be BS, Nxy, Nt
        stack.append(var)
    stack = torch.stack(stack, dim=1)
    return stack

class SpatioTemporalDataset(Dataset):
    #Rewritten for data of shape [BS, Nxy, Nt]
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
        self.batch_size, self.variables, self.xy_dim, self.time_steps = self.data.shape
        
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
        input_sequence = self.data[sample_idx, :, :, start_idx:end_idx]
        
        # Get target sequence
        target_start = end_idx
        target_end = target_start + self.prediction_steps
        target_sequence = self.data[sample_idx, :, :, target_start:target_end]
        
        return input_sequence, target_sequence


# %% 
def flow_past_cylinder(configuration):
    #Incompressible Flow
    n_sims = configuration['Data']['ntrain']
    data_loc = '/pitagora/home/userexternal/vgopakum/NOs_for_POs/Data'
    data =  np.load(data_loc + '/flow_past_cylinder/rawData.npy', allow_pickle=True)
    meshPosition =  np.loadtxt(data_loc + '/flow_past_cylinder/meshPosition_all.txt')

    u = np.transpose(data['x'][...,0].astype(np.float32)[:n_sims], (0, 2, 1))
    v = np.transpose(data['x'][...,1].astype(np.float32)[:n_sims], (0, 2, 1))
    p = np.transpose(data['x'][...,2].astype(np.float32)[:n_sims], (0, 2, 1))

    x, y = meshPosition[:, 0], meshPosition[:, 1]
    dt = torch.tensor(configuration['Physics']['dt'], dtype=torch.float)

    fields = stacked_fields([u,v])

    mass = data['mass']
    reynolds = data['para']
    edge_attr, edge_index = data['edge_attr'], data['edge_index']

    #Slicing the data to reduce the size.
    fields = fields[...,::configuration['Physics']['t_slice']]
    dt = dt*configuration['Physics']['t_slice']

    return fields, x, y, dt, mass, reynolds, edge_attr, edge_index
