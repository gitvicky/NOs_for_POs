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
    def __init__(self, data, params=None, input_window=64, prediction_steps=1):
        """
        Initialize the dataset for spatiotemporal sequence prediction.
        
        Args:
            data (numpy.ndarray or torch.Tensor): Input data of shape (batch_size, variables, xy_dim, time_steps)
            params (numpy.ndarray or torch.Tensor, optional): Parameters associated with each trajectory,
                shape (batch_size, n_params) or (batch_size,) for single parameter
            input_window (int): Number of time steps to use as input
            prediction_steps (int): Number of steps to predict ahead
        """
        if isinstance(data, np.ndarray):
            self.data = torch.tensor(data, dtype=torch.float32)
        else:
            self.data = data.float()
        
        # Handle parameters
        if params is not None:
            if isinstance(params, np.ndarray):
                self.params = torch.tensor(params, dtype=torch.float32)
            else:
                self.params = params.float()
            
            # Ensure params is 2D (batch_size, n_params)
            if self.params.ndim == 1:
                self.params = self.params.unsqueeze(1)
            
            # Validate params shape matches data batch size
            if self.params.shape[0] != data.shape[0]:
                raise ValueError(f"params batch size ({self.params.shape[0]}) must match "
                               f"data batch size ({data.shape[0]})")
        else:
            self.params = None
        
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
            tuple: ([input_sequence, param_vector], target_sequence) if params is provided
                (input_sequence, target_sequence) if params is None
                - input_sequence: Tensor of shape (variables, xy_dim, input_window)
                - param_vector: Tensor of shape (n_params,) - parameters for this trajectory
                - target_sequence: Tensor of shape (variables, xy_dim, prediction_steps)
        """
        sample_idx, start_idx = self.indices[idx]
        
        # Get input sequence
        end_idx = start_idx + self.input_window
        input_sequence = self.data[sample_idx, :, :, start_idx:end_idx]
        
        # Get target sequence
        target_start = end_idx
        target_end = target_start + self.prediction_steps
        target_sequence = self.data[sample_idx, :, :, target_start:target_end]
        
        if self.params is not None:
            # Get the parameter vector for this trajectory
            param_vector = self.params[sample_idx]
            return [input_sequence, param_vector], target_sequence
        else:
            return input_sequence, target_sequence


class DatasetWithParams(Dataset):
    def __init__(self, inputs, params, targets):
        self.inputs = inputs
        self.targets = targets
        
                # Handle parameters
        if params is not None:
            if isinstance(params, np.ndarray):
                self.params = torch.tensor(params, dtype=torch.float32)
            else:
                self.params = params.float()
            
            # Ensure params is 2D (batch_size, n_params)
            if self.params.ndim == 1:
                self.params = self.params.unsqueeze(1)
            
            # Validate params shape matches data batch size
            if self.params.shape[0] != self.inputs.shape[0]:
                raise ValueError(f"params batch size ({self.params.shape[0]}) must match "
                               f"data batch size ({self.inputs.shape[0]})")
        else:
            self.params = None
        

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return [self.inputs[idx], self.params[idx]], self.targets[idx]
    

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
    fields = fields.permute(0, 1, 3, 2)

    mass = data['mass']
    viscosity = data['para'][:n_sims]
    edge_attr, edge_index = data['edge_attr'], data['edge_index']

    #Slicing the data to reduce the size.
    fields = fields[...,::configuration['Physics']['t_slice']]
    dt = dt*configuration['Physics']['t_slice']

    return fields, x, y, dt, mass, viscosity, edge_attr, edge_index

# %%
def Wave_Spectral(configuration):
    #Testing with NS_Spectral (for now)
    n_sims = configuration['Data']['ntrain']
    data_loc = '/pitagora_work/FUPB1_UKAEA_ML/vgopakum/Data/'
    data =  np.load(data_loc + '/Spectral_Wave_data_LHS.npz')

    fields = data['u'].astype(np.float32)[:n_sims]
    x = data['x']
    y = data['y']
    t = data['t']
    dt = torch.tensor(t[1] - t[0], dtype=torch.float)

    #Slicing the data to reduce the size.
    fields = fields[:,::configuration['Physics']['x_slice'],::configuration['Physics']['y_slice'],::configuration['Physics']['t_slice']]
    x = x[::configuration['Physics']['x_slice']]
    y = x[::configuration['Physics']['y_slice']]
    dt = dt*configuration['Physics']['t_slice']

    xx, yy = np.meshgrid(x,y, indexing='ij')
    X, Y = xx.reshape(-1), yy.reshape(-1)
    X, Y = torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

    fields = fields.reshape(fields.shape[0], fields.shape[1], -1)
    fields = torch.tensor(fields, dtype=torch.float32).unsqueeze(1)
    fields = fields.permute(0, 1, 3, 2)
    wave_velocity = np.ones(n_sims)

    return fields, X, Y, dt, wave_velocity



#Testing on Diffusion Data built from pypde. 
import sys 
sys.path.append('/pitagora/home/userexternal/vgopakum/NOs_for_POs/Data/')
def diffusion_pypde(configuration):
    from pypde.generate_diffusion_dataset import load_dataset
    data = load_dataset('/pitagora/home/userexternal/vgopakum/NOs_for_POs/Data/pypde/diffusion_dataset.hdf5')
    fields = data['fields']
    dt = data['metadata']['dt']
    diffusivities = data['diffusivities']
    
    x = np.linspace(0,1,data['metadata']['grid_size'])
    y = np.linspace(0,1,data['metadata']['grid_size'])
    xx, yy = np.meshgrid(x, y, indexing='ij')
    x, y = xx.flatten(), yy.flatten()
    x, y = torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    fields = torch.tensor(fields, dtype=torch.float32).flatten(start_dim=-2).unsqueeze(1)
    fields = fields.permute(0, 1, 3, 2)

    #Slicing the data to reduce the size.
    fields = fields[...,::configuration['Physics']['t_slice']]
    dt = dt*configuration['Physics']['t_slice']

    return fields, x, y, dt, diffusivities


# %%
