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
def FDS_Carpark(configuration):
    # ntrain = configuration['Data']['ntrain']
    data_loc = '/home/ir-gopa2/rds/rds-ukaea-ap001/ir-gopa2/Code/NOs_for_POs/Data/FDS'
    
    # Temperature data
    if configuration['Physics']['variable'] == 'Temperature':
        data = np.load(data_loc + '/FDS_Carpark_temp_time_average.npz')
        data = np.load(data_loc + '/TEMPS_COMBINED.npz')
        temp = data['temperatures']
        temp = np.nan_to_num(temp)
        temp = temp.astype(np.float32)
        var = temp
        
    #Visibility data
    if configuration['Physics']['variable'] == 'Smoke':
        data = np.load(data_loc + '/MULTI_VIS_825.npz')
        vis = data['visibility']
        vis = np.nan_to_num(vis)
        vis = np.delete(vis, [267, 333, 510, 530, 629, 655], axis=0)
        vis = vis.astype(np.float32)
        var = vis

    fire_loc = data['fire_locations']

    ntrain = len(var)
    T = torch.tensor(var, dtype=torch.float32)
    x = torch.arange(0, 101, 1.0)
    y = torch.arange(0, 31, 1.0)
    z = torch.tensor((2, 6, 10, 14, 18), dtype=torch.float32)
    t = torch.arange(0, 1800, 15)
    vent_open_time = 120 #Vent Opening time
    

    if configuration['Model']['arch'] == 'AE' or configuration['Model']['arch'] == 'VAE':
        xx, yy, zz = torch.meshgrid(x, y, z)
        ins = torch.stack((xx, yy, zz))
        conds = torch.tensor(fire_loc[:ntrain], dtype=torch.float32)
        outs = T[:ntrain, ..., 80:81]
        print(ins.shape, conds.shape, outs.shape)
        return ins, conds, outs

    else: 
        if configuration['Train']['odesolve']['method'] == 'AR':

            # fields = stacked_fields([T])
            fields = T 
            fields = fields.permute(0, 3, 1, 2, 4)[:ntrain,...,::configuration['Physics']['t_slice']]#[...,4:]
            t = t[::configuration['Physics']['t_slice']]#[4:]
            dt = t[1] - t[0]
            return fields, x, y, z, t, dt, fire_loc, vent_open_time
        
        elif configuration['Train']['odesolve']['method'] == 'None':
            fields = T 
            ins = fields.permute(0, 3, 1, 2, 4)[...,0:1][:ntrain,...,::configuration['Physics']['t_slice']]
            outs = fields.permute(0, 3, 1, 2, 4)[...,80:81][:ntrain,...,::configuration['Physics']['t_slice']]
            # fields = torch.cat((ins, outs), dim=-1)  
            t = t[::configuration['Physics']['t_slice']][...,80:81]
            dt = 20*60
            return ins, fire_loc[:ntrain], outs



# %%
