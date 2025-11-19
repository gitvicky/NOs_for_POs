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
import torch
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data
# Import helper functions from the provided utils.py content
from Models.utils import get_graph 

class GraphSpatioTemporalDataset(Dataset):
    def __init__(self, data, pos, params=None, input_window=64, prediction_steps=1, connectivity_radius=0.1):
        """
        Initializes the dataset, integrating the sliding window logic with PyG Data objects.
        
        Args:
            data (torch.Tensor): Input data of shape (Batch, Variables, Nodes, Time)
            pos (torch.Tensor/np.ndarray): Spatial coordinates of shape (Nodes, Dimensions)
            params (torch.Tensor/np.ndarray, optional): Global parameters (Batch, n_params)
            input_window (int): Number of time steps to use as input
            prediction_steps (int): Number of steps to predict ahead
            connectivity_radius (float): Radius for edge creation, passed to get_graph.
        """
        # 1. Data and Parameter Setup
        self.data = torch.as_tensor(data, dtype=torch.float32)
        self.pos = torch.as_tensor(pos, dtype=torch.float32)
        
        if params is not None:
            self.params = torch.as_tensor(params, dtype=torch.float32)
            if self.params.ndim == 1: self.params = self.params.unsqueeze(1)
        else:
            self.params = None
            
        self.input_window = input_window
        self.prediction_steps = prediction_steps
        self.connectivity_radius = connectivity_radius
        
        # Dimensions: [Batch, Vars, Nodes, Time]
        self.B, self.V, self.N, self.T = self.data.shape
        
        # 2. Pre-compute Static Graph Edges and Attributes using get_graph
        # Since node positions (pos) are static, we calculate edges once.
        # Calling get_graph with x_out=None (single discretization)
        self.edge_index, self.edge_attr = get_graph(
            x_in=self.pos,  # Use static positions
            x_out=None,     # Single discretization
            radius=self.connectivity_radius
        )
        # Note: get_graph returns detached tensors, which is good practice.

        # 3. Create Sliding Window Indices
        self.indices = self._create_indices()

    def _create_indices(self):
        """Create valid start indices for the sliding windows."""
        total_len = self.input_window + self.prediction_steps
        if total_len > self.T:
             raise ValueError(f"Required length ({total_len}) exceeds time steps ({self.T})")
        
        valid_indices = []
        # For each sample in the batch
        for b in range(self.B):
            # Create sliding windows along temporal dimension
            max_start = self.T - total_len + 1
            sample_indices = [(b, t) for t in range(max_start)]
            valid_indices.extend(sample_indices)
            
        return valid_indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        b_idx, start_idx = self.indices[idx]
        
        # --- A. Prepare Input Sequence (X) ---
        end_idx = start_idx + self.input_window
        # Slice: [Vars, Nodes, Window]
        x_seq = self.data[b_idx, :, :, start_idx:end_idx]
        
        # Aggregate the time window into the feature dimension (required for GNN)
        # 1. Permute to [Nodes, Vars, Window]
        x_seq = x_seq.permute(1, 0, 2) 
        # 2. Flatten to [Nodes, Vars * Window] (This is the feature vector x for the GNN)
        x_flat = x_seq.reshape(self.N, -1)
        
        # --- B. Prepare Target Sequence (Y) ---
        target_start = end_idx
        target_end = target_start + self.prediction_steps
        # Slice: [Vars, Nodes, Pred_Steps]
        y_seq = self.data[b_idx, :, :, target_start:target_end]
        
        # Flatten Target similarly: [Nodes, Vars * Pred_Steps] (This is the target y for the GNN)
        y_seq = y_seq.permute(1, 0, 2)
        y_flat = y_seq.reshape(self.N, -1)

        # --- C. Construct Graph Data Object ---
        graph_data = Data(
            x=x_flat, 
            edge_index=self.edge_index.clone().detach(),  # Use pre-computed static graph structure
            edge_attr=self.edge_attr.clone().detach(),    # Use pre-computed static edge features
            pos=self.pos.clone().detach(),
            y=y_flat
        )
        
        # Attach global parameters if they exist
        if self.params is not None:
            # Add a 'u' (global attribute) field for the parameters of this trajectory
            graph_data.params = self.params[b_idx].unsqueeze(0)
            
        return graph_data
    
# from torch_geometric.loader import DataLoader

# # --- 1. Mock Data Setup ---
# # Batch=10, Vars=2 (e.g., velocity_x, velocity_y), Nodes=100, Time=200
# batch_size, vars, nodes, time = 10, 2, 100, 200
# raw_data = torch.randn(batch_size, vars, nodes, time)
# positions = torch.rand(nodes, 2) # xy coordinates
# params = torch.rand(batch_size, 1) # e.g., Reynolds number

# # --- 2. Initialize Dataset ---
# dataset = GraphSpatioTemporalDataset(
#     data=raw_data,
#     pos=positions,
#     params=params,
#     input_window=1,     # Look at past 10 steps
#     prediction_steps=1,  # Predict next 1 step
#     connectivity_radius=0.1
# )

# # --- 3. Create DataLoader ---
# # Use PyG DataLoader!
# loader = DataLoader(dataset, batch_size=5, shuffle=True)

# # --- 4. Iterate ---
# for batch in loader:
#     # batch.x shape: [batch_size * nodes, vars * input_window]
#     # batch.edge_index shape: [2, num_edges_in_batch]
#     # batch.batch: vector indicating which graph each node belongs to
#     print(f"Input shape: {batch.x.shape}") 
#     print(f"Target shape: {batch.y.shape}")
    
#     # If you pass this to a GNN:
#     # out = model(batch.x, batch.edge_index)
#     break


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
    X, Y = torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    dt = torch.tensor(configuration['Physics']['dt'], dtype=torch.float)

    fields = stacked_fields([u,v])
    fields = fields.permute(0, 1, 3, 2)

    mass = data['mass']
    viscosity = data['para'][:n_sims]
    edge_attr, edge_index = data['edge_attr'], data['edge_index']

    #Slicing the data to reduce the size.
    fields = fields[...,::configuration['Physics']['t_slice']]
    dt = dt*configuration['Physics']['t_slice']

    return fields, X, Y, dt, mass, viscosity, edge_attr, edge_index

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
