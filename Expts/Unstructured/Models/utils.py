import numpy as np
import torch
from torch_geometric.data import Data

def get_graph_data(u_in: np.array ,x_in: np.array, r: float) -> Data:
    """
    Create a graph from input node positions and features.

    Args:
    x_in (torch.Tensor): Input tensor of shape (num_nodes, dim) representing node positions.
    u_in (torch.Tensor): Input tensor of shape (num_nodes, features) representing node features.
    r (float): Maximum distance for edge creation between nodes. - Radius of the Sphere from GNO

    Returns:
    Data: A PyTorch Geometric Data object containing:
        - x (torch.Tensor): Node features of shape (num_nodes, 1)
        - edge_index (torch.Tensor): Graph connectivity in COO format of shape (2, num_edges)
        - edge_attr (torch.Tensor): Edge features of shape (num_edges, 2 * dim)

    Notes:
    - The function creates edges between nodes that are within 'r' distance of each other.
    - Edge attributes are the concatenated positions of the connected nodes.
    """
    # Reshape node features to (num_nodes, 1)
    node_features = torch.from_numpy(u_in).float()

    # Ensure x_in is a 2D tensor
    x_in = torch.tensor(x_in).squeeze()
    if x_in.dim() == 1:
        x_in = x_in.unsqueeze(1)

    # Compute pairwise distances between nodes
    pwd = torch.cdist(x_in, x_in).squeeze()

    # Create edges for nodes within r distance
    edge_index = torch.stack(torch.where(pwd <= r))
    edge_index = torch.tensor(edge_index, dtype=torch.long, device=x_in.device)

    # Compute edge attributes (concatenated positions of connected nodes)
    edge_attr = torch.cat([x_in[edge_index[0]], x_in[edge_index[1]]], dim=-1)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    # Create and return the Data object
    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)


def get_graph(x_in, x_out=None, radius=0.1):
    """
    Create graph connectivity and edge features for either single or dual discretization setups.

    Args:
        x_in (torch.Tensor): Input tensor of shape (num_nodes_in, dim) representing input node positions
        x_out (torch.Tensor, optional): Output tensor of shape (num_nodes_out, dim) representing output 
            node positions. If None, creates a graph using only x_in nodes.
        radius (float): Maximum distance for edge creation between nodes (default: 0.1)

    Returns:
        tuple: (edge_index, edge_attr)
            - edge_index (torch.Tensor): Graph connectivity in COO format of shape (2, num_edges)
            - edge_attr (torch.Tensor): Edge features of shape (num_edges, 2 * dim)

    Notes:
        - For single discretization (x_out=None): Creates edges between x_in nodes within radius
        - For dual discretization: Creates edges between x_in and x_out nodes within radius
        - Edge attributes are the concatenated positions of connected nodes
        - For dual discretization, output node indices are offset by N_in to maintain unique indexing
    """

    if x_out is None:
        x_in = torch.tensor(x_in)
        # Single discretization case
        x_in = x_in.squeeze()
        # Compute pairwise distances between input nodes
        pwd = torch.cdist(x_in, x_in).squeeze()
        # Create edges for nodes within radius distance
        edge_index = torch.stack(torch.where(pwd <= radius))
        edge_index = torch.tensor(edge_index, dtype=torch.long, device=x_in.device)
        # Compute edge attributes by concatenating connected node positions
        edge_attr = torch.cat([x_in[edge_index[0].T], x_in[edge_index[1].T]], dim=-1)
    else:
        x_in = torch.tensor(x_in)
        x_out = torch.tensor(x_out)
        # Dual discretization case
        x_in = x_in.squeeze()
        x_out = x_out.squeeze()
        N_in = x_in.shape[0]  # Store number of input nodes for index offsetting
        # Compute distances between input and output nodes
        pwd = torch.cdist(x_in, x_out).squeeze()
        # Create edges between input and output nodes within radius
        edge_index = torch.stack(torch.where(pwd <= radius))
        edge_index = torch.tensor(edge_index, dtype=torch.long, device=x_in.device)
        # Compute edge attributes using input and output node positions
        edge_attr = torch.cat([x_in[edge_index[0].T], x_out[edge_index[1].T]], dim=-1)
        # Offset output node indices by N_in to ensure unique indexing
        edge_index[1, :] = edge_index[1, :] + N_in

    # Return detached tensors to prevent gradient computation
    return edge_index.detach(), edge_attr.detach()