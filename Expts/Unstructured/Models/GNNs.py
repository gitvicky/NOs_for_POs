import torch 
import torch.nn as nn 
from torch_geometric.nn import  MessagePassing, GCNConv, NNConv
from torch_geometric.utils import add_self_loops
import torch.nn.functional as F

#Graph Convolutions
class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(GCN, self).__init__()
        self.num_layers = num_layers
        
        # Input layer
        self.convs = nn.ModuleList([GCNConv(in_channels, hidden_channels)])
        
        # Hidden layers
        self.convs.extend([GCNConv(hidden_channels, hidden_channels) for _ in range(num_layers - 2)])
        
        # Output layer
        self.convs.append(GCNConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.num_layers - 1:  # Apply ReLU to all but the last layer
                x = F.relu(x)
        return x
 
  
#NNconv with edge attributes as well. 
class NNConvNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, edge_dim):
        super(NNConvNet, self).__init__()
        self.num_layers = num_layers
        
        # Define the neural networks for each NNConv layer
        self.nns = nn.ModuleList()
        self.convs = nn.ModuleList()
        
        # Input layer
        self.nns.append(nn.Sequential(
            nn.Linear(edge_dim, hidden_channels * in_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels * in_channels, hidden_channels * in_channels)
        ))
        self.convs.append(NNConv(in_channels, hidden_channels, self.nns[-1], aggr='mean'))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.nns.append(nn.Sequential(
                nn.Linear(edge_dim, hidden_channels * hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels * hidden_channels, hidden_channels * hidden_channels)
            ))
            self.convs.append(NNConv(hidden_channels, hidden_channels, self.nns[-1], aggr='mean'))
        
        # Output layer
        self.nns.append(nn.Sequential(
            nn.Linear(edge_dim, out_channels * hidden_channels),
            nn.ReLU(),
            nn.Linear(out_channels * hidden_channels, out_channels * hidden_channels)
        ))
        self.convs.append(NNConv(hidden_channels, out_channels, self.nns[-1], aggr='mean'))

    def forward(self, x, edge_index, edge_attr):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr)
            if i < self.num_layers - 1:  # Apply ReLU to all but the last layer
                x = F.relu(x)
        return x
    

class CustomGNN(MessagePassing):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(CustomGNN, self).__init__(aggr='add')
        self.num_layers = num_layers

        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(in_channels, hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_channels, hidden_channels))
        
        # Output layer
        self.layers.append(nn.Linear(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        # Add self-loops to the adjacency matrix
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        for i in range(self.num_layers - 1):
            x = self.propagate(edge_index, x=x, layer_index=i)
            x = F.relu(x)

        # Final layer
        x = self.propagate(edge_index, x=x, layer_index=self.num_layers - 1)
        return x

    def message(self, x_j, layer_index):
        return self.layers[layer_index](x_j)

    def update(self, aggr_out):
        return aggr_out


# #Custom Message Passing Layer
# class custom_GNN(MessagePassing):
#     def __init__(self, in_channels, hidden_channels, out_channels):
#       super(custom_GNN, self).__init__(aggr='add')


#       self.net_1 = nn.Linear(in_channels, hidden_channels)
#       self.net_2 = nn.Linear(hidden_channels, out_channels)


#     def forward(self, x, edge_index):
#       #Add self-loops to the adjacency matrix.
#       # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

#       out = self.propagate(edge_index, x=x)
#       return out

#     def message(self, x_j, norm):
#       message = F.relu(self.net_1(x_j))
#       # message = self.net_2(message)
#       return message

#     def update(self, aggr_out):
#       aggr_out = self.net_2(aggr_out)
#       return aggr_out

#GNO - Zongyi's code
class MLP(torch.nn.Module):
    def __init__(self, layers, nonlinearity, out_nonlinearity=None, normalize=False):
        super(MLP, self).__init__()

        self.n_layers = len(layers) - 1

        assert self.n_layers >= 1

        self.layers = nn.ModuleList()

        for j in range(self.n_layers):
            self.layers.append(nn.Linear(layers[j], layers[j + 1]))

            if j != self.n_layers - 1:
                if normalize:
                    self.layers.append(nn.BatchNorm1d(layers[j + 1]))

                self.layers.append(nonlinearity())

        if out_nonlinearity is not None:
            self.layers.append(out_nonlinearity())

    def forward(self, x):
        for _, l in enumerate(self.layers):
            x = l(x)
        return x


class GNO_zl(nn.Module):
    def __init__(self, in_channel=3, width=32, mid_width=64, out_channel=3, r=0.1):
        super().__init__()

        kernel1 = MLP(
            [in_channel * 4, mid_width // 2, mid_width, width**2], torch.nn.GELU
        )
        kernel2 = MLP(
            [in_channel * 4, mid_width // 2, mid_width, width**2], torch.nn.GELU
        )
        kernel3 = MLP(
            [in_channel * 4, mid_width // 2, mid_width, width**2], torch.nn.GELU
        )
        kernel4 = MLP(
            [in_channel * 4, mid_width // 2, mid_width, width**2], torch.nn.GELU
        )
        self.conv1 = NNConv(width, width, kernel1, aggr="mean")
        self.conv2 = NNConv(width, width, kernel2, aggr="mean")
        self.conv3 = NNConv(width, width, kernel3, aggr="mean")
        self.conv4 = NNConv(width, width, kernel4, aggr="mean")

        self.linear1 = nn.Linear(in_channel, width)
        self.linear2 = nn.Linear(in_channel+1, width)
        self.to_output = torch.nn.Linear(width, out_channel)
        self.activation = F.gelu
        self.r = r

    def get_graph(self, x_in, x_out=None):
        if x_out is None:
            x_in = x_in.squeeze()
            pwd = torch.cdist(x_in, x_in).squeeze()
            edge_index = torch.stack(torch.where(pwd <= self.r))
            edge_index = torch.tensor(edge_index, dtype=torch.long, device=x_in.device)
            edge_attr = torch.cat([x_in[edge_index[0].T], x_in[edge_index[1].T]], dim=-1)
        else:
            x_in = x_in.squeeze()
            x_out = x_out.squeeze()
            N_in = x_in.shape[0]
            pwd = torch.cdist(x_in, x_out).squeeze()
            edge_index = torch.stack(torch.where(pwd <= self.r))
            edge_index = torch.tensor(edge_index, dtype=torch.long, device=x_in.device)
            edge_attr = torch.cat([x_in[edge_index[0].T], x_out[edge_index[1].T]], dim=-1)
            edge_index[1, :] = edge_index[1, :] + N_in
        return edge_index.detach(), edge_attr.detach()

    def forward(self, u_in, x_in=None, x_out=None):
        """
        u_in: (N_in, C)
        x_in: (N_in, d) or None
        When both x_in and x_out are None, the first input is just vectices and there's no feature associated to the vertices.
        Synthesize features.
        x_out: (N_out, d) or None
        """
        if x_in is None:
            x_in = u_in
            u_in = self.linear(x_in)
        if x_out is None:
            edge_index, edge_attr = self.get_graph(x_in)
            u = u_in
        else:
            N_in = x_in.shape[1]
            edge_index, edge_attr = self.get_graph(x_in, x_out)
            u_in = self.linear1(u_in)
            u_out = self.linear2(x_out)  # x_out (B, N_out, C)
            u = torch.cat([u_in, u_out], dim=1)  # x_out (B, N_in + N_out, C)
        u = u.squeeze()
        u = self.conv1(u, edge_index, edge_attr)
        u = self.activation(u)
        u = self.conv2(u, edge_index, edge_attr)
        u = self.activation(u)
        u = self.conv3(u, edge_index, edge_attr)
        u = self.activation(u)
        u = self.conv4(u, edge_index, edge_attr)
        u = u.unsqueeze(0)

        if x_out is not None:
            u = u[:, N_in:, :]

        u_out = self.to_output(u)
        return u_out
