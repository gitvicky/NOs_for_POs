import torch 
import torch.nn as nn

count_parameters = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)

class LearnableMatrix(nn.Module):
    def __init__(self, xsize=3, ysize=3, features=2, init_type='random'):
        """
        Create a learnable square matrix of size MxM.
        
        Args:
            size (int): The size of the square matrix (M)
            init_type (str): Initialization strategy - 'random', 'identity', 'zeros', or 'ones'
        """
        super(LearnableMatrix, self).__init__()
        
        # Create a parameter tensor of size MxM
        self.weight = nn.Parameter(torch.empty(features, xsize, ysize))
        
        # Initialize the matrix based on the specified strategy
        if init_type == 'random':
            # Xavier/Glorot initialization
            nn.init.xavier_uniform_(self.weight)
        elif init_type == 'identity':
            # Initialize as identity matrix
            nn.init.eye_(self.weight)
        elif init_type == 'zeros':
            # Initialize with zeros
            nn.init.zeros_(self.weight)
        elif init_type == 'ones':
            # Initialize with ones
            nn.init.ones_(self.weight)
        else:
            raise ValueError(f"Unknown initialization type: {init_type}")
    
    def forward(self, x=None):
        """
        Forward pass to return the matrix itself or apply it to input x.
        
        Args:
            x (torch.Tensor, optional): If provided, multiply matrix with x. 
                                    Expected shape: (batch_size, size)
        
        Returns:
            torch.Tensor: The matrix itself or the result of matrix multiplication
        """
        if x is None:
            return self.weight
        return x @ self.weight  # Matrix multiplication
    
    def get_matrix(self):
        """
        Get the current value of the matrix.
        
        Returns:
            torch.Tensor: The matrix as a tensor
        """
        return self.weight.detach()
    
    def count_params(self):
        """
        Count the number of parameters in the matrix.
        
        Returns:
            int: The number of parameters
        """
        return self.weight.numel()
