import torch 
import torch.nn as nn
import torch.nn.functional as F
count_parameters = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)

class Matrix2d(nn.Module):
    def __init__(self, xsize=3, ysize=3, features=2, init_type='random'):
        """
        Create a learnable square matrix of size MxM.
        
        Args:
            size (int): The size of the square matrix (M)
            init_type (str): Initialization strategy - 'random', 'identity', 'zeros', or 'ones'
        """
        super(Matrix2d, self).__init__()
        
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


from Utils.boundary_conditions import BoundaryManager

class Convolution2d(nn.Module):
    def __init__(self, kernel_size=3, features=2, init_type='random', boundary_type='periodic'):
        """
        Create a learnable convolution with configurable kernel and boundary conditions.
        
        Args:
            kernel_size (int): The size of the convolution kernel
            features (int): Number of feature channels
            init_type (str): Initialization strategy - 'random', 'identity', 'zer   os', or 'ones'
            boundary_type (str): Type of boundary condition - 'dirichlet', 'neumann', 'periodic', 'symmetric'
        """
        super(Convolution2d, self).__init__()
        
        # Create a parameter tensor for convolutional kernel
        self.kernel = nn.Parameter(torch.empty(features, features, kernel_size, kernel_size))
        self.features = features
        self.kernel_size = kernel_size
        
        # Initialize the kernel based on the specified strategy
        if init_type == 'random':
            # Xavier/Glorot initialization
            nn.init.xavier_uniform_(self.kernel)
        elif init_type == 'identity':
            # Initialize to approximate identity operation
            # (for convolution, this is a kernel with 1 in center, 0 elsewhere)
            nn.init.zeros_(self.kernel)
            # Set center pixel to 1 for each feature map
            for i in range(features):
                self.kernel[i, i, kernel_size//2, kernel_size//2] = 1.0
        elif init_type == 'zeros':
            # Initialize with zeros
            nn.init.zeros_(self.kernel)
        elif init_type == 'ones':
            # Initialize with ones
            nn.init.ones_(self.kernel)
        else:
            raise ValueError(f"Unknown initialization type: {init_type}")
        
        # Set up boundary manager from the boundary_conditions.py
        self.boundary_manager = BoundaryManager(kernel_size)
        self.boundary_manager.set_all_boundaries(boundary_type)
 
    def apply_boundary_padding(self, x):
        """
        Apply padding to the input tensor based on boundary conditions.
        
        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Padded tensor ready for convolution
        """
        batch_size = x.shape[0]
        channels = x.shape[1]
        
        # Process each batch and channel separately for proper boundary handling
        result = []
        for b in range(batch_size):
            channels_result = []
            for c in range(channels):
                # Extract 2D slice and apply boundary padding
                x_slice = x[b, c]  # (height, width)
                padded_slice = self.boundary_manager.pad_signal(x_slice)
                channels_result.append(padded_slice.unsqueeze(0))
            
            # Stack channels back together
            batch_result = torch.cat(channels_result, dim=0).unsqueeze(0)
            result.append(batch_result)
        
        # Stack batches back together
        return torch.cat(result, dim=0)
    
    def get_kernel(self):
        """
        Get the current value of the convolution kernel.
        
        Returns:
            torch.Tensor: The kernel as a tensor
        """
        return self.kernel.detach()
    
    def count_params(self):
        """
        Count the number of parameters in the kernel.
        
        Returns:
            int: The number of parameters
        """
        return self.kernel.numel()
    
    def set_boundary_type(self, boundary_type, value=0.0):
        """
        Set the boundary condition type for all sides.
        
        Args:
            boundary_type (str): Type of boundary condition - 'dirichlet', 'neumann', 'periodic', 'symmetric'
            value (float): Value for Dirichlet boundary condition (default 0.0)
        """
        self.boundary_manager.set_all_boundaries(boundary_type, value)
    
    def set_specific_boundary(self, side, bc_type, value=0.0):
        """
        Set boundary condition for a specific side.
        
        Args:
            side (str): One of 'left', 'right', 'top', 'bottom'
            bc_type (str): Boundary condition type
            value (float): Value for Dirichlet boundary condition
        """
        self.boundary_manager.set_boundary_type(side, bc_type, value)



    def forward(self, x=None):
        """
        Forward pass to apply convolution with proper boundary conditions.
        
        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, features, height, width)
                              If None, return the current kernel
        
        Returns:
            torch.Tensor: Result of convolution with same shape as input x
        """
        if x is None:
            return self.kernel
        
        # Input shape validation and reshaping if needed
        original_shape = x.shape
        original_ndim = len(original_shape)
        
        # Handle different input shapes
        if original_ndim == 2:  # (height, width)
            x = x.unsqueeze(0).unsqueeze(0)  # -> (1, 1, height, width)
        elif original_ndim == 3:  # (batch, height, width)
            x = x.unsqueeze(1)  # -> (batch, 1, height, width)
        
        batch_size = x.shape[0]
        x_channels = x.shape[1]
        
        # Handle the case where input channels don't match kernel features
        if x_channels != self.features:
            # Either repeat the input channels or use only first 'features' channels
            if x_channels < self.features:
                # Repeat input channels to match feature count
                x = x.repeat(1, self.features // x_channels + 1, 1, 1)[:, :self.features, :, :]
            else:
                # Use only the first 'features' channels
                x = x[:, :self.features, :, :]
        
        # Apply padding based on boundary conditions
        padded_x = self.apply_boundary_padding(x)
        
        # Now perform convolution directly using F.conv2d
        output = F.conv2d(padded_x, self.kernel)
        
        # Restore original shape dimensionality
        if original_ndim == 2:
            output = output.squeeze(0).squeeze(0)  # -> (height, width)
        elif original_ndim == 3:
            output = output.squeeze(1)  # -> (batch, height, width)
        
        return output

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, init_type='random'):
        """
        Create a learnable 2D Fourier layer that performs FFT, linear transform, and Inverse FFT.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            modes1 (int): Number of Fourier modes to multiply in first dimension, at most floor(N/2) + 1
            modes2 (int): Number of Fourier modes to multiply in second dimension
            init_type (str): Initialization strategy - 'random', 'zeros', or 'ones'
        """
        super(SpectralConv2d, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        
        # Scaling factor for initialization
        self.scale = (1 / in_channels)
        
        # Parameters for the first and second sets of modes
        self.weights1 = nn.Parameter(torch.empty(in_channels, out_channels, 
                                               modes1, modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(torch.empty(in_channels, out_channels, 
                                               modes1, modes2, dtype=torch.cfloat))
        
        # Initialize weights based on the specified strategy
        if init_type == 'random':
            # Random initialization with scaling
            nn.init.uniform_(self.weights1.real, -0.5, 0.5)
            nn.init.uniform_(self.weights1.imag, -0.5, 0.5)
            nn.init.uniform_(self.weights2.real, -0.5, 0.5)
            nn.init.uniform_(self.weights2.imag, -0.5, 0.5)
            self.weights1.data *= self.scale
            self.weights2.data *= self.scale
        elif init_type == 'zeros':
            # Initialize with zeros
            nn.init.zeros_(self.weights1.real)
            nn.init.zeros_(self.weights1.imag)
            nn.init.zeros_(self.weights2.real)
            nn.init.zeros_(self.weights2.imag)
        elif init_type == 'ones':
            # Initialize with ones (with scaling)
            nn.init.ones_(self.weights1.real)
            nn.init.zeros_(self.weights1.imag)
            nn.init.ones_(self.weights2.real)
            nn.init.zeros_(self.weights2.imag)
            self.weights1.data *= self.scale
            self.weights2.data *= self.scale
        else:
            raise ValueError(f"Unknown initialization type: {init_type}")
    
    def compl_mul2d(self, input, weights):
        """
        Complex multiplication between input and weights.
        
        Args:
            input (torch.Tensor): Input tensor of shape (batch, in_channel, x, y)
            weights (torch.Tensor): Weight tensor of shape (in_channel, out_channel, x, y)
            
        Returns:
            torch.Tensor: Result of complex multiplication with shape (batch, out_channel, x, y)
        """
        return torch.einsum("bixy,ioxy->boxy", input, weights)
    
    def forward(self, x=None):
        """
        Forward pass to apply spectral convolution.
        
        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, in_channels, height, width)
                              If None, return the current weights
        
        Returns:
            torch.Tensor: Result of spectral convolution with shape (batch_size, out_channels, height, width)
        """
        if x is None:
            return self.weights1, self.weights2
        
        # Input shape validation and reshaping if needed
        original_shape = x.shape
        original_ndim = len(original_shape)
        
        # Handle different input shapes
        if original_ndim == 2:  # (height, width)
            x = x.unsqueeze(0).unsqueeze(0)  # -> (1, 1, height, width)
        elif original_ndim == 3:  # (batch, height, width)
            x = x.unsqueeze(1)  # -> (batch, 1, height, width)
        
        batchsize = x.shape[0]
        
        # Handle the case where input channels don't match expected in_channels
        x_channels = x.shape[1]
        if x_channels != self.in_channels:
            # Either repeat the input channels or use only first 'in_channels' channels
            if x_channels < self.in_channels:
                # Repeat input channels to match in_channels count
                x = x.repeat(1, self.in_channels // x_channels + 1, 1, 1)[:, :self.in_channels, :, :]
            else:
                # Use only the first 'in_channels' channels
                x = x[:, :self.in_channels, :, :]
        
        # Compute Fourier coefficients
        x_ft = torch.fft.rfft2(x)
        
        # Initialize output Fourier coefficients
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1,
                          dtype=torch.cfloat, device=x.device)
        
        # Multiply relevant Fourier modes
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        
        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        
        # Restore original shape dimensionality
        if original_ndim == 2:
            x = x.squeeze(0).squeeze(0)  # -> (height, width)
        elif original_ndim == 3:
            x = x.squeeze(1)  # -> (batch, height, width)
        
        return x
    
    def get_weights(self):
        """
        Get the current value of the Fourier weights.
        
        Returns:
            tuple: A tuple containing the two weight tensors (weights1, weights2)
        """
        return self.weights1.detach(), self.weights2.detach()
    
    def count_params(self):
        """
        Count the number of parameters in the Fourier weights.
        
        Returns:
            int: The number of parameters (real + imaginary components)
        """
        # Each complex number has 2 parameters (real and imaginary parts)
        return 2 * (self.weights1.numel() + self.weights2.numel())
    
    
class MLP2d(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        """
        Create a 2D Multi-Layer Perceptron using 1x1 convolutions.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            mid_channels (int): Number of channels in the hidden layer
        """
        super(MLP2d, self).__init__()
        
        # First layer: in_channels -> mid_channels with 1x1 convolution
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
        
        # Second layer: mid_channels -> out_channels with 1x1 convolution
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)
        
        # Activation function - GELU (Gaussian Error Linear Unit)
        self.activation = F.gelu

    def forward(self, x):
        """
        Forward pass of the MLP2d network.
        
        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, in_channels, height, width)
        
        Returns:
            torch.Tensor: Output tensor with shape (batch_size, out_channels, height, width)
        """
        # Apply first convolution
        x = self.mlp1(x)
        
        # Apply activation function
        x = self.activation(x)
        
        # Apply second convolution
        x = self.mlp2(x)
        
        return x


class FNO2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, grid='arbitrary'):
        """
        Create a 2D Fourier Neural Operator layer.
        
        The FNO2d combines spectral convolution with residual connections.
        
        Args:
            modes1 (int): Number of Fourier modes to multiply in first dimension
            modes2 (int): Number of Fourier modes to multiply in second dimension
            width (int): Number of channels to use throughout the network
            grid (str): Type of grid to use, defaults to 'arbitrary'
        """
        super(FNO2d, self).__init__()

        self.modes1 = modes1  # Number of Fourier modes in first dimension
        self.modes2 = modes2  # Number of Fourier modes in second dimension
        self.in_channels = in_channels  # Channel width
        self.out_channels = out_channels  # Channel width
        self.grid = grid      # Grid type
        
        # Spectral convolution component
        self.conv = SpectralConv2d(self.in_channels, self.out_channels, self.modes1, self.modes2)
        
        # MLP component for non-linear processing
        self.mlp = MLP2d(self.in_channels, self.out_channels, self.in_channels*2)
        
        # Standard convolution for residual connection
        self.w = nn.Conv2d(self.in_channels, self.out_channels, 1)
        
        # # Coordinate grid encoder
        # self.b = nn.Conv2d(2, self.out_channels, 1)


    def forward(self, x, grid=None):
        """
        Forward pass of the FNO2d layer.
        
        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, width, height, width)
            grid (torch.Tensor, optional): Coordinate grid tensor with shape 
                                          (batch_size, 2, height, width)
                                          where 2 represents the x and y coordinates.
                                          If None, a grid will be created based on 
                                          the self.grid setting.
        
        Returns:
            torch.Tensor: Output tensor with shape (batch_size, width, height, width)
        """
        shape = x.shape
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        
        # Create grid if not provided
        if grid is None:
            if self.grid == 'arbitrary':
                # Create normalized coordinate grid
                gridx = torch.tensor(torch.linspace(0, 1, size_x), dtype=torch.float, device=x.device)
                gridy = torch.tensor(torch.linspace(0, 1, size_y), dtype=torch.float, device=x.device)
                
                # Meshgrid for 2D coordinates
                gridx, gridy = torch.meshgrid(gridx, gridy, indexing='ij')
                
                # Stack and reshape to create the grid tensor
                grid = torch.stack([gridx, gridy], dim=-1)
                grid = grid.reshape(1, size_x, size_y, 2).permute(0, 3, 1, 2)
                grid = grid.repeat(batchsize, 1, 1, 1)
        
        # Apply spectral convolution followed by MLP
        x1 = self.conv(x)
        x1 = self.mlp(x1)
        
        # Apply residual connection
        x2 = self.w(x)
        
        # # Incorporate coordinate information
        # x3 = self.b(grid)
        
        # Combine all branches
        x = x1 + x2 #+ x3
        
        return x