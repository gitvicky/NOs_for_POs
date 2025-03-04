# %% 
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

class BoundaryManager:
    """
    A class to manage boundary conditions for generic convolution operations.
    """
    
    SUPPORTED_TYPES = [
        'dirichlet',    # Fixed value at boundary
        'neumann',      # Zero gradient at boundary
        'periodic',     # Periodic boundary
        'symmetric',    # Reflection at boundary (alias for symmetric)
        'free_slip',    # Zero normal, zero gradient tangential (for vector fields)
        'outflow'       # Zero gradient (alias for neumann)
    ]
    
    def __init__(self, kernel_size):
        """
        Initialize the boundary manager
        
        Args:
            kernel_size: Tuple (height, width) of kernel size or int
        """
        # Store kernel size
        if isinstance(kernel_size, int):
            self.kernel_height = kernel_size
            self.kernel_width = kernel_size
        else:
            self.kernel_height, self.kernel_width = kernel_size
            
        # Calculate padding size
        self.pad_left = self.kernel_width // 2
        self.pad_right = self.kernel_width // 2
        self.pad_top = self.kernel_height // 2
        self.pad_bottom = self.kernel_height // 2
        
        # Default boundary conditions (periodic on all sides)
        self.boundary_types = {
            'left': 'periodic',
            'right': 'periodic',
            'top': 'periodic',
            'bottom': 'periodic'
        }
        
        # Default boundary values (for Dirichlet)
        self.boundary_values = {
            'left': 0.0,
            'right': 0.0,
            'top': 0.0,
            'bottom': 0.0
        }
        
    def set_boundary_type(self, side, bc_type, value=0.0):
        """
        Set boundary condition type for a specific side
        
        Args:
            side: One of 'left', 'right', 'top', 'bottom'
            bc_type: Boundary condition type from SUPPORTED_TYPES
            value: Value for Dirichlet boundary condition (default 0.0)
        """
        if side not in ['left', 'right', 'top', 'bottom']:
            raise ValueError(f"Unknown side: {side}. Use 'left', 'right', 'top', or 'bottom'")
            
        if bc_type.lower() not in self.SUPPORTED_TYPES:
            raise ValueError(f"Unsupported boundary type: {bc_type}")
            
        self.boundary_types[side] = bc_type.lower()
        self.boundary_values[side] = value
        
    def set_all_boundaries(self, bc_type, value=0.0):
        """Set all boundaries to the same type and value"""
        for side in ['left', 'right', 'top', 'bottom']:
            self.set_boundary_type(side, bc_type, value)
    
    def pad_signal(self, signal):
        """
        Apply padding to a signal based on boundary conditions
        
        Args:
            signal: Input tensor of shape [batch, channel, height, width] or [height, width]
            
        Returns:
            Padded tensor with appropriate boundary conditions
        """
        # Store original shape to restore later
        original_shape = signal.shape
        original_ndim = len(original_shape)
        
        # Convert to 4D if input is 2D
        if original_ndim == 2:
            signal = signal.unsqueeze(0).unsqueeze(0)
        
        # Start with the signal itself
        result = signal
        
        # Apply padding for each side independently
        # Left boundary
        if self.pad_left > 0:
            bc_type = self.boundary_types['left']
            bc_value = self.boundary_values['left']
            
            if bc_type == 'dirichlet':
                pad = (self.pad_left, 0, 0, 0)
                result = F.pad(result, pad, mode='constant', value=bc_value)
            elif bc_type in ['neumann', 'outflow']:
                pad = (self.pad_left, 0, 0, 0)
                result = F.pad(result, pad, mode='replicate')
            elif bc_type == 'periodic':
                # For periodic, we need to pull from the right side
                left_padding = result[:, :, :, -self.pad_left:]
                result = torch.cat([left_padding, result], dim=3)
            elif bc_type == 'symmetric':
                pad = (self.pad_left, 0, 0, 0)
                result = F.pad(result, pad, mode='reflect')
        
        # Right boundary
        if self.pad_right > 0:
            bc_type = self.boundary_types['right']
            bc_value = self.boundary_values['right']
            
            if bc_type == 'dirichlet':
                pad = (0, self.pad_right, 0, 0)
                result = F.pad(result, pad, mode='constant', value=bc_value)
            elif bc_type in ['neumann', 'outflow']:
                pad = (0, self.pad_right, 0, 0)
                result = F.pad(result, pad, mode='replicate')
            elif bc_type == 'periodic':
                # For periodic, we need to pull from the left side
                right_padding = result[:, :, :, :self.pad_right]
                result = torch.cat([result, right_padding], dim=3)
            elif bc_type == 'symmetric':
                pad = (0, self.pad_right, 0, 0)
                result = F.pad(result, pad, mode='reflect')
        
        # Top boundary
        if self.pad_top > 0:
            bc_type = self.boundary_types['top']
            bc_value = self.boundary_values['top']
            
            if bc_type == 'dirichlet':
                pad = (0, 0, self.pad_top, 0)
                result = F.pad(result, pad, mode='constant', value=bc_value)
            elif bc_type in ['neumann', 'outflow']:
                pad = (0, 0, self.pad_top, 0)
                result = F.pad(result, pad, mode='replicate')
            elif bc_type == 'periodic':
                # For periodic, we need to pull from the bottom
                top_padding = result[:, :, -self.pad_top:, :]
                result = torch.cat([top_padding, result], dim=2)
            elif bc_type == 'symmetric':
                pad = (0, 0, self.pad_top, 0)
                result = F.pad(result, pad, mode='reflect')
        
        # Bottom boundary
        if self.pad_bottom > 0:
            bc_type = self.boundary_types['bottom']
            bc_value = self.boundary_values['bottom']
            
            if bc_type == 'dirichlet':
                pad = (0, 0, 0, self.pad_bottom)
                result = F.pad(result, pad, mode='constant', value=bc_value)
            elif bc_type in ['neumann', 'outflow']:
                pad = (0, 0, 0, self.pad_bottom)
                result = F.pad(result, pad, mode='replicate')
            elif bc_type == 'periodic':
                # For periodic, we need to pull from the top
                bottom_padding = result[:, :, :self.pad_bottom, :]
                result = torch.cat([result, bottom_padding], dim=2)
            elif bc_type == 'symmetric':
                pad = (0, 0, 0, self.pad_bottom)
                result = F.pad(result, pad, mode='reflect')
        
        # Restore original shape dimension
        if original_ndim == 2:
            result = result.squeeze(0).squeeze(0)
            
        return result
    
    def apply_convolution(self, signal, kernel):
        """
        Apply convolution with specified kernel and boundary conditions
        
        Args:
            signal: Input tensor of shape [height, width]
            kernel: Convolution kernel of shape [height, width]
            
        Returns:
            Convolution result
        """
        # Convert kernel to PyTorch format
        kernel_torch = kernel.unsqueeze(0).unsqueeze(0)
        
        # Pad signal with boundary conditions
        padded_signal = self.pad_signal(signal)
        
        # Convert to 4D if needed
        if len(padded_signal.shape) == 2:
            padded_signal = padded_signal.unsqueeze(0).unsqueeze(0)
        
        # Apply convolution
        result = F.conv2d(padded_signal, kernel_torch)
        
        # Restore original shape
        result = result.squeeze(0).squeeze(0)
        
        return result
