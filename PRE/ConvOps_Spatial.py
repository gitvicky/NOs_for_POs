#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
24th May, 2024

Wrapper for Implementing Convolutional Operator as the Differential and Integral Operator 
- prefefined using a Finite Difference Scheme. 

Data used for all operations should be in the shape: BS, Nx, Ny
"""
import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from fft_conv_pytorch.fft_conv import * 

def get_stencil(dims, deriv_order, taylor_order=2):
    if dims == 1:
        if deriv_order == 0:  # Identity convolution
            return torch.tensor([
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0]
            ], dtype=torch.float32)
        elif deriv_order == 2 and taylor_order == 2:
            return torch.tensor([
                [0, 1, 0],
                [0, -2, 0],
                [0, 1, 0]
            ], dtype=torch.float32)
        elif deriv_order == 1 and taylor_order == 2:
            return torch.tensor([
                [0, -1, 0],
                [0, 0, 0],
                [0, 1, 0]
            ], dtype=torch.float32)
    elif dims == 2:
        if deriv_order == 2 and taylor_order == 2:
            return torch.tensor([
                [0., 1., 0.],
                [1., -4., 1.],
                [0., 1., 0.]
            ], dtype=torch.float32)
        elif deriv_order == 2 and taylor_order == 4:
            return torch.tensor([
                [0, 0, -1/12, 0, 0],
                [0, 0, 4/3, 0, 0],
                [-1/12, 4/3, -5/2, 4/3, -1/12],
                [0, 0, 4/3, 0, 0],
                [0, 0, -1/12, 0, 0]
            ], dtype=torch.float32)
        elif deriv_order == 2 and taylor_order == 6:
            return torch.tensor([
                [0, 0, 0, 1/90, 0, 0, 0],
                [0, 0, 0, -3/20, 0, 0, 0],
                [0, 0, 0, 3/2, 0, 0, 0],
                [1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90],
                [0, 0, 0, 3/2, 0, 0, 0],
                [0, 0, 0, -3/20, 0, 0, 0],
                [0, 0, 0, 1/90, 0, 0, 0]
            ], dtype=torch.float32)

    raise ValueError("Invalid stencil parameters")

def pad_kernel(grid, kernel):#Could go into the deriv conv class
    kernel_size = kernel.shape[0]
    bs, nt, nx, ny = grid.shape[0], grid.shape[1], grid.shape[2], grid.shape[3]
    return torch.nn.functional.pad(kernel, (0, nx - kernel_size, 0, ny-kernel_size, 0, nt-kernel_size), "constant", 0)

class ConvOperator():
    """
    A class for performing convolutions as a derivative or integrative operation on a given domain.
    By default the class instance evaluates the derivative

    Args:
        domain (str or tuple): The domain across which the derivative is taken.
            Can be 't' for time domain or ('x', 'y') for spatial domain.
        order (int): The order of derivation.
    """
    def __init__(self, domain=None, order=None, scale=1.0, taylor_order=2, conv='direct', device='cpu', requires_grad=False):

        try: 
            self.domain = domain #Axis across with the derivative is taken. 
            self.dims = len(self.domain) #Domain size
            self.order = order #order of derivation
            self.stencil = get_stencil(self.dims, self.order, taylor_order)

            if self.domain == 'x':
                self.axis = 0
            elif self.domain == 'y':
                self.axis = 1
            elif self.domain == ('x','y'):
                self.axis = 0
            else:
                raise ValueError("Invalid Domain. Must be either x,y or their combination")
            
            self.kernel = self.stencil
            self.kernel = scale*self.kernel
            self.kernel = self.kernel.to(device)

            if requires_grad==True:
                self.kernel.requires_grad_ = True 

        except:
            pass


        if conv == 'direct': 
            self.conv = self.convolution
        elif conv == 'spectral':
            self.conv = self.spectral_convolution
        else:
            raise ValueError("Unknown Convolution Method")

    def convolution(self, field, kernel=None):
        """
        Performs 3D derivative convolution.

        Args:
            f (torch.Tensor): The input field tensor.
            k (torch.Tensor): The convolution kernel tensor.

        Returns:
            torch.Tensor: The result of the 3D derivative convolution.
        """
        if kernel != None: 
            self.kernel = kernel

        conv = F.conv2d(field.unsqueeze(1), self.kernel.unsqueeze(0).unsqueeze(0), padding=(self.kernel.shape[0]//2, self.kernel.shape[1]//2))
        return conv.squeeze(1)
    

    def spectral_convolution(self, field, kernel=None, inverse=False):
        """
        Performs spectral convolution using the convolution theorem 

        f * g = \hat{f} . \hat{g}

        Args:
            f (torch.Tensor): The input field tensor.
            k (torch.Tensor): The convolution kernel tensor.

        Returns:
            torch.Tensor: The result of the 3D derivative convolution.
        """ 
        if kernel is not None:
            self.kernel = kernel

        # Add channel dimension for conv1d
        if field.dim() == 3:
            field = field.unsqueeze(1)

        kernel = self.kernel.unsqueeze(0).unsqueeze(0)
        convfft = fft_conv(field, kernel, padding=(self.kernel.shape[0]//2, self.kernel.shape[1]//2), inverse=inverse)

        return convfft.squeeze(1)


    def differentiate(self, field, kernel=None, correlation=False, slice_pad=True):

        """
        Performs Convolution using the convolution theorem. Manual Implementation. 
        
        f * g = \hat{f} . \hat{g}
        
        Args:
            field (torch.Tensor): The input field tensor.
            kernel (torch.Tensor): The convolution kernel tensor.
        Returns:
            torch.Tensor: The result of the 3D differentation operation. 
        """
        if kernel is not None:
            self.kernel = kernel

        # Add channel dimension for conv1d
        if field.dim() == 3:
            field = field.unsqueeze(1)

        pad_size = self.kernel.size(-1) // 2
        padded_field = F.pad(field, (pad_size,pad_size,pad_size,pad_size), mode='constant')

        field_fft = torch.fft.rfftn(padded_field.float(), dim=tuple(range(2, field.ndim)))
        kernel = self.kernel.unsqueeze(0).unsqueeze(0)

        kernel_padding = [
            pad
            for i in reversed(range(2, padded_field.ndim))
            for pad in [0, padded_field.size(i) - kernel.size(i)]
        ]
        padded_kernel = F.pad(kernel, kernel_padding)

        kernel_fft = torch.fft.rfftn(padded_kernel.float(), dim = tuple(range(2, field.ndim)))

        if correlation == True:
            kernel_fft.imag *= -1

        output = irfftn(field_fft * kernel_fft, dim=tuple(range(2, field.ndim)))

            # Remove extra padded values
        if slice_pad == True:
            crop_slices = [slice(None), slice(None)] + [
                slice(0, (padded_field.size(i) - kernel.size(i) + 1), 1)#stride=1
                for i in range(2, padded_field.ndim)
            ]

            output = output[crop_slices].contiguous()

        return output.squeeze(1)


    def integrate(self, field, kernel=None, correlation=False, slice_pad=False, eps=1e-6):

        """
        Performs Integration using the convolution theorem 3D derivative convolution.
        
        f * g * h = f  ; h =  1 / (g+eps)
        
        Args:
            field (torch.Tensor): The input field tensor.
            kernel (torch.Tensor): The convolution kernel tensor.
            eps (float): Avoiding NaNs
        Returns:
            torch.Tensor: The result of the 3D Integration Operation. 
        """
        
        if kernel is not None:
                self.kernel = kernel

        # Add channel dimension for conv1d
        if field.dim() == 3:
            field = field.unsqueeze(1)

        pad_size = self.kernel.size(-1) // 2
        padded_field = F.pad(field, (pad_size,pad_size,pad_size,pad_size), mode='constant')

        field_fft = torch.fft.rfftn(padded_field, dim=tuple(range(2, field.ndim)))
        kernel = self.kernel.unsqueeze(0).unsqueeze(0)

        kernel_padding = [
            pad
            for i in reversed(range(2, padded_field.ndim))
            for pad in [0, padded_field.size(i) - kernel.size(i)]
        ]
        padded_kernel = F.pad(kernel, kernel_padding)

        kernel_fft = torch.fft.rfftn(padded_kernel, dim = tuple(range(2, field.ndim)))
        inv_kernel_fft = 1 / (kernel_fft + eps)

        if correlation == True:
            inv_kernel_fft.imag *= -1 

        output = irfftn(field_fft * inv_kernel_fft, dim=tuple(range(2, field.ndim)))

            # Remove extra padded values
        if slice_pad == True:
            crop_slices = [slice(None), slice(None)] + [
                slice(0, (padded_field.size(i) - kernel.size(i) + 1), 1)#stride=1
                for i in range(2, padded_field.ndim)
            ]


            output = output[crop_slices].contiguous()

        return output.squeeze(1)



    def forward(self, field):
        """
        Performs the forward pass of the derivative convolution.

        Args:
            field (torch.Tensor): The input field tensor.

        Returns:
            torch.Tensor: The result of the derivative convolution.
        """
        return self.conv(field, self.kernel)

    
    def __call__(self, inputs):
        """
        Performs the forward pass computation when the instance is called.

        Args:
            inputs (torch.Tensor): The input tensor to the derivative convolution. 
            -has to be in shape (BS, Nt, Nx, Ny)

        Returns:
            torch.Tensor: The result of the derivative convolution.
        """
        outputs = self.forward(inputs)
        return outputs

# %% 
# #Example Usage

# import torch
# from matplotlib import pyplot as plt

# # One-liner 2D Gaussian function
# x, y = torch.meshgrid(torch.linspace(-5, 5, 100), torch.linspace(-5, 5, 100), indexing='ij')
# amplitude = torch.tensor(1.0, dtype=torch.float32)
# x_mean = torch.tensor(0.0, dtype=torch.float32)
# y_mean = torch.tensor(0.0, dtype=torch.float32)
# x_sigma = torch.tensor(1.0, dtype=torch.float32)
# y_sigma = torch.tensor(1.0, dtype=torch.float32)
# theta = torch.tensor(0.0, dtype=torch.float32)

# gaussian_2d = lambda x, y: amplitude * torch.exp(-0.5 * (((x - x_mean) * torch.cos(theta) - (y - y_mean) * torch.sin(theta))**2 / x_sigma**2 + ((x - x_mean) * torch.sin(theta) + (y - y_mean) * torch.cos(theta))**2 / y_sigma**2))
# signal = gaussian_2d(x, y).unsqueeze(0)

# # %% 

# D = ConvOperator(domain=('x','y'), order=2)
# direct_conv= D(signal)
# spectral_conv = D.spectral_convolution(signal)
# manual_conv = D.differentiate(signal, correlation=True, slice_pad=True)

# # %% 
# #Inverse 
# diff = D.differentiate(signal, correlation=True, slice_pad=True)
# integ = D.integrate(diff, correlation=False, slice_pad=True)

# plt.imshow(integ[0] - signal[0])
# plt.colorbar()

# # %%
