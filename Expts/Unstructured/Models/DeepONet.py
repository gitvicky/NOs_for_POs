# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 22 Apr, 2024
@author: DeepXde
@modified: vgopakum

DeepOnet

"""      
import numpy as np 
import torch 
import torch.nn as nn 
import torch.functional as F 

import operator
from functools import reduce
from functools import partial
from collections import OrderedDict

# %%
################################################################
# DeepOnet - Code modified from DeepXDE
################################################################

class FNN(torch.nn.Module):
    """Fully-connected neural network."""

    def __init__(self, input_size, output_size, width, num_layers, activation=torch.nn.GELU()):
        super().__init__()

        self.linears = torch.nn.ModuleList()
        self.linears.append(torch.nn.Linear(input_size, width, dtype=torch.float32))
        for i in range(1, num_layers-1):
            self.linears.append(
                torch.nn.Linear(
                    width, width, dtype=torch.float32
                )
            )
        self.linears.append(torch.nn.Linear(width, output_size, dtype=torch.float32))
        self.activation = activation 

    def forward(self, inputs):
        x = inputs
        for j, linear in enumerate(self.linears[:-1]):
            x = self.activation(linear(x))
        x = self.linears[-1](x)
        return x


class DeepONet(torch.nn.Module):
    """Deep operator network.

    Args:
        layer_sizes_branch: A list of integers as the width of a fully connected network,
            or `(dim, f)` where `dim` is the input dimension and `f` is a network
            function. The width of the last layer in the branch and trunk net should be
            equal.
        layer_sizes_trunk (list): A list of integers as the width of a fully connected
            network.
        activation: If `activation` is a ``string``, then the same activation is used in
            both trunk and branch nets. If `activation` is a ``dict``, then the trunk
            net uses the activation `activation["trunk"]`, and the branch net uses
            `activation["branch"]`.
    """

    def __init__(
        self,
        in_branch,
        width_branch,
        layers_branch, 
        out_branch,
        in_trunk,
        width_trunk,
        layers_trunk, 
        out_trunk,
        activation=torch.tanh
        ):
        super(DeepONet, self).__init__()

        # Fully connected network
        self.activation = activation
        self.branch = FNN(in_branch, out_branch, width_branch, layers_branch)
        self.trunk = FNN(in_trunk, out_trunk, width_trunk, layers_trunk)
        self.b = torch.nn.parameter.Parameter(torch.tensor(0.0))

    def forward(self, x_func, x_loc):
            
            # Branch net to encode the input function
            branch_output = self.branch(x_func)
            # Trunk net to encode the domain of the output function
            trunk_output  = self.trunk(x_loc)

            # Dot product
            if branch_output.shape[-1] != trunk_output.shape[-1]:
                raise AssertionError(
                    "Output sizes of branch net and trunk net do not match."
                )
            x = torch.einsum('bl,nl->bn', branch_output, trunk_output)
            # x = torch.unsqueeze(x, -1)
            # Add bias
            # x += self.b            
            # x = torch.sum(branch_output * trunk_output, dim=-1)

            return x

    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))

        return c

# #Example Usage
# model = DeepONet(in_branch=100,
#         width_branch=256,
#         layers_branch=4, 
#         out_branch=100,
#         in_trunk=2,
#         width_trunk=256,
#         layers_trunk=4, 
#         out_trunk=100)

# trunk_in = torch.randn(100, 2)
# branch_in = torch.randn(20, 100)
# output = model(branch_in, trunk_in)
# print(output.shape)

class DON(torch.nn.Module):
    def __init__(
        self, 
        in_channels,
        out_channels,
        trunk_width,
        trunk_depth, 
        branch_width,
        branch_depth,
        x_in, 
        y_in
    ):
        super().__init__()
        
        self.deeponets = nn.ModuleList([
                DeepONet(in_branch=len(x_in),
                width_branch=branch_width,
                layers_branch=branch_depth, 
                out_branch=256,
                in_trunk=2,
                width_trunk=trunk_width,
                layers_trunk=trunk_depth, 
                out_trunk=256)
                for _ in range(in_channels)
                ])
        

        self.coords = torch.stack([x_in, y_in], dim=-1)

    def forward(self, X):
        X = X[..., -1].permute(0, 2, 1)
        batch_size = X.shape[0]
        outputs = []
        for var, don in enumerate(self.deeponets):
            outputs.append(don(X[...,var], self.coords))
                    
        output = torch.stack(outputs, dim=1).unsqueeze(-1)
        return output 
    
    def count_params(self):
        """Count the number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# #Example Usage
# x = torch.linspace(0, 1, 32)
# y = torch.linspace(0, 1, 32)
# xx, yy = torch.meshgrid(x, y)
# x_in, y_in = xx.flatten(), yy.flatten()

# model = DON(
#         in_channels=2,
#         out_channels=2,
#         trunk_width=32,
#         trunk_depth=4, 
#         branch_width=32,
#         branch_depth=4,
#         x_in=x_in, 
#         y_in=y_in
# )


# u_in = torch.randn(20, 2, 1024, 1)
# output = model(u_in)
# print(output.shape)


# %%

class MIONet(torch.nn.Module):
    def __init__(
        self, 
        in_channels,
        out_channels,
        trunk_width,
        trunk_depth, 
        branch_width,
        branch_depth,
        x_in, 
        y_in
    ):
        super().__init__()
        
        self.trunk = FNN(input_size=2, output_size=1, width=trunk_width, num_layers=trunk_depth)
        self.branches = nn.ModuleList([
            FNN(input_size=in_channels, output_size=1, width=branch_width, num_layers=branch_depth)
            for _ in range(out_channels)
            ])
        self.coords = torch.stack([x_in, y_in], dim=-1)

    def forward(self, X):
        X = X[0][..., -1].permute(0, 2, 1)
        batch_size = X.shape[0]
        trunked = self.trunk(self.coords)
        outputs = []
        for branch in self.branches:
            branched = torch.einsum('bpo, po->bp', branch(X), trunked) #Dot product across branch and trunk
            outputs.append(branched)
                    
        output = torch.stack(outputs, dim=1).unsqueeze(-1)
        return output 
    
    def count_params(self):
        """Count the number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# x = torch.linspace(0, 1, 32)
# y = torch.linspace(0, 1, 32)
# xx, yy = torch.meshgrid(x, y)
# x_in, y_in = xx.flatten(), yy.flatten()
# u_in = torch.randn(20, 2, 1024, 1)
# model = MIONet(2, 2, 32, 4, 32, 4, x_in, y_in)
# output = model(u_in)
# print(output.shape)


# %%
