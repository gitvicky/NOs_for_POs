#Imports

# %% 

import shutil
import os
import yaml 
import argparse

import sys
sys.path.append("..")
sys.path.append("../..")

import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn 

# %% 
#Loading config
def parse_args():
    parser = argparse.ArgumentParser(description='Training script with YAML config')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    return parser.parse_args()

args = parse_args()
with open(args.config, 'r') as f:
    configuration = yaml.safe_load(f)

#Loading dataset
from data_loaders import *
pde = configuration['Physics']['pde']
if pde == 'Incomp. Navier-Stokes':
    fields, x, y, dt, mass, params, edge_attr, edge_index = flow_past_cylinder(configuration)
if pde == 'Wave':
    fields, x, y, dt, params = Wave_Spectral(configuration)

t = torch.arange(0, configuration['Data']['t_out']*dt, dt)
fields = fields[...,:configuration['Data']['t_out']]

print("Data shape: " + str(fields.shape))

# %% 
# Normalising the data -- using the same normalisations for inputs and outputs
normalizer = nn.Identity()
fields_encoded = normalizer(fields)

from sklearn.model_selection import train_test_split
train_in, test_in, train_out, test_out = train_test_split(fields_encoded[...,:configuration['Data']['t_in']], fields_encoded[...,configuration['Data']['t_in']:configuration['Data']['t_out']], test_size=configuration['Data']['test_train_split'], random_state=42)
params_train, params_test = params[:int(configuration['Data']['ntrain']*(1-0.2))], params[-int(configuration['Data']['ntrain']*(0.2)):]

train_data = torch.cat((train_in, train_out), dim=-1)#Merging for creating the windowed dataset.
input_window = configuration['Train']['input_length']
prediction_steps = configuration['Train']['rollout_length'] - 1 
train_dataset = SpatioTemporalDataset(train_data, params_train, input_window, prediction_steps)
test_dataset = DatasetWithParams(test_in, params_test, test_out)

#Setting up the data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=configuration['Data']['batch_size'], shuffle=True, pin_memory=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=configuration['Data']['batch_size'], shuffle=False, pin_memory=True, num_workers=4)
print("Training Input: " + str(train_in.shape))
print("Training Output: " + str(train_out.shape))

# %% 
#Model Definition 
# MultiInputOutput Network

class FNN(torch.nn.Module):
    """Fully-connected neural network."""

    def __init__(self, input_size, output_size, width, num_layers, activation=torch.tanh):
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
        batch_size = X.shape[0]
        trunked = self.trunk(self.coords)
        outputs = []
        for branch in self.branches:
            branched = torch.einsum('bpo, po->bp', branch(X), trunked) #Dot product across branch and trunk
            outputs.append(branched)
                    
        output = torch.stack(outputs, dim=1)
        return output 


# %% 
#Instantiating the Model
model = MIONet(2, 2, 32, 4, 32, 4, x_in, y_in)
