# %% 
import torch
import torch.nn as nn
from typing import List, Dict, Union, Optional, Tuple, Any

import sys 
sys.path.append('..')
from learnable_matrices import * 
from Utils.boundary_conditions import BoundaryManager

def sequential_model(
    configuration: Dict[str, Any],
    num_layers: int = 1,
    activation: Optional[str] = None,
    layer_widths: Optional[List[int]] = None,
    dropout_rate: Optional[float] = None,
    layer_types: Optional[List[str]] = None,
) -> nn.Sequential:
    """
    Build a flexible nn.Sequential model with arbitrary number and types of layers.
    
    Args:
        configuration (Dict): Configuration dictionary containing model and physics parameters
        num_layers (int): Number of learnable layers to include (default: 1)
        activation (str, optional): Activation function between layers ('gelu', 'relu', 'tanh', etc.)
        layer_widths (List[int], optional): Width of each layer (if None, uses configuration values)
        dropout_rate (float, optional): Dropout rate to apply between layers (if None, no dropout)
        layer_types (List[str], optional): List of layer types to use ('matrix', 'conv', etc.)
                                          If None, uses configuration['Model']['arch'] for all layers
    
    Returns:
        nn.Sequential: Model with the specified architecture and number of layers
    """
    layers = []
    
    # Set defaults based on configuration if not provided
    if layer_types is None:
        # Use the same architecture for all layers
        layer_types = [configuration['Model']['arch']] * num_layers
    elif len(layer_types) < num_layers:
        # Extend layer_types to match num_layers by repeating the last element
        layer_types.extend([layer_types[-1]] * (num_layers - len(layer_types)))
    
    # Map activation function names to their classes
    activation_map = {
        'gelu': nn.GELU,
        'relu': nn.ReLU,
        'tanh': nn.Tanh,
        'sigmoid': nn.Sigmoid,
        'leaky_relu': lambda: nn.LeakyReLU(0.2),
        'elu': nn.ELU,
        'selu': nn.SELU,
        'none': None
    }
    
    # Get the activation function class if specified
    activation_fn = None
    if activation and activation.lower() in activation_map:
        activation_fn = activation_map[activation.lower()]
    
    # Build layers according to the specified types
    for i in range(num_layers):
        layer_type = layer_types[i]
        
        # Get layer width if specified, otherwise use configuration values
        width = None
        if layer_widths and i < len(layer_widths):
            width = layer_widths[i]
        
        # Add the appropriate layer based on type
        if layer_type == 'Matrix':
            layers.append(
                Matrix2d(
                    xsize=configuration['Physics']['Nx'],
                    ysize=configuration['Physics']['Ny'],
                    features=width if width else configuration['Physics']['variables'],
                    init_type=configuration['Model'].get('init_type', 'random')
                )
            )
        elif layer_type == 'Conv':
            layers.append(
                Convolution2d(
                    kernel_size=configuration['Model']['kernel_size'],
                    features=width if width else configuration['Physics']['variables'],
                    init_type=configuration['Model'].get('init', 'random'),
                    boundary_type=configuration['Model'].get('boundary_type', 'periodic')
                )
            )

        elif layer_type == 'SpectralConv':
            layers.append(
                SpectralConv2d(
                    in_channels=configuration['Model']['in_vars'],
                    out_channels=configuration['Model']['out_vars'],
                    modes1=configuration['Model']['modes'],
                    modes2=configuration['Model']['modes'],
                    init_type=configuration['Model'].get('init', 'random'),
                )
            )
        
        elif layer_type == 'FNO':
            layers.append(
                FNO2d(
                    in_channels=configuration['Model']['in_vars'],
                    out_channels=configuration['Model']['out_vars'],
                    modes1=configuration['Model']['modes'],
                    modes2=configuration['Model']['modes'],
                    # init_type=configuration['Model'].get('init', 'random'),
                )
            )
        

        elif layer_type == 'SelfAttention':
            layers.append(
                SelfAttention2d(
                            channels=configuration['Model']['channels'],
                            heads=configuration['Model']['heads'],  
                            dropout=configuration['Model']['dropout'],
                            init_type=configuration['Model'].get('init', 'random'),
                )
            )
        else:
            raise ValueError(f"Unknown layer type: {layer_type}")
        
        # Add activation after each layer except the last one
        if activation_fn and i < num_layers - 1:
            layers.append(activation_fn())
            
            # Add dropout if specified
            if dropout_rate and dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
    
    # Create and return the sequential model
    return nn.Sequential(*layers)


def build_model(configuration: Dict[str, Any]) -> nn.Module:
    """
    Build a model based on the configuration dictionary.
    This is a higher-level function that uses build_flexible_sequential_model internally.
    
    Args:
        configuration (Dict): Configuration dictionary containing model and physics parameters
    
    Returns:
        nn.Module: The constructed model
    """
    # Extract model-specific configuration
    model_config = configuration.get('Model', {})
    architecture = model_config.get('arch', 'Conv')
    num_layers = model_config.get('n_layers', 1)
    activation = model_config.get('act', None)
    dropout_rate = model_config.get('dropout', 0.0)
    
    # For more complex architectures with different layer sizes
    layer_widths = model_config.get('layer_widths', None)
    layer_types = model_config.get('layer_types', None)
    
    # Handle special case of a custom defined architecture
    if architecture == 'custom':
        # Use the explicit layer types defined in the configuration
        if layer_types is None:
            raise ValueError("For 'custom' architecture, 'layer_types' must be specified")
    else:
        # For predefined architectures
        if layer_types is None:
            layer_types = [architecture] * num_layers
    
    # Build the model using the flexible builder
    return sequential_model(
        configuration=configuration,
        num_layers=num_layers,
        activation=activation,
        layer_widths=layer_widths,
        dropout_rate=dropout_rate,
        layer_types=layer_types
    )

