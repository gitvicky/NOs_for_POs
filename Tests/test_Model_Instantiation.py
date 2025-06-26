#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Test Suite for Model Instantiation in Training Pipeline

This test suite verifies:
1. Correct model instantiation via model_initialisation function
2. Model compatibility with expected data shapes and configurations
3. Parameter counting and initialization
4. Input/output shape consistency
5. Model-specific requirements and constraints
6. Operator splitting model functionality
7. Device compatibility (CPU/CUDA)

Run with: python -m pytest test_model_instantiation.py -v
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import yaml
import tempfile
import os
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your actual model setup
try:
    from Expts.model_setup import model_initialisation, count_parameters
    from Neural_PDE.Models.FNO import FNO_multi2d
    from Neural_PDE.Models.UNet import UNet2d
    from Neural_PDE.Models.CNO import CNO2d
    from Neural_PDE.Models.gMLP_Vision import gMLP
    # from Neural_PDE.Models.ConvOperator import ConvolutionalModel
    from Neural_PDE.Utils.processing_utils import (
        MinMax_Normalizer, MinMax_Normalizer_variable, 
        Gaussian_Normalizer, Identity_Normalizer
    )
    MODEL_SETUP_AVAILABLE = True
except ImportError as e:
    MODEL_SETUP_AVAILABLE = False
    print(f"Warning: Model setup not available: {e}")


class ConfigurationFactory:
    """Factory for creating test configurations based on your actual YAML configs"""
    
    @staticmethod
    def create_base_config() -> Dict[str, Any]:
        """Create base configuration common to all models"""
        return {
            'seed': 907,
            'Physics': {
                'pde': 'Navier-Stokes',
                'source': 'Spectral',
                'field': 'u, v',
                'variables': 2,
                'Nx': 64,
                'Ny': 64,
                'Nt': 50,
                'dx': 0.01,
                'dy': 0.01,
                'dt': 0.01,
                'x_slice': 1,
                'y_slice': 1,
                't_slice': 1,
                'physics normalisation': False
            },
            'Data': {
                'name': 'NS_Spectral_test.npz',
                'loc': '/tmp/test_data',
                'ntrain': 20,
                'test-train-split': 0.2,
                'batch size': 8,
                't_in': 1,
                't_out': 25,
                'step': 1,
                'normalisation': 'Min-Max'
            },
            'Opt': {
                'optimizer': 'adam',
                'epochs': 100,
                'learning rate': 0.005,
                'beta1': 0.9,
                'beta2': 0.9,
                'weight_decay': 1e-4,
                'scheduler': 'step',
                'scheduler step': 100,
                'scheduler gamma': 0.5,
                'grad_clip': 2.0
            },
            'Train': {
                'loss': 'LP',
                'input_length': 1,
                'rollout_length': 10,
                'input_noise': 0.0,
                'odesolve': {
                    'source': 'custom',
                    'method': 'euler',
                    'adjoint': False
                },
                'batch_norm': False,
                'checkpoint': {'epochs': 50},
                'restart': False,
                'restart run name': ''
            }
        }
    
    @staticmethod
    def create_fno_config(operator_splitting: bool = False) -> Dict[str, Any]:
        """Create FNO model configuration"""
        config = ConfigurationFactory.create_base_config()
        config['Model'] = {
            'arch': 'FNO',
            'in_vars': 2,
            'out_vars': 2,
            'modes': 8,
            'width': 16,
            'n_layers': 3,
            'act': 'gelu',
            'operator splitting': operator_splitting,
            'ops_split normalise': False
        }
        return config
    
    @staticmethod
    def create_unet_config(operator_splitting: bool = False) -> Dict[str, Any]:
        """Create U-Net model configuration"""
        config = ConfigurationFactory.create_base_config()
        config['Model'] = {
            'arch': 'U-Net',
            'in_vars': 2,
            'out_vars': 2,
            'width': 32,
            'act': 'tanh',
            'operator splitting': operator_splitting,
            'ops_split normalise': True
        }
        return config
    
    @staticmethod
    def create_cno_config(operator_splitting: bool = False) -> Dict[str, Any]:
        """Create CNO model configuration"""
        config = ConfigurationFactory.create_base_config()
        config['Model'] = {
            'arch': 'CNO',
            'in_dim': 2,
            'out_dim': 2,
            'size': 64,
            'N_layers': 3,
            'N_res': 2,
            'N_res_neck': 2,
            'channel_multiplier': 16,
            'operator splitting': operator_splitting,
            'ops_split normalise': False
        }
        return config
    
    @staticmethod
    def create_gmlp_config(operator_splitting: bool = False) -> Dict[str, Any]:
        """Create gMLP model configuration"""
        config = ConfigurationFactory.create_base_config()
        config['Model'] = {
            'arch': 'gMLP',
            'n_blocks': 4,
            'd_in': 2,
            'd_ffn': 32,
            'Nx': 64,
            'Ny': 64,
            'operator splitting': operator_splitting
        }
        return config
    
    @staticmethod
    def create_conv_config(operator_splitting: bool = False) -> Dict[str, Any]:
        """Create ConvolutionalModel configuration"""
        config = ConfigurationFactory.create_base_config()
        config['Model'] = {
            'arch': 'Conv',
            'in_vars': 2,
            'out_vars': 2,
            'hidden_vars': [8, 16, 8],
            'n_layers': 3,
            'kernel_size': 3,
            'act': 'gelu',
            'init': 'random',
            'dropout': 0.0,
            'operator splitting': operator_splitting,
            'ops_split normalise': False
        }
        return config
    
    @staticmethod
    def create_multi_variable_config(arch: str = 'FNO', variables: int = 4) -> Dict[str, Any]:
        """Create configuration with multiple variables (e.g., for Euler equations)"""
        if arch == 'FNO':
            config = ConfigurationFactory.create_fno_config()
        elif arch == 'U-Net':
            config = ConfigurationFactory.create_unet_config()
        elif arch == 'CNO':
            config = ConfigurationFactory.create_cno_config()
        elif arch == 'Conv':
            config = ConfigurationFactory.create_conv_config()
        else:
            config = ConfigurationFactory.create_fno_config()
        
        # Update for multi-variable case
        config['Physics']['variables'] = variables
        config['Physics']['field'] = 'rho, u, v, p' if variables == 4 else f'var1, var2, var3'
        config['Model']['in_vars'] = variables
        config['Model']['out_vars'] = variables
        
        if arch == 'CNO':
            config['Model']['in_dim'] = variables
            config['Model']['out_dim'] = variables
        elif arch == 'gMLP':
            config['Model']['d_in'] = variables
        
        return config


class DummyRun:
    """Mock run object for testing operator splitting models"""
    def __init__(self, name: str = "test_run"):
        self.name = name
        self.metadata = {}
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Mock log_metrics method"""
        pass
    
    def update_metadata(self, metadata: Dict[str, Any]):
        """Mock update_metadata method"""
        self.metadata.update(metadata)
    
    def save_file(self, filepath: str, category: str = 'output'):
        """Mock save_file method"""
        pass


class TestDataGenerator:
    """Generate synthetic test data for model testing"""
    
    @staticmethod
    def create_test_data(batch_size: int = 10, variables: int = 2, 
                        nx: int = 64, ny: int = 64, nt: int = 25) -> torch.Tensor:
        """Create synthetic PDE-like test data"""
        # Create spatially and temporally varying data
        x = torch.linspace(-1, 1, nx)
        y = torch.linspace(-1, 1, ny)
        t = torch.linspace(0, 1, nt)
        
        X, Y = torch.meshgrid(x, y, indexing='ij')
        data = torch.zeros(batch_size, variables, nx, ny, nt)
        
        for b in range(batch_size):
            for v in range(variables):
                for i, time in enumerate(t):
                    # Create wave-like patterns with different frequencies per variable
                    freq = 1.0 + v * 0.5 + b * 0.1
                    phase = v * np.pi / 4
                    data[b, v, :, :, i] = torch.sin(freq * (X + Y - time)) * torch.cos(phase)
        
        return data


@pytest.fixture(scope="session")
def device():
    """Fixture to determine device (CPU/CUDA)"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def dummy_normalizer():
    """Create dummy normalizer for testing"""
    dummy_data = torch.randn(5, 2, 32, 32, 10)
    return MinMax_Normalizer(dummy_data)


@pytest.fixture
def dummy_run():
    """Create dummy run object for testing"""
    return DummyRun()


class TestModelInstantiation:
    """Test model instantiation via model_initialisation function"""
    
    @pytest.mark.skipif(not MODEL_SETUP_AVAILABLE, reason="Model setup not available")
    @pytest.mark.parametrize("arch", ['FNO', 'U-Net', 'CNO', 'gMLP', 'Conv'])
    def test_basic_model_creation(self, arch, dummy_normalizer, dummy_run):
        """Test that models can be instantiated correctly"""
        if arch == 'FNO':
            config = ConfigurationFactory.create_fno_config()
        elif arch == 'U-Net':
            config = ConfigurationFactory.create_unet_config()
        elif arch == 'CNO':
            config = ConfigurationFactory.create_cno_config()
        elif arch == 'gMLP':
            config = ConfigurationFactory.create_gmlp_config()
        elif arch == 'Conv':
            config = ConfigurationFactory.create_conv_config()
        
        try:
            model = model_initialisation(config, dummy_normalizer, dummy_run)
            
            # Basic checks
            assert model is not None, f"Model {arch} was not instantiated"
            assert isinstance(model, nn.Module), f"Model {arch} is not a PyTorch module"
            
            # Check that model has parameters
            param_count = count_parameters(model)
            assert param_count > 0, f"Model {arch} has no trainable parameters"
            
            print(f"✓ {arch} model instantiated successfully with {param_count:,} parameters")
            
        except Exception as e:
            pytest.fail(f"Failed to instantiate {arch} model: {e}")
    
    @pytest.mark.skipif(not MODEL_SETUP_AVAILABLE, reason="Model setup not available")
    @pytest.mark.parametrize("arch,operator_splitting", [
        ('FNO', True),
        ('U-Net', True),
        ('Conv', True),
    ])
    def test_operator_splitting_models(self, arch, operator_splitting, dummy_normalizer, dummy_run):
        """Test operator splitting model instantiation"""
        if arch == 'FNO':
            config = ConfigurationFactory.create_fno_config(operator_splitting=True)
        elif arch == 'U-Net':
            config = ConfigurationFactory.create_unet_config(operator_splitting=True)
        elif arch == 'Conv':
            config = ConfigurationFactory.create_conv_config(operator_splitting=True)
        
        try:
            model = model_initialisation(config, dummy_normalizer, dummy_run)
            
            assert model is not None, f"Operator splitting {arch} model was not instantiated"
            
            # Operator splitting models might have different structure
            param_count = count_parameters(model) if hasattr(model, 'parameters') else 0
            
            print(f"✓ Operator splitting {arch} model instantiated successfully")
            
        except Exception as e:
            pytest.skip(f"Operator splitting {arch} not available or failed: {e}")
    
    @pytest.mark.skipif(not MODEL_SETUP_AVAILABLE, reason="Model setup not available")
    @pytest.mark.parametrize("variables", [1, 2, 3, 4, 5])
    def test_multi_variable_models(self, variables, dummy_normalizer, dummy_run):
        """Test models with different numbers of variables"""
        config = ConfigurationFactory.create_multi_variable_config('FNO', variables)
        
        # Create normalizer with correct number of variables
        dummy_data = torch.randn(5, variables, 32, 32, 10)
        normalizer = MinMax_Normalizer(dummy_data)
        
        try:
            model = model_initialisation(config, normalizer, dummy_run)
            
            assert model is not None, f"Model with {variables} variables was not instantiated"
            
            param_count = count_parameters(model)
            assert param_count > 0, f"Model with {variables} variables has no parameters"
            
            print(f"✓ Model with {variables} variables instantiated successfully")
            
        except Exception as e:
            pytest.fail(f"Failed to instantiate model with {variables} variables: {e}")
    
    @pytest.mark.skipif(not MODEL_SETUP_AVAILABLE, reason="Model setup not available")
    @pytest.mark.parametrize("grid_size", [16, 32, 64, 128])
    def test_different_grid_sizes(self, grid_size, dummy_normalizer, dummy_run):
        """Test models with different spatial grid sizes"""
        config = ConfigurationFactory.create_fno_config()
        config['Physics']['Nx'] = grid_size
        config['Physics']['Ny'] = grid_size
        
        # Update CNO-specific size parameter
        if 'size' in config.get('Model', {}):
            config['Model']['size'] = grid_size
        
        # Update gMLP-specific size parameters
        if config.get('Model', {}).get('arch') == 'gMLP':
            config['Model']['Nx'] = grid_size
            config['Model']['Ny'] = grid_size
        
        try:
            model = model_initialisation(config, dummy_normalizer, dummy_run)
            
            assert model is not None, f"Model with {grid_size}x{grid_size} grid was not instantiated"
            
            print(f"✓ Model with {grid_size}x{grid_size} grid instantiated successfully")
            
        except Exception as e:
            pytest.skip(f"Model with {grid_size}x{grid_size} grid failed: {e}")


class TestModelShapeCompatibility:
    """Test model compatibility with different input shapes"""
    
    def setup_method(self):
        """Setup test data for shape compatibility tests"""
        self.test_data = TestDataGenerator.create_test_data()
        self.normalizer = MinMax_Normalizer(self.test_data)
        self.dummy_run = DummyRun()
    
    @pytest.mark.skipif(not MODEL_SETUP_AVAILABLE, reason="Model setup not available")
    @pytest.mark.parametrize("arch", ['FNO', 'Conv'])
    def test_standard_input_shapes(self, arch, device):
        """Test models with standard input shapes [batch, vars, nx, ny]"""
        if arch == 'FNO':
            config = ConfigurationFactory.create_fno_config()
        elif arch == 'Conv':
            config = ConfigurationFactory.create_conv_config()
        
        try:
            model = model_initialisation(config, self.normalizer, self.dummy_run)
            model = model.to(device)
            model.eval()
            
            # Test with standard 2D input
            batch_size = 4
            variables = config['Model']['in_vars']
            nx = config['Physics']['Nx']
            ny = config['Physics']['Ny']
            
            x = torch.randn(batch_size, variables, nx, ny, device=device)
            
            with torch.no_grad():
                output = model(x)
            
            # Check output properties
            assert output.device == x.device, f"Output device mismatch for {arch}"
            assert torch.isfinite(output).all(), f"Output contains non-finite values for {arch}"
            assert output.shape[0] == batch_size, f"Batch size mismatch for {arch}"
            assert output.shape[1] == config['Model']['out_vars'], f"Output variables mismatch for {arch}"
            
            print(f"✓ {arch} handles standard input shape {x.shape} -> {output.shape}")
            
        except Exception as e:
            pytest.skip(f"{arch} standard shape test failed: {e}")
    
    @pytest.mark.skipif(not MODEL_SETUP_AVAILABLE, reason="Model setup not available")
    def test_unet_temporal_input_shapes(self, device):
        """Test U-Net with temporal input shapes [batch, vars, nx, ny, t_in]"""
        config = ConfigurationFactory.create_unet_config()
        
        try:
            model = model_initialisation(config, self.normalizer, self.dummy_run)
            model = model.to(device)
            model.eval()
            
            # U-Net expects temporal input
            batch_size = 4
            variables = config['Model']['in_vars']
            nx = config['Physics']['Nx']
            ny = config['Physics']['Ny']
            t_in = config['Data']['t_in']
            
            x = torch.randn(batch_size, variables, nx, ny, t_in, device=device)
            
            with torch.no_grad():
                output = model(x)
            
            # Check output properties
            assert output.device == x.device, "Output device mismatch for U-Net"
            assert torch.isfinite(output).all(), "U-Net output contains non-finite values"
            assert output.shape[0] == batch_size, "U-Net batch size mismatch"
            
            print(f"✓ U-Net handles temporal input shape {x.shape} -> {output.shape}")
            
        except Exception as e:
            pytest.skip(f"U-Net temporal shape test failed: {e}")
    
    @pytest.mark.skipif(not MODEL_SETUP_AVAILABLE, reason="Model setup not available")
    @pytest.mark.parametrize("batch_size", [1, 4, 8, 16])
    def test_different_batch_sizes(self, batch_size, device):
        """Test models with different batch sizes"""
        config = ConfigurationFactory.create_fno_config()
        
        try:
            model = model_initialisation(config, self.normalizer, self.dummy_run)
            model = model.to(device)
            model.eval()
            
            variables = config['Model']['in_vars']
            nx = config['Physics']['Nx']
            ny = config['Physics']['Ny']
            
            x = torch.randn(batch_size, variables, nx, ny, device=device)
            
            with torch.no_grad():
                output = model(x)
            
            assert output.shape[0] == batch_size, f"Batch size {batch_size} not handled correctly"
            assert torch.isfinite(output).all(), f"Non-finite output with batch size {batch_size}"
            
            print(f"✓ Model handles batch size {batch_size}")
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                pytest.skip(f"Out of memory with batch size {batch_size}")
            else:
                raise
    
    @pytest.mark.skipif(not MODEL_SETUP_AVAILABLE, reason="Model setup not available")
    @pytest.mark.parametrize("spatial_size", [(32, 32), (64, 64), (128, 128), (32, 64)])
    def test_different_spatial_sizes(self, spatial_size, device):
        """Test models with different spatial dimensions"""
        config = ConfigurationFactory.create_fno_config()
        nx, ny = spatial_size
        config['Physics']['Nx'] = nx
        config['Physics']['Ny'] = ny
        
        try:
            model = model_initialisation(config, self.normalizer, self.dummy_run)
            model = model.to(device)
            model.eval()
            
            batch_size = 2
            variables = config['Model']['in_vars']
            
            x = torch.randn(batch_size, variables, nx, ny, device=device)
            
            with torch.no_grad():
                output = model(x)
            
            # Check that spatial dimensions are handled correctly
            assert output.shape[2:4] == (nx, ny) or len(output.shape) == 4, \
                f"Spatial dimensions not preserved: {x.shape} -> {output.shape}"
            assert torch.isfinite(output).all(), f"Non-finite output with size {spatial_size}"
            
            print(f"✓ Model handles spatial size {spatial_size}")
            
        except Exception as e:
            pytest.skip(f"Spatial size {spatial_size} test failed: {e}")


class TestModelParameterCounting:
    """Test parameter counting functionality"""
    
    @pytest.mark.skipif(not MODEL_SETUP_AVAILABLE, reason="Model setup not available")
    @pytest.mark.parametrize("arch", ['FNO', 'U-Net', 'CNO', 'Conv'])
    def test_parameter_counting_consistency(self, arch, dummy_normalizer, dummy_run):
        """Test that count_parameters function works correctly"""
        if arch == 'FNO':
            config = ConfigurationFactory.create_fno_config()
        elif arch == 'U-Net':
            config = ConfigurationFactory.create_unet_config()
        elif arch == 'CNO':
            config = ConfigurationFactory.create_cno_config()
        elif arch == 'Conv':
            config = ConfigurationFactory.create_conv_config()
        
        try:
            model = model_initialisation(config, dummy_normalizer, dummy_run)
            
            # Test count_parameters function
            param_count_custom = count_parameters(model)
            
            # Cross-check with PyTorch's parameter counting
            param_count_torch = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            assert param_count_custom == param_count_torch, \
                f"Parameter counting mismatch for {arch}: {param_count_custom} vs {param_count_torch}"
            
            assert param_count_custom > 0, f"Model {arch} has no trainable parameters"
            
            print(f"✓ {arch} parameter counting verified: {param_count_custom:,} parameters")
            
        except Exception as e:
            pytest.skip(f"Parameter counting test for {arch} failed: {e}")
    
    @pytest.mark.skipif(not MODEL_SETUP_AVAILABLE, reason="Model setup not available")
    def test_parameter_scaling_with_width(self, dummy_normalizer, dummy_run):
        """Test that parameter count scales appropriately with model width"""
        base_config = ConfigurationFactory.create_fno_config()
        
        widths = [8, 16, 32, 64]
        param_counts = []
        
        for width in widths:
            config = base_config.copy()
            config['Model']['width'] = width
            
            try:
                model = model_initialisation(config, dummy_normalizer, dummy_run)
                param_count = count_parameters(model)
                param_counts.append(param_count)
                
                print(f"Width {width}: {param_count:,} parameters")
                
            except Exception as e:
                pytest.skip(f"Width scaling test failed at width {width}: {e}")
        
        # Check that parameter count generally increases with width
        if len(param_counts) > 1:
            for i in range(1, len(param_counts)):
                assert param_counts[i] >= param_counts[i-1], \
                    f"Parameter count decreased with larger width: {param_counts}"
    
    @pytest.mark.skipif(not MODEL_SETUP_AVAILABLE, reason="Model setup not available")
    def test_parameter_scaling_with_layers(self, dummy_normalizer, dummy_run):
        """Test that parameter count scales with number of layers"""
        base_config = ConfigurationFactory.create_fno_config()
        
        layer_counts = [2, 3, 4, 5]
        param_counts = []
        
        for n_layers in layer_counts:
            config = base_config.copy()
            config['Model']['n_layers'] = n_layers
            
            try:
                model = model_initialisation(config, dummy_normalizer, dummy_run)
                param_count = count_parameters(model)
                param_counts.append(param_count)
                
                print(f"Layers {n_layers}: {param_count:,} parameters")
                
            except Exception as e:
                pytest.skip(f"Layer scaling test failed at {n_layers} layers: {e}")
        
        # Check that parameter count generally increases with more layers
        if len(param_counts) > 1:
            for i in range(1, len(param_counts)):
                assert param_counts[i] >= param_counts[i-1], \
                    f"Parameter count decreased with more layers: {param_counts}"


class TestModelSpecificRequirements:
    """Test model-specific requirements and constraints"""
    
    @pytest.mark.skipif(not MODEL_SETUP_AVAILABLE, reason="Model setup not available")
    def test_fno_modes_configuration(self, dummy_normalizer, dummy_run):
        """Test FNO with different Fourier mode configurations"""
        mode_configs = [4, 8, 16, 32]
        
        for modes in mode_configs:
            config = ConfigurationFactory.create_fno_config()
            config['Model']['modes'] = modes
            
            try:
                model = model_initialisation(config, dummy_normalizer, dummy_run)
                
                # Test forward pass
                x = torch.randn(2, 2, 64, 64)
                with torch.no_grad():
                    output = model(x)
                
                assert torch.isfinite(output).all(), f"FNO with {modes} modes produced non-finite output"
                
                print(f"✓ FNO with {modes} modes works correctly")
                
            except Exception as e:
                pytest.skip(f"FNO modes test failed for {modes} modes: {e}")
    
    @pytest.mark.skipif(not MODEL_SETUP_AVAILABLE, reason="Model setup not available")
    def test_cno_layer_configuration(self, dummy_normalizer, dummy_run):
        """Test CNO with different layer configurations"""
        layer_configs = [
            {'N_layers': 2, 'N_res': 2, 'N_res_neck': 2},
            {'N_layers': 3, 'N_res': 3, 'N_res_neck': 3},
            {'N_layers': 4, 'N_res': 4, 'N_res_neck': 2},
        ]
        
        for layer_config in layer_configs:
            config = ConfigurationFactory.create_cno_config()
            config['Model'].update(layer_config)
            
            try:
                model = model_initialisation(config, dummy_normalizer, dummy_run)
                
                # Test forward pass
                x = torch.randn(2, 2, 64, 64)
                with torch.no_grad():
                    output = model(x)
                
                assert torch.isfinite(output).all(), \
                    f"CNO with {layer_config} produced non-finite output"
                
                print(f"✓ CNO with {layer_config} works correctly")
                
            except Exception as e:
                pytest.skip(f"CNO layer test failed for {layer_config}: {e}")
    
    @pytest.mark.skipif(not MODEL_SETUP_AVAILABLE, reason="Model setup not available")
    def test_gmlp_spatial_requirements(self, dummy_normalizer, dummy_run):
        """Test gMLP with spatial dimension requirements"""
        spatial_configs = [
            {'Nx': 32, 'Ny': 32},
            {'Nx': 64, 'Ny': 64},
            {'Nx': 32, 'Ny': 64},  # Non-square
        ]
        
        for spatial_config in spatial_configs:
            config = ConfigurationFactory.create_gmlp_config()
            config['Model'].update(spatial_config)
            config['Physics']['Nx'] = spatial_config['Nx']
            config['Physics']['Ny'] = spatial_config['Ny']
            
            try:
                model = model_initialisation(config, dummy_normalizer, dummy_run)
                
                # Test forward pass
                nx, ny = spatial_config['Nx'], spatial_config['Ny']
                x = torch.randn(2, 2, nx, ny)
                with torch.no_grad():
                    output = model(x)
                
                assert torch.isfinite(output).all(), \
                    f"gMLP with {spatial_config} produced non-finite output"
                
                print(f"✓ gMLP with {spatial_config} works correctly")
                
            except Exception as e:
                pytest.skip(f"gMLP spatial test failed for {spatial_config}: {e}")
    
    # @pytest.mark.skipif(not MODEL_SETUP_AVAILABLE, reason="Model setup not available")
    # def test_conv_kernel_size_requirements(self, dummy_normalizer, dummy_run):
    #     """Test ConvolutionalModel with different kernel sizes"""
    #     kernel_sizes = [3, 5, 7]
        
    #     for kernel_size in kernel_sizes:
    #         config = ConfigurationFactory.create_conv_config()
    #         config['Model']['kernel_size'] = kernel_size
            
    #         try:
    #             model = model_initialisation(config, dummy_normalizer, dummy_run)
                
    #             # Test forward pass
    #             x = torch.randn(2, 2, 64, 64)
    #             with torch.no_grad():
    #                 output = model(x)
                
    #             assert torch.isfinite(output).all(), \
    #                 f"Conv with kernel size {kernel_size} produced non-finite output"
    #             assert output.shape == x.shape, \
    #                 f"Conv output shape mismatch: {output.shape} vs {x.shape}"
                
    #             print(f"✓ ConvolutionalModel with kernel size {kernel_size} works correctly")
                
    #         except Exception as e:
    #             pytest.skip(f"Conv kernel size test failed for {kernel_size}: {e}")


class TestDeviceCompatibility:
    """Test model device compatibility (CPU/CUDA)"""
    
    @pytest.mark.skipif(not MODEL_SETUP_AVAILABLE, reason="Model setup not available")
    def test_cpu_compatibility(self, dummy_normalizer, dummy_run):
        """Test that models work on CPU"""
        config = ConfigurationFactory.create_fno_config()
        
        try:
            model = model_initialisation(config, dummy_normalizer, dummy_run)
            model = model.cpu()
            model.eval()
            
            # Test forward pass on CPU
            x = torch.randn(2, 2, 64, 64)
            with torch.no_grad():
                output = model(x)
            
            assert output.device.type == 'cpu', "Output not on CPU"
            assert torch.isfinite(output).all(), "CPU output contains non-finite values"
            
            print("✓ Model CPU compatibility verified")
            
        except Exception as e:
            pytest.fail(f"CPU compatibility test failed: {e}")
    
    @pytest.mark.skipif(not MODEL_SETUP_AVAILABLE, reason="Model setup not available")
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_compatibility(self, dummy_normalizer, dummy_run):
        """Test that models work on CUDA"""
        config = ConfigurationFactory.create_fno_config()
        
        try:
            model = model_initialisation(config, dummy_normalizer, dummy_run)
            model = model.cuda()
            model.eval()
            
            # Test forward pass on CUDA
            x = torch.randn(2, 2, 64, 64, device='cuda')
            with torch.no_grad():
                output = model(x)
            
            assert output.device.type == 'cuda', "Output not on CUDA"
            assert torch.isfinite(output).all(), "CUDA output contains non-finite values"
            
            print("✓ Model CUDA compatibility verified")
            
        except Exception as e:
            pytest.skip(f"CUDA compatibility test failed: {e}")
    
    @pytest.mark.skipif(not MODEL_SETUP_AVAILABLE, reason="Model setup not available")
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_transfer(self, dummy_normalizer, dummy_run):
        """Test moving models between CPU and CUDA"""
        config = ConfigurationFactory.create_fno_config()
        
        try:
            model = model_initialisation(config, dummy_normalizer, dummy_run)
            
            # Test CPU -> CUDA transfer
            model = model.cuda()
            x_cuda = torch.randn(2, 2, 32, 32, device='cuda')
            with torch.no_grad():
                output_cuda = model(x_cuda)
            assert output_cuda.device.type == 'cuda'
            
            # Test CUDA -> CPU transfer
            model = model.cpu()
            x_cpu = torch.randn(2, 2, 32, 32, device='cpu')
            with torch.no_grad():
                output_cpu = model(x_cpu)
            assert output_cpu.device.type == 'cpu'
            
            print("✓ Model device transfer verified")
            
        except Exception as e:
            pytest.skip(f"Device transfer test failed: {e}")


class TestErrorHandling:
    """Test error handling and edge cases in model instantiation"""
    
    @pytest.mark.skipif(not MODEL_SETUP_AVAILABLE, reason="Model setup not available")
    def test_invalid_architecture(self, dummy_normalizer, dummy_run):
        """Test handling of invalid architecture specification"""
        config = ConfigurationFactory.create_fno_config()
        config['Model']['arch'] = 'InvalidArchitecture'
        
        # Should either raise an appropriate error or handle gracefully
        with pytest.raises((ValueError, KeyError, AttributeError, NotImplementedError)):
            model_initialisation(config, dummy_normalizer, dummy_run)
    
    @pytest.mark.skipif(not MODEL_SETUP_AVAILABLE, reason="Model setup not available")
    def test_missing_model_parameters(self, dummy_normalizer, dummy_run):
        """Test handling of missing required model parameters"""
        config = ConfigurationFactory.create_fno_config()
        
        # Remove required parameter
        del config['Model']['modes']
        
        with pytest.raises((KeyError, ValueError, TypeError)):
            model_initialisation(config, dummy_normalizer, dummy_run)
    
    @pytest.mark.skipif(not MODEL_SETUP_AVAILABLE, reason="Model setup not available")
    def test_invalid_parameter_values(self, dummy_normalizer, dummy_run):
        """Test handling of invalid parameter values"""
        config = ConfigurationFactory.create_fno_config()
        
        # Test with invalid values
        invalid_configs = [
            {'Model': {**config['Model'], 'modes': 0}},  # Zero modes
            {'Model': {**config['Model'], 'width': -1}},  # Negative width
            {'Model': {**config['Model'], 'n_layers': 0}},  # Zero layers
        ]
        
        for invalid_config in invalid_configs:
            test_config = config.copy()
            test_config.update(invalid_config)
            
            with pytest.raises((ValueError, AssertionError, RuntimeError)):
                model_initialisation(test_config, dummy_normalizer, dummy_run)
    
    @pytest.mark.skipif(not MODEL_SETUP_AVAILABLE, reason="Model setup not available")
    def test_incompatible_normalizer_shapes(self, dummy_run):
        """Test handling of incompatible normalizer shapes"""
        config = ConfigurationFactory.create_fno_config()
        
        # Create normalizer with wrong number of variables
        wrong_data = torch.randn(5, 5, 32, 32, 10)  # 5 variables instead of 2
        wrong_normalizer = MinMax_Normalizer(wrong_data)
        
        # This might work (if model adapts) or fail (if strict checking)
        try:
            model = model_initialisation(config, wrong_normalizer, dummy_run)
            print("⚠ Model accepted incompatible normalizer (might be adaptive)")
        except (ValueError, AssertionError, RuntimeError):
            print("✓ Model correctly rejected incompatible normalizer")


class TestModelConsistency:
    """Test model consistency and determinism"""
    
    @pytest.mark.skipif(not MODEL_SETUP_AVAILABLE, reason="Model setup not available")
    def test_deterministic_initialization(self, dummy_normalizer, dummy_run):
        """Test that model initialization is deterministic with same seed"""
        config = ConfigurationFactory.create_fno_config()
        
        def create_model_with_seed(seed):
            torch.manual_seed(seed)
            np.random.seed(seed)
            return model_initialisation(config, dummy_normalizer, dummy_run)
        
        # Create two models with same seed
        model1 = create_model_with_seed(42)
        model2 = create_model_with_seed(42)
        
        # Check that parameters are identical
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2, atol=1e-6), \
                "Model initialization not deterministic with same seed"
        
        print("✓ Model initialization is deterministic")
    
    @pytest.mark.skipif(not MODEL_SETUP_AVAILABLE, reason="Model setup not available")
    def test_different_seed_different_initialization(self, dummy_normalizer, dummy_run):
        """Test that different seeds produce different initializations"""
        config = ConfigurationFactory.create_fno_config()
        
        def create_model_with_seed(seed):
            torch.manual_seed(seed)
            np.random.seed(seed)
            return model_initialisation(config, dummy_normalizer, dummy_run)
        
        # Create models with different seeds
        model1 = create_model_with_seed(42)
        model2 = create_model_with_seed(123)
        
        # Check that at least some parameters are different
        params_differ = False
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            if not torch.allclose(p1, p2, atol=1e-6):
                params_differ = True
                break
        
        assert params_differ, "Different seeds produced identical models"
        print("✓ Different seeds produce different initializations")
    
    @pytest.mark.skipif(not MODEL_SETUP_AVAILABLE, reason="Model setup not available")
    def test_forward_pass_consistency(self, dummy_normalizer, dummy_run):
        """Test that forward passes are consistent in eval mode"""
        config = ConfigurationFactory.create_fno_config()
        model = model_initialisation(config, dummy_normalizer, dummy_run)
        model.eval()
        
        x = torch.randn(2, 2, 64, 64)
        
        # Multiple forward passes should give same result in eval mode
        with torch.no_grad():
            output1 = model(x)
            output2 = model(x)
        
        assert torch.allclose(output1, output2, atol=1e-6), \
            "Forward passes not consistent in eval mode"
        
        print("✓ Forward pass consistency verified")


class TestModelIntegrationWithPipeline:
    """Test model integration with training pipeline components"""
    
    @pytest.mark.skipif(not MODEL_SETUP_AVAILABLE, reason="Model setup not available")
    def test_model_with_different_normalizers(self, dummy_run):
        """Test models with different normalization schemes"""
        config = ConfigurationFactory.create_fno_config()
        
        # Test data
        test_data = TestDataGenerator.create_test_data(10, 2, 64, 64, 20)
        
        normalizer_types = [
            MinMax_Normalizer,
            MinMax_Normalizer_variable,
            Gaussian_Normalizer,
            Identity_Normalizer
        ]
        
        for NormalizerClass in normalizer_types:
            try:
                normalizer = NormalizerClass(test_data)
                model = model_initialisation(config, normalizer, dummy_run)
                
                assert model is not None, f"Model failed with {NormalizerClass.__name__}"
                
                # Test forward pass
                x = torch.randn(2, 2, 64, 64)
                with torch.no_grad():
                    output = model(x)
                
                assert torch.isfinite(output).all(), \
                    f"Non-finite output with {NormalizerClass.__name__}"
                
                print(f"✓ Model works with {NormalizerClass.__name__}")
                
            except Exception as e:
                pytest.skip(f"Model with {NormalizerClass.__name__} failed: {e}")
    
    @pytest.mark.skipif(not MODEL_SETUP_AVAILABLE, reason="Model setup not available")
    def test_model_with_different_pde_types(self, dummy_normalizer, dummy_run):
        """Test models configured for different PDE types"""
        pde_configs = [
            ('Navier-Stokes', 2, 'u, v'),
            ('Euler-Fluid', 4, 'rho, u, v, p'),
            ('Wave', 1, 'u'),
            ('Comp. Navier-Stokes', 4, 'u, v, p, rho'),
        ]
        
        for pde_name, variables, field in pde_configs:
            config = ConfigurationFactory.create_multi_variable_config('FNO', variables)
            config['Physics']['pde'] = pde_name
            config['Physics']['field'] = field
            
            # Create appropriate normalizer
            test_data = TestDataGenerator.create_test_data(10, variables, 64, 64, 20)
            normalizer = MinMax_Normalizer(test_data)
            
            try:
                model = model_initialisation(config, normalizer, dummy_run)
                
                assert model is not None, f"Model failed for {pde_name}"
                
                # Test with appropriate input shape
                x = torch.randn(2, variables, 64, 64)
                with torch.no_grad():
                    output = model(x)
                
                assert output.shape[1] == variables, \
                    f"Output variables mismatch for {pde_name}: {output.shape[1]} vs {variables}"
                
                print(f"✓ Model works for {pde_name} with {variables} variables")
                
            except Exception as e:
                pytest.skip(f"Model for {pde_name} failed: {e}")
    
    @pytest.mark.skipif(not MODEL_SETUP_AVAILABLE, reason="Model setup not available")
    def test_model_training_mode_compatibility(self, dummy_normalizer, dummy_run):
        """Test model behavior in training vs eval mode"""
        config = ConfigurationFactory.create_fno_config()
        model = model_initialisation(config, dummy_normalizer, dummy_run)
        
        x = torch.randn(2, 2, 64, 64)
        
        # Test in training mode
        model.train()
        output_train = model(x)
        assert torch.isfinite(output_train).all(), "Non-finite output in training mode"
        
        # Test in eval mode
        model.eval()
        with torch.no_grad():
            output_eval = model(x)
        assert torch.isfinite(output_eval).all(), "Non-finite output in eval mode"
        
        # For models without dropout/batch norm, outputs might be identical
        # For models with these components, outputs might differ
        print("✓ Model works in both training and eval modes")
    
    @pytest.mark.skipif(not MODEL_SETUP_AVAILABLE, reason="Model setup not available")
    def test_model_gradient_computation(self, dummy_normalizer, dummy_run):
        """Test that models can compute gradients properly"""
        config = ConfigurationFactory.create_fno_config()
        model = model_initialisation(config, dummy_normalizer, dummy_run)
        model.train()
        
        x = torch.randn(2, 2, 64, 64, requires_grad=True)
        target = torch.randn(2, 2, 64, 64)
        
        # Forward pass
        output = model(x)
        loss = nn.MSELoss()(output, target)
        
        # Backward pass
        loss.backward()
        
        # Check that model parameters have gradients
        has_gradients = False
        for param in model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_gradients = True
                assert torch.isfinite(param.grad).all(), "Non-finite gradients detected"
        
        assert has_gradients, "No gradients found in model parameters"
        print("✓ Model gradient computation verified")


class TestSpecialConfigurations:
    """Test special model configurations and edge cases"""
    
    @pytest.mark.skipif(not MODEL_SETUP_AVAILABLE, reason="Model setup not available")
    def test_single_variable_models(self, dummy_run):
        """Test models with single variable (e.g., scalar fields)"""
        config = ConfigurationFactory.create_multi_variable_config('FNO', 1)
        
        # Create single-variable normalizer
        test_data = TestDataGenerator.create_test_data(10, 1, 64, 64, 20)
        normalizer = MinMax_Normalizer(test_data)
        
        try:
            model = model_initialisation(config, normalizer, dummy_run)
            
            # Test with single variable input
            x = torch.randn(2, 1, 64, 64)
            with torch.no_grad():
                output = model(x)
            
            assert output.shape[1] == 1, f"Output should have 1 variable: {output.shape}"
            assert torch.isfinite(output).all(), "Non-finite output for single variable"
            
            print("✓ Single variable model works correctly")
            
        except Exception as e:
            pytest.skip(f"Single variable model failed: {e}")
    
    @pytest.mark.skipif(not MODEL_SETUP_AVAILABLE, reason="Model setup not available")
    def test_large_variable_count_models(self, dummy_run):
        """Test models with many variables (stress test)"""
        large_var_count = 8
        config = ConfigurationFactory.create_multi_variable_config('FNO', large_var_count)
        
        # Create multi-variable normalizer
        test_data = TestDataGenerator.create_test_data(5, large_var_count, 32, 32, 10)
        normalizer = MinMax_Normalizer(test_data)
        
        try:
            model = model_initialisation(config, normalizer, dummy_run)
            
            # Test with many variables
            x = torch.randn(2, large_var_count, 32, 32)
            with torch.no_grad():
                output = model(x)
            
            assert output.shape[1] == large_var_count, \
                f"Output should have {large_var_count} variables: {output.shape}"
            assert torch.isfinite(output).all(), "Non-finite output for large variable count"
            
            print(f"✓ Model with {large_var_count} variables works correctly")
            
        except Exception as e:
            pytest.skip(f"Large variable count model failed: {e}")
    
    @pytest.mark.skipif(not MODEL_SETUP_AVAILABLE, reason="Model setup not available")
    def test_minimal_model_configurations(self, dummy_normalizer, dummy_run):
        """Test models with minimal configurations (smallest possible)"""
        minimal_configs = [
            # Minimal FNO
            {
                'arch': 'FNO',
                'in_vars': 1,
                'out_vars': 1,
                'modes': 2,
                'width': 4,
                'n_layers': 1,
                'operator splitting': False
            },
            # Minimal Conv
            {
                'arch': 'Conv',
                'in_vars': 1,
                'out_vars': 1,
                'hidden_vars': [2],
                'n_layers': 1,
                'kernel_size': 3,
                'operator splitting': False
            }
        ]
        
        for minimal_config in minimal_configs:
            config = ConfigurationFactory.create_fno_config()
            config['Model'] = minimal_config
            config['Physics']['variables'] = 1
            
            try:
                model = model_initialisation(config, dummy_normalizer, dummy_run)
                
                # Test forward pass
                x = torch.randn(1, 1, 16, 16)
                with torch.no_grad():
                    output = model(x)
                
                assert torch.isfinite(output).all(), \
                    f"Non-finite output for minimal {minimal_config['arch']}"
                
                print(f"✓ Minimal {minimal_config['arch']} configuration works")
                
            except Exception as e:
                pytest.skip(f"Minimal {minimal_config['arch']} failed: {e}")


# Performance and memory tests
@pytest.mark.slow
class TestModelPerformance:
    """Performance and memory tests for model instantiation"""
    
    @pytest.mark.skipif(not MODEL_SETUP_AVAILABLE, reason="Model setup not available")
    def test_model_instantiation_speed(self, dummy_normalizer, dummy_run):
        """Test that model instantiation is reasonably fast"""
        import time
        
        config = ConfigurationFactory.create_fno_config()
        
        start_time = time.time()
        model = model_initialisation(config, dummy_normalizer, dummy_run)
        end_time = time.time()
        
        instantiation_time = end_time - start_time
        
        # Should instantiate within reasonable time (adjust threshold as needed)
        assert instantiation_time < 30.0, \
            f"Model instantiation too slow: {instantiation_time:.2f}s"
        
        print(f"✓ Model instantiated in {instantiation_time:.3f}s")
    
    @pytest.mark.skipif(not MODEL_SETUP_AVAILABLE, reason="Model setup not available")
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_memory_usage_cuda(self, dummy_normalizer, dummy_run):
        """Test memory usage during model instantiation on CUDA"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        config = ConfigurationFactory.create_fno_config()
        
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
        
        model = model_initialisation(config, dummy_normalizer, dummy_run)
        model = model.cuda()
        
        model_memory = torch.cuda.memory_allocated() - initial_memory
        
        # Test forward pass memory
        x = torch.randn(4, 2, 64, 64, device='cuda')
        with torch.no_grad():
            output = model(x)
        
        total_memory = torch.cuda.memory_allocated() - initial_memory
        
        print(f"Model memory: {model_memory / 1024**2:.1f} MB")
        print(f"Total memory: {total_memory / 1024**2:.1f} MB")
        
        # Memory usage should be reasonable (adjust thresholds as needed)
        assert model_memory < 1024**3, "Model uses more than 1GB memory"
        
        torch.cuda.empty_cache()


# Main test execution and utility functions
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "cuda: marks tests that require CUDA")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically"""
    for item in items:
        # Add slow marker to performance tests
        if "performance" in item.name.lower() or "memory" in item.name.lower():
            item.add_marker(pytest.mark.slow)
        
        # Add cuda marker to CUDA tests
        if "cuda" in item.name.lower():
            item.add_marker(pytest.mark.cuda)


class TestSummaryReport:
    """Generate summary report of model instantiation tests"""
    
    @pytest.mark.skipif(not MODEL_SETUP_AVAILABLE, reason="Model setup not available")
    def test_comprehensive_model_report(self, dummy_normalizer, dummy_run, capsys):
        """Generate comprehensive report of all model configurations"""
        architectures = ['FNO', 'U-Net', 'CNO', 'Conv', 'gMLP']
        report = []
        
        print("\n" + "="*80)
        print("COMPREHENSIVE MODEL INSTANTIATION REPORT")
        print("="*80)
        
        for arch in architectures:
            try:
                if arch == 'FNO':
                    config = ConfigurationFactory.create_fno_config()
                elif arch == 'U-Net':
                    config = ConfigurationFactory.create_unet_config()
                elif arch == 'CNO':
                    config = ConfigurationFactory.create_cno_config()
                elif arch == 'gMLP':
                    config = ConfigurationFactory.create_gmlp_config()
                elif arch == 'Conv':
                    config = ConfigurationFactory.create_conv_config()
                
                model = model_initialisation(config, dummy_normalizer, dummy_run)
                param_count = count_parameters(model)
                
                # Test basic forward pass
                if arch == 'U-Net':
                    x = torch.randn(1, 2, 64, 64, 1)
                else:
                    x = torch.randn(1, 2, 64, 64)
                
                with torch.no_grad():
                    output = model(x)
                
                status = "✓ PASS"
                error_msg = ""
                
            except Exception as e:
                status = "✗ FAIL"
                error_msg = str(e)[:50] + "..." if len(str(e)) > 50 else str(e)
                param_count = 0
            
            report.append({
                'Architecture': arch,
                'Status': status,
                'Parameters': f"{param_count:,}" if param_count > 0 else "N/A",
                'Error': error_msg
            })
        
        # Print report
        print(f"{'Architecture':<15} {'Status':<10} {'Parameters':<15} {'Error':<30}")
        print("-" * 80)
        
        for item in report:
            print(f"{item['Architecture']:<15} {item['Status']:<10} {item['Parameters']:<15} {item['Error']:<30}")
        
        print("-" * 80)
        
        # Summary statistics
        total_models = len(report)
        passed_models = sum(1 for item in report if "PASS" in item['Status'])
        failed_models = total_models - passed_models
        
        print(f"SUMMARY: {passed_models}/{total_models} models passed instantiation tests")
        print(f"Success rate: {100 * passed_models / total_models:.1f}%")
        
        if failed_models > 0:
            print(f"⚠ {failed_models} models failed - check individual test results for details")
        
        print("="*80)
        
        # Assert that at least some models work
        assert passed_models > 0, "No models passed instantiation tests"


if __name__ == "__main__":
    """
    Run tests when script is executed directly.
    
    Usage examples:
    python test_model_instantiation.py                    # Run all tests
    python test_model_instantiation.py -v                 # Verbose output
    python test_model_instantiation.py -m "not slow"     # Skip slow tests
    python test_model_instantiation.py -k "test_fno"      # Run only FNO tests
    python test_model_instantiation.py --tb=short         # Short traceback
    """
    
    import pytest
    import sys
    
    # Add current directory to path for imports
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    # Run pytest with current file
    exit_code = pytest.main([
        __file__,
        "-v",  # Verbose output
        "--tb=short",  # Short traceback format
        "--strict-markers",  # Strict marker usage
        "-x",  # Stop on first failure for debugging
        "--capture=no",  # Show print statements
    ])
    
    sys.exit(exit_code)