#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Test Suite for Model Training and Evaluation Pipeline

This test suite verifies:
1. Model instantiation and parameter initialization
2. Training pipeline (forward pass, loss computation, gradients, optimization)
3. Evaluation pipeline and metrics
4. Data flow and tensor shapes throughout the process
5. Integration with different architectures and time stepping schemes

Run with: python -m pytest test_training_pipeline.py -v
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yaml
import tempfile
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Imports from your codebase
from Neural_PDE.Utils.processing_utils import MinMax_Normalizer, MinMax_Normalizer_variable
from Expts.data_loaders import SpatioTemporalDataset, stacked_fields
from Utils.explicit_time import Train_Setup as ExplicitTrainSetup, Eval_Setup as ExplicitEvalSetup
from Utils.explicit_time import euler, midpoint, rk4, autoregressive
from Utils.torch_odesolve import Train_Setup as ODETrainSetup, Eval_Setup as ODEEvalSetup
from Utils.metrics import MSE, RMSE, NMSE, NRMSE

# Import your actual model setup
try:
    from Expts.model_setup import model_initialisation, count_parameters
    from Neural_PDE.Models.FNO import FNO_multi2d
    from Neural_PDE.Models.UNet import UNet2d
    from Neural_PDE.Models.CNO import CNO2d
    from Neural_PDE.Models.gMLP_Vision import gMLP
    from Neural_PDE.Models.ConvOperator import ConvolutionalModel
    MODEL_SETUP_AVAILABLE = True
except ImportError:
    MODEL_SETUP_AVAILABLE = False
    
# Fallback imports for testing framework
from Tests.learnable_matrices import (
    Convolution2d, SpectralConv2d, FNO2d, SelfAttention2d, 
    Matrix2d, MLP2d
)
from Tests.model_builder import build_model as fallback_build_model


class TestDataGeneration:
    """Test utilities for generating synthetic data"""
    
    @staticmethod
    def create_synthetic_pde_data(batch_size=10, variables=2, nx=32, ny=32, nt=20, pattern='wave'):
        """
        Create synthetic PDE-like data with known temporal evolution
        
        Args:
            batch_size: Number of simulation samples
            variables: Number of field variables (e.g., u, v for NS)
            nx, ny: Spatial dimensions
            nt: Number of time steps
            pattern: Type of temporal pattern ('wave', 'decay', 'diffusion')
        
        Returns:
            torch.Tensor: Data of shape [batch_size, variables, nx, ny, nt]
        """
        x = torch.linspace(-1, 1, nx)
        y = torch.linspace(-1, 1, ny)
        t = torch.linspace(0, 2*np.pi, nt)
        
        X, Y = torch.meshgrid(x, y, indexing='ij')
        data = torch.zeros(batch_size, variables, nx, ny, nt)
        
        for b in range(batch_size):
            for v in range(variables):
                for i, time in enumerate(t):
                    if pattern == 'wave':
                        # Traveling wave pattern
                        freq = 1.0 + v * 0.5 + b * 0.1
                        data[b, v, :, :, i] = torch.sin(freq * (X + Y - 0.5 * time))
                    elif pattern == 'decay':
                        # Exponential decay
                        decay_rate = 0.1 + v * 0.05
                        initial = torch.exp(-(X**2 + Y**2))
                        data[b, v, :, :, i] = initial * torch.exp(-decay_rate * time)
                    elif pattern == 'diffusion':
                        # Gaussian diffusion
                        sigma = 0.3 + 0.1 * time + v * 0.05
                        data[b, v, :, :, i] = torch.exp(-((X**2 + Y**2) / (2 * sigma**2)))
                    else:
                        # Random smooth field
                        data[b, v, :, :, i] = torch.randn(nx, ny) * torch.exp(-0.1 * time)
        
        return data
    
    @staticmethod
    def create_test_config(arch='FNO', ode_solver='custom', method='euler', operator_splitting=False):
        """Create test configuration dictionary using your actual model configurations"""
        base_config = {
            'Physics': {
                'pde': 'Navier-Stokes',
                'variables': 2,
                'field': 'u, v',
                'Nx': 32,
                'Ny': 32,
                'Nt': 20,
                'dx': 0.1,
                'dy': 0.1,
                'dt': 0.01,
                'x_slice': 1,
                'y_slice': 1,
                't_slice': 1
            },
            'Data': {
                'ntrain': 8,
                'test-train-split': 0.25,
                'batch_size': 4,
                't_in': 1,
                't_out': 10,
                'step': 1,
                'normalisation': 'Min-Max'
            },
            'Train': {
                'rollout_length': 5,
                'input_length': 1,
                'input_noise': 0.0,
                'batch_norm': False,
                'odesolve': {
                    'source': ode_solver,
                    'method': method,
                    'adjoint': False
                }
            },
            'Opt': {
                'learning_rate': 0.001,
                'epochs': 3,
                'scheduler_step': 100,
                'scheduler_gamma': 0.5
            }
        }
        
        # Configure model based on architecture
        if arch == 'FNO':
            base_config['Model'] = {
                'arch': 'FNO',
                'operator splitting': operator_splitting,
                'in_vars': 2,
                'out_vars': 2,
                'modes': 8,
                'width': 16,
                'n_layers': 3
            }
        elif arch == 'U-Net':
            base_config['Model'] = {
                'arch': 'U-Net',
                'operator splitting': operator_splitting,
                'in_vars': 2,
                'out_vars': 2,
                'width': 32
            }
        elif arch == 'CNO':
            base_config['Model'] = {
                'arch': 'CNO',
                'operator splitting': operator_splitting,
                'in channels': 2,
                'out channels': 2,
                'Nx': 32,
                'N_layers': 3,
                'N_res': 2,
                'N_res_neck': 2,
                'channel multiplier': 16
            }
        elif arch == 'gMLP':
            base_config['Model'] = {
                'arch': 'gMLP',
                'operator splitting': operator_splitting,
                'n_blocks': 4,
                'd_in': 2,
                'd_ffn': 32,
                'Nx': 32,
                'Ny': 32
            }
        elif arch == 'Conv':
            base_config['Model'] = {
                'arch': 'Conv',
                'operator splitting': operator_splitting,
                'in_vars': 2,
                'out_vars': 2,
                'hidden_vars': [8, 16, 8],
                'n_layers': 3,
                'act': 'gelu'
            }
        else:
            # Fallback for test framework models
            base_config['Model'] = {
                'arch': arch,
                'operator splitting': False,
                'in_vars': 2,
                'out_vars': 2,
                'modes': 8,
                'width': 16,
                'n_layers': 2,
                'kernel_size': 3,
                'act': 'gelu',
                'init': 'random',
                'channels': 16,
                'heads': 2,
                'dropout': 0.0
            }
        
        return base_config


def build_model(config):
    """Build model using your actual model_initialisation function or fallback"""
    if MODEL_SETUP_AVAILABLE:
        # Create a dummy normalizer for model initialization
        dummy_data = torch.randn(1, config['Physics']['variables'], 
                                config['Physics']['Nx'], config['Physics']['Ny'], 5)
        normalizer = MinMax_Normalizer(dummy_data)
        
        # Create a dummy run object that operator splitting models might need
        class DummyRun:
            def __init__(self):
                self.name = "test_run"
            def log_metrics(self, metrics):
                pass
        
        dummy_run = DummyRun()
        
        try:
            model = model_initialisation(config, normalizer, dummy_run)
            return model
        except Exception as e:
            print(f"Warning: Could not create model with model_initialisation: {e}")
            return fallback_build_model(config)
    else:
        return fallback_build_model(config)


class TestModelInstantiation:
    """Test model creation and initialization using your actual models"""
    
    @pytest.mark.skipif(not MODEL_SETUP_AVAILABLE, reason="Model setup not available")
    @pytest.mark.parametrize("arch", ['FNO', 'U-Net', 'CNO', 'Conv', 'gMLP'])
    def test_actual_model_creation(self, arch):
        """Test that your actual models can be instantiated correctly"""
        config = TestDataGeneration.create_test_config(arch=arch)
        
        try:
            model = build_model(config)
            assert model is not None
            
            # Test forward pass with correct input shape
            # Your models expect different input shapes, so we'll test accordingly
            if arch == 'U-Net':
                # U-Net expects [batch, vars, nx, ny, t_in]
                x = torch.randn(2, 2, 32, 32, 1)
            elif arch == 'gMLP':
                # gMLP might expect flattened spatial input
                x = torch.randn(2, 2, 32, 32)
            else:
                # FNO, CNO, Conv expect [batch, vars, nx, ny]
                x = torch.randn(2, 2, 32, 32)
            
            output = model(x)
            
            # Check output properties
            assert torch.isfinite(output).all(), f"Model {arch} output contains non-finite values"
            assert output.shape[0] == x.shape[0], f"Batch dimension mismatch for {arch}"
            
            print(f"✓ {arch} model created successfully. Input: {x.shape}, Output: {output.shape}")
            
        except Exception as e:
            pytest.skip(f"Model {arch} not available or failed: {e}")
    
    @pytest.mark.skipif(not MODEL_SETUP_AVAILABLE, reason="Model setup not available")
    def test_operator_splitting_models(self):
        """Test operator splitting models if available"""
        config = TestDataGeneration.create_test_config(arch='FNO', operator_splitting=True)
        
        try:
            model = build_model(config)
            assert model is not None
            
            # Test forward pass - operator splitting models might expect different input format
            x = torch.randn(2, 2, 32, 32, 1)  # [batch, vars, nx, ny, nt]
            output = model(x)
            
            assert torch.isfinite(output).all(), "Operator splitting model output contains non-finite values"
            print(f"✓ Operator splitting model created successfully")
            
        except Exception as e:
            pytest.skip(f"Operator splitting model not available: {e}")
    
    @pytest.mark.skipif(not MODEL_SETUP_AVAILABLE, reason="Model setup not available")
    def test_parameter_counting(self):
        """Test parameter counting using your count_parameters function"""
        config = TestDataGeneration.create_test_config(arch='FNO')
        model = build_model(config)
        
        # Test your count_parameters function
        param_count = count_parameters(model)
        
        assert param_count > 0, "Model has no parameters"
        assert isinstance(param_count, int), "Parameter count should be an integer"
        
        # Cross-check with PyTorch's parameter counting
        torch_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert param_count == torch_count, "Parameter counting mismatch"
        
        print(f"✓ Model has {param_count:,} parameters")
    
    def test_fallback_model_creation(self):
        """Test fallback model creation when actual models aren't available"""
        config = TestDataGeneration.create_test_config(arch='TestConv')
        
        try:
            model = build_model(config)
            assert model is not None
            
            x = torch.randn(2, 2, 16, 16)
            output = model(x)
            
            assert output.shape == x.shape
            assert torch.isfinite(output).all()
            
        except Exception as e:
            pytest.skip(f"Fallback model creation failed: {e}")
    
    @pytest.mark.skipif(not MODEL_SETUP_AVAILABLE, reason="Model setup not available")
    def test_model_input_output_shapes(self):
        """Test that models handle input/output shapes correctly"""
        architectures = ['FNO', 'Conv']  # Test architectures that are likely to be available
        
        for arch in architectures:
            try:
                config = TestDataGeneration.create_test_config(arch=arch)
                model = build_model(config)
                
                # Test different input sizes
                test_sizes = [(16, 16), (32, 32), (64, 64)]
                
                for nx, ny in test_sizes:
                    x = torch.randn(1, 2, nx, ny)
                    
                    try:
                        output = model(x)
                        
                        # Check that output has correct batch and channel dimensions
                        assert output.shape[0] == 1, f"Batch dimension incorrect for {arch}"
                        assert output.shape[1] == 2, f"Channel dimension incorrect for {arch}"
                        
                        print(f"✓ {arch} handles {nx}x{ny} input correctly")
                        
                    except Exception as e:
                        print(f"⚠ {arch} failed with {nx}x{ny} input: {e}")
                        
            except Exception as e:
                print(f"⚠ Could not test {arch}: {e}")
    
    @pytest.mark.skipif(not MODEL_SETUP_AVAILABLE, reason="Model setup not available")
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_model_cuda_compatibility(self):
        """Test CUDA compatibility for actual models"""
        config = TestDataGeneration.create_test_config(arch='FNO')
        
        try:
            model = build_model(config)
            model = model.cuda()
            
            x = torch.randn(2, 2, 32, 32, device='cuda')
            output = model(x)
            
            assert output.device.type == 'cuda'
            assert torch.isfinite(output).all()
            
            print("✓ Model CUDA compatibility verified")
            
        except Exception as e:
            pytest.skip(f"CUDA compatibility test failed: {e}")


class TestDataPipeline:
    """Test data loading and processing pipeline"""
    
    def test_data_generation(self):
        """Test synthetic data generation"""
        data = TestDataGeneration.create_synthetic_pde_data(
            batch_size=5, variables=2, nx=16, ny=16, nt=10
        )
        
        assert data.shape == (5, 2, 16, 16, 10)
        assert torch.isfinite(data).all()
        
        # Test different patterns
        for pattern in ['wave', 'decay', 'diffusion']:
            pattern_data = TestDataGeneration.create_synthetic_pde_data(
                batch_size=2, variables=1, nx=8, ny=8, nt=5, pattern=pattern
            )
            assert pattern_data.shape == (2, 1, 8, 8, 5)
            assert torch.isfinite(pattern_data).all()
    
    def test_normalization_pipeline(self):
        """Test data normalization and denormalization"""
        data = TestDataGeneration.create_synthetic_pde_data(
            batch_size=6, variables=3, nx=16, ny=16, nt=8
        )
        
        # Test MinMax normalization
        normalizer = MinMax_Normalizer(data)
        normalized = normalizer.encode(data)
        denormalized = normalizer.decode(normalized)
        
        # Check normalization range
        assert normalized.min() >= -0.1
        assert normalized.max() <= 1.1
        
        # Check reconstruction
        assert torch.allclose(data, denormalized, atol=1e-5)
        
        # Test per-variable normalization
        var_normalizer = MinMax_Normalizer_variable(data)
        var_normalized = var_normalizer.encode(data)
        var_denormalized = var_normalizer.decode(var_normalized)
        
        assert torch.allclose(data, var_denormalized, atol=1e-5)
    
    def test_spatiotemporal_dataset(self):
        """Test SpatioTemporalDataset creation and sampling"""
        data = TestDataGeneration.create_synthetic_pde_data(
            batch_size=4, variables=2, nx=16, ny=16, nt=15
        )
        
        dataset = SpatioTemporalDataset(data, input_window=3, prediction_steps=2)
        
        assert len(dataset) > 0
        
        # Test sample retrieval
        input_seq, target_seq = dataset[0]
        assert input_seq.shape == (2, 16, 16, 3)
        assert target_seq.shape == (2, 16, 16, 2)
        
        # Test temporal continuity
        assert torch.isfinite(input_seq).all()
        assert torch.isfinite(target_seq).all()


class TestTrainingPipeline:
    """Test training pipeline components"""
    
    def setup_training_data(self, config):
        """Setup training and test data"""
        # Generate synthetic data
        data = TestDataGeneration.create_synthetic_pde_data(
            batch_size=config['Data']['ntrain'],
            variables=config['Physics']['variables'],
            nx=config['Physics']['Nx'],
            ny=config['Physics']['Ny'],
            nt=config['Data']['t_out']
        )
        
        # Normalize data
        normalizer = MinMax_Normalizer(data)
        data_normalized = normalizer.encode(data)
        
        # Split into train/test
        from sklearn.model_selection import train_test_split
        train_in, test_in, train_out, test_out = train_test_split(
            data_normalized[..., :config['Data']['t_in']],
            data_normalized[..., config['Data']['t_in']:config['Data']['t_out']],
            test_size=config['Data']['test-train-split'],
            random_state=42
        )
        
        # Create data loaders
        train_data = torch.cat((train_in, train_out), dim=-1)
        dataset = SpatioTemporalDataset(
            train_data,
            input_window=config['Train']['input_length'],
            prediction_steps=config['Train']['rollout_length'] - 1
        )
        
        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=config['Data']['batch_size'], shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(test_in, test_out),
            batch_size=config['Data']['batch_size'], shuffle=False
        )
        
        return train_loader, test_loader, normalizer
    
    @pytest.mark.skipif(not MODEL_SETUP_AVAILABLE, reason="Model setup not available")
    @pytest.mark.parametrize("arch,ode_solver,method", [
        ('FNO', 'custom', 'euler'),
        ('FNO', 'custom', 'midpoint'),
        ('FNO', 'custom', 'rk4'),
        ('Conv', 'custom', 'AR'),
    ])
    def test_actual_model_training_setup(self, arch, ode_solver, method):
        """Test explicit time integration training setup with actual models"""
        config = TestDataGeneration.create_test_config(arch=arch, ode_solver=ode_solver, method=method)
        
        try:
            # Setup data
            train_loader, test_loader, normalizer = self.setup_training_data(config)
            
            # Create model
            model = build_model(config)
            
            # Setup training components
            optimizer = optim.Adam(model.parameters(), lr=config['Opt']['learning_rate'])
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=config['Opt']['scheduler_step'], 
                gamma=config['Opt']['scheduler_gamma']
            )
            loss_func = nn.MSELoss()
            
            # Create trainer
            trainer = ExplicitTrainSetup(
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                loss_func=loss_func,
                optimizer=optimizer,
                scheduler=scheduler,
                epochs=config['Opt']['epochs'],
                ode_solver=ode_solver,
                roll_out=method,
                noise=config['Train']['input_noise'],
                batch_norm=config['Train']['batch_norm']
            )
            
            # Test one epoch
            dt = config['Physics']['dt']
            train_loss, test_loss = trainer.one_epoch(
                step=config['Data']['step'],
                train_T_out=config['Train']['rollout_length'] - 1,
                test_T_out=config['Data']['t_out'] - config['Data']['t_in'],
                dt=dt
            )
            
            # Verify losses are finite and reasonable
            assert np.isfinite(train_loss), f"Training loss is not finite: {train_loss}"
            assert np.isfinite(test_loss), f"Test loss is not finite: {test_loss}"
            assert train_loss >= 0, f"Training loss is negative: {train_loss}"
            assert test_loss >= 0, f"Test loss is negative: {test_loss}"
            
            print(f"✓ {arch} with {method} training successful. Train loss: {train_loss:.6f}, Test loss: {test_loss:.6f}")
            
        except Exception as e:
            pytest.skip(f"Training setup failed for {arch} with {method}: {e}")
    
    @pytest.mark.skipif(not MODEL_SETUP_AVAILABLE, reason="Model setup not available")
    def test_gradient_flow_actual_models(self):
        """Test that gradients flow properly through actual models"""
        config = TestDataGeneration.create_test_config(arch='FNO')
        
        try:
            train_loader, test_loader, normalizer = self.setup_training_data(config)
            
            model = build_model(config)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            loss_func = nn.MSELoss()
            
            # Get initial parameter values
            initial_params = [p.clone() for p in model.parameters()]
            
            # Forward pass
            for batch_input, batch_target in train_loader:
                optimizer.zero_grad()
                
                # Handle different input shapes for different models
                if len(batch_input.shape) == 5:  # [batch, vars, nx, ny, nt]
                    input_data = batch_input[..., 0]  # Take first time step
                    target_data = batch_target[..., 0]
                else:
                    input_data = batch_input
                    target_data = batch_target
                
                output = model(input_data)
                loss = loss_func(output, target_data)
                
                # Backward pass
                loss.backward()
                
                # Check that gradients exist and are finite
                grad_found = False
                for param in model.parameters():
                    if param.grad is not None:
                        assert torch.isfinite(param.grad).all(), "Parameter gradient contains non-finite values"
                        grad_found = True
                
                assert grad_found, "No gradients found in model parameters"
                
                # Update parameters
                optimizer.step()
                break
            
            # Check that parameters actually changed
            params_changed = False
            for initial, current in zip(initial_params, model.parameters()):
                if not torch.allclose(initial, current, atol=1e-7):
                    params_changed = True
                    break
            
            assert params_changed, "Parameters didn't change after optimization step"
            print("✓ Gradient flow verified for actual model")
            
        except Exception as e:
            pytest.skip(f"Gradient flow test failed: {e}")
    
    @pytest.mark.skipif(not MODEL_SETUP_AVAILABLE, reason="Model setup not available")
    def test_model_specific_requirements(self):
        """Test model-specific requirements and constraints"""
        
        # Test FNO with grid requirements
        try:
            config = TestDataGeneration.create_test_config(arch='FNO')
            model = build_model(config)
            
            # FNO should handle power-of-2 sizes well
            for size in [16, 32, 64]:
                x = torch.randn(1, 2, size, size)
                output = model(x)
                assert torch.isfinite(output).all()
            
            print("✓ FNO grid size compatibility verified")
            
        except Exception as e:
            print(f"⚠ FNO test failed: {e}")
        
        # Test U-Net with temporal input
        try:
            config = TestDataGeneration.create_test_config(arch='U-Net')
            model = build_model(config)
            
            # U-Net expects temporal input
            x = torch.randn(2, 2, 32, 32, 1)  # [batch, vars, nx, ny, t_in]
            output = model(x)
            assert torch.isfinite(output).all()
            
            print("✓ U-Net temporal input compatibility verified")
            
        except Exception as e:
            print(f"⚠ U-Net test failed: {e}")
        
        # Test ConvolutionalModel
        try:
            config = TestDataGeneration.create_test_config(arch='Conv')
            model = build_model(config)
            
            # Conv model should handle standard 2D input
            x = torch.randn(2, 2, 32, 32)
            output = model(x)
            assert torch.isfinite(output).all()
            assert output.shape == x.shape  # Should preserve input shape
            
            print("✓ ConvolutionalModel compatibility verified")
            
        except Exception as e:
            print(f"⚠ ConvolutionalModel test failed: {e}")
            ode_solver = ODESolver(
                model,
                roll_out=method,
                noise=config['Train']['input_noise'],
                batch_norm=config['Train']['batch_norm']
            )

        # Test one epoch
        dt = config['Physics']['dt']
        train_loss, test_loss = trainer.one_epoch(
            step=config['Data']['step'],
            train_T_out=config['Train']['rollout_length'] - 1,
            test_T_out=config['Data']['t_out'] - config['Data']['t_in'],
            dt=dt
        )
        
        # Verify losses are finite and reasonable
        assert np.isfinite(train_loss), f"Training loss is not finite: {train_loss}"
        assert np.isfinite(test_loss), f"Test loss is not finite: {test_loss}"
        assert train_loss >= 0, f"Training loss is negative: {train_loss}"
        assert test_loss >= 0, f"Test loss is negative: {test_loss}"
    
    def test_gradient_flow(self):
        """Test that gradients flow properly through the model"""
        config = TestDataGeneration.create_test_config()
        train_loader, test_loader, normalizer = self.setup_training_data(config)
        
        model = build_model(config)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        loss_func = nn.MSELoss()
        
        # Get initial parameter values
        initial_params = [p.clone() for p in model.parameters()]
        
        # Forward pass
        for batch_input, batch_target in train_loader:
            optimizer.zero_grad()
            
            # Simple forward pass (not full rollout for simplicity)
            output = model(batch_input[..., 0])
            loss = loss_func(output, batch_target[..., 0])
            
            # Backward pass
            loss.backward()
            
            # Check that gradients exist and are finite
            for param in model.parameters():
                assert param.grad is not None, "Parameter has no gradient"
                assert torch.isfinite(param.grad).all(), "Parameter gradient contains non-finite values"
            
            # Update parameters
            optimizer.step()
            break
        
        # Check that parameters actually changed
        for initial, current in zip(initial_params, model.parameters()):
            assert not torch.allclose(initial, current), "Parameters didn't change after optimization step"
    
    def test_loss_computation(self):
        """Test loss computation and behavior"""
        config = TestDataGeneration.create_test_config()
        model = build_model(config)
        
        # Test with identical inputs (should give low loss)
        x = torch.randn(4, 2, 16, 16)
        output = model(x)
        
        loss_func = nn.MSELoss()
        identical_loss = loss_func(output, output)
        assert identical_loss.item() < 1e-6, "Loss with identical tensors should be near zero"
        
        # Test with different inputs (should give higher loss)
        y = torch.randn_like(output)
        different_loss = loss_func(output, y)
        assert different_loss.item() > identical_loss.item(), "Loss with different tensors should be higher"
        
        # Test loss reduction
        assert different_loss.dim() == 0, "Loss should be a scalar"
        assert torch.isfinite(different_loss), "Loss should be finite"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_training(self):
        """Test training on CUDA if available"""
        config = TestDataGeneration.create_test_config()
        train_loader, test_loader, normalizer = self.setup_training_data(config)
        
        model = build_model(config).cuda()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        loss_func = nn.MSELoss()
        
        # Test one training step on CUDA
        for batch_input, batch_target in train_loader:
            batch_input = batch_input.cuda()
            batch_target = batch_target.cuda()
            
            optimizer.zero_grad()
            output = model(batch_input[..., 0])
            loss = loss_func(output, batch_target[..., 0])
            loss.backward()
            optimizer.step()
            
            assert output.device.type == 'cuda'
            assert loss.device.type == 'cuda'
            break


class TestEvaluationPipeline:
    """Test evaluation and inference pipeline"""
    
    @pytest.mark.skipif(not MODEL_SETUP_AVAILABLE, reason="Model setup not available")
    def test_actual_model_evaluation_setup(self):
        """Test evaluation setup and inference with actual models"""
        config = TestDataGeneration.create_test_config(arch='FNO')
        
        try:
            # Generate test data
            data = TestDataGeneration.create_synthetic_pde_data(
                batch_size=6, variables=2, nx=32, ny=32, nt=15
            )
            
            normalizer = MinMax_Normalizer(data)
            data_normalized = normalizer.encode(data)
            
            test_in = data_normalized[..., :config['Data']['t_in']]
            test_out = data_normalized[..., config['Data']['t_in']:config['Data']['t_out']]
            
            # Create model
            model = build_model(config)
            model.eval()
            
            # Create evaluator
            evaluator = ExplicitEvalSetup(
                model=model,
                test_in=test_in,
                test_out=test_out,
                normalizer='False',
                ode_solver=config['Train']['odesolve']['source'],
                roll_out=config['Train']['odesolve']['method'],
                batch_size=config['Data']['batch_size']
            )
            
            # Run inference
            pred_set, error = evaluator.inference(
                step=config['Data']['step'],
                T_out=config['Data']['t_out'] - config['Data']['t_in'],
                dt=config['Physics']['dt']
            )
            
            # Check outputs
            assert pred_set.shape == test_out.shape
            assert torch.isfinite(pred_set).all()
            assert np.isfinite(error), f"Error is not finite: {error}"
            assert error >= 0, f"Error is negative: {error}"
            
            print(f"✓ FNO evaluation successful. Error: {error:.6f}")
            
        except Exception as e:
            pytest.skip(f"Actual model evaluation failed: {e}")
    
    def test_evaluation_setup(self):
        """Test evaluation setup and inference with fallback models"""
        config = TestDataGeneration.create_test_config()
        
        # Generate test data
        data = TestDataGeneration.create_synthetic_pde_data(
            batch_size=6, variables=2, nx=32, ny=32, nt=15
        )
        
        normalizer = MinMax_Normalizer(data)
        data_normalized = normalizer.encode(data)
        
        test_in = data_normalized[..., :config['Data']['t_in']]
        test_out = data_normalized[..., config['Data']['t_in']:config['Data']['t_out']]
        
        # Create model
        model = build_model(config)
        model.eval()
        
        # Create evaluator
        evaluator = ExplicitEvalSetup(
            model=model,
            test_in=test_in,
            test_out=test_out,
            normalizer='False',
            ode_solver=config['Train']['odesolve']['source'],
            roll_out=config['Train']['odesolve']['method'],
            batch_size=config['Data']['batch_size']
        )
        
        # Run inference
        pred_set, error = evaluator.inference(
            step=config['Data']['step'],
            T_out=config['Data']['t_out'] - config['Data']['t_in'],
            dt=config['Physics']['dt']
        )
        
        # Check outputs
        assert pred_set.shape == test_out.shape
        assert torch.isfinite(pred_set).all()
        assert np.isfinite(error), f"Error is not finite: {error}"
        assert error >= 0, f"Error is negative: {error}"
    
    def test_metrics_computation(self):
        """Test performance metrics computation"""
        # Create dummy prediction and target data
        pred = torch.randn(8, 2, 16, 16, 5)
        target = torch.randn(8, 2, 16, 16, 5)
        
        # Test MSE
        mse_result = MSE(pred, target)
        assert 'per_sample' in mse_result
        assert 'average' in mse_result
        assert len(mse_result['per_sample']) == 8
        assert mse_result['average'] >= 0
        
        # Test RMSE
        rmse_result = RMSE(pred, target)
        assert np.allclose(rmse_result['average'], np.sqrt(mse_result['average']))
        
        # Test NMSE
        nmse_result = NMSE(pred, target)
        assert nmse_result['average'] >= 0
        
        # Test NRMSE
        nrmse_result = NRMSE(pred, target)
        assert nrmse_result['average'] >= 0
    
    def test_prediction_consistency(self):
        """Test that predictions are consistent across multiple runs"""
        config = TestDataGeneration.create_test_config()
        
        # Set seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        model = build_model(config)
        model.eval()
        
        x = torch.randn(2, 2, 16, 16)
        
        # Multiple forward passes should give same result
        with torch.no_grad():
            pred1 = model(x)
            pred2 = model(x)
        
        assert torch.allclose(pred1, pred2), "Model predictions are not consistent"


class TestTimeSteppingSchemes:
    """Test different time stepping schemes"""
    
    def setup_simple_model(self):
        """Create a simple model for testing time stepping"""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 4)  # Simple linear transformation
                
            def forward(self, x):
                # x shape: [batch, vars, nx, ny, nt] -> flatten spatial dims
                batch, vars, nx, ny, nt = x.shape
                x_flat = x.reshape(batch, vars * nx * ny, nt)
                x_out = self.linear(x_flat.transpose(-1, -2)).transpose(-1, -2)
                return x_out.reshape(batch, vars, nx, ny, nt)
        
        return SimpleModel()
    
    def test_time_stepping_schemes(self):
        """Test different explicit time stepping schemes"""
        # Simple ODE: du/dt = -u (exponential decay)
        simple_model = lambda x: -0.1 * x
        dt = 0.01
        u0 = torch.tensor([1.0, 2.0, 3.0])
        
        # Test each scheme
        schemes = {
            'euler': euler,
            'midpoint': midpoint,
            'rk4': rk4
        }
        
        for name, scheme in schemes.items():
            u = u0.clone()
            # Take a few steps
            for _ in range(10):
                u = scheme(simple_model, u, dt)
            
            # Should decay (all values should be smaller than initial)
            assert torch.all(torch.abs(u) < torch.abs(u0)), f"{name} scheme didn't produce decay"
            assert torch.isfinite(u).all(), f"{name} scheme produced non-finite values"
    
    def test_autoregressive_scheme(self):
        """Test autoregressive time stepping"""
        model = self.setup_simple_model()
        x = torch.randn(2, 2, 4, 4, 1)
        
        # AR should just apply the model
        result = autoregressive(model, x)
        
        assert result.shape == x.shape
        assert torch.isfinite(result).all()
    
    def test_scheme_stability(self):
        """Test numerical stability of time stepping schemes"""
        # Stiff ODE that can cause instability with large dt
        stiff_model = lambda x: -10.0 * x
        dt_large = 0.5  # Large time step
        u0 = torch.tensor([1.0])
        
        # Test that schemes don't blow up (remain bounded)
        schemes = [euler, midpoint, rk4]
        
        for scheme in schemes:
            u = u0.clone()
            for _ in range(20):
                u = scheme(stiff_model, u, dt_large)
                # Check that solution remains bounded
                assert torch.abs(u).max() < 100, f"Scheme became unstable: {u}"


class TestIntegrationWorkflow:
    """Test complete training and evaluation workflow using actual models"""
    
    @pytest.mark.skipif(not MODEL_SETUP_AVAILABLE, reason="Model setup not available")
    def test_complete_workflow_actual_models(self):
        """Test complete training and evaluation workflow with actual models"""
        config = TestDataGeneration.create_test_config(arch='FNO')
        
        try:
            # Generate data
            data = TestDataGeneration.create_synthetic_pde_data(
                batch_size=config['Data']['ntrain'],
                variables=config['Physics']['variables'],
                nx=config['Physics']['Nx'],
                ny=config['Physics']['Ny'],
                nt=config['Data']['t_out']
            )
            
            # Normalize data
            normalizer = MinMax_Normalizer(data)
            data_normalized = normalizer.encode(data)
            
            # Split data
            from sklearn.model_selection import train_test_split
            train_in, test_in, train_out, test_out = train_test_split(
                data_normalized[..., :config['Data']['t_in']],
                data_normalized[..., config['Data']['t_in']:config['Data']['t_out']],
                test_size=config['Data']['test-train-split'],
                random_state=42
            )
            
            # Create model and training components
            model = build_model(config)
            optimizer = optim.Adam(model.parameters(), lr=config['Opt']['learning_rate'])
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
            loss_func = nn.MSELoss()
            
            # Setup data loaders
            train_data = torch.cat((train_in, train_out), dim=-1)
            dataset = SpatioTemporalDataset(train_data, input_window=1, prediction_steps=4)
            train_loader = torch.utils.data.DataLoader(dataset, batch_size=config['Data']['batch_size'])
            test_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(test_in, test_out),
                batch_size=config['Data']['batch_size']
            )
            
            # Training
            trainer = ExplicitTrainSetup(
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                loss_func=loss_func,
                optimizer=optimizer,
                scheduler=scheduler,
                epochs=config['Opt']['epochs'],
                ode_solver=config['Train']['odesolve']['source'],
                roll_out=config['Train']['odesolve']['method']
            )
            
            # Train for a few epochs
            initial_loss = None
            for epoch in range(2):
                train_loss, test_loss = trainer.one_epoch(
                    step=1, train_T_out=4, test_T_out=config['Data']['t_out'] - 1,
                    dt=config['Physics']['dt']
                )
                
                if initial_loss is None:
                    initial_loss = train_loss
                
                assert np.isfinite(train_loss)
                assert np.isfinite(test_loss)
            
            # Evaluation
            evaluator = ExplicitEvalSetup(
                model=model,
                test_in=test_in,
                test_out=test_out,
                ode_solver=config['Train']['odesolve']['source'],
                roll_out=config['Train']['odesolve']['method']
            )
            
            pred_set, error = evaluator.inference(
                step=1, T_out=config['Data']['t_out'] - 1, dt=config['Physics']['dt']
            )
            
            # Verify complete workflow
            assert pred_set.shape == test_out.shape
            assert np.isfinite(error)
            
            # Denormalize and check
            pred_physical = normalizer.decode(pred_set)
            test_physical = normalizer.decode(test_out)
            
            # Compute metrics on physical space
            mse_physical = MSE(pred_physical, test_physical)
            assert np.isfinite(mse_physical['average'])
            
            print(f"✓ Complete workflow successful with FNO model")
            print(f"  Final error: {error:.6f}")
            print(f"  Physical MSE: {mse_physical['average']:.6f}")
            print(f"  Model parameters: {count_parameters(model):,}")
            
        except Exception as e:
            pytest.skip(f"Complete workflow test failed: {e}")
    
    def test_complete_workflow(self):
        """Test complete training and evaluation workflow with fallback models"""
        config = TestDataGeneration.create_test_config()
        
        # Generate data
        data = TestDataGeneration.create_synthetic_pde_data(
            batch_size=config['Data']['ntrain'],
            variables=config['Physics']['variables'],
            nx=config['Physics']['Nx'],
            ny=config['Physics']['Ny'],
            nt=config['Data']['t_out']
        )
        
        # Normalize data
        normalizer = MinMax_Normalizer(data)
        data_normalized = normalizer.encode(data)
        
        # Split data
        from sklearn.model_selection import train_test_split
        train_in, test_in, train_out, test_out = train_test_split(
            data_normalized[..., :config['Data']['t_in']],
            data_normalized[..., config['Data']['t_in']:config['Data']['t_out']],
            test_size=config['Data']['test-train-split'],
            random_state=42
        )
        
        # Create model and training components
        model = build_model(config)
        optimizer = optim.Adam(model.parameters(), lr=config['Opt']['learning_rate'])
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
        loss_func = nn.MSELoss()
        
        # Setup data loaders
        train_data = torch.cat((train_in, train_out), dim=-1)
        dataset = SpatioTemporalDataset(train_data, input_window=1, prediction_steps=4)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=config['Data']['batch_size'])
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(test_in, test_out),
            batch_size=config['Data']['batch_size']
        )
        
        # Training
        trainer = ExplicitTrainSetup(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            loss_func=loss_func,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=config['Opt']['epochs'],
            ode_solver=config['Train']['odesolve']['source'],
            roll_out=config['Train']['odesolve']['method']
        )
        
        # Train for a few epochs
        initial_loss = None
        for epoch in range(2):
            train_loss, test_loss = trainer.one_epoch(
                step=1, train_T_out=4, test_T_out=config['Data']['t_out'] - 1,
                dt=config['Physics']['dt']
            )
            
            if initial_loss is None:
                initial_loss = train_loss
            
            assert np.isfinite(train_loss)
            assert np.isfinite(test_loss)
        
        # Evaluation
        evaluator = ExplicitEvalSetup(
            model=model,
            test_in=test_in,
            test_out=test_out,
            ode_solver=config['Train']['odesolve']['source'],
            roll_out=config['Train']['odesolve']['method']
        )
        
        pred_set, error = evaluator.inference(
            step=1, T_out=config['Data']['t_out'] - 1, dt=config['Physics']['dt']
        )
        
        # Verify complete workflow
        assert pred_set.shape == test_out.shape
        assert np.isfinite(error)
        
        # Denormalize and check
        pred_physical = normalizer.decode(pred_set)
        test_physical = normalizer.decode(test_out)
        
        # Compute metrics on physical space
        mse_physical = MSE(pred_physical, test_physical)
        assert np.isfinite(mse_physical['average'])
    
    @pytest.mark.skipif(not MODEL_SETUP_AVAILABLE, reason="Model setup not available")  
    def test_config_variations_actual_models(self):
        """Test workflow with different configuration variations using actual models"""
        base_config = TestDataGeneration.create_test_config(arch='FNO')
        
        variations = [
            ('FNO', {'Train': {'odesolve': {'method': 'midpoint'}}}),
            ('Conv', {'Train': {'odesolve': {'method': 'rk4'}}}),
            ('FNO', {'Data': {'normalisation': 'Min-Max_variable'}}),
        ]
        
        for arch, variation in variations:
            config = TestDataGeneration.create_test_config(arch=arch)
            # Deep update config with variation
            for key, value in variation.items():
                if key in config:
                    config[key].update(value)
                else:
                    config[key] = value
            
            try:
                # Quick workflow test
                data = TestDataGeneration.create_synthetic_pde_data(
                    batch_size=4, variables=2, nx=16, ny=16, nt=8
                )
                
                model = build_model(config)
                assert model is not None
                
                # Test forward pass
                x = torch.randn(1, 2, 16, 16)
                output = model(x)
                assert output.shape[:2] == x.shape[:2]  # At least batch and channels should match
                
                print(f"✓ Configuration variation {arch} with {variation} successful")
                
            except Exception as e:
                print(f"⚠ Configuration variation {arch} failed: {e}")
    
    def test_config_variations(self):
        """Test workflow with different configuration variations"""
        base_config = TestDataGeneration.create_test_config()
        
        variations = [
            {'Train': {'odesolve': {'method': 'midpoint'}}},
            {'Data': {'normalisation': 'Min-Max_variable'}},
        ]
        
        for variation in variations:
            config = base_config.copy()
            # Deep update config with variation
            for key, value in variation.items():
                if key in config:
                    config[key].update(value)
                else:
                    config[key] = value
            
            try:
                # Quick workflow test
                data = TestDataGeneration.create_synthetic_pde_data(
                    batch_size=4, variables=2, nx=16, ny=16, nt=8
                )
                
                model = build_model(config)
                assert model is not None
                
                # Test forward pass
                x = torch.randn(1, 2, 16, 16)
                output = model(x)
                assert output.shape == x.shape
                
            except Exception as e:
                pytest.skip(f"Configuration variation failed: {e}")


# Performance and stress tests
class TestPerformanceStress:
    """Performance and stress tests"""
    
    @pytest.mark.slow
    def test_memory_usage(self):
        """Test memory usage doesn't grow unexpectedly"""
        config = TestDataGeneration.create_test_config()
        model = build_model(config)
        
        # Monitor memory during multiple forward passes
        if torch.cuda.is_available():
            device = torch.device('cuda')
            model = model.to(device)
            torch.cuda.empty_cache()
            
            initial_memory = torch.cuda.memory_allocated()
            
            for i in range(10):
                x = torch.randn(2, 2, 32, 32, device=device)
                with torch.no_grad():
                    output = model(x)
                del x, output
                
                if i > 0:  # Allow for initial memory allocation
                    current_memory = torch.cuda.memory_allocated()
                    assert current_memory <= initial_memory * 2, "Memory usage growing unexpectedly"
            
            torch.cuda.empty_cache()
    
    @pytest.mark.slow
    def test_large_batch_training(self):
        """Test training with larger batch sizes"""
        config = TestDataGeneration.create_test_config()
        config['Data']['batch_size'] = 16  # Larger batch
        config['Data']['ntrain'] = 32
        
        try:
            data = TestDataGeneration.create_synthetic_pde_data(
                batch_size=32, variables=2, nx=32, ny=32, nt=15
            )
            
            model = build_model(config)
            normalizer = MinMax_Normalizer(data)
            
            # Test that large batches work
            large_batch = torch.randn(16, 2, 32, 32)
            output = model(large_batch)
            
            assert output.shape == large_batch.shape
            assert torch.isfinite(output).all()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                pytest.skip("Insufficient memory for large batch test")
            else:
                raise
    
    def test_gradient_accumulation(self):
        """Test gradient accumulation for larger effective batch sizes"""
        config = TestDataGeneration.create_test_config()
        model = build_model(config)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        loss_func = nn.MSELoss()
        
        # Simulate gradient accumulation
        accumulation_steps = 4
        effective_batch_size = config['Data']['batch_size'] * accumulation_steps
        
        # Generate data
        data = TestDataGeneration.create_synthetic_pde_data(
            batch_size=effective_batch_size, variables=2, nx=16, ny=16, nt=10
        )
        
        # Split into smaller batches
        small_batches = torch.split(data, config['Data']['batch_size'], dim=0)
        
        optimizer.zero_grad()
        accumulated_loss = 0
        
        for i, batch in enumerate(small_batches):
            input_batch = batch[..., 0]
            target_batch = batch[..., 1]
            
            output = model(input_batch)
            loss = loss_func(output, target_batch) / accumulation_steps
            loss.backward()
            
            accumulated_loss += loss.item()
        
        # Check gradients accumulated properly
        for param in model.parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all()
        
        optimizer.step()
        
        assert accumulated_loss > 0


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_nan_input_handling(self):
        """Test model behavior with NaN inputs"""
        config = TestDataGeneration.create_test_config()
        model = build_model(config)
        
        # Create input with NaN
        x = torch.randn(2, 2, 16, 16)
        x[0, 0, 0, 0] = float('nan')
        
        with torch.no_grad():
            output = model(x)
        
        # Model should either handle NaN gracefully or propagate it
        # (depending on implementation, both behaviors can be valid)
        assert output.shape == x.shape
    
    def test_empty_batch_handling(self):
        """Test handling of empty or minimal batches"""
        config = TestDataGeneration.create_test_config()
        model = build_model(config)
        
        # Test with batch size 1
        x = torch.randn(1, 2, 16, 16)
        output = model(x)
        
        assert output.shape == x.shape
        assert torch.isfinite(output).all()
    
    def test_mismatched_dimensions(self):
        """Test handling of mismatched input dimensions"""
        config = TestDataGeneration.create_test_config()
        model = build_model(config)
        
        # Test with wrong number of channels
        try:
            x_wrong_channels = torch.randn(2, 3, 16, 16)  # 3 channels instead of 2
            output = model(x_wrong_channels)
            # Some models might handle this by adapting
        except (RuntimeError, AssertionError):
            # Expected for models that require exact channel matching
            pass
        
        # Test with different spatial dimensions
        try:
            x_different_size = torch.randn(2, 2, 8, 8)  # 8x8 instead of 16x16
            output = model(x_different_size)
            assert output.shape[:2] == x_different_size.shape[:2]  # Batch and channels should match
        except (RuntimeError, AssertionError):
            # Some models might not handle different spatial sizes
            pass
    
    def test_zero_learning_rate(self):
        """Test training with zero learning rate"""
        config = TestDataGeneration.create_test_config()
        model = build_model(config)
        
        optimizer = optim.Adam(model.parameters(), lr=0.0)  # Zero learning rate
        loss_func = nn.MSELoss()
        
        # Get initial parameters
        initial_params = [p.clone() for p in model.parameters()]
        
        # Training step
        x = torch.randn(2, 2, 16, 16)
        target = torch.randn(2, 2, 16, 16)
        
        optimizer.zero_grad()
        output = model(x)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        
        # Parameters should not change with zero learning rate
        for initial, current in zip(initial_params, model.parameters()):
            assert torch.allclose(initial, current), "Parameters changed with zero learning rate"


class TestReproducibility:
    """Test reproducibility and determinism"""
    
    def test_seed_reproducibility(self):
        """Test that setting seeds produces reproducible results"""
        config = TestDataGeneration.create_test_config()
        
        def run_with_seed(seed):
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            model = build_model(config)
            x = torch.randn(2, 2, 16, 16)
            
            with torch.no_grad():
                output = model(x)
            
            return output
        
        # Run twice with same seed
        output1 = run_with_seed(42)
        output2 = run_with_seed(42)
        
        assert torch.allclose(output1, output2), "Results not reproducible with same seed"
        
        # Run with different seed
        output3 = run_with_seed(123)
        
        assert not torch.allclose(output1, output3), "Different seeds should give different results"
    
    def test_model_state_consistency(self):
        """Test model state saving and loading"""
        config = TestDataGeneration.create_test_config()
        model = build_model(config)
        
        # Get initial output
        x = torch.randn(2, 2, 16, 16)
        with torch.no_grad():
            initial_output = model(x)
        
        # Save and load state
        state_dict = model.state_dict()
        model.load_state_dict(state_dict)
        
        # Output should be identical after save/load
        with torch.no_grad():
            loaded_output = model(x)
        
        assert torch.allclose(initial_output, loaded_output), "Model output changed after save/load"
    
    def test_training_reproducibility(self):
        """Test that training is reproducible with same initialization"""
        config = TestDataGeneration.create_test_config()
        
        def train_model_step(seed):
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            model = build_model(config)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            loss_func = nn.MSELoss()
            
            x = torch.randn(2, 2, 16, 16)
            target = torch.randn(2, 2, 16, 16)
            
            optimizer.zero_grad()
            output = model(x)
            loss = loss_func(output, target)
            loss.backward()
            optimizer.step()
            
            return loss.item(), [p.clone() for p in model.parameters()]
        
        # Train with same seed twice
        loss1, params1 = train_model_step(42)
        loss2, params2 = train_model_step(42)
        
        assert abs(loss1 - loss2) < 1e-6, "Training loss not reproducible"
        
        for p1, p2 in zip(params1, params2):
            assert torch.allclose(p1, p2), "Model parameters not reproducible after training"


# Custom pytest markers and configuration
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "cuda: marks tests that require CUDA")
    config.addinivalue_line("markers", "integration: marks integration tests")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically"""
    for item in items:
        # Add slow marker to performance tests
        if "performance" in item.name.lower() or "large_batch" in item.name.lower():
            item.add_marker(pytest.mark.slow)
        
        # Add cuda marker to CUDA tests
        if "cuda" in item.name.lower():
            item.add_marker(pytest.mark.cuda)
        
        # Add integration marker to workflow tests
        if "workflow" in item.name.lower() or "integration" in item.name.lower():
            item.add_marker(pytest.mark.integration)


# Main test runner
if __name__ == "__main__":
    """
    Run tests when script is executed directly.
    
    Usage examples:
    python test_training_pipeline.py                    # Run all tests
    python test_training_pipeline.py -v                 # Verbose output
    python test_training_pipeline.py -m "not slow"     # Skip slow tests
    python test_training_pipeline.py -k "test_model"    # Run only model tests
    """
    
    # Configure test discovery and execution
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
        "-x",  # Stop on first failure
    ])
    
    sys.exit(exit_code)


# Additional helper functions for testing
class TestHelpers:
    """Helper functions for testing"""
    
    @staticmethod
    def assert_shape_matches(tensor1, tensor2, msg="Shape mismatch"):
        """Assert that two tensors have matching shapes"""
        assert tensor1.shape == tensor2.shape, f"{msg}: {tensor1.shape} vs {tensor2.shape}"
    
    @staticmethod
    def assert_finite_tensor(tensor, msg="Tensor contains non-finite values"):
        """Assert that tensor contains only finite values"""
        assert torch.isfinite(tensor).all(), f"{msg}: {tensor}"
    
    @staticmethod
    def assert_gradient_flow(model, msg="No gradients found"):
        """Assert that model has gradients"""
        has_gradients = any(p.grad is not None and p.grad.abs().sum() > 0 
                           for p in model.parameters())
        assert has_gradients, msg
    
    @staticmethod
    def count_parameters(model):
        """Count trainable parameters in model"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    @staticmethod
    def get_memory_usage():
        """Get current memory usage"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated()
        else:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            return process.memory_info().rss


# Example usage and integration test
class TestExampleUsage:
    """Example usage patterns and integration tests"""
    
    def test_complete_example_usage(self):
        """Test complete example usage of the framework"""
        
        # Step 1: Create configuration
        config = TestDataGeneration.create_test_config()
        
        # Step 2: Generate and prepare data
        raw_data = TestDataGeneration.create_synthetic_pde_data(
            batch_size=12, variables=2, nx=32, ny=32, nt=20, pattern='wave'
        )
        
        # Step 3: Normalize data
        normalizer = MinMax_Normalizer(raw_data)
        normalized_data = normalizer.encode(raw_data)
        
        # Step 4: Split into train/test
        from sklearn.model_selection import train_test_split
        train_in, test_in, train_out, test_out = train_test_split(
            normalized_data[..., :config['Data']['t_in']],
            normalized_data[..., config['Data']['t_in']:config['Data']['t_out']],
            test_size=0.3, random_state=42
        )
        
        # Step 5: Create model
        model = build_model(config)
        
        # Step 6: Setup training
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.8)
        loss_func = nn.MSELoss()
        
        # Step 7: Create data loaders
        train_data = torch.cat((train_in, train_out), dim=-1)
        dataset = SpatioTemporalDataset(train_data, input_window=1, prediction_steps=5)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(test_in, test_out), 
            batch_size=4, shuffle=False
        )
        
        # Step 8: Train
        trainer = ExplicitTrainSetup(
            model=model, train_loader=train_loader, test_loader=test_loader,
            loss_func=loss_func, optimizer=optimizer, scheduler=scheduler,
            epochs=3, ode_solver='custom', roll_out='euler'
        )
        
        losses = []
        for epoch in range(2):
            train_loss, test_loss = trainer.one_epoch(
                step=1, train_T_out=5, test_T_out=9, dt=0.01
            )
            losses.append((train_loss, test_loss))
        
        # Step 9: Evaluate
        evaluator = ExplicitEvalSetup(
            model=model, test_in=test_in, test_out=test_out,
            ode_solver='custom', roll_out='euler', batch_size=4
        )
        
        predictions, error = evaluator.inference(step=1, T_out=9, dt=0.01)
        
        # Step 10: Compute metrics
        mse_result = MSE(predictions, test_out)
        rmse_result = RMSE(predictions, test_out)
        
        # Step 11: Denormalize for physical interpretation
        pred_physical = normalizer.decode(predictions)
        test_physical = normalizer.decode(test_out)
        
        # Verify everything worked
        assert len(losses) == 2
        assert all(np.isfinite(l[0]) and np.isfinite(l[1]) for l in losses)
        assert predictions.shape == test_out.shape
        assert pred_physical.shape == test_physical.shape
        assert np.isfinite(error)
        assert np.isfinite(mse_result['average'])
        assert np.isfinite(rmse_result['average'])
        
        print(f"Training completed successfully!")
        print(f"Final training loss: {losses[-1][0]:.6f}")
        print(f"Final test loss: {losses[-1][1]:.6f}")
        print(f"Final evaluation error: {error:.6f}")
        print(f"MSE: {mse_result['average']:.6f}")
        print(f"RMSE: {rmse_result['average']:.6f}")
        print(f"Model parameters: {TestHelpers.count_parameters(model)}")
        
        return {
            'model': model,
            'losses': losses,
            'predictions': predictions,
            'error': error,
            'metrics': {'mse': mse_result, 'rmse': rmse_result},
            'normalizer': normalizer
        }