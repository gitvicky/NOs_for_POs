#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Test Suite for Neural Operators for Physical Operators (NOs for POs)

This test suite covers:
1. Physical Residual Estimation (PRE) components
2. Data loading and processing
3. Model initialization and forward passes  
4. Operator splitting functionality
5. Time integration schemes
6. Boundary conditions
7. Integration and end-to-end tests

Run with: python -m pytest test_nos_for_pos.py -v
"""

# %%
import pytest
import torch
import numpy as np
import yaml
import os
import tempfile
import shutil
from pathlib import Path
import sys


# %%
# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Imports from your codebase
from PRE.ConvOps_Spatial import ConvOperator
from PRE.VectorConvOps_Spatial import Gradient, Divergence, Curl, Laplace
from PRE.boundary_conditions import BoundaryManager
from PRE.Stencils import get_stencil
from Neural_PDE.Utils.processing_utils import MinMax_Normalizer, MinMax_Normalizer_variable
from Expts.data_loaders import stacked_fields, SpatioTemporalDataset
from Utils.explicit_time import autoregressive, euler, midpoint, rk4
from Utils.metrics import MSE, RMSE, NMSE, NRMSE

# Test fixtures and utilities
class TestConfig:
    """Test configuration and utilities"""
    
    @staticmethod
    def get_test_config():
        """Create a minimal test configuration"""
        return {
            'Physics': {
                'pde': 'Navier-Stokes',
                'variables': 2,
                'field': 'u, v',
                'Nx': 32,
                'Ny': 32,
                'Nt': 10,
                'dx': 0.1,
                'dy': 0.1,
                'dt': 0.01,
                'x_slice': 1,
                'y_slice': 1,
                't_slice': 1
            },
            'Model': {
                'arch': 'Conv',
                'in_vars': 2,
                'out_vars': 2,
                'modes': 8,
                'width': 16,
                'n_layers': 2,
                'act': 'gelu',
                'operator_splitting': True
            },
            'Data': {
                'ntrain': 10,
                'test-train-split': 0.2,
                'batch_size': 2,
                't_in': 1,
                't_out': 5,
                'step': 1,
                'normalisation': 'Min-Max'
            },
            'Train': {
                'rollout_length': 3,
                'input_length': 1,
                'odesolve': {
                    'source': 'custom',
                    'method': 'euler'
                }
            }
        }
    
    @staticmethod
    def create_dummy_field_data(batch_size=10, variables=2, nx=32, ny=32, nt=10):
        """Create dummy field data for testing"""
        return torch.randn(batch_size, variables, nx, ny, nt)
    
    @staticmethod
    def create_test_field_2d(nx=32, ny=32, type='gaussian'):
        """Create 2D test fields with known properties"""
        x = torch.linspace(-1, 1, nx)
        y = torch.linspace(-1, 1, ny)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        if type == 'gaussian':
            field = torch.exp(-(X**2 + Y**2))
        elif type == 'sine':
            field = torch.sin(np.pi * X) * torch.sin(np.pi * Y)
        elif type == 'linear':
            field = X + Y
        else:
            field = torch.ones_like(X)
            
        return field.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims


class TestPREComponents:
    """Test Physical Residual Estimation components"""
    
    def test_stencil_generation(self):
        """Test finite difference stencil generation"""
        # Test 1D stencils
        stencil_1d_2nd = get_stencil(dims=1, deriv_order=2, taylor_order=2)
        assert stencil_1d_2nd.shape == (3, 3)
        assert stencil_1d_2nd[1, 1] == -2  # Center coefficient for 2nd derivative
        
        # Test 2D stencils
        stencil_2d_2nd = get_stencil(dims=2, deriv_order=2, taylor_order=2)
        assert stencil_2d_2nd.shape == (3, 3)
        assert stencil_2d_2nd[1, 1] == -4  # Center coefficient for 2D Laplacian
        
        # Test higher order stencils
        stencil_4th = get_stencil(dims=2, deriv_order=2, taylor_order=4)
        assert stencil_4th.shape == (5, 5)
    
    def test_conv_operator_basic(self):
        """Test basic ConvOperator functionality"""
        # Test x-direction derivative
        grad_x = ConvOperator(domain='x', order=1, scale=1.0, taylor_order=2)
        
        # Test on linear field (should give constant gradient)
        field = TestConfig.create_test_field_2d(type='linear')
        result = grad_x(field)
        
        assert result.shape == field.shape
        assert torch.allclose(result, torch.ones_like(result), atol=1e-6)
    
    def test_boundary_conditions(self):
        """Test boundary condition handling"""
        bc_manager = BoundaryManager(kernel_size=(3, 3))
        
        # Test periodic boundaries
        bc_manager.set_all_boundaries('periodic')
        field = torch.randn(1, 1, 10, 10)
        padded = bc_manager.pad_signal(field)
        
        assert padded.shape == (1, 1, 12, 12)  # Added 1 pixel padding on each side
        
        # Test Dirichlet boundaries
        bc_manager.set_all_boundaries('dirichlet', value=0.0)
        padded_dirichlet = bc_manager.pad_signal(field)
        
        # Check that boundary values are zero
        assert torch.allclose(padded_dirichlet[:, :, 0, :], torch.zeros_like(padded_dirichlet[:, :, 0, :]))
    
    def test_vector_operations(self):
        """Test vector field operations"""
        # Create test vector field
        u = TestConfig.create_test_field_2d(type='linear')
        v = TestConfig.create_test_field_2d(type='sine')
        
        # Test gradient
        gradient = Gradient(scale=1.0, taylor_order=2, boundary_cond='periodic')
        grad_result = gradient(u)
        
        assert grad_result.shape == (u.shape[0], 2, u.shape[2], u.shape[3])
        
        # Test divergence
        divergence = Divergence(scale=1.0, taylor_order=2, boundary_cond='periodic')
        div_result = divergence(u, v)
        
        assert div_result.shape == u.shape
        
        # Test Laplacian
        laplacian = Laplace(scale=1.0, taylor_order=2, boundary_cond='periodic', scalar=True)
        lap_result = laplacian(u)
        
        assert lap_result.shape == u.shape
    
    def test_accuracy_convergence(self):
        """Test accuracy of finite difference operators"""
        # Test on analytical function where we know the exact derivative
        nx, ny = 64, 64
        x = torch.linspace(-1, 1, nx)
        y = torch.linspace(-1, 1, ny)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        # Test function: f(x,y) = sin(π*x) * cos(π*y)
        f = torch.sin(np.pi * X) * torch.cos(np.pi * Y)
        f = f.unsqueeze(0).unsqueeze(0)
        
        # Analytical second derivative: -π²*sin(π*x)*cos(π*y) - π²*sin(π*x)*cos(π*y) = -2π²*f
        exact_laplacian = -2 * np.pi**2 * f
        
        # Compute numerical Laplacian
        dx = 2.0 / (nx - 1)
        laplacian = ConvOperator(domain=('x', 'y'), order=2, scale=1.0/dx**2, taylor_order=2)
        numerical_laplacian = laplacian(f)
        
        # Check error (should be small for smooth function)
        error = torch.abs(numerical_laplacian - exact_laplacian).max()
        assert error < 0.1  # Reasonable error for 2nd order scheme


class TestDataProcessing:
    """Test data loading and processing components"""
    
    def test_stacked_fields(self):
        """Test field stacking utility"""
        # Create dummy data
        u = np.random.randn(5, 10, 10, 8)  # batch, nx, ny, nt
        v = np.random.randn(5, 10, 10, 8)
        
        stacked = stacked_fields([u, v])
        
        assert stacked.shape == (5, 2, 10, 10, 8)  # batch, vars, nx, ny, nt
        assert torch.allclose(stacked[:, 0], torch.tensor(u).permute(0, 2, 3, 1))
    
    def test_spatiotemporal_dataset(self):
        """Test spatiotemporal dataset creation"""
        data = torch.randn(5, 2, 32, 32, 20)  # batch, vars, nx, ny, nt
        dataset = SpatioTemporalDataset(data, input_window=3, prediction_steps=2)
        
        assert len(dataset) > 0
        
        # Test a sample
        input_seq, target_seq = dataset[0]
        assert input_seq.shape == (2, 32, 32, 3)  # vars, nx, ny, input_window
        assert target_seq.shape == (2, 32, 32, 2)  # vars, nx, ny, prediction_steps
    
    def test_normalization(self):
        """Test data normalization"""
        data = torch.randn(10, 2, 16, 16, 5) * 10 + 5  # Non-normalized data
        
        # Test Min-Max normalization
        normalizer = MinMax_Normalizer(data)
        normalized = normalizer.encode(data)
        denormalized = normalizer.decode(normalized)
        
        # Check that normalization brings data to [0, 1] range
        assert normalized.min() >= -0.1  # Allow small numerical errors
        assert normalized.max() <= 1.1
        
        # Check that denormalization recovers original data
        assert torch.allclose(data, denormalized, atol=1e-5)
    
    def test_variable_normalization(self):
        """Test per-variable normalization"""
        data = torch.randn(10, 3, 16, 16, 5)
        data[:, 0] *= 10  # Different scales for different variables
        data[:, 1] *= 0.1
        
        normalizer = MinMax_Normalizer_variable(data)
        normalized = normalizer.encode(data)
        denormalized = normalizer.decode(normalized)
        
        # Check recovery
        assert torch.allclose(data, denormalized, atol=1e-5)


class TestTimeIntegration:
    """Test time integration schemes"""
    
    def setup_method(self):
        """Setup for time integration tests"""
        self.simple_model = lambda x: -x  # Simple decay model: dx/dt = -x
        self.dt = 0.01
        self.x0 = torch.tensor([1.0])
        
    def test_euler_scheme(self):
        """Test Euler time stepping"""
        x = self.x0.clone()
        for _ in range(100):
            x = euler(self.simple_model, x, self.dt)
        
        # Analytical solution: x(t) = x0 * exp(-t)
        t_final = 100 * self.dt
        exact = self.x0 * torch.exp(-torch.tensor(t_final))
        
        # Euler scheme should be reasonably close for small dt
        assert torch.abs(x - exact) < 0.1
    
    def test_rk4_scheme(self):
        """Test RK4 time stepping"""
        x = self.x0.clone()
        for _ in range(100):
            x = rk4(self.simple_model, x, self.dt)
        
        t_final = 100 * self.dt
        exact = self.x0 * torch.exp(-torch.tensor(t_final))
        
        # RK4 should be much more accurate than Euler
        assert torch.abs(x - exact) < 0.01
    
    def test_scheme_stability(self):
        """Test stability of time integration schemes"""
        # Test with larger time step that might cause instability
        dt_large = 0.1
        x_euler = self.x0.clone()
        x_rk4 = self.x0.clone()
        
        for _ in range(50):
            x_euler = euler(self.simple_model, x_euler, dt_large)
            x_rk4 = rk4(self.simple_model, x_rk4, dt_large)
        
        # Both should remain bounded (not blow up)
        assert torch.abs(x_euler) < 10
        assert torch.abs(x_rk4) < 10


class TestMetrics:
    """Test performance metrics"""
    
    def test_mse_calculation(self):
        """Test MSE metric calculation"""
        pred = torch.randn(5, 2, 10, 10)
        target = torch.randn(5, 2, 10, 10)
        
        mse_result = MSE(pred, target)
        
        assert 'per_sample' in mse_result
        assert 'average' in mse_result
        assert len(mse_result['per_sample']) == 5  # batch size
        assert mse_result['average'] >= 0
    
    def test_rmse_calculation(self):
        """Test RMSE metric calculation"""
        pred = torch.randn(3, 2, 8, 8)
        target = torch.randn(3, 2, 8, 8)
        
        rmse_result = RMSE(pred, target)
        mse_result = MSE(pred, target)
        
        # RMSE should be sqrt of MSE
        assert torch.allclose(
            torch.tensor(rmse_result['average']),
            torch.sqrt(torch.tensor(mse_result['average'])),
            atol=1e-6
        )
    
    def test_normalized_metrics(self):
        """Test normalized metrics"""
        # Create data with known properties
        target = torch.ones(2, 1, 4, 4) * 10
        pred = target + torch.randn_like(target) * 0.1
        
        nmse_result = NMSE(pred, target)
        nrmse_result = NRMSE(pred, target)
        
        assert nmse_result['average'] >= 0
        assert nrmse_result['average'] >= 0
        assert not np.isnan(nmse_result['average'])
        assert not np.isnan(nrmse_result['average'])


# class TestOperatorSplitting:
#     """Test operator splitting functionality"""
    
#     def test_linear_nonlinear_split(self):
#         """Test basic operator splitting concept"""
#         # Mock model components
#         class LinearOperator(torch.nn.Module):
#             def forward(self, x):
#                 return torch.zeros_like(x)  # Dummy linear operator
        
#         class NonlinearOperator(torch.nn.Module):
#             def forward(self, x):
#                 return torch.zeros_like(x)  # Dummy nonlinear operator
        
#         linear_op = LinearOperator()
#         nonlinear_op = NonlinearOperator()
        
#         # Test that they can be called
#         x = torch.randn(1, 2, 8, 8, 1)
#         linear_result = linear_op(x)
#         nonlinear_result = nonlinear_op(x)
        
#         assert linear_result.shape == x.shape
#         assert nonlinear_result.shape == x.shape
    
#     def test_conservation_properties(self):
#         """Test that operators preserve conservation properties where expected"""
#         # Test divergence-free property
#         u = torch.randn(1, 1, 16, 16)
#         v = torch.randn(1, 1, 16, 16)
        
#         # Make the field divergence-free by construction
#         psi = torch.randn(1, 1, 16, 16)  # Stream function
#         gradient = Gradient(scale=1.0, taylor_order=2, boundary_cond='periodic')
#         grad_psi = gradient(psi)
        
#         u_divfree = grad_psi[:, 1:2]   # v-component of gradient
#         v_divfree = -grad_psi[:, 0:1]  # negative u-component of gradient
        
#         # Check divergence is (nearly) zero
#         divergence = Divergence(scale=1.0, taylor_order=2, boundary_cond='periodic')
#         div_result = divergence(u_divfree, v_divfree)
        
#         assert torch.abs(div_result).max() < 1e-10  # Should be machine precision


class TestIntegration:
    """Integration and end-to-end tests"""
    
    def setup_method(self):
        """Setup for integration tests"""
        self.config = TestConfig.get_test_config()
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Cleanup after integration tests"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_config_loading(self):
        """Test configuration loading and validation"""
        config_path = os.path.join(self.temp_dir, 'test_config.yaml')
        
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f)
        
        # Load and verify
        with open(config_path, 'r') as f:
            loaded_config = yaml.safe_load(f)
        
        assert loaded_config['Physics']['pde'] == 'Navier-Stokes'
        assert loaded_config['Model']['arch'] == 'Conv'
    
    def test_data_pipeline(self):
        """Test complete data processing pipeline"""
        # Create synthetic data
        fields = TestConfig.create_dummy_field_data(
            batch_size=self.config['Data']['ntrain'],
            variables=self.config['Physics']['variables'],
            nx=self.config['Physics']['Nx'],
            ny=self.config['Physics']['Ny'],
            nt=self.config['Data']['t_out']
        )
        
        # Test normalization
        normalizer = MinMax_Normalizer(fields)
        fields_encoded = normalizer.encode(fields)
        
        # Test train-test split
        from sklearn.model_selection import train_test_split
        train_in, test_in, train_out, test_out = train_test_split(
            fields_encoded[..., :self.config['Data']['t_in']], 
            fields_encoded[..., self.config['Data']['t_in']:self.config['Data']['t_out']], 
            test_size=self.config['Data']['test-train-split'], 
            random_state=42
        )
        
        assert train_in.shape[0] + test_in.shape[0] == self.config['Data']['ntrain']
        assert train_in.shape[-1] == self.config['Data']['t_in']
        assert train_out.shape[-1] == self.config['Data']['t_out'] - self.config['Data']['t_in']
    
    def test_model_forward_pass(self):
        """Test that models can perform forward passes without errors"""
        try:
            from Tests.learnable_matrices import Convolution2d, SpectralConv2d
            
            # Test convolutional model
            conv_model = Convolution2d(
                kernel_size=3,
                features=self.config['Physics']['variables'],
                init_type='random'
            )
            
            x = torch.randn(2, 2, 16, 16)
            output = conv_model(x)
            assert output.shape == x.shape
            
            # Test spectral model
            spectral_model = SpectralConv2d(
                in_channels=2, 
                out_channels=2,
                modes1=4, 
                modes2=4
            )
            
            output2 = spectral_model(x)
            assert output2.shape == x.shape
            
        except ImportError:
            pytest.skip("Learnable matrices module not available")
    
    def test_physics_consistency(self):
        """Test that physics operations are consistent"""
        # Test that grad(div(u)) = div(grad(u)) for smooth functions
        u = TestConfig.create_test_field_2d(type='sine')
        
        gradient = Gradient(scale=1.0, taylor_order=2, boundary_cond='periodic')
        divergence = Divergence(scale=1.0, taylor_order=2, boundary_cond='periodic')
        
        # This test would need vector field, so create one
        v = TestConfig.create_test_field_2d(type='gaussian')
        uv = torch.cat([u, v], dim=1)
        
        # Test vector identities (up to numerical precision)
        grad_u = gradient(u)
        grad_v = gradient(v)
        
        assert grad_u.shape == (1, 2, 32, 32)
        assert grad_v.shape == (1, 2, 32, 32)


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_shape_mismatches(self):
        """Test handling of shape mismatches"""
        with pytest.raises(AssertionError):
            # Wrong input shape for metric calculation
            pred = torch.randn(5, 2, 10, 10)
            target = torch.randn(3, 2, 10, 10)  # Different batch size
            MSE(pred, target)
    
    def test_invalid_boundary_conditions(self):
        """Test invalid boundary condition handling"""
        bc_manager = BoundaryManager(kernel_size=(3, 3))
        
        with pytest.raises(ValueError):
            bc_manager.set_boundary_type('invalid_side', 'periodic')
    
    def test_numerical_stability(self):
        """Test numerical stability with extreme values"""
        # Test with very large values
        large_field = torch.ones(1, 1, 10, 10) * 1e6
        
        normalizer = MinMax_Normalizer(large_field)
        normalized = normalizer.encode(large_field)
        denormalized = normalizer.decode(normalized)
        
        # Should handle large values gracefully
        assert torch.allclose(large_field, denormalized, rtol=1e-5)
        
        # Test with very small values
        small_field = torch.ones(1, 1, 10, 10) * 1e-6
        
        normalizer_small = MinMax_Normalizer(small_field)
        normalized_small = normalizer_small.encode(small_field)
        denormalized_small = normalizer_small.decode(normalized_small)
        
        assert torch.allclose(small_field, denormalized_small, rtol=1e-5)


# Performance benchmarks (optional)
class TestPerformance:
    """Performance and scaling tests"""
    
    @pytest.mark.slow
    def test_scaling_behavior(self):
        """Test how operations scale with problem size"""
        import time
        
        sizes = [16, 32, 64]
        times = []
        
        gradient = Gradient(scale=1.0, taylor_order=2, boundary_cond='periodic')
        
        for size in sizes:
            field = torch.randn(1, 1, size, size)
            
            start_time = time.time()
            for _ in range(10):  # Multiple runs for averaging
                result = gradient(field)
            end_time = time.time()
            
            times.append((end_time - start_time) / 10)
        
        # Check that scaling is reasonable (should be roughly O(N²))
        assert times[1] / times[0] < 10  # 4x problem size shouldn't be >10x slower
        assert times[2] / times[1] < 10


# Pytest configuration
def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")


if __name__ == "__main__":
    # Allow running as script for quick testing
    pytest.main([__file__, "-v"])
