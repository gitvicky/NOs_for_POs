#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pytest Test Suite for ConvOperator Finite Differences

Comprehensive comparison between custom ConvOperator implementation and findiff package.
Modified to use difference-based assertions instead of error ratios.
"""
import os 
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

tolerance = 1e-3  # Global tolerance for numerical comparisons

import pytest
import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import warnings
from typing import Tuple, List, Dict, Any, Callable

# Try to import findiff
try:
    from findiff import FinDiff
    FINDIFF_AVAILABLE = True
except ImportError:
    FINDIFF_AVAILABLE = False

# Try to import your ConvOperator
try:
    import sys
    sys.path.append('..')
    from PRE.ConvOps_Spatial import ConvOperator
    CONVOPERATOR_AVAILABLE = True
except ImportError:
    CONVOPERATOR_AVAILABLE = False


@pytest.fixture(scope="session")
def device():
    """Fixture to determine device (CPU/CUDA)"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def test_functions():
    """Fixture providing analytical test functions and their derivatives"""
    
    functions = {
        'polynomial': {
            'func': lambda x, y: x**4 + 2*x**3*y + x*y**3 + y**4,
            'dfdx': lambda x, y: 4*x**3 + 6*x**2*y + y**3,
            'dfdy': lambda x, y: 2*x**3 + 3*x*y**2 + 4*y**3,
            'd2fdx2': lambda x, y: 12*x**2 + 12*x*y,
            'd2fdy2': lambda x, y: 6*x*y + 12*y**2,
            'd2fdxdy': lambda x, y: 6*x**2 + 3*y**2,
            'laplacian': lambda x, y: 12*x**2 + 12*x*y + 6*x*y + 12*y**2,
            'description': 'High-order polynomial'
        },
        'gaussian': {
            'func': lambda x, y: np.exp(-(x**2 + y**2)),
            'dfdx': lambda x, y: -2*x * np.exp(-(x**2 + y**2)),
            'dfdy': lambda x, y: -2*y * np.exp(-(x**2 + y**2)),
            'd2fdx2': lambda x, y: (4*x**2 - 2) * np.exp(-(x**2 + y**2)),
            'd2fdy2': lambda x, y: (4*y**2 - 2) * np.exp(-(x**2 + y**2)),
            'd2fdxdy': lambda x, y: 4*x*y * np.exp(-(x**2 + y**2)),
            'laplacian': lambda x, y: (4*x**2 + 4*y**2 - 4) * np.exp(-(x**2 + y**2)),
            'description': 'Smooth Gaussian function'
        },
        'trigonometric': {
            'func': lambda x, y: np.sin(2*np.pi*x) * np.cos(2*np.pi*y),
            'dfdx': lambda x, y: 2*np.pi * np.cos(2*np.pi*x) * np.cos(2*np.pi*y),
            'dfdy': lambda x, y: -2*np.pi * np.sin(2*np.pi*x) * np.sin(2*np.pi*y),
            'd2fdx2': lambda x, y: -4*np.pi**2 * np.sin(2*np.pi*x) * np.cos(2*np.pi*y),
            'd2fdy2': lambda x, y: -4*np.pi**2 * np.sin(2*np.pi*x) * np.cos(2*np.pi*y),
            'd2fdxdy': lambda x, y: -4*np.pi**2 * np.cos(2*np.pi*x) * np.sin(2*np.pi*y),
            'laplacian': lambda x, y: -8*np.pi**2 * np.sin(2*np.pi*x) * np.cos(2*np.pi*y),
            'description': 'Oscillatory trigonometric function'
        },
        'exponential': {
            'func': lambda x, y: np.exp(x + 0.5*y),
            'dfdx': lambda x, y: np.exp(x + 0.5*y),
            'dfdy': lambda x, y: 0.5 * np.exp(x + 0.5*y),
            'd2fdx2': lambda x, y: np.exp(x + 0.5*y),
            'd2fdy2': lambda x, y: 0.25 * np.exp(x + 0.5*y),
            'd2fdxdy': lambda x, y: 0.5 * np.exp(x + 0.5*y),
            'laplacian': lambda x, y: 1.25 * np.exp(x + 0.5*y),
            'description': 'Exponential function'
        }
    }
    
    return functions


@pytest.fixture
def create_test_grid():
    """Fixture to create test grids"""
    def _create_grid(nx: int = 101, ny: int = 101, 
                    domain: Tuple[float, float, float, float] = (-2, 2, -2, 2)):
        x_min, x_max, y_min, y_max = domain
        x = np.linspace(x_min, x_max, nx)
        y = np.linspace(y_min, y_max, ny)
        X, Y = np.meshgrid(x, y, indexing='ij')
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        return X, Y, dx, dy
    
    return _create_grid


class TestFirstDerivatives:
    """Test class for first derivative operations"""
    
    @pytest.mark.skipif(not CONVOPERATOR_AVAILABLE, reason="ConvOperator not available")
    @pytest.mark.skipif(not FINDIFF_AVAILABLE, reason="findiff package not available")
    @pytest.mark.parametrize("func_name", ['polynomial', 'gaussian', 'trigonometric'])
    @pytest.mark.parametrize("taylor_order", [2, 4, 6])
    @pytest.mark.parametrize("grid_size", [51, 101])
    def test_first_derivative_x_accuracy(self, test_functions, create_test_grid, device, 
                                        func_name, taylor_order, grid_size):
        """Test first derivative in x-direction accuracy"""
        
        # Create test data
        X, Y, dx, dy = create_test_grid(grid_size, grid_size)
        test_func = test_functions[func_name]
        
        f = test_func['func'](X, Y)
        dfdx_analytical = test_func['dfdx'](X, Y)
        
        # Convert to torch tensor
        f_torch = torch.tensor(f, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        
        # Your method
        conv_op_x = ConvOperator(scale=1/dx, domain='x', order=1, 
                                taylor_order=taylor_order, device=device)
        dfdx_yours = conv_op_x(f_torch)[0, 0].detach().cpu().numpy()
        
        # findiff method
        d_dx = FinDiff(0, dx, 1, acc=taylor_order)
        dfdx_findiff = d_dx(f)
        
        # Calculate cropped regions (excluding boundary points)
        pad = taylor_order // 2
        if dfdx_yours.shape != dfdx_analytical.shape:
            offset = (dfdx_analytical.shape[0] - dfdx_yours.shape[0]) // 2
            dfdx_analytical_cropped = dfdx_analytical[offset:-offset, offset:-offset]
        else:
            dfdx_analytical_cropped = dfdx_analytical[pad:-pad, pad:-pad]
            dfdx_yours = dfdx_yours[pad:-pad, pad:-pad]
        
        dfdx_findiff_cropped = dfdx_findiff[pad:-pad, pad:-pad]
        
        # Check difference between your method and findiff
        diff_yours_findiff = np.abs(dfdx_yours - dfdx_findiff_cropped)
        max_diff = np.max(diff_yours_findiff)
        
        assert max_diff < tolerance, f"Difference between your method and findiff too large: {max_diff}"
        
        # Optional: Still check that both methods are reasonably accurate
        error_yours = np.sqrt(np.mean((dfdx_yours - dfdx_analytical_cropped)**2))
        error_findiff = np.sqrt(np.mean((dfdx_findiff_cropped - dfdx_analytical_cropped)**2))
        
        assert error_yours < 1e0, f"Your method error too high: {error_yours:.2e}"
        assert error_findiff < 1e0, f"findiff error too high: {error_findiff:.2e}"
        
        # For polynomial functions with sufficient order, error should be very small
        if func_name == 'polynomial' and taylor_order >= 4:
            assert error_yours < tolerance, f"Polynomial should be exact: {error_yours:.2e}"
    
    @pytest.mark.skipif(not CONVOPERATOR_AVAILABLE, reason="ConvOperator not available")
    @pytest.mark.skipif(not FINDIFF_AVAILABLE, reason="findiff package not available")
    @pytest.mark.parametrize("func_name", ['polynomial', 'gaussian'])
    @pytest.mark.parametrize("taylor_order", [2, 4, 6])
    def test_first_derivative_y_accuracy(self, test_functions, create_test_grid, device,
                                        func_name, taylor_order):
        """Test first derivative in y-direction accuracy"""
        
        X, Y, dx, dy = create_test_grid(101, 101)
        test_func = test_functions[func_name]
        
        f = test_func['func'](X, Y)
        dfdy_analytical = test_func['dfdy'](X, Y)
        
        f_torch = torch.tensor(f, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        
        # Your method
        conv_op_y = ConvOperator(scale=1/dy, domain='y', order=1, 
                                taylor_order=taylor_order, device=device)
        dfdy_yours = conv_op_y(f_torch)[0, 0].detach().cpu().numpy()
        
        # findiff method
        d_dy = FinDiff(1, dy, 1, acc=taylor_order)
        dfdy_findiff = d_dy(f)
        
        # Calculate cropped regions
        pad = taylor_order // 2
        if dfdy_yours.shape != dfdy_analytical.shape:
            offset = (dfdy_analytical.shape[0] - dfdy_yours.shape[0]) // 2
            dfdy_analytical_cropped = dfdy_analytical[offset:-offset, offset:-offset]
        else:
            dfdy_analytical_cropped = dfdy_analytical[pad:-pad, pad:-pad]
            dfdy_yours = dfdy_yours[pad:-pad, pad:-pad]
        
        dfdy_findiff_cropped = dfdy_findiff[pad:-pad, pad:-pad]
        
        # Check difference between your method and findiff
        diff_yours_findiff = np.abs(dfdy_yours - dfdy_findiff_cropped)
        max_diff = np.max(diff_yours_findiff)
        
        assert max_diff < tolerance, f"Difference between your method and findiff too large: {max_diff}"
        
        # Optional: Still check that both methods are reasonably accurate
        error_yours = np.sqrt(np.mean((dfdy_yours - dfdy_analytical_cropped)**2))
        error_findiff = np.sqrt(np.mean((dfdy_findiff_cropped - dfdy_analytical_cropped)**2))
        
        assert error_yours < 1e0, f"Your method error too high: {error_yours:.2e}"
        assert error_findiff < 1e0, f"findiff error too high: {error_findiff:.2e}"


class TestSecondDerivatives:
    """Test class for second derivative operations"""
    
    @pytest.mark.skipif(not CONVOPERATOR_AVAILABLE, reason="ConvOperator not available")
    @pytest.mark.skipif(not FINDIFF_AVAILABLE, reason="findiff package not available")
    @pytest.mark.parametrize("func_name", ['polynomial', 'gaussian'])
    @pytest.mark.parametrize("taylor_order", [2, 4, 6])
    def test_laplacian_accuracy(self, test_functions, create_test_grid, device,
                               func_name, taylor_order):
        """Test Laplacian (2D second derivative) accuracy"""
        
        X, Y, dx, dy = create_test_grid(101, 101)
        test_func = test_functions[func_name]
        
        f = test_func['func'](X, Y)
        laplacian_analytical = test_func['laplacian'](X, Y)
        
        f_torch = torch.tensor(f, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        
        # Your method - 2D Laplacian
        conv_op_laplacian = ConvOperator(scale=1.0/dx**2, domain=('x', 'y'), order=2, 
                                        taylor_order=taylor_order, device=device)
        laplacian_yours = conv_op_laplacian(f_torch)[0, 0].detach().cpu().numpy()
        
        # findiff method
        laplacian_op = FinDiff(0, dx, 2, acc=taylor_order) + FinDiff(1, dy, 2, acc=taylor_order)
        laplacian_findiff = laplacian_op(f)
        
        # Calculate cropped regions
        pad = taylor_order // 2
        if laplacian_yours.shape != laplacian_analytical.shape:
            offset = (laplacian_analytical.shape[0] - laplacian_yours.shape[0]) // 2
            laplacian_analytical_cropped = laplacian_analytical[offset:-offset, offset:-offset]
        else:
            laplacian_analytical_cropped = laplacian_analytical[pad:-pad, pad:-pad]
            laplacian_yours = laplacian_yours[pad:-pad, pad:-pad]
        
        laplacian_findiff_cropped = laplacian_findiff[pad:-pad, pad:-pad]
        
        # Check difference between your method and findiff
        diff_yours_findiff = np.abs(laplacian_yours - laplacian_findiff_cropped)
        max_diff = np.max(diff_yours_findiff)
        
        assert max_diff < tolerance, f"Difference between your method and findiff too large: {max_diff}"
        
        # Optional: Still check that both methods are reasonably accurate
        error_yours = np.sqrt(np.mean((laplacian_yours - laplacian_analytical_cropped)**2))
        error_findiff = np.sqrt(np.mean((laplacian_findiff_cropped - laplacian_analytical_cropped)**2))
        
        assert error_yours < 1e0, f"Your method error too high: {error_yours:.2e}"
        assert error_findiff < 1e0, f"findiff error too high: {error_findiff:.2e}"
        
        # For polynomial with sufficient order, should be very accurate
        if func_name == 'polynomial' and taylor_order >= 4:
            assert error_yours < tolerance, f"Polynomial Laplacian should be very accurate: {error_yours:.2e}"
    
    @pytest.mark.skipif(not CONVOPERATOR_AVAILABLE, reason="ConvOperator not available")
    def test_laplacian_edge_cases(self, device):
        """Test Laplacian with edge cases"""
        
        # Test with zero function
        zeros = torch.zeros(1, 1, 50, 50, device=device)
        conv_op = ConvOperator(scale=1.0, domain=('x', 'y'), order=2, 
                              taylor_order=4, device=device)
        result = conv_op(zeros)
        
        assert torch.allclose(result, torch.zeros_like(result), atol=tolerance)
        
        # Test with constant function
        ones = torch.ones(1, 1, 50, 50, device=device)
        result = conv_op(ones)
        
        # Laplacian of constant should be zero
        assert torch.allclose(result, torch.zeros_like(result), atol=tolerance)


class TestConvergenceAnalysis:
    """Test class for convergence analysis"""
    
    @pytest.mark.skipif(not CONVOPERATOR_AVAILABLE, reason="ConvOperator not available")
    @pytest.mark.skipif(not FINDIFF_AVAILABLE, reason="findiff package not available")
    @pytest.mark.parametrize("func_name", ['gaussian'])
    @pytest.mark.parametrize("taylor_order", [4, 6])
    @pytest.mark.slow
    def test_convergence_first_derivative(self, test_functions, create_test_grid, device,
                                         func_name, taylor_order):
        """Test convergence behavior for first derivatives with grid refinement"""
        
        grid_sizes = [21, 41, 81]
        errors_yours = []
        errors_findiff = []
        grid_spacings = []
        
        test_func = test_functions[func_name]
        
        for grid_size in grid_sizes:
            X, Y, dx, dy = create_test_grid(grid_size, grid_size)
            
            f = test_func['func'](X, Y)
            dfdx_analytical = test_func['dfdx'](X, Y)
            
            # Your method
            f_torch = torch.tensor(f, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            conv_op = ConvOperator(scale=1/dx, domain='x', order=1, 
                                  taylor_order=taylor_order, device=device)
            dfdx_yours = conv_op(f_torch)[0, 0].detach().cpu().numpy()
            
            # findiff method
            d_dx = FinDiff(0, dx, 1, acc=taylor_order)
            dfdx_findiff = d_dx(f)
            
            # Calculate errors
            pad = taylor_order // 2
            if dfdx_yours.shape != dfdx_analytical.shape:
                offset = (dfdx_analytical.shape[0] - dfdx_yours.shape[0]) // 2
                dfdx_analytical_cropped = dfdx_analytical[offset:-offset, offset:-offset]
            else:
                dfdx_analytical_cropped = dfdx_analytical[pad:-pad, pad:-pad]
                dfdx_yours = dfdx_yours[pad:-pad, pad:-pad]
            
            dfdx_findiff_cropped = dfdx_findiff[pad:-pad, pad:-pad]
            
            error_yours = np.sqrt(np.mean((dfdx_yours - dfdx_analytical_cropped)**2))
            error_findiff = np.sqrt(np.mean((dfdx_findiff_cropped - dfdx_analytical_cropped)**2))
            
            errors_yours.append(error_yours)
            errors_findiff.append(error_findiff)
            grid_spacings.append(dx)
        
        # Check convergence: errors should decrease with finer grid
        assert errors_yours[1] < errors_yours[0], "Error should decrease with grid refinement"
        assert errors_yours[2] < errors_yours[1], "Error should continue decreasing"
        assert errors_findiff[1] < errors_findiff[0], "findiff error should also decrease"
        
        # Estimate convergence rate
        log_dx = np.log(np.array(grid_spacings))
        log_err_yours = np.log(np.array(errors_yours))
        
        # Should have approximately the expected convergence rate
        if len(log_dx) >= 3:
            rate_yours = np.polyfit(log_dx, log_err_yours, 1)[0]
            # For first derivatives, expect rate close to taylor_order
            assert rate_yours > 1.0, f"Convergence rate too low: {rate_yours:.2f}"
    
    @pytest.mark.skipif(not CONVOPERATOR_AVAILABLE, reason="ConvOperator not available")
    @pytest.mark.skipif(not FINDIFF_AVAILABLE, reason="findiff package not available")
    @pytest.mark.slow
    def test_convergence_laplacian(self, test_functions, create_test_grid, device):
        """Test convergence behavior for Laplacian"""
        
        grid_sizes = [21, 41, 81]
        errors_yours = []
        errors_findiff = []
        
        test_func = test_functions['gaussian']
        taylor_order = 4
        
        for grid_size in grid_sizes:
            X, Y, dx, dy = create_test_grid(grid_size, grid_size)
            
            f = test_func['func'](X, Y)
            laplacian_analytical = test_func['laplacian'](X, Y)
            
            # Your method
            f_torch = torch.tensor(f, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            conv_op = ConvOperator(scale=1.0/dx**2, domain=('x', 'y'), order=2, 
                                  taylor_order=taylor_order, device=device)
            laplacian_yours = conv_op(f_torch)[0, 0].detach().cpu().numpy()
            
            # findiff method
            laplacian_op = FinDiff(0, dx, 2, acc=taylor_order) + FinDiff(1, dy, 2, acc=taylor_order)
            laplacian_findiff = laplacian_op(f)
            
            # Calculate errors
            pad = taylor_order // 2
            if laplacian_yours.shape != laplacian_analytical.shape:
                offset = (laplacian_analytical.shape[0] - laplacian_yours.shape[0]) // 2
                laplacian_analytical_cropped = laplacian_analytical[offset:-offset, offset:-offset]
            else:
                laplacian_analytical_cropped = laplacian_analytical[pad:-pad, pad:-pad]
                laplacian_yours = laplacian_yours[pad:-pad, pad:-pad]
            
            laplacian_findiff_cropped = laplacian_findiff[pad:-pad, pad:-pad]
            
            error_yours = np.sqrt(np.mean((laplacian_yours - laplacian_analytical_cropped)**2))
            error_findiff = np.sqrt(np.mean((laplacian_findiff_cropped - laplacian_analytical_cropped)**2))
            
            errors_yours.append(error_yours)
            errors_findiff.append(error_findiff)
        
        # Check convergence
        assert errors_yours[1] < errors_yours[0], "Laplacian error should decrease with refinement"
        assert errors_yours[2] < errors_yours[1], "Laplacian error should continue decreasing"


@pytest.mark.benchmark
class TestPerformance:
    """Test class for performance comparisons"""
    
    @pytest.mark.skipif(not CONVOPERATOR_AVAILABLE, reason="ConvOperator not available")
    @pytest.mark.parametrize("grid_size", [101, 201])
    def test_first_derivative_performance(self, create_test_grid, device, grid_size):
        """Test first derivative performance"""
        
        X, Y, dx, dy = create_test_grid(grid_size, grid_size)
        f = np.exp(-(X**2 + Y**2))
        f_torch = torch.tensor(f, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        
        conv_op = ConvOperator(scale=1/dx, domain='x', order=1, 
                              taylor_order=4, device=device)
        
        # Warm up
        _ = conv_op(f_torch)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Time the operation
        num_runs = 10
        start_time = time.time()
        for _ in range(num_runs):
            result = conv_op(f_torch)
            if device.type == 'cuda':
                torch.cuda.synchronize()
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs
        
        # Performance should be reasonable
        max_time = 0.1 * (grid_size / 100)**2  # Rough scaling estimate
        assert avg_time < max_time, f"Performance too slow: {avg_time:.4f}s for {grid_size}x{grid_size}"
    
    @pytest.mark.skipif(not CONVOPERATOR_AVAILABLE or not FINDIFF_AVAILABLE, 
                       reason="Both ConvOperator and findiff required")
    @pytest.mark.parametrize("grid_size", [51, 101])
    def test_performance_comparison(self, create_test_grid, device, grid_size):
        """Compare performance between your method and findiff"""
        
        X, Y, dx, dy = create_test_grid(grid_size, grid_size)
        f = np.exp(-(X**2 + Y**2))
        f_torch = torch.tensor(f, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        
        # Time your method
        conv_op = ConvOperator(scale=1/dx, domain='x', order=1, 
                              taylor_order=4, device=device)
        
        # Warm up
        _ = conv_op(f_torch)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        num_runs = 5
        start_time = time.time()
        for _ in range(num_runs):
            result_yours = conv_op(f_torch)
            if device.type == 'cuda':
                torch.cuda.synchronize()
        time_yours = (time.time() - start_time) / num_runs
        
        # Time findiff method
        d_dx = FinDiff(0, dx, 1, acc=4)
        
        start_time = time.time()
        for _ in range(num_runs):
            result_findiff = d_dx(f)
        time_findiff = (time.time() - start_time) / num_runs
        
        # Both should complete in reasonable time
        assert time_yours < 1.0, f"Your method too slow: {time_yours:.4f}s"
        assert time_findiff < 1.0, f"findiff too slow: {time_findiff:.4f}s"
        
        # Calculate speedup factor
        speedup = time_findiff / time_yours
        
        # Log the results (will show in verbose mode)
        print(f"\nGrid {grid_size}x{grid_size}: Yours={time_yours:.4f}s, "
              f"findiff={time_findiff:.4f}s, speedup={speedup:.2f}x")


class TestComparisonMethods:
    """Test class for comprehensive method comparisons"""
    
    @pytest.mark.skipif(not CONVOPERATOR_AVAILABLE or not FINDIFF_AVAILABLE,
                       reason="Both ConvOperator and findiff required")
    @pytest.mark.parametrize("func_name", ['polynomial', 'gaussian'])
    @pytest.mark.parametrize("derivative_type", ['first_x', 'first_y', 'laplacian'])
    @pytest.mark.parametrize("taylor_order", [4, 6])
    def test_accuracy_comparison(self, test_functions, create_test_grid, device,
                                func_name, derivative_type, taylor_order):
        """Comprehensive accuracy comparison between methods"""
        
        X, Y, dx, dy = create_test_grid(81, 81)
        test_func = test_functions[func_name]
        
        f = test_func['func'](X, Y)
        f_torch = torch.tensor(f, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        
        if derivative_type == 'first_x':
            analytical = test_func['dfdx'](X, Y)
            
            # Your method
            conv_op = ConvOperator(scale=1/dx, domain='x', order=1, 
                                  taylor_order=taylor_order, device=device)
            result_yours = conv_op(f_torch)[0, 0].detach().cpu().numpy()
            
            # findiff method
            op_findiff = FinDiff(0, dx, 1, acc=taylor_order)
            result_findiff = op_findiff(f)
            
        elif derivative_type == 'first_y':
            analytical = test_func['dfdy'](X, Y)
            
            # Your method
            conv_op = ConvOperator(scale=1/dy, domain='y', order=1, 
                                  taylor_order=taylor_order, device=device)
            result_yours = conv_op(f_torch)[0, 0].detach().cpu().numpy()
            
            # findiff method
            op_findiff = FinDiff(1, dy, 1, acc=taylor_order)
            result_findiff = op_findiff(f)
            
        else:  # laplacian
            analytical = test_func['laplacian'](X, Y)
            
            # Your method
            conv_op = ConvOperator(scale=1.0/dx**2, domain=('x', 'y'), order=2, 
                                  taylor_order=taylor_order, device=device)
            result_yours = conv_op(f_torch)[0, 0].detach().cpu().numpy()
            
            # findiff method
            op_findiff = FinDiff(0, dx, 2, acc=taylor_order) + FinDiff(1, dy, 2, acc=taylor_order)
            result_findiff = op_findiff(f)
        
        # Calculate cropped regions
        pad = 3
        if result_yours.shape != analytical.shape:
            offset = (analytical.shape[0] - result_yours.shape[0]) // 2
            analytical_cropped = analytical[offset:-offset, offset:-offset]
        else:
            analytical_cropped = analytical[pad:-pad, pad:-pad]
            result_yours = result_yours[pad:-pad, pad:-pad]
        
        result_findiff_cropped = result_findiff[pad:-pad, pad:-pad]
        
        # Check difference between your method and findiff
        diff_yours_findiff = np.abs(result_yours - result_findiff_cropped)
        max_diff = np.max(diff_yours_findiff)
        
        assert max_diff < tolerance, f"Difference between your method and findiff for {derivative_type} too large: {max_diff}"
        
        # Optional: Still check that both methods are reasonably accurate
        error_yours = np.sqrt(np.mean((result_yours - analytical_cropped)**2))
        error_findiff = np.sqrt(np.mean((result_findiff_cropped - analytical_cropped)**2))
        
        assert error_yours < 1e0, f"Your {derivative_type} error too high: {error_yours:.2e}"
        assert error_findiff < 1e0, f"findiff {derivative_type} error too high: {error_findiff:.2e}"


# Test configuration and utility functions
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "benchmark: mark test as a benchmark test")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers and organize tests"""
    for item in items:
        # Add slow marker to convergence tests
        if "convergence" in item.name:
            item.add_marker(pytest.mark.slow)
        
        # Add benchmark marker to performance tests
        if "performance" in item.name:
            item.add_marker(pytest.mark.benchmark)


# Quick test for manual execution
def run_quick_test():
    """Quick test that can be run without the full test suite"""
    
    print("Running quick ConvOperator test...")
    
    if not CONVOPERATOR_AVAILABLE:
        print("ConvOperator not available")
        return
    
    # Simple test case
    nx, ny = 51, 51
    x = np.linspace(-2, 2, nx)
    y = np.linspace(-2, 2, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    dx = x[1] - x[0]
    
    # Test function: f(x,y) = sin(πx) * cos(πy)
    f = np.sin(np.pi * X) * np.cos(np.pi * Y)
    dfdx_analytical = np.pi * np.cos(np.pi * X) * np.cos(np.pi * Y)
    
    f_torch = torch.tensor(f, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    for order in [2, 4, 6]:
        try:
            conv_op = ConvOperator(scale=1/dx, domain='x', order=1, taylor_order=order)
            dfdx_yours = conv_op(f_torch)[0, 0].numpy()
            
            # Calculate error
            if dfdx_yours.shape != dfdx_analytical.shape:
                offset = (dfdx_analytical.shape[0] - dfdx_yours.shape[0]) // 2
                dfdx_analytical_cropped = dfdx_analytical[offset:-offset, offset:-offset]
            else:
                pad = 2
                dfdx_analytical_cropped = dfdx_analytical[pad:-pad, pad:-pad]
                dfdx_yours = dfdx_yours[pad:-pad, pad:-pad]
            
            error = np.sqrt(np.mean((dfdx_yours - dfdx_analytical_cropped)**2))
            print(f"Order {order}: Error = {error:.2e}")
            
        except Exception as e:
            print(f"Order {order}: Error - {e}")


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v"])