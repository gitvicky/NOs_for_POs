#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pytest Test Suite for Vector Operations

Compares custom vector operations with findiff package using pytest framework.
Modified to use difference-based assertions instead of error ratios.
"""
import os 
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import torch
import matplotlib.pyplot as plt
import time
from typing import Tuple, Optional, Dict, Any


tolerance = 1e-3  # Tolerance for numerical comparisons

# Try to import findiff
try:
    from findiff import Divergence as FindiffDivergence
    from findiff import Curl as FindiffCurl
    from findiff import Laplacian as FindiffLaplacian
    FINDIFF_AVAILABLE = True
except ImportError:
    FINDIFF_AVAILABLE = False

# Try to import your vector operations
try:
    import sys
    sys.path.append('..')
    from PRE.VectorConvOps_Spatial import Divergence, Curl, Laplace
    VECTOR_OPS_AVAILABLE = True
except ImportError:
    VECTOR_OPS_AVAILABLE = False


@pytest.fixture(scope="session")
def device():
    """Fixture to determine device (CPU/CUDA)"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def simple_vector_field():
    """Fixture to create a simple test vector field"""
    def _create_field(size=51):
        x = np.linspace(-1, 1, size)
        y = np.linspace(-1, 1, size)
        X, Y = np.meshgrid(x, y, indexing='ij')
        dx = x[1] - x[0]
        
        # Simple polynomial field: u = x², v = xy
        u = X**2
        v = X * Y
        
        # Analytical results
        div_exact = 2*X + X  # ∂u/∂x + ∂v/∂y = 2x + x = 3x
        curl_exact = Y - 0  # ∂v/∂x - ∂u/∂y = y - 0 = y
        
        return X, Y, u, v, div_exact, curl_exact, dx
    
    return _create_field


@pytest.fixture
def scalar_test_field():
    """Fixture to create a scalar test field for Laplacian"""
    def _create_field(size=51):
        x = np.linspace(-1, 1, size)
        y = np.linspace(-1, 1, size)
        X, Y = np.meshgrid(x, y, indexing='ij')
        dx = x[1] - x[0]
        
        f = X**2 + Y**2
        laplacian_exact = np.ones_like(f) * 4  # ∇²(x² + y²) = 2 + 2 = 4
        
        return X, Y, f, laplacian_exact, dx
    
    return _create_field


class TestVectorOperations:
    """Test class for vector operations"""
    
    @pytest.mark.skipif(not VECTOR_OPS_AVAILABLE, reason="Vector operations module not available")
    @pytest.mark.skipif(not FINDIFF_AVAILABLE, reason="findiff package not available")
    @pytest.mark.parametrize("taylor_order", [2, 4, 6])
    @pytest.mark.parametrize("grid_size", [51, 101])
    def test_divergence_accuracy(self, simple_vector_field, device, taylor_order, grid_size):
        """Test divergence operation accuracy against findiff"""
        X, Y, u, v, div_exact, _, dx = simple_vector_field(grid_size)
        
        # Your method
        u_torch = torch.tensor(u, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        v_torch = torch.tensor(v, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        
        div_op = Divergence(domain=('x', 'y'), order=1, scale=1/dx, 
                          taylor_order=taylor_order, device=device)
        div_yours = div_op(u_torch, v_torch)[0, 0].detach().cpu().numpy()
        
        # findiff method
        div_findiff_op = FindiffDivergence(h=[dx, dx], acc=taylor_order)
        div_findiff = div_findiff_op(np.stack((u, v)))
        
        # Calculate cropped regions (handle boundary effects)
        pad = 3
        if div_yours.shape != div_exact.shape:
            offset = (div_exact.shape[0] - div_yours.shape[0]) // 2
            div_exact_cropped = div_exact[offset:-offset, offset:-offset]
        else:
            div_exact_cropped = div_exact[pad:-pad, pad:-pad]
            div_yours = div_yours[pad:-pad, pad:-pad]
        
        div_findiff_cropped = div_findiff[pad:-pad, pad:-pad]
        
        # Check difference between your method and findiff
        diff_yours_findiff = np.abs(div_yours - div_findiff_cropped)
        max_diff = np.max(diff_yours_findiff)
        
        assert max_diff < tolerance, f"Difference between your method and findiff too large: {max_diff}"
        
        # Optional: Still check that both methods are reasonably accurate
        error_yours = np.sqrt(np.mean((div_yours - div_exact_cropped)**2))
        error_findiff = np.sqrt(np.mean((div_findiff_cropped - div_exact_cropped)**2))
        
        assert error_yours < 1e-1, f"Your method error too high: {error_yours}"
        assert error_findiff < 1e-1, f"findiff error too high: {error_findiff}"

    # @pytest.mark.skipif(not VECTOR_OPS_AVAILABLE, reason="Vector operations module not available")
    # @pytest.mark.skipif(not FINDIFF_AVAILABLE, reason="findiff package not available")
    # @pytest.mark.parametrize("taylor_order", [2, 4, 6])
    # def test_curl_accuracy(self, simple_vector_field, device, taylor_order):
    #     """Test curl operation accuracy against findiff"""
    #     X, Y, u, v, _, curl_exact, dx = simple_vector_field()
        
    #     # Your method
    #     u_torch = torch.tensor(u, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    #     v_torch = torch.tensor(v, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        
    #     curl_op = Curl(domain=('x', 'y'), order=1, scale=1/dx, 
    #                   taylor_order=taylor_order, device=device)
    #     curl_yours = curl_op(u_torch, v_torch)[0, 0].detach().cpu().numpy()
        
    #     # findiff method
    #     curl_findiff_op = FindiffCurl(h=[dx, dx], acc=taylor_order)
    #     curl_findiff = curl_findiff_op(np.stack((u, v)))
        
    #     # Calculate cropped regions
    #     pad = 3
    #     if curl_yours.shape != curl_exact.shape:
    #         offset = (curl_exact.shape[0] - curl_yours.shape[0]) // 2
    #         curl_exact_cropped = curl_exact[offset:-offset, offset:-offset]
    #     else:
    #         curl_exact_cropped = curl_exact[pad:-pad, pad:-pad]
    #         curl_yours = curl_yours[pad:-pad, pad:-pad]
        
    #     curl_findiff_cropped = curl_findiff[pad:-pad, pad:-pad]
        
    #     # Check difference between your method and findiff
    #     diff_yours_findiff = np.abs(curl_yours - curl_findiff_cropped)
    #     max_diff = np.max(diff_yours_findiff)
        
    #     assert max_diff < tolerance, f"Difference between your method and findiff too large: {max_diff}"

    @pytest.mark.skipif(not VECTOR_OPS_AVAILABLE, reason="Vector operations module not available")
    @pytest.mark.skipif(not FINDIFF_AVAILABLE, reason="findiff package not available")
    @pytest.mark.parametrize("taylor_order", [2, 4, 6])
    def test_laplacian_accuracy(self, scalar_test_field, device, taylor_order):
        """Test Laplacian operation accuracy against findiff"""
        X, Y, f, laplacian_exact, dx = scalar_test_field()
        
        # Your method
        f_torch = torch.tensor(f, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        
        laplace_op = Laplace(domain=('x', 'y'), order=2, scale=1/dx**2, 
                           taylor_order=taylor_order, scalar=True, device=device)
        laplace_yours = laplace_op(f_torch)[0, 0].detach().cpu().numpy()
        
        # findiff method
        laplace_findiff_op = FindiffLaplacian(h=[dx, dx], acc=taylor_order)
        laplace_findiff = laplace_findiff_op(f)
        
        # Calculate cropped regions
        pad = 3
        if laplace_yours.shape != laplacian_exact.shape:
            offset = (laplacian_exact.shape[0] - laplace_yours.shape[0]) // 2
            laplacian_exact_cropped = laplacian_exact[offset:-offset, offset:-offset]
        else:
            laplacian_exact_cropped = laplacian_exact[pad:-pad, pad:-pad]
            laplace_yours = laplace_yours[pad:-pad, pad:-pad]
        
        laplace_findiff_cropped = laplace_findiff[pad:-pad, pad:-pad]
        
        # Check difference between your method and findiff
        diff_yours_findiff = np.abs(laplace_yours - laplace_findiff_cropped)
        max_diff = np.max(diff_yours_findiff)
        
        assert max_diff < tolerance, f"Difference between your method and findiff too large: {max_diff}"
        
        # Optional: Still check that both methods are reasonably accurate
        error_yours = np.sqrt(np.mean((laplace_yours - laplacian_exact_cropped)**2))
        error_findiff = np.sqrt(np.mean((laplace_findiff_cropped - laplacian_exact_cropped)**2))
        
        assert error_yours < 1e-1, f"Your method error too high: {error_yours}"
        assert error_findiff < 1e-1, f"findiff error too high: {error_findiff}"

    @pytest.mark.skipif(not VECTOR_OPS_AVAILABLE, reason="Vector operations module not available")
    def test_divergence_performance(self, simple_vector_field, device):
        """Test divergence operation performance"""
        X, Y, u, v, _, _, dx = simple_vector_field(201)  # Larger grid for performance test
        
        u_torch = torch.tensor(u, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        v_torch = torch.tensor(v, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        
        div_op = Divergence(domain=('x', 'y'), order=1, scale=1/dx, 
                          taylor_order=4, device=device)
        
        # Warm up
        _ = div_op(u_torch, v_torch)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Time the operation
        start_time = time.time()
        for _ in range(10):
            result = div_op(u_torch, v_torch)
            if device.type == 'cuda':
                torch.cuda.synchronize()
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10
        
        # Performance should be reasonable (less than 1 second for 201x201 grid)
        assert avg_time < 1.0, f"Performance too slow: {avg_time:.4f}s per operation"

    @pytest.mark.skipif(not VECTOR_OPS_AVAILABLE, reason="Vector operations module not available")
    @pytest.mark.skipif(not FINDIFF_AVAILABLE, reason="findiff package not available")
    def test_convergence_divergence(self, simple_vector_field, device):
        """Test convergence behavior for divergence with grid refinement"""
        grid_sizes = [21, 41, 81]
        errors_yours = []
        errors_findiff = []
        
        for size in grid_sizes:
            X, Y, u, v, div_exact, _, dx = simple_vector_field(size)
            
            # Your method
            u_torch = torch.tensor(u, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            v_torch = torch.tensor(v, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            
            div_op = Divergence(domain=('x', 'y'), order=1, scale=1/dx, 
                              taylor_order=4, device=device)
            div_yours = div_op(u_torch, v_torch)[0, 0].detach().cpu().numpy()
            
            # findiff method
            div_findiff_op = FindiffDivergence(h=[dx, dx], acc=4)
            div_findiff = div_findiff_op(np.stack((u, v)))
            
            # Calculate errors
            pad = 2
            if div_yours.shape != div_exact.shape:
                offset = (div_exact.shape[0] - div_yours.shape[0]) // 2
                div_exact_cropped = div_exact[offset:-offset, offset:-offset]
            else:
                div_exact_cropped = div_exact[pad:-pad, pad:-pad]
                div_yours = div_yours[pad:-pad, pad:-pad]
            
            div_findiff_cropped = div_findiff[pad:-pad, pad:-pad]
            
            error_yours = np.sqrt(np.mean((div_yours - div_exact_cropped)**2))
            error_findiff = np.sqrt(np.mean((div_findiff_cropped - div_exact_cropped)**2))
            
            errors_yours.append(error_yours)
            errors_findiff.append(error_findiff)
        
        # Check that errors decrease with grid refinement
        assert errors_yours[1] < errors_yours[0], "Error should decrease with finer grid"
        assert errors_yours[2] < errors_yours[1], "Error should continue decreasing"
        assert errors_findiff[1] < errors_findiff[0], "findiff error should also decrease"

    @pytest.mark.skipif(not VECTOR_OPS_AVAILABLE, reason="Vector operations module not available")
    def test_divergence_edge_cases(self, device):
        """Test divergence with edge cases"""
        # Test with zero field
        zeros = torch.zeros(1, 1, 50, 50, device=device)
        div_op = Divergence(domain=('x', 'y'), order=1, scale=1.0, taylor_order=4, device=device)
        result = div_op(zeros, zeros)
        
        # Should be zero everywhere
        assert torch.allclose(result, torch.zeros_like(result), atol=tolerance)
        
        # Test with constant field
        ones = torch.ones(1, 1, 50, 50, device=device)
        result = div_op(ones, ones)
        
        # Divergence of constant field should be zero
        assert torch.allclose(result, torch.zeros_like(result), atol=tolerance)


class TestVectorOperationsComparison:
    """Test class for comparing your implementation with findiff"""
    
    @pytest.mark.skipif(not VECTOR_OPS_AVAILABLE or not FINDIFF_AVAILABLE, 
                       reason="Both vector operations and findiff required")
    @pytest.mark.parametrize("operation", ["divergence", "curl", "laplacian"])
    @pytest.mark.parametrize("taylor_order", [4, 6])
    def test_accuracy_comparison(self, operation, taylor_order, device):
        """Compare accuracy between your method and findiff for different operations"""
        size = 81
        
        if operation in ["divergence"]:#, "curl"]:
            # Create vector field
            x = np.linspace(-1, 1, size)
            y = np.linspace(-1, 1, size)
            X, Y = np.meshgrid(x, y, indexing='ij')
            dx = x[1] - x[0]
            
            u = X**2
            v = X * Y
            
            if operation == "divergence":
                exact = 2*X + X  # div = ∂u/∂x + ∂v/∂y = 2x + x = 3x
                
                # Your method
                u_torch = torch.tensor(u, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                v_torch = torch.tensor(v, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                op_yours = Divergence(domain=('x', 'y'), order=1, scale=1/dx, 
                                    taylor_order=taylor_order, device=device)
                result_yours = op_yours(u_torch, v_torch)[0, 0].detach().cpu().numpy()
                
                # findiff method
                op_findiff = FindiffDivergence(h=[dx, dx], acc=taylor_order)
                result_findiff = op_findiff(np.stack((u, v)))
                
            else:  # curl
                exact = Y - 0  # curl = ∂v/∂x - ∂u/∂y = y - 0 = y
                
                # Your method
                u_torch = torch.tensor(u, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                v_torch = torch.tensor(v, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                op_yours = Curl(domain=('x', 'y'), order=1, scale=1/dx, 
                              taylor_order=taylor_order, device=device)
                result_yours = op_yours(u_torch, v_torch)[0, 0].detach().cpu().numpy()
                
                # findiff method
                op_findiff = FindiffCurl(h=[dx, dx], acc=taylor_order)
                result_findiff = op_findiff(np.stack((u, v)))
        
        else:  # laplacian
            x = np.linspace(-1, 1, size)
            y = np.linspace(-1, 1, size)
            X, Y = np.meshgrid(x, y, indexing='ij')
            dx = x[1] - x[0]
            
            f = X**2 + Y**2
            exact = np.ones_like(f) * 4  # ∇²(x² + y²) = 4
            
            # Your method
            f_torch = torch.tensor(f, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            op_yours = Laplace(domain=('x', 'y'), order=2, scale=1/dx**2, 
                             taylor_order=taylor_order, scalar=True, device=device)
            result_yours = op_yours(f_torch)[0, 0].detach().cpu().numpy()
            
            # findiff method
            op_findiff = FindiffLaplacian(h=[dx, dx], acc=taylor_order)
            result_findiff = op_findiff(f)
        
        # Calculate cropped regions
        pad = 3
        if result_yours.shape != exact.shape:
            offset = (exact.shape[0] - result_yours.shape[0]) // 2
            exact_cropped = exact[offset:-offset, offset:-offset]
        else:
            exact_cropped = exact[pad:-pad, pad:-pad]
            result_yours = result_yours[pad:-pad, pad:-pad]
        
        result_findiff_cropped = result_findiff[pad:-pad, pad:-pad]
        
        # Check difference between your method and findiff
        diff_yours_findiff = np.abs(result_yours - result_findiff_cropped)
        max_diff = np.max(diff_yours_findiff)
        
        assert max_diff < tolerance, f"Difference between your method and findiff for {operation} too large: {max_diff}"
        
        # Optional: Still check that both methods are reasonably accurate
        error_yours = np.sqrt(np.mean((result_yours - exact_cropped)**2))
        error_findiff = np.sqrt(np.mean((result_findiff_cropped - exact_cropped)**2))
        
        assert error_yours < 1e-1, f"Your {operation} error too high: {error_yours}"
        assert error_findiff < 1e-1, f"findiff {operation} error too high: {error_findiff}"


@pytest.mark.benchmark
class TestVectorOperationsPerformance:
    """Performance benchmark tests"""
    
    @pytest.mark.skipif(not VECTOR_OPS_AVAILABLE, reason="Vector operations module not available")
    @pytest.mark.parametrize("grid_size", [101, 201, 401])
    def test_divergence_scaling(self, grid_size, device):
        """Test how divergence performance scales with grid size"""
        x = np.linspace(-1, 1, grid_size)
        y = np.linspace(-1, 1, grid_size)
        X, Y = np.meshgrid(x, y, indexing='ij')
        dx = x[1] - x[0]
        
        u = X**2
        v = X * Y
        
        u_torch = torch.tensor(u, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        v_torch = torch.tensor(v, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        
        div_op = Divergence(domain=('x', 'y'), order=1, scale=1/dx, 
                          taylor_order=4, device=device)
        
        # Warm up
        _ = div_op(u_torch, v_torch)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Time multiple runs
        num_runs = 5
        start_time = time.time()
        for _ in range(num_runs):
            result = div_op(u_torch, v_torch)
            if device.type == 'cuda':
                torch.cuda.synchronize()
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs
        
        # Performance should scale reasonably with grid size
        # (this is a rough check - actual scaling depends on implementation details)
        max_time = 0.1 * (grid_size / 100)**2  # Quadratic scaling assumption
        assert avg_time < max_time, f"Performance scaling poor for grid {grid_size}: {avg_time:.4f}s"


# Utility functions for pytest
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line("markers", "benchmark: mark test as a benchmark test")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add skip markers"""
    for item in items:
        # Add slow marker to convergence tests
        if "convergence" in item.name:
            item.add_marker(pytest.mark.slow)
        
        # Add benchmark marker to performance tests
        if "performance" in item.name or "scaling" in item.name:
            item.add_marker(pytest.mark.benchmark)


# Test data generation functions for use in tests
def generate_test_fields():
    """Generate various test fields for comprehensive testing"""
    test_fields = {}
    
    # Polynomial field
    def poly_field(X, Y):
        u = X**3 + Y**2
        v = X*Y + Y**3
        div_exact = 3*X**2 + X + 3*Y**2
        curl_exact = Y + 3*Y**2 - 2*Y
        return u, v, div_exact, curl_exact
    
    # Trigonometric field
    def trig_field(X, Y):
        u = np.sin(np.pi*X) * np.cos(np.pi*Y)
        v = np.cos(np.pi*X) * np.sin(np.pi*Y)
        div_exact = np.pi*np.cos(np.pi*X)*np.cos(np.pi*Y) + np.pi*np.cos(np.pi*X)*np.cos(np.pi*Y)
        curl_exact = -np.pi*np.sin(np.pi*X)*np.sin(np.pi*Y) - np.pi*np.sin(np.pi*X)*np.sin(np.pi*Y)
        return u, v, div_exact, curl_exact
    
    test_fields['polynomial'] = poly_field
    test_fields['trigonometric'] = trig_field
    
    return test_fields


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v"])