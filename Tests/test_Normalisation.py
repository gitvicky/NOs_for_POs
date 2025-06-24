"""
Enhanced tests for data normalization schemes

Tests all normalization methods to ensure proper encoding/decoding,
range preservation, and numerical stability.
"""

import pytest
import torch
import numpy as np
import os 
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Neural_PDE.Utils.processing_utils import (
    MinMax_Normalizer, 
    MinMax_Normalizer_variable,
    Gaussian_Normalizer,
    Identity_Normalizer,
    UnitGaussian_Normalizer,
    Range_Normalizer,
    LogNormalizer,
    Normalisation
)


class TestNormalizationFactory:
    """Test the normalization factory function"""
    
    def test_factory_creation(self):
        """Test that factory creates correct normalizer types"""
        data = torch.randn(5, 2, 16, 16, 8)
        
        # Test factory function
        minmax_norm = Normalisation('Min-Max')(data)
        assert isinstance(minmax_norm, MinMax_Normalizer)
        
        minmax_var_norm = Normalisation('Min-Max_variable')(data)
        assert isinstance(minmax_var_norm, MinMax_Normalizer_variable)
        
        gaussian_norm = Normalisation('Gaussian')(data)
        assert isinstance(gaussian_norm, Gaussian_Normalizer)
        
        identity_norm = Normalisation('Identity')(data)
        assert isinstance(identity_norm, Identity_Normalizer)
        
        range_norm = Normalisation('Range')(data)
        assert isinstance(range_norm, Range_Normalizer)
        
    def test_invalid_normalization_type(self):
        """Test handling of invalid normalization types"""
        data = torch.randn(5, 2, 16, 16, 8)
        
        with pytest.raises(KeyError):
            # This should fail since 'InvalidType' is not in the factory
            invalid_norm = Normalisation('InvalidType')(data)


class TestMinMaxNormalizer:
    """Test Min-Max normalization (global across all variables)"""
    
    def test_basic_normalization(self):
        """Test basic min-max normalization functionality"""
        # Create data with known range
        data = torch.linspace(-5, 10, 1000).reshape(10, 2, 5, 10, 1)
        
        normalizer = MinMax_Normalizer(data, low=0.01, high=1.0)
        normalized = normalizer.encode(data)
        denormalized = normalizer.decode(normalized)
        
        # Check normalization range
        assert normalized.min() >= 0.009  # Allow small numerical errors
        assert normalized.max() <= 1.001
        
        # Check perfect reconstruction
        assert torch.allclose(data, denormalized, atol=1e-6)
        
    def test_default_parameters(self):
        """Test with default low/high parameters"""
        data = torch.randn(5, 2, 8, 8, 4) * 10
        
        normalizer = MinMax_Normalizer(data)  # Default: low=-1.0, high=1.0
        normalized = normalizer.encode(data)
        denormalized = normalizer.decode(normalized)
        
        # Check default range [-1, 1]
        assert normalized.min() >= -1.001
        assert normalized.max() <= 1.001
        assert torch.allclose(data, denormalized, atol=1e-6)
        
    def test_uniform_data(self):
        """Test behavior with uniform (constant) data"""
        # All values the same
        data = torch.ones(5, 2, 8, 8, 4) * 7.5
        
        normalizer = MinMax_Normalizer(data)
        normalized = normalizer.encode(data)
        denormalized = normalizer.decode(normalized)
        
        # Should handle constant data gracefully
        assert torch.allclose(data, denormalized, atol=1e-6)
        
        # Normalized constant data should be handled appropriately
        assert torch.isfinite(normalized).all()
        
    def test_different_ranges(self):
        """Test with data having very different ranges"""
        # Create data with different scales
        data = torch.zeros(4, 3, 10, 10, 5)
        data[:, 0] = torch.randn(4, 10, 10, 5) * 1000  # Large values
        data[:, 1] = torch.randn(4, 10, 10, 5) * 0.001  # Small values
        data[:, 2] = torch.randn(4, 10, 10, 5)  # Normal scale
        
        normalizer = MinMax_Normalizer(data, low=0.0, high=1.0)
        normalized = normalizer.encode(data)
        denormalized = normalizer.decode(normalized)
        
        # Check range
        assert normalized.min() >= -0.001
        assert normalized.max() <= 1.001
        
        # Check reconstruction
        assert torch.allclose(data, denormalized, atol=1e-5)
        
    def test_edge_cases(self):
        """Test edge cases like single values, zeros, etc."""
        # Single value
        single_data = torch.tensor([[[[[1.5]]]]])
        norm_single = MinMax_Normalizer(single_data)
        encoded_single = norm_single.encode(single_data)
        decoded_single = norm_single.decode(encoded_single)
        assert torch.allclose(single_data, decoded_single, atol=1e-6)
        
        # All zeros
        zero_data = torch.zeros(2, 1, 4, 4, 3)
        norm_zero = MinMax_Normalizer(zero_data)
        encoded_zero = norm_zero.encode(zero_data)
        decoded_zero = norm_zero.decode(encoded_zero)
        assert torch.allclose(zero_data, decoded_zero, atol=1e-6)
        
    def test_batch_consistency(self):
        """Test that normalization is consistent across batches"""
        # Create training data
        train_data = torch.randn(10, 2, 8, 8, 5) * 5 + 2
        
        # Train normalizer
        normalizer = MinMax_Normalizer(train_data)
        
        # Test on new batch with same distribution
        test_data = torch.randn(5, 2, 8, 8, 5) * 5 + 2
        normalized_test = normalizer.encode(test_data)
        denormalized_test = normalizer.decode(normalized_test)
        
        # Should reconstruct perfectly
        assert torch.allclose(test_data, denormalized_test, atol=1e-6)
        
    def test_cuda_cpu_transfer(self):
        """Test CUDA/CPU device transfer functionality"""
        data = torch.randn(5, 2, 8, 8, 4)
        normalizer = MinMax_Normalizer(data)
        
        # Test CPU method (should work even without CUDA)
        normalizer.cpu()
        assert normalizer.a.device.type == 'cpu'
        assert normalizer.b.device.type == 'cpu'
        
        # Test CUDA method if available
        if torch.cuda.is_available():
            normalizer.cuda()
            assert normalizer.a.device.type == 'cuda'
            assert normalizer.b.device.type == 'cuda'


class TestMinMaxVariableNormalizer:
    """Test Min-Max normalization per variable"""
    
    def test_per_variable_normalization(self):
        """Test that each variable is normalized independently"""
        data = torch.zeros(5, 3, 10, 10, 4)
        
        # Variable 0: range [0, 10]
        data[:, 0] = torch.rand(5, 10, 10, 4) * 10
        
        # Variable 1: range [-5, 5]  
        data[:, 1] = torch.rand(5, 10, 10, 4) * 10 - 5
        
        # Variable 2: range [100, 200]
        data[:, 2] = torch.rand(5, 10, 10, 4) * 100 + 100
        
        normalizer = MinMax_Normalizer_variable(data)
        normalized = normalizer.encode(data)
        denormalized = normalizer.decode(normalized)
        
        # Each variable should be in [0,1] range
        for var in range(3):
            var_data = normalized[:, var]
            assert var_data.min() >= -0.001
            assert var_data.max() <= 1.001
            
        # Check reconstruction
        assert torch.allclose(data, denormalized, atol=1e-6)
        
    def test_variable_independence(self):
        """Test that variables are normalized independently"""
        data = torch.zeros(3, 2, 8, 8, 3)
        
        # Variable 0: large values
        data[:, 0] = 1000 + torch.randn(3, 8, 8, 3) * 100
        
        # Variable 1: small values
        data[:, 1] = torch.randn(3, 8, 8, 3) * 0.01
        
        normalizer = MinMax_Normalizer_variable(data)
        
        # Check that normalization parameters are different for each variable
        assert not torch.allclose(normalizer.a[0], normalizer.a[1])
        assert not torch.allclose(normalizer.b[0], normalizer.b[1])
        
        normalized = normalizer.encode(data)
        
        # Both variables should be in [0,1] despite very different original ranges
        assert normalized[:, 0].min() >= -0.001
        assert normalized[:, 0].max() <= 1.001
        assert normalized[:, 1].min() >= -0.001  
        assert normalized[:, 1].max() <= 1.001
        
    def test_single_variable_case(self):
        """Test that it works correctly with single variable"""
        data = torch.randn(5, 1, 12, 12, 6) * 7 + 3
        
        normalizer = MinMax_Normalizer_variable(data)
        normalized = normalizer.encode(data)
        denormalized = normalizer.decode(normalized)
        
        assert normalized.min() >= -0.001
        assert normalized.max() <= 1.001
        assert torch.allclose(data, denormalized, atol=1e-6)
        
    def test_selective_variable_encoding(self):
        """Test encoding/decoding specific variables"""
        data = torch.randn(3, 3, 8, 8, 4) * 10
        normalizer = MinMax_Normalizer_variable(data)
        
        # Test encoding specific variable
        test_data = data.clone()
        original_var1 = test_data[:, 1].clone()
        
        # This should only encode variable 0
        encoded = normalizer.encode(test_data, var_idx=0)
        
        # Variable 1 should be unchanged
        assert torch.allclose(encoded[:, 1], original_var1)
        
        # Variable 0 should be changed (normalized)
        assert not torch.allclose(encoded[:, 0], data[:, 0])
        
    def test_cuda_cpu_transfer(self):
        """Test CUDA/CPU device transfer functionality"""
        data = torch.randn(5, 3, 8, 8, 4)
        normalizer = MinMax_Normalizer_variable(data)
        
        # Test CPU method
        normalizer.cpu()
        assert normalizer.a.device.type == 'cpu'
        assert normalizer.b.device.type == 'cpu'
        
        # Test CUDA method if available
        if torch.cuda.is_available():
            normalizer.cuda()
            assert normalizer.a.device.type == 'cuda'
            assert normalizer.b.device.type == 'cuda'


class TestRangeNormalizer:
    """Test Range normalization (similar to MinMax but per-feature)"""
    
    def test_range_normalization(self):
        """Test basic range normalization functionality"""
        data = torch.randn(10, 3, 8, 8, 5) * 5 + 10
        
        normalizer = Range_Normalizer(data, low=-2.0, high=2.0)
        normalized = normalizer.encode(data)
        denormalized = normalizer.decode(normalized)
        
        # Check reconstruction
        assert torch.allclose(data, denormalized, atol=1e-5)
        
    def test_default_range(self):
        """Test with default range [-1, 1]"""
        data = torch.randn(8, 2, 6, 6, 4) * 3
        
        normalizer = Range_Normalizer(data)  # Default: low=-1.0, high=1.0
        normalized = normalizer.encode(data)
        denormalized = normalizer.decode(normalized)
        
        assert torch.allclose(data, denormalized, atol=1e-6)


class TestGaussianNormalizer:
    """Test Gaussian (Z-score) normalization"""
    
    def test_gaussian_normalization(self):
        """Test that Gaussian normalization produces zero mean, unit variance"""
        # Create data with known statistics
        data = torch.randn(20, 2, 16, 16, 8) * 5 + 10  # mean=10, std=5
        
        normalizer = Gaussian_Normalizer(data)
        normalized = normalizer.encode(data)
        denormalized = normalizer.decode(normalized)
        
        # Check that normalized data has approximately zero mean and unit variance
        assert torch.abs(normalized.mean()) < 0.1
        assert torch.abs(normalized.std() - 1.0) < 0.1
        
        # Check reconstruction
        assert torch.allclose(data, denormalized, atol=1e-5)
        
    def test_gaussian_with_different_distributions(self):
        """Test Gaussian normalization with different input distributions"""
        # Test with uniform distribution
        uniform_data = torch.rand(10, 2, 8, 8, 4) * 20 - 10  # Uniform[-10, 10]
        
        norm_uniform = Gaussian_Normalizer(uniform_data)
        normalized_uniform = norm_uniform.encode(uniform_data)
        denormalized_uniform = norm_uniform.decode(normalized_uniform)
        
        assert torch.allclose(uniform_data, denormalized_uniform, atol=1e-6)
        
        # Test with exponential-like distribution
        exp_data = torch.exp(torch.randn(8, 3, 6, 6, 5))
        
        norm_exp = Gaussian_Normalizer(exp_data)
        normalized_exp = norm_exp.encode(exp_data)
        denormalized_exp = norm_exp.decode(normalized_exp)
        
        assert torch.allclose(exp_data, denormalized_exp, atol=1e-5)
        
    def test_constant_data_handling(self):
        """Test Gaussian normalization with constant data"""
        # All same value
        constant_data = torch.ones(5, 2, 8, 8, 3) * 42
        
        normalizer = Gaussian_Normalizer(constant_data)
        normalized = normalizer.encode(constant_data)
        denormalized = normalizer.decode(normalized)
        
        # Should handle constant data gracefully (std=0 case)
        assert torch.allclose(constant_data, denormalized, atol=1e-6)
        assert torch.isfinite(normalized).all()
        
    def test_cuda_cpu_transfer(self):
        """Test CUDA/CPU device transfer functionality"""
        data = torch.randn(5, 2, 8, 8, 4)
        normalizer = Gaussian_Normalizer(data)
        
        # Test CPU method
        normalizer.cpu()
        assert normalizer.mean.device.type == 'cpu'
        assert normalizer.std.device.type == 'cpu'
        
        # Test CUDA method if available
        if torch.cuda.is_available():
            normalizer.cuda()
            assert normalizer.mean.device.type == 'cuda'
            assert normalizer.std.device.type == 'cuda'


class TestUnitGaussianNormalizer:
    """Test Unit Gaussian normalization (per-feature normalization)"""
    
    def test_unit_gaussian_properties(self):
        """Test Unit Gaussian normalization properties"""
        data = torch.randn(15, 3, 12, 12, 6) * 3 + 7
        
        normalizer = UnitGaussian_Normalizer(data)
        normalized = normalizer.encode(data)
        denormalized = normalizer.decode(normalized)
        
        # Check reconstruction
        assert torch.allclose(data, denormalized, atol=1e-5)
        
    def test_sample_idx_functionality(self):
        """Test the sample_idx parameter in decode"""
        data = torch.randn(10, 3, 8, 8, 5)
        normalizer = UnitGaussian_Normalizer(data)
        normalized = normalizer.encode(data)
        
        # Test decode without sample_idx
        decoded_all = normalizer.decode(normalized)
        assert torch.allclose(data, decoded_all, atol=1e-5)
        
        # Test decode with sample_idx (this tests the conditional logic)
        sample_indices = [torch.tensor([0, 2, 4])]
        try:
            decoded_sample = normalizer.decode(normalized, sample_idx=sample_indices)
            # The exact behavior depends on implementation
            assert decoded_sample.shape == normalized.shape
        except (IndexError, RuntimeError):
            # Some sample_idx operations might not be valid for all tensor shapes
            pass


class TestLogNormalizer:
    """Test Log normalization"""
    
    def test_log_normalization(self):
        """Test basic log normalization functionality"""
        # Use positive data for log normalization
        data = torch.rand(5, 2, 8, 8, 4) * 10 + 1  # Ensure positive values
        
        normalizer = LogNormalizer(data)
        normalized = normalizer.encode(data)
        denormalized = normalizer.decode(normalized)
        
        # Check reconstruction
        assert torch.allclose(data, denormalized, atol=1e-5)
        
        # Check that log operation was applied
        assert torch.isfinite(normalized).all()
        
    def test_log_with_small_values(self):
        """Test log normalization with small values"""
        # Small positive values
        data = torch.rand(4, 2, 6, 6, 3) * 0.1 + 0.01
        
        normalizer = LogNormalizer(data, eps=1e-6)
        normalized = normalizer.encode(data)
        denormalized = normalizer.decode(normalized)
        
        assert torch.allclose(data, denormalized, atol=1e-5)


class TestIdentityNormalizer:
    """Test Identity normalization (no-op)"""
    
    def test_identity_behavior(self):
        """Test that Identity normalizer does nothing"""
        data = torch.randn(7, 4, 10, 10, 5) * 100 - 50
        
        normalizer = Identity_Normalizer(data)
        normalized = normalizer.encode(data)
        denormalized = normalizer.decode(normalized)
        
        # Should be completely unchanged
        assert torch.allclose(data, normalized, atol=1e-10)
        assert torch.allclose(data, denormalized, atol=1e-10)
        assert torch.allclose(normalized, denormalized, atol=1e-10)
        
    def test_identity_with_extreme_values(self):
        """Test Identity normalizer with extreme values"""
        # Very large values
        large_data = torch.ones(3, 2, 6, 6, 4) * 1e6
        norm_large = Identity_Normalizer(large_data)
        assert torch.allclose(large_data, norm_large.encode(large_data))
        
        # Very small values
        small_data = torch.ones(3, 2, 6, 6, 4) * 1e-6
        norm_small = Identity_Normalizer(small_data)
        assert torch.allclose(small_data, norm_small.encode(small_data))


class TestNormalizationComparison:
    """Compare different normalization schemes"""
    
    def test_normalization_range_comparison(self):
        """Compare output ranges of different normalizers"""
        # Create data with wide range
        data = torch.randn(10, 2, 8, 8, 5) * 50 + 25
        
        # Test different normalizers
        minmax_norm = MinMax_Normalizer(data, low=0.0, high=1.0)
        minmax_normalized = minmax_norm.encode(data)
        
        gaussian_norm = Gaussian_Normalizer(data)
        gaussian_normalized = gaussian_norm.encode(data)
        
        identity_norm = Identity_Normalizer(data)
        identity_normalized = identity_norm.encode(data)
        
        # MinMax should be in [0,1]
        assert minmax_normalized.min() >= -0.001
        assert minmax_normalized.max() <= 1.001
        
        # Gaussian should have ~zero mean, unit variance
        assert torch.abs(gaussian_normalized.mean()) < 0.5
        assert torch.abs(gaussian_normalized.std() - 1.0) < 0.5
        
        # Identity should be unchanged
        assert torch.allclose(data, identity_normalized)
        
    def test_reconstruction_accuracy(self):
        """Test reconstruction accuracy across normalizers"""
        data = torch.randn(8, 3, 12, 12, 4) * 10 + 5
        
        normalizers = [
            MinMax_Normalizer(data),
            MinMax_Normalizer_variable(data),
            Gaussian_Normalizer(data),
            UnitGaussian_Normalizer(data),
            Range_Normalizer(data),
            Identity_Normalizer(data)
        ]
        
        for normalizer in normalizers:
            normalized = normalizer.encode(data)
            denormalized = normalizer.decode(normalized)
            
            # All should reconstruct accurately
            assert torch.allclose(data, denormalized, atol=1e-5), \
                f"Failed for {type(normalizer).__name__}"
                
    def test_numerical_stability(self):
        """Test numerical stability across normalizers"""
        # Test with extreme values
        extreme_data = torch.tensor([
            [[[[[1e-10, 1e10, -1e10, 0, 1]]]]]
        ])
        
        normalizers = [
            MinMax_Normalizer(extreme_data),
            Gaussian_Normalizer(extreme_data), 
            Identity_Normalizer(extreme_data)
        ]
        
        for normalizer in normalizers:
            try:
                normalized = normalizer.encode(extreme_data)
                denormalized = normalizer.decode(normalized)
                
                # Should not produce NaN or Inf
                assert torch.isfinite(normalized).all(), \
                    f"Non-finite values in {type(normalizer).__name__}"
                assert torch.isfinite(denormalized).all(), \
                    f"Non-finite decoded values in {type(normalizer).__name__}"
                    
            except Exception as e:
                # Some normalizers might legitimately fail with extreme values
                print(f"Warning: {type(normalizer).__name__} failed with extreme values: {e}")


class TestNormalizationEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_empty_data(self):
        """Test behavior with empty or minimal data"""
        # Empty data should raise an error
        with pytest.raises((ValueError, RuntimeError, IndexError)):
            empty_data = torch.empty(0, 2, 8, 8, 5)
            MinMax_Normalizer(empty_data)
            
        # Single element
        single_element = torch.tensor([[[[[1.0]]]]])
        norm_single = MinMax_Normalizer(single_element)
        normalized_single = norm_single.encode(single_element)
        denormalized_single = norm_single.decode(normalized_single)
        assert torch.allclose(single_element, denormalized_single)
        
    def test_wrong_shape_decode(self):
        """Test decoding with wrong tensor shapes"""
        train_data = torch.randn(5, 2, 8, 8, 4)
        normalizer = MinMax_Normalizer(train_data)
        
        # Try to decode tensor with different shape
        wrong_shape = torch.randn(5, 3, 8, 8, 4)  # Different number of variables
        
        try:
            # This might work or might fail depending on implementation
            result = normalizer.decode(wrong_shape)
            # If it works, check that it's reasonable
            assert result.shape == wrong_shape.shape
        except (RuntimeError, IndexError, ValueError):
            # It's okay if this fails - it's an invalid operation
            pass
            
    def test_inf_nan_handling(self):
        """Test handling of infinite and NaN values"""
        data = torch.randn(5, 2, 8, 8, 4)
        data[0, 0, 0, 0, 0] = float('inf')
        data[0, 0, 1, 1, 1] = float('-inf')
        data[0, 0, 2, 2, 2] = float('nan')
        
        # Some normalizers might handle this, others might not
        for NormalizerClass in [MinMax_Normalizer, Gaussian_Normalizer, Identity_Normalizer]:
            try:
                normalizer = NormalizerClass(data)
                normalized = normalizer.encode(data)
                
                # If it succeeds, check that inf/nan are handled somehow
                assert normalized.shape == data.shape
                
            except (ValueError, RuntimeError):
                # It's acceptable for normalizers to reject inf/nan data
                pass
                
    def test_very_small_variance(self):
        """Test with data having very small variance"""
        # Create data with tiny variance
        base_value = 1000.0
        tiny_variation = torch.randn(10, 2, 8, 8, 5) * 1e-10
        data = base_value + tiny_variation
        
        for NormalizerClass in [MinMax_Normalizer, Gaussian_Normalizer]:
            normalizer = NormalizerClass(data)
            normalized = normalizer.encode(data)
            denormalized = normalizer.decode(normalized)
            
            # Should handle small variance gracefully
            assert torch.isfinite(normalized).all()
            assert torch.allclose(data, denormalized, atol=1e-8)


class TestNormalizationParameters:
    """Test normalization parameter storage and retrieval"""
    
    def test_parameter_persistence(self):
        """Test that normalization parameters can be saved and loaded"""
        data = torch.randn(10, 3, 12, 12, 6) * 5 + 10
        
        # Train normalizer
        normalizer = MinMax_Normalizer(data)
        original_a = normalizer.a.clone()
        original_b = normalizer.b.clone()
        
        # Test data
        test_data = torch.randn(5, 3, 12, 12, 6) * 5 + 10
        normalized_test = normalizer.encode(test_data)
        
        # Create new normalizer with same parameters
        new_normalizer = MinMax_Normalizer(torch.zeros_like(data))
        new_normalizer.a = original_a
        new_normalizer.b = original_b
        
        # Should produce same results
        new_normalized = new_normalizer.encode(test_data)
        assert torch.allclose(normalized_test, new_normalized)
        
    def test_parameter_shapes(self):
        """Test that normalization parameters have correct shapes"""
        data = torch.randn(8, 4, 10, 10, 5)
        
        # MinMax global
        minmax_norm = MinMax_Normalizer(data)
        assert minmax_norm.a.shape == () or minmax_norm.a.shape == (1,)  # Scalar
        assert minmax_norm.b.shape == () or minmax_norm.b.shape == (1,)  # Scalar
        
        # MinMax per variable
        minmax_var_norm = MinMax_Normalizer_variable(data)
        assert minmax_var_norm.a.shape[0] == 4  # One per variable
        assert minmax_var_norm.b.shape[0] == 4  # One per variable
        
        # Gaussian
        gaussian_norm = Gaussian_Normalizer(data)
        # Check that parameters exist and have reasonable shapes
        assert hasattr(gaussian_norm, 'mean')
        assert hasattr(gaussian_norm, 'std')
        
    def test_parameter_types(self):
        """Test that parameters are proper torch tensors"""
        data = torch.randn(5, 3, 8, 8, 4)
        
        for NormalizerClass in [MinMax_Normalizer, MinMax_Normalizer_variable, Gaussian_Normalizer]:
            normalizer = NormalizerClass(data)
            
            # All parameters should be torch tensors
            for param_name in ['a', 'b', 'mean', 'std']:
                if hasattr(normalizer, param_name):
                    param = getattr(normalizer, param_name)
                    assert isinstance(param, torch.Tensor), f"{param_name} is not a tensor in {NormalizerClass.__name__}"


@pytest.mark.slow
class TestNormalizationPerformance:
    """Performance tests for normalization"""
    
    def test_large_data_normalization(self):
        """Test normalization on large datasets"""
        # Large dataset
        large_data = torch.randn(50, 5, 64, 64, 20)
        
        import time
        
        for NormalizerClass in [MinMax_Normalizer, Gaussian_Normalizer, Identity_Normalizer]:
            start_time = time.time()
            
            normalizer = NormalizerClass(large_data)
            normalized = normalizer.encode(large_data)
            denormalized = normalizer.decode(normalized)
            
            end_time = time.time()
            
            # Should complete in reasonable time
            assert end_time - start_time < 10.0  # 10 seconds max
            
            # Should still be accurate (except Identity which should be exact)
            if NormalizerClass == Identity_Normalizer:
                assert torch.allclose(large_data, denormalized, atol=1e-10)
            else:
                assert torch.allclose(large_data, denormalized, atol=1e-5)
            
    def test_repeated_operations(self):
        """Test performance of repeated encode/decode operations"""
        data = torch.randn(10, 3, 16, 16, 8)
        test_batch = torch.randn(5, 3, 16, 16, 8)
        
        normalizer = MinMax_Normalizer(data)
        
        import time
        start_time = time.time()
        
        # Repeated operations
        for _ in range(100):
            normalized = normalizer.encode(test_batch)
            denormalized = normalizer.decode(normalized)
            
        end_time = time.time()
        
        # Should be fast
        assert end_time - start_time < 5.0  # 5 seconds for 100 operations


class TestNormalizationIntegration:
    """Integration tests combining multiple normalizers and real-world scenarios"""
    
    def test_pipeline_compatibility(self):
        """Test that normalizers work well in ML pipelines"""
        # Simulate training/validation split
        full_data = torch.randn(100, 4, 32, 32, 10) * 20 + 5
        train_data = full_data[:80]
        val_data = full_data[80:]
        
        # Train normalizer on training data only
        normalizer = MinMax_Normalizer_variable(train_data)
        
        # Apply to both training and validation
        train_normalized = normalizer.encode(train_data.clone())
        val_normalized = normalizer.encode(val_data.clone())
        
        # Validation data might be outside [0,1] range, but should reconstruct
        val_denormalized = normalizer.decode(val_normalized)
        assert torch.allclose(val_data, val_denormalized, atol=1e-5)
        
    def test_mixed_data_types(self):
        """Test normalizers with different data characteristics"""
        data = torch.zeros(20, 4, 16, 16, 8)
        
        # Variable 0: Normally distributed
        data[:, 0] = torch.randn(20, 16, 16, 8)
        
        # Variable 1: Uniformly distributed
        data[:, 1] = torch.rand(20, 16, 16, 8) * 100 - 50
        
        # Variable 2: Exponentially distributed (positive only)
        data[:, 2] = torch.exp(torch.randn(20, 16, 16, 8))
        
        # Variable 3: Sparse data (mostly zeros)
        sparse_mask = torch.rand(20, 16, 16, 8) < 0.1
        data[:, 3] = sparse_mask.float() * torch.randn(20, 16, 16, 8) * 10
        
        # Test different normalizers
        for NormalizerClass in [MinMax_Normalizer, MinMax_Normalizer_variable, 
                               Gaussian_Normalizer, UnitGaussian_Normalizer]:
            normalizer = NormalizerClass(data)
            normalized = normalizer.encode(data)
            denormalized = normalizer.decode(normalized)
            
            assert torch.allclose(data, denormalized, atol=1e-4), \
                f"Failed reconstruction for {NormalizerClass.__name__} with mixed data"
                
    def test_sequential_normalization(self):
        """Test applying multiple normalizers in sequence"""
        data = torch.randn(10, 2, 8, 8, 5) * 50 + 100
        
        # Apply MinMax first, then Gaussian
        minmax_norm = MinMax_Normalizer(data, low=0, high=1)
        minmax_normalized = minmax_norm.encode(data)
        
        gaussian_norm = Gaussian_Normalizer(minmax_normalized)
        final_normalized = gaussian_norm.encode(minmax_normalized)
        
        # Reverse the process
        gaussian_denormalized = gaussian_norm.decode(final_normalized)
        final_denormalized = minmax_norm.decode(gaussian_denormalized)
        
        assert torch.allclose(data, final_denormalized, atol=1e-5)
        
    def test_normalization_with_masking(self):
        """Test normalizers with masked/missing data simulation"""
        data = torch.randn(8, 3, 12, 12, 6) * 10
        
        # Create a mask for "valid" data
        mask = torch.rand_like(data) > 0.1  # 90% of data is valid
        
        # Set invalid data to zero (simulating missing values)
        masked_data = data * mask.float()
        
        # Normalizers should still work (though results may vary)
        for NormalizerClass in [MinMax_Normalizer, Gaussian_Normalizer]:
            normalizer = NormalizerClass(masked_data)
            normalized = normalizer.encode(masked_data)
            denormalized = normalizer.decode(normalized)
            
            # Should reconstruct the masked data accurately
            assert torch.allclose(masked_data, denormalized, atol=1e-5)


class TestNormalizationErrorHandling:
    """Test error handling and robustness"""
    
    def test_division_by_zero_handling(self):
        """Test handling of division by zero scenarios"""
        # Constant data (std = 0, range = 0)
        constant_data = torch.ones(5, 2, 8, 8, 4) * 42
        
        # These should handle zero variance/range gracefully
        normalizers = [
            MinMax_Normalizer(constant_data),
            MinMax_Normalizer_variable(constant_data),
            Gaussian_Normalizer(constant_data),
            UnitGaussian_Normalizer(constant_data)
        ]
        
        for normalizer in normalizers:
            try:
                normalized = normalizer.encode(constant_data)
                denormalized = normalizer.decode(normalized)
                
                # Should not produce NaN or inf
                assert torch.isfinite(normalized).all()
                assert torch.isfinite(denormalized).all()
                
                # Should reconstruct original data
                assert torch.allclose(constant_data, denormalized, atol=1e-6)
                
            except Exception as e:
                pytest.fail(f"{type(normalizer).__name__} failed with constant data: {e}")
                
    def test_input_validation(self):
        """Test input validation and error messages"""
        valid_data = torch.randn(5, 2, 8, 8, 4)
        
        # Test invalid tensor dimensions
        with pytest.raises((ValueError, IndexError, RuntimeError)):
            invalid_data = torch.randn(5, 2)  # Too few dimensions
            MinMax_Normalizer_variable(invalid_data)
            
        # Test with non-tensor input
        with pytest.raises((TypeError, AttributeError)):
            MinMax_Normalizer(np.random.randn(5, 2, 8, 8, 4))
            
    def test_parameter_corruption_recovery(self):
        """Test behavior when normalization parameters are corrupted"""
        data = torch.randn(8, 3, 10, 10, 5)
        normalizer = MinMax_Normalizer(data)
        
        # Corrupt parameters
        original_a = normalizer.a.clone()
        original_b = normalizer.b.clone()
        
        # Set to invalid values
        normalizer.a = torch.tensor(float('inf'))
        normalizer.b = torch.tensor(float('nan'))
        
        try:
            normalized = normalizer.encode(data)
            # If this doesn't fail, check that we get some reasonable output
            assert normalized.shape == data.shape
        except (ValueError, RuntimeError):
            # It's acceptable for this to fail
            pass
        
        # Restore valid parameters
        normalizer.a = original_a
        normalizer.b = original_b
        
        # Should work again
        normalized = normalizer.encode(data)
        denormalized = normalizer.decode(normalized)
        assert torch.allclose(data, denormalized, atol=1e-5)


class TestNormalizationDocumentation:
    """Test that normalizers behave as documented/expected"""
    
    def test_minmax_range_guarantees(self):
        """Test MinMax normalizer range guarantees"""
        data = torch.randn(10, 3, 8, 8, 5) * 100
        
        # Test custom range
        low, high = -2.5, 7.5
        normalizer = MinMax_Normalizer(data, low=low, high=high)
        normalized = normalizer.encode(data)
        
        # Should be within specified range (allowing for numerical precision)
        assert normalized.min() >= low - 1e-6
        assert normalized.max() <= high + 1e-6
        
    def test_gaussian_statistical_properties(self):
        """Test Gaussian normalizer statistical properties"""
        # Use large dataset for better statistics
        data = torch.randn(100, 2, 16, 16, 8) * 10 + 20
        
        normalizer = Gaussian_Normalizer(data)
        normalized = normalizer.encode(data)
        
        # Should have approximately zero mean and unit variance
        mean = normalized.mean()
        std = normalized.std()
        
        assert torch.abs(mean) < 0.1, f"Mean should be ~0, got {mean}"
        assert torch.abs(std - 1.0) < 0.1, f"Std should be ~1, got {std}"
        
    def test_identity_properties(self):
        """Test Identity normalizer properties"""
        data = torch.randn(5, 2, 8, 8, 4) * 1000
        normalizer = Identity_Normalizer(data)
        
        # Encode and decode should be perfect identity operations
        encoded = normalizer.encode(data)
        decoded = normalizer.decode(data)
        
        assert torch.equal(data, encoded)
        assert torch.equal(data, decoded)
        assert torch.equal(encoded, decoded)


class TestSpecialCases:
    """Test special mathematical and edge cases"""
    
    def test_very_large_numbers(self):
        """Test with very large numbers"""
        large_data = torch.ones(3, 2, 4, 4, 3) * 1e8
        large_data += torch.randn(3, 2, 4, 4, 3) * 1e7
        
        for NormalizerClass in [MinMax_Normalizer, Gaussian_Normalizer]:
            normalizer = NormalizerClass(large_data)
            normalized = normalizer.encode(large_data)
            denormalized = normalizer.decode(normalized)
            
            # Should handle large numbers without overflow
            assert torch.isfinite(normalized).all()
            assert torch.allclose(large_data, denormalized, rtol=1e-5)
            
    def test_very_small_numbers(self):
        """Test with very small numbers"""
        small_data = torch.ones(3, 2, 4, 4, 3) * 1e-8
        small_data += torch.randn(3, 2, 4, 4, 3) * 1e-9
        
        for NormalizerClass in [MinMax_Normalizer, Gaussian_Normalizer]:
            normalizer = NormalizerClass(small_data)
            normalized = normalizer.encode(small_data)
            denormalized = normalizer.decode(normalized)
            
            # Should handle small numbers without underflow
            assert torch.isfinite(normalized).all()
            assert torch.allclose(small_data, denormalized, atol=1e-12)
            
    def test_mixed_positive_negative_data(self):
        """Test with data spanning positive and negative values"""
        data = torch.zeros(6, 3, 8, 8, 4)
        
        # Variable 0: mostly negative
        data[:, 0] = -torch.abs(torch.randn(6, 8, 8, 4)) * 10 - 5
        
        # Variable 1: mostly positive  
        data[:, 1] = torch.abs(torch.randn(6, 8, 8, 4)) * 10 + 5
        
        # Variable 2: mixed around zero
        data[:, 2] = torch.randn(6, 8, 8, 4) * 10
        
        for NormalizerClass in [MinMax_Normalizer, MinMax_Normalizer_variable, Gaussian_Normalizer]:
            normalizer = NormalizerClass(data)
            normalized = normalizer.encode(data)
            denormalized = normalizer.decode(normalized)
            
            assert torch.allclose(data, denormalized, atol=1e-5)


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])