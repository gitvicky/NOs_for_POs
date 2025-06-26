"""
Comprehensive test suite for SpatioTemporalDataset class.

Tests the dataset functionality with sequential spatio-temporal data
to ensure proper windowing, indexing, and data integrity.
"""

import pytest
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import sys
import os


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Expts.data_loaders import SpatioTemporalDataset


def create_sequential_test_data(num_sims=3, num_vars=2, Nx=16, Ny=16, Nt=100, pattern_type="linear"):
    """
    Create synthetic sequential spatio-temporal data for testing.
    
    Args:
        num_sims (int): Number of simulations
        num_vars (int): Number of variables
        Nx, Ny (int): Spatial dimensions
        Nt (int): Number of time steps
        pattern_type (str): Type of temporal pattern ("linear", "sinusoidal", "exponential")
    
    Returns:
        numpy.ndarray: Data of shape (num_sims, num_vars, Nx, Ny, Nt)
    """
    data = np.zeros((num_sims, num_vars, Nx, Ny, Nt))
    
    for sim in range(num_sims):
        for var in range(num_vars):
            for x in range(Nx):
                for y in range(Ny):
                    # Create unique but predictable patterns for each location and variable
                    base_value = (sim + 1) * 10 + var * 5 + x * 0.1 + y * 0.01
                    
                    for t in range(Nt):
                        if pattern_type == "linear":
                            # Linear progression over time
                            data[sim, var, x, y, t] = base_value + t * 0.1
                        elif pattern_type == "sinusoidal":
                            # Sinusoidal pattern with different frequencies
                            frequency = 0.1 + var * 0.05 + (x + y) * 0.001
                            data[sim, var, x, y, t] = base_value + np.sin(frequency * t) * 5
                        elif pattern_type == "exponential":
                            # Exponential decay/growth
                            growth_rate = 0.01 + var * 0.005
                            data[sim, var, x, y, t] = base_value * np.exp(growth_rate * t)
                        else:
                            # Random walk
                            if t == 0:
                                data[sim, var, x, y, t] = base_value
                            else:
                                data[sim, var, x, y, t] = data[sim, var, x, y, t-1] + np.random.normal(0, 0.1)
    
    return data


class TestSpatioTemporalDataset:
    """Test suite for SpatioTemporalDataset"""
    
    def test_basic_initialization(self):
        """Test basic dataset initialization"""
        # Create test data
        data = create_sequential_test_data(num_sims=2, num_vars=3, Nx=8, Ny=8, Nt=50)
        
        # Initialize dataset
        dataset = SpatioTemporalDataset(data, input_window=10, prediction_steps=5)
        
        # Check basic properties
        assert dataset.batch_size == 2
        assert dataset.variables == 3
        assert dataset.x_dim == 8
        assert dataset.y_dim == 8
        assert dataset.time_steps == 50
        assert dataset.input_window == 10
        assert dataset.prediction_steps == 5
        
        # Check data type conversion
        assert isinstance(dataset.data, torch.Tensor)
        assert dataset.data.dtype == torch.float32
        
    def test_indices_creation(self):
        """Test that indices are created correctly"""
        data = create_sequential_test_data(num_sims=2, num_vars=2, Nx=4, Ny=4, Nt=20)
        dataset = SpatioTemporalDataset(data, input_window=5, prediction_steps=3)
        
        # With input_window=5 and prediction_steps=3, we need 8 consecutive time steps
        # So for Nt=20, we should have 20-8+1=13 valid starting positions per simulation
        # With 2 simulations, total should be 2*13=26
        expected_length = 2 * (20 - 8 + 1)
        assert len(dataset.indices) == expected_length
        assert len(dataset) == expected_length
        
        # Check that indices are properly formatted
        for sample_idx, start_idx in dataset.indices:
            assert 0 <= sample_idx < 2  # Valid simulation index
            assert 0 <= start_idx <= 12  # Valid start index
            
    def test_getitem_functionality(self):
        """Test that __getitem__ returns correct data shapes and values"""
        data = create_sequential_test_data(num_sims=2, num_vars=3, Nx=6, Ny=6, Nt=30, pattern_type="linear")
        dataset = SpatioTemporalDataset(data, input_window=8, prediction_steps=4)
        
        # Get first sample
        input_seq, target_seq = dataset[0]
        
        # Check shapes
        assert input_seq.shape == (3, 6, 6, 8)  # (variables, x_dim, y_dim, input_window)
        assert target_seq.shape == (3, 6, 6, 4)  # (variables, x_dim, y_dim, prediction_steps)
        
        # Check data types
        assert isinstance(input_seq, torch.Tensor)
        assert isinstance(target_seq, torch.Tensor)
        assert input_seq.dtype == torch.float32
        assert target_seq.dtype == torch.float32
        
    def test_temporal_continuity(self):
        """Test that input and target sequences are temporally continuous"""
        # Create data with a clear linear pattern for easy verification
        data = create_sequential_test_data(num_sims=1, num_vars=1, Nx=4, Ny=4, Nt=20, pattern_type="linear")
        dataset = SpatioTemporalDataset(data, input_window=5, prediction_steps=3)
        
        # Get the first sample (should start at t=0)
        input_seq, target_seq = dataset[0]
        
        # For linear pattern, values should increase by 0.1 each time step
        # Check a specific spatial location (0,0) for variable 0
        input_values = input_seq[0, 0, 0, :].numpy()  # Time steps 0-4
        target_values = target_seq[0, 0, 0, :].numpy()  # Time steps 5-7
        
        # Verify temporal continuity
        for i in range(len(input_values) - 1):
            diff = input_values[i+1] - input_values[i]
            assert abs(diff - 0.1) < 1e-6, f"Input sequence not continuous at step {i}"
            
        for i in range(len(target_values) - 1):
            diff = target_values[i+1] - target_values[i]
            assert abs(diff - 0.1) < 1e-6, f"Target sequence not continuous at step {i}"
            
        # Check continuity between input and target
        last_input = input_values[-1]
        first_target = target_values[0]
        diff = first_target - last_input
        assert abs(diff - 0.1) < 1e-6, "Gap between input and target sequences"
        
    def test_different_window_sizes(self):
        """Test dataset with different window sizes"""
        data = create_sequential_test_data(num_sims=1, num_vars=2, Nx=8, Ny=8, Nt=50)
        
        # Test various combinations
        test_cases = [
            (1, 1),    # Minimal case
            (10, 5),   # Standard case
            (20, 1),   # Large input, small prediction
            (5, 15),   # Small input, large prediction
            (25, 25),  # Equal input and prediction
        ]
        
        for input_window, prediction_steps in test_cases:
            if input_window + prediction_steps <= 50:  # Ensure valid configuration
                dataset = SpatioTemporalDataset(data, input_window=input_window, prediction_steps=prediction_steps)
                
                # Check that we can get samples
                assert len(dataset) > 0
                
                # Get a sample and check shapes
                input_seq, target_seq = dataset[0]
                assert input_seq.shape == (2, 8, 8, input_window)
                assert target_seq.shape == (2, 8, 8, prediction_steps)
                
    def test_multiple_simulations(self):
        """Test that dataset correctly handles multiple simulations"""
        data = create_sequential_test_data(num_sims=3, num_vars=2, Nx=4, Ny=4, Nt=20)
        dataset = SpatioTemporalDataset(data, input_window=6, prediction_steps=4)
        
        # With 3 simulations and 20-10+1=11 windows per simulation, should have 33 samples
        expected_samples = 3 * 11
        assert len(dataset) == expected_samples
        
        # Check that we get samples from different simulations
        simulation_indices = set()
        for i in range(len(dataset)):
            sample_idx, _ = dataset.indices[i]
            simulation_indices.add(sample_idx)
            
        assert simulation_indices == {0, 1, 2}  # Should include all 3 simulations
        
    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        data = create_sequential_test_data(num_sims=1, num_vars=1, Nx=4, Ny=4, Nt=10)
        
        # Case 1: Exact fit (input_window + prediction_steps = Nt)
        dataset1 = SpatioTemporalDataset(data, input_window=6, prediction_steps=4)
        assert len(dataset1) == 1  # Only one valid window
        
        # Case 2: Minimum viable case
        dataset2 = SpatioTemporalDataset(data, input_window=1, prediction_steps=1)
        assert len(dataset2) == 9  # 10-2+1 = 9 valid windows
        
        # Case 3: Invalid case (should have 0 samples)
        dataset3 = SpatioTemporalDataset(data, input_window=8, prediction_steps=5)  # Requires 13 time steps
        assert len(dataset3) == 0
        
    def test_data_integrity(self):
        """Test that original data is preserved correctly"""
        # Create data with known pattern
        data = create_sequential_test_data(num_sims=2, num_vars=1, Nx=3, Ny=3, Nt=15, pattern_type="linear")
        dataset = SpatioTemporalDataset(data, input_window=5, prediction_steps=2)
        
        # Get multiple samples and verify they match original data
        for idx in range(min(5, len(dataset))):
            input_seq, target_seq = dataset[idx]
            sim_idx, start_idx = dataset.indices[idx]
            
            # Check that input sequence matches original data
            original_input = data[sim_idx, :, :, :, start_idx:start_idx+5]
            assert np.allclose(input_seq.numpy(), original_input), f"Input mismatch at index {idx}"
            
            # Check that target sequence matches original data
            original_target = data[sim_idx, :, :, :, start_idx+5:start_idx+7]
            assert np.allclose(target_seq.numpy(), original_target), f"Target mismatch at index {idx}"
            
    def test_dataloader_compatibility(self):
        """Test that dataset works with PyTorch DataLoader"""
        data = create_sequential_test_data(num_sims=2, num_vars=3, Nx=8, Ny=8, Nt=30)
        dataset = SpatioTemporalDataset(data, input_window=10, prediction_steps=5)
        
        # Create DataLoader
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        
        # Test that we can iterate through the dataloader
        batches_processed = 0
        for batch_input, batch_target in dataloader:
            # Check batch shapes
            batch_size = batch_input.shape[0]
            assert batch_input.shape == (batch_size, 3, 8, 8, 10)
            assert batch_target.shape == (batch_size, 3, 8, 8, 5)
            
            batches_processed += 1
            if batches_processed >= 3:  # Test a few batches
                break
                
        assert batches_processed > 0, "No batches were processed"
        
    def test_different_data_patterns(self):
        """Test dataset with different temporal patterns"""
        patterns = ["linear", "sinusoidal", "exponential"]
        
        for pattern in patterns:
            data = create_sequential_test_data(num_sims=1, num_vars=2, Nx=6, Ny=6, Nt=25, pattern_type=pattern)
            dataset = SpatioTemporalDataset(data, input_window=8, prediction_steps=4)
            
            # Should be able to create dataset and get samples
            assert len(dataset) > 0
            
            input_seq, target_seq = dataset[0]
            assert input_seq.shape == (2, 6, 6, 8)
            assert target_seq.shape == (2, 6, 6, 4)
            
            # Check that data is finite and reasonable
            assert torch.isfinite(input_seq).all()
            assert torch.isfinite(target_seq).all()
            
    def test_memory_efficiency(self):
        """Test dataset with larger data to check memory efficiency"""
        # Create moderately large dataset
        data = create_sequential_test_data(num_sims=5, num_vars=4, Nx=32, Ny=32, Nt=100)
        dataset = SpatioTemporalDataset(data, input_window=20, prediction_steps=10)
        
        # Should handle large dataset without issues
        assert len(dataset) > 0
        
        # Test random access
        indices_to_test = [0, len(dataset)//4, len(dataset)//2, len(dataset)-1]
        for idx in indices_to_test:
            input_seq, target_seq = dataset[idx]
            assert input_seq.shape == (4, 32, 32, 20)
            assert target_seq.shape == (4, 32, 32, 10)
            
    def test_reproducibility(self):
        """Test that dataset access is deterministic and reproducible"""
        data = create_sequential_test_data(num_sims=2, num_vars=2, Nx=8, Ny=8, Nt=30, pattern_type="linear")
        dataset = SpatioTemporalDataset(data, input_window=10, prediction_steps=5)
        
        # Get the same sample multiple times
        sample1_input, sample1_target = dataset[5]
        sample2_input, sample2_target = dataset[5]
        
        # Should be identical
        assert torch.equal(sample1_input, sample2_input)
        assert torch.equal(sample1_target, sample2_target)
        
        # Different indices should give different results (unless they happen to be the same data)
        sample3_input, sample3_target = dataset[10]
        # These might be different simulations or time windows
        
    def test_sliding_window_progression(self):
        """Test that sliding windows progress correctly"""
        data = create_sequential_test_data(num_sims=1, num_vars=1, Nx=2, Ny=2, Nt=15, pattern_type="linear")
        dataset = SpatioTemporalDataset(data, input_window=3, prediction_steps=2)
        
        # Should have 15-5+1=11 samples for 1 simulation
        assert len(dataset) == 11
        
        # Check that consecutive samples represent shifted windows
        for i in range(min(3, len(dataset)-1)):
            input1, target1 = dataset[i]
            input2, target2 = dataset[i+1]
            
            # The second input's first timestep should equal first input's second timestep
            # (sliding window effect)
            assert torch.allclose(input1[0, 0, 0, 1:], input2[0, 0, 0, :-1])


# Additional utility tests
class TestDataCreation:
    """Test the synthetic data creation function"""
    
    def test_data_shape(self):
        """Test that created data has correct shape"""
        data = create_sequential_test_data(num_sims=3, num_vars=2, Nx=16, Ny=16, Nt=50)
        assert data.shape == (3, 2, 16, 16, 50)
        
    def test_different_patterns(self):
        """Test that different patterns produce different data"""
        linear_data = create_sequential_test_data(num_sims=1, num_vars=1, Nx=4, Ny=4, Nt=20, pattern_type="linear")
        sin_data = create_sequential_test_data(num_sims=1, num_vars=1, Nx=4, Ny=4, Nt=20, pattern_type="sinusoidal")
        
        # Should be different
        assert not np.allclose(linear_data, sin_data)
        
    def test_temporal_progression(self):
        """Test that linear pattern actually progresses linearly"""
        data = create_sequential_test_data(num_sims=1, num_vars=1, Nx=2, Ny=2, Nt=10, pattern_type="linear")
        
        # Check a specific location
        time_series = data[0, 0, 0, 0, :]
        differences = np.diff(time_series)
        
        # All differences should be approximately 0.1 for linear pattern
        assert np.allclose(differences, 0.1, atol=1e-10)


if __name__ == "__main__":
    # Run the tests
    print("Running SpatioTemporalDataset tests...")
    
    # You can run individual test classes like this:
    test_dataset = TestSpatioTemporalDataset()
    test_dataset.test_basic_initialization()
    test_dataset.test_indices_creation()
    test_dataset.test_getitem_functionality()
    test_dataset.test_temporal_continuity()
    test_dataset.test_multiple_simulations()
    test_dataset.test_dataloader_compatibility()
    
    print("All tests passed!")
    
    # Or use pytest for more comprehensive testing:
    # pytest.main([__file__, "-v"])