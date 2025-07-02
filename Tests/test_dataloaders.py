
# %% 
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from data_loaders import SpatioTemporalDataset

def test_spatiotemporal_dataset():
    """
    Comprehensive test suite for SpatioTemporalDataset
    """
    print("Testing SpatioTemporalDataset...")
    
    # Test 1: Basic functionality with simple data
    print("\n1. Testing basic functionality...")
    batch_size, variables, x_dim, y_dim, time_steps = 2, 3, 4, 5, 20
    input_window, prediction_steps = 5, 2
    
    # Create test data with known pattern (time index as values)
    data = np.zeros((batch_size, variables, x_dim, y_dim, time_steps))
    for t in range(time_steps):
        data[:, :, :, :, t] = t  # Each time step has value = time index
    
    dataset = SpatioTemporalDataset(data, input_window=input_window, prediction_steps=prediction_steps)
    
    
    # Check dataset length
    expected_samples = batch_size * (time_steps - input_window - prediction_steps + 1)
    assert len(dataset) == expected_samples, f"Expected {expected_samples} samples, got {len(dataset)}"
    print(f"✓ Dataset length correct: {len(dataset)} samples")
    
    # Test 2: Check sample shapes
    print("\n2. Testing sample shapes...")
    input_seq, target_seq = dataset[0]
    expected_input_shape = (variables, x_dim, y_dim, input_window)
    expected_target_shape = (variables, x_dim, y_dim, prediction_steps)
    
    assert input_seq.shape == expected_input_shape, f"Input shape mismatch: {input_seq.shape} vs {expected_input_shape}"
    assert target_seq.shape == expected_target_shape, f"Target shape mismatch: {target_seq.shape} vs {expected_target_shape}"
    print(f"✓ Input shape: {input_seq.shape}")
    print(f"✓ Target shape: {target_seq.shape}")
    
    # Test 3: Check temporal continuity
    print("\n3. Testing temporal continuity...")
    input_seq, target_seq = dataset[0]  # First sample from first batch item
    
    # Input should be time steps 0-4, target should be 5-6
    assert torch.all(input_seq == torch.arange(input_window).view(1, 1, 1, -1)), "Input sequence values incorrect"
    assert torch.all(target_seq == torch.arange(input_window, input_window + prediction_steps).view(1, 1, 1, -1)), "Target sequence values incorrect"
    print("✓ Temporal continuity verified for first sample")
    
    # Test 4: Check sliding window behavior
    print("\n4. Testing sliding window behavior...")
    input_seq1, target_seq1 = dataset[0]  # First window
    input_seq2, target_seq2 = dataset[1]  # Second window (should be shifted by 1)
    
    # Second input should start 1 step later
    expected_input2 = torch.arange(1, input_window + 1).view(1, 1, 1, -1)
    expected_target2 = torch.arange(input_window + 1, input_window + prediction_steps + 1).view(1, 1, 1, -1)
    
    assert torch.all(input_seq2 == expected_input2), "Second input sequence incorrect"
    assert torch.all(target_seq2 == expected_target2), "Second target sequence incorrect"
    print("✓ Sliding window behavior verified")
    
    # Test 5: Test different batch samples
    print("\n5. Testing different batch samples...")
    samples_per_batch = time_steps - input_window - prediction_steps + 1
    first_batch_first_idx = 0  # First sample from batch 0 (time 0-4)
    second_batch_first_idx = samples_per_batch  # First sample from batch 1 (also time 0-4)
    
    input_seq_b1, target_seq_b1 = dataset[first_batch_first_idx]  # First sample from batch 0
    input_seq_b2, target_seq_b2 = dataset[second_batch_first_idx]  # First sample from batch 1
    
    # Both should have same temporal pattern since our test data is identical across batches
    # and we're comparing the same temporal window (both start at time 0)
    assert torch.allclose(input_seq_b1, input_seq_b2), "Cross-batch consistency failed"
    assert torch.allclose(target_seq_b1, target_seq_b2), "Cross-batch target consistency failed"
    print("✓ Cross-batch consistency verified")
    
    # Test 6: Edge cases
    print("\n6. Testing edge cases...")
    
    # Test with minimum valid parameters
    min_data = np.random.randn(1, 1, 2, 2, 3)  # Minimal data
    min_dataset = SpatioTemporalDataset(min_data, input_window=1, prediction_steps=1)
    assert len(min_dataset) == 2, "Minimum case failed"  # Should have 2 samples: (0->1) and (1->2)
    print("✓ Minimum parameter case works")
    
    # Test error case - window too large
    try:
        invalid_dataset = SpatioTemporalDataset(min_data, input_window=2, prediction_steps=2)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✓ Correctly caught error for invalid parameters: {str(e)}")
    
    # Test 7: DataLoader compatibility
    print("\n7. Testing DataLoader compatibility...")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    batch_input, batch_target = next(iter(dataloader))
    
    expected_batch_input_shape = (4, variables, x_dim, y_dim, input_window)
    expected_batch_target_shape = (4, variables, x_dim, y_dim, prediction_steps)
    
    assert batch_input.shape == expected_batch_input_shape, f"Batch input shape mismatch: {batch_input.shape}"
    assert batch_target.shape == expected_batch_target_shape, f"Batch target shape mismatch: {batch_target.shape}"
    print(f"✓ DataLoader compatibility verified")
    print(f"  Batch input shape: {batch_input.shape}")
    print(f"  Batch target shape: {batch_target.shape}")
    
    # Test 8: Multi-step prediction
    print("\n8. Testing multi-step prediction...")
    multi_step_dataset = SpatioTemporalDataset(data, input_window=3, prediction_steps=5)
    input_seq, target_seq = multi_step_dataset[0]
    
    assert input_seq.shape[-1] == 3, "Multi-step input window incorrect"
    assert target_seq.shape[-1] == 5, "Multi-step prediction incorrect"
    print("✓ Multi-step prediction verified")
    
    print("\n🎉 All tests passed! Your SpatioTemporalDataset implementation is correct.")
    
    # Summary statistics
    print(f"\nDataset Summary:")
    print(f"  Original data shape: {data.shape}")
    print(f"  Input window: {input_window}, Prediction steps: {prediction_steps}")
    print(f"  Total samples generated: {len(dataset)}")
    print(f"  Samples per batch item: {samples_per_batch}")


def demo_usage():
    """
    Demonstrate typical usage of the dataset
    """
    print("\n" + "="*50)
    print("DEMO: Typical Usage")
    print("="*50)
    
    # Create sample weather data (temperature, humidity, pressure over time)
    batch_size, variables, x_dim, y_dim, time_steps = 3, 3, 10, 10, 100
    
    # Simulate some realistic spatiotemporal data
    np.random.seed(42)
    data = np.random.randn(batch_size, variables, x_dim, y_dim, time_steps)
    
    # Add some temporal correlation to make it more realistic
    for t in range(1, time_steps):
        data[:, :, :, :, t] = 0.8 * data[:, :, :, :, t-1] + 0.2 * data[:, :, :, :, t]
    
    # Create dataset
    dataset = SpatioTemporalDataset(data, input_window=10, prediction_steps=3)
    
    print(f"Dataset created with {len(dataset)} samples")
    print(f"Each input sequence: {dataset[0][0].shape}")
    print(f"Each target sequence: {dataset[0][1].shape}")
    
    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
    
    print(f"\nDataLoader created with batch_size=8")
    print(f"Number of batches: {len(dataloader)}")
    
    # Show first batch
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        print(f"\nBatch {batch_idx + 1}:")
        print(f"  Input batch shape: {inputs.shape}")
        print(f"  Target batch shape: {targets.shape}")
        if batch_idx == 0:  # Only show first batch
            break


if __name__ == "__main__":
    test_spatiotemporal_dataset()
    demo_usage()