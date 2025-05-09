import numpy as np

def convert_to_numpy(tensor):
    """
    Convert a PyTorch tensor to NumPy array if needed.
    
    Args:
        tensor: Input tensor or array
        
    Returns:
        NumPy array
    """
    if hasattr(tensor, 'detach') and hasattr(tensor, 'cpu') and hasattr(tensor, 'numpy'):
        return tensor.detach().cpu().numpy()
    return np.asarray(tensor)

def MSE(prediction, target):
    """
    Calculate Mean Squared Error (MSE) for batched data.
    
    Args:
        prediction: NumPy array or PyTorch tensor of shape [batch_size, ...]
        target: NumPy array or PyTorch tensor of shape [batch_size, ...]
        
    Returns:
        Dictionary containing per-sample MSE values and batch average
    """
    # Convert to numpy if tensors
    prediction = convert_to_numpy(prediction)
    target = convert_to_numpy(target)
    
    # Check shapes match
    if prediction.shape != target.shape:
        raise ValueError(f"Prediction shape {prediction.shape} does not match target shape {target.shape}")
    
    batch_size = prediction.shape[0]
    
    # Reshape to [batch_size, -1]
    pred_flat = prediction.reshape(batch_size, -1)
    target_flat = target.reshape(batch_size, -1)
    
    # Vectorized MSE calculation
    squared_diff = (pred_flat - target_flat)**2
    mse_values = np.mean(squared_diff, axis=1)
    
    # Calculate average across batch
    avg_mse = np.mean(mse_values)
    
    return {
        'per_sample': mse_values,
        'average': avg_mse
    }

def RMSE(prediction, target):
    """
    Calculate Root Mean Squared Error (RMSE) for batched data.
    
    Args:
        prediction: NumPy array or PyTorch tensor of shape [batch_size, ...]
        target: NumPy array or PyTorch tensor of shape [batch_size, ...]
        
    Returns:
        Dictionary containing per-sample RMSE values and batch average
    """
    # Get MSE first
    mse_result = MSE(prediction, target)
    
    # Take square root of MSE values
    rmse_values = np.sqrt(mse_result['per_sample'])
    avg_rmse = np.mean(rmse_values)
    
    return {
        'per_sample': rmse_values,
        'average': avg_rmse
    }

def NMSE(prediction, target, normalization='variance'):
    """
    Calculate Normalized Mean Squared Error (NMSE) for batched data.
    
    Args:
        prediction: NumPy array or PyTorch tensor of shape [batch_size, ...]
        target: NumPy array or PyTorch tensor of shape [batch_size, ...]
        normalization: Method for normalization (default: 'variance')
                      Options: 'variance', 'mean_squared', 'max_squared', 'min_max_squared'
        
    Returns:
        Dictionary containing per-sample NMSE values and batch average
    """
    # Convert to numpy if tensors
    prediction = convert_to_numpy(prediction)
    target = convert_to_numpy(target)
    
    # Check shapes match
    if prediction.shape != target.shape:
        raise ValueError(f"Prediction shape {prediction.shape} does not match target shape {target.shape}")
    
    batch_size = prediction.shape[0]
    
    # Reshape to [batch_size, -1]
    pred_flat = prediction.reshape(batch_size, -1)
    target_flat = target.reshape(batch_size, -1)
    
    # Get MSE values
    squared_diff = (pred_flat - target_flat)**2
    mse_values = np.mean(squared_diff, axis=1)
    
    # Initialize normalization factors array with NaN values
    norm_factors = np.full_like(mse_values, np.nan)
    
    if normalization == 'variance':
        # Normalize by variance of target values (traditional NMSE)
        target_var = np.var(target_flat, axis=1)
        positive_var_mask = target_var > 0
        norm_factors[positive_var_mask] = target_var[positive_var_mask]
    
    elif normalization == 'mean_squared':
        # Normalize by squared mean of target values
        target_mean = np.mean(target_flat, axis=1)
        valid_mean = np.abs(target_mean) > 0
        norm_factors[valid_mean] = target_mean[valid_mean]**2
    
    elif normalization == 'max_squared':
        # Normalize by squared max value of each target
        target_max = np.max(np.abs(target_flat), axis=1)
        valid_max = target_max > 0
        norm_factors[valid_max] = target_max[valid_max]**2
    
    elif normalization == 'min_max_squared':
        # Normalize by squared range of target values
        target_min = np.min(target_flat, axis=1)
        target_max = np.max(target_flat, axis=1)
        target_range = target_max - target_min
        valid_range = target_range > 0
        norm_factors[valid_range] = target_range[valid_range]**2
    
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")
    
    # Calculate NMSE only where normalization factor is valid
    nmse_values = np.full_like(mse_values, np.nan)
    valid_norm = ~np.isnan(norm_factors) & (norm_factors > 0)
    nmse_values[valid_norm] = mse_values[valid_norm] / norm_factors[valid_norm]
    
    # Calculate average NMSE, ignoring NaN values
    avg_nmse = np.nanmean(nmse_values)
    
    return {
        'per_sample': nmse_values,
        'average': avg_nmse
    }

def NRMSE(prediction, target, normalization='min_max'):
    """
    Calculate Normalized Root Mean Squared Error (NRMSE) for batched data.
    
    Args:
        prediction: NumPy array or PyTorch tensor of shape [batch_size, ...]
        target: NumPy array or PyTorch tensor of shape [batch_size, ...]
        normalization: Method for normalization (default: 'min_max')
                      Options: 'min_max', 'mean', 'std', 'range'
        
    Returns:
        Dictionary containing per-sample NRMSE values and batch average
    """
    # Convert to numpy if tensors
    prediction = convert_to_numpy(prediction)
    target = convert_to_numpy(target)
    
    # Check shapes match
    if prediction.shape != target.shape:
        raise ValueError(f"Prediction shape {prediction.shape} does not match target shape {target.shape}")
    
    batch_size = prediction.shape[0]
    
    # Get RMSE values
    rmse_result = RMSE(prediction, target)
    rmse_values = rmse_result['per_sample']
    
    # Reshape to [batch_size, -1]
    target_flat = target.reshape(batch_size, -1)
    
    # Initialize normalization factors array
    norm_factors = np.ones_like(rmse_values)
    
    if normalization == 'min_max':
        # Normalize by range of target values for each sample
        target_min = np.min(target_flat, axis=1)
        target_max = np.max(target_flat, axis=1)
        target_range = target_max - target_min
        # Avoid division by zero
        valid_range = target_range > 0
        norm_factors[valid_range] = target_range[valid_range]
    elif normalization == 'mean':
        # Normalize by mean of target values
        target_mean = np.mean(target_flat, axis=1)
        # Avoid division by zero
        valid_mean = np.abs(target_mean) > 0
        norm_factors[valid_mean] = np.abs(target_mean[valid_mean])
    elif normalization == 'std':
        # Normalize by standard deviation of target values
        target_std = np.std(target_flat, axis=1)
        # Avoid division by zero
        valid_std = target_std > 0
        norm_factors[valid_std] = target_std[valid_std]
    elif normalization == 'range':
        # Fixed normalization by target data range
        norm_factors = np.ones_like(rmse_values)
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")
    
    # Calculate NRMSE
    nrmse_values = rmse_values / norm_factors
    
    # Handle any remaining infinity or NaN values
    nrmse_values = np.nan_to_num(nrmse_values, nan=np.nan, posinf=np.nan, neginf=np.nan)
    
    # Calculate average NRMSE, ignoring NaN values
    avg_nrmse = np.nanmean(nrmse_values)
    
    return {
        'per_sample': nrmse_values,
        'average': avg_nrmse
    }



def calculate_all_metrics(prediction, target):
    """
    Calculate MSE, RMSE, and NMSE across batched tensors.
    
    Args:
        prediction: NumPy array or PyTorch tensor of shape [batch_size, ...]
        target: NumPy array or PyTorch tensor of shape [batch_size, ...]
        
    Returns:
        Dictionary containing MSE, RMSE, and NMSE values for each item in the batch
        and the average across the batch
    """
    return {
        'mse': MSE(prediction, target),
        'rmse': RMSE(prediction, target),
        'nmse': NMSE(prediction, target)
    }

# %% 

