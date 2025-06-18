import torch

def get_stencil(dims, deriv_order, taylor_order=2):
    """
    Get finite difference stencils for various derivative orders and accuracy orders.
    
    Args:
        dims (int): Spatial dimensions (1 or 2)
        deriv_order (int): Order of derivative (0, 1, or 2)
        taylor_order (int): Order of accuracy (2, 4, 6, 8, 10)
    
    Returns:
        torch.Tensor: Finite difference stencil
    """
    
    if dims == 1:
        # 1D Stencils
        if deriv_order == 0:  # Identity convolution
            return torch.tensor([
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0]
            ], dtype=torch.float32)
            
        elif deriv_order == 1:
            if taylor_order == 2:
                return torch.tensor([
                    [0, -1/2, 0],
                    [0, 0, 0],
                    [0, 1/2, 0]
                ], dtype=torch.float32)
            elif taylor_order == 4:
                return torch.tensor([
                    [0, 0, 1/12, 0, 0],
                    [0, 0, -2/3, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 2/3, 0, 0],
                    [0, 0, -1/12, 0, 0]
                ], dtype=torch.float32)
            elif taylor_order == 6:
                return torch.tensor([
                    [0, 0, 0, -1/60, 0, 0, 0],
                    [0, 0, 0, 3/20, 0, 0, 0],
                    [0, 0, 0, -3/4, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 3/4, 0, 0, 0],
                    [0, 0, 0, -3/20, 0, 0, 0],
                    [0, 0, 0, 1/60, 0, 0, 0]
                ], dtype=torch.float32)
            elif taylor_order == 8:
                return torch.tensor([
                    [0, 0, 0, 0, 1/280, 0, 0, 0, 0],
                    [0, 0, 0, 0, -4/105, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1/5, 0, 0, 0, 0],
                    [0, 0, 0, 0, -4/5, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 4/5, 0, 0, 0, 0],
                    [0, 0, 0, 0, -1/5, 0, 0, 0, 0],
                    [0, 0, 0, 0, 4/105, 0, 0, 0, 0],
                    [0, 0, 0, 0, -1/280, 0, 0, 0, 0]
                ], dtype=torch.float32)
            elif taylor_order == 10:
                return torch.tensor([
                    [0, 0, 0, 0, 0, -1/1260, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 5/504, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, -5/84, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 5/21, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, -5/6, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 5/6, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, -5/21, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 5/84, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, -5/504, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1/1260, 0, 0, 0, 0, 0]
                ], dtype=torch.float32)
                
        elif deriv_order == 2:
            if taylor_order == 2:
                return torch.tensor([
                    [0, 1, 0],
                    [0, -2, 0],
                    [0, 1, 0]
                ], dtype=torch.float32)
            elif taylor_order == 4:
                return torch.tensor([
                    [0, 0, -1/12, 0, 0],
                    [0, 0, 4/3, 0, 0],
                    [0, 0, -5/2, 0, 0],
                    [0, 0, 4/3, 0, 0],
                    [0, 0, -1/12, 0, 0]
                ], dtype=torch.float32)
            elif taylor_order == 6:
                return torch.tensor([
                    [0, 0, 0, 1/90, 0, 0, 0],
                    [0, 0, 0, -3/20, 0, 0, 0],
                    [0, 0, 0, 3/2, 0, 0, 0],
                    [0, 0, 0, -49/18, 0, 0, 0],
                    [0, 0, 0, 3/2, 0, 0, 0],
                    [0, 0, 0, -3/20, 0, 0, 0],
                    [0, 0, 0, 1/90, 0, 0, 0]
                ], dtype=torch.float32)
            elif taylor_order == 8:
                return torch.tensor([
                    [0, 0, 0, 0, -1/560, 0, 0, 0, 0],
                    [0, 0, 0, 0, 8/315, 0, 0, 0, 0],
                    [0, 0, 0, 0, -1/5, 0, 0, 0, 0],
                    [0, 0, 0, 0, 8/5, 0, 0, 0, 0],
                    [0, 0, 0, 0, -205/72, 0, 0, 0, 0],
                    [0, 0, 0, 0, 8/5, 0, 0, 0, 0],
                    [0, 0, 0, 0, -1/5, 0, 0, 0, 0],
                    [0, 0, 0, 0, 8/315, 0, 0, 0, 0],
                    [0, 0, 0, 0, -1/560, 0, 0, 0, 0]
                ], dtype=torch.float32)
            elif taylor_order == 10:
                return torch.tensor([
                    [0, 0, 0, 0, 0, 1/3150, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, -5/1008, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 5/126, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, -5/21, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 5/3, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, -5269/1800, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 5/3, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, -5/21, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 5/126, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, -5/1008, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1/3150, 0, 0, 0, 0, 0]
                ], dtype=torch.float32)
                
    elif dims == 2:
        # 2D Stencils - Cross-shaped stencils for isotropic operators
        if deriv_order == 0:  # Identity convolution
            return torch.tensor([
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0]
            ], dtype=torch.float32)
        elif deriv_order == 1:
            if taylor_order == 2:
                return torch.tensor([
                    [0., -1/2., 0.],
                    [-1/2, 0, 1/2],
                    [0., 1/2, 0.]
                ], dtype=torch.float32)
            elif taylor_order == 4:
                return torch.tensor([
                    [0., 0., 1/12., 0., 0.],
                    [0., 0., -2/3., 0., 0.],
                    [1/12., -2/3., 0., 2/3., -1/12.],
                    [0., 0., 2/3., 0., 0.],
                    [0., 0., -1/12., 0., 0.]
                ], dtype=torch.float32)
            elif taylor_order == 6:
                return torch.tensor([
                    [0., 0., 0., -1/60., 0., 0., 0.],
                    [0., 0., 0., 3/20., 0., 0., 0.],
                    [0., 0., 0., -3/4., 0., 0., 0.],
                    [-1/60., 3/20., -3/4., 0., 3/4., -3/20., 1/60.],
                    [0., 0., 0., 3/4., 0., 0., 0.],
                    [0., 0., 0., -3/20., 0., 0., 0.],
                    [0., 0., 0., 1/60., 0., 0., 0.]
                ], dtype=torch.float32)
            elif taylor_order == 8:
                return torch.tensor([
                    [0., 0., 0., 0., 1/280., 0., 0., 0., 0.],
                    [0., 0., 0., 0., -4/105., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 1/5., 0., 0., 0., 0.],
                    [0., 0., 0., 0., -4/5., 0., 0., 0., 0.],
                    [1/280., -4/105., 1/5., -4/5., 0., 4/5., -1/5., 4/105., -1/280.],
                    [0., 0., 0., 0., 4/5., 0., 0., 0., 0.],
                    [0., 0., 0., 0., -1/5., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 4/105., 0., 0., 0., 0.],
                    [0., 0., 0., 0., -1/280., 0., 0., 0., 0.]
                ], dtype=torch.float32)
                
        elif deriv_order == 2:
            if taylor_order == 2:
                return torch.tensor([
                    [0., 1., 0.],
                    [1., -4., 1.],
                    [0., 1., 0.]
                ], dtype=torch.float32)
            elif taylor_order == 4:
                return torch.tensor([
                    [0, 0, -1/12, 0, 0],
                    [0, 0, 4/3, 0, 0],
                    [-1/12, 4/3, -5, 4/3, -1/12],
                    [0, 0, 4/3, 0, 0],
                    [0, 0, -1/12, 0, 0]
                ], dtype=torch.float32)
            elif taylor_order == 6:
                return torch.tensor([
                    [0, 0, 0, 1/90, 0, 0, 0],
                    [0, 0, 0, -3/20, 0, 0, 0],
                    [0, 0, 0, 3/2, 0, 0, 0],
                    [1/90, -3/20, 3/2, -49/9, 3/2, -3/20, 1/90],
                    [0, 0, 0, 3/2, 0, 0, 0],
                    [0, 0, 0, -3/20, 0, 0, 0],
                    [0, 0, 0, 1/90, 0, 0, 0]
                ], dtype=torch.float32)
            elif taylor_order == 8:
                return torch.tensor([
                    [0, 0, 0, 0, -1/560, 0, 0, 0, 0],
                    [0, 0, 0, 0, 8/315, 0, 0, 0, 0],
                    [0, 0, 0, 0, -1/5, 0, 0, 0, 0],
                    [0, 0, 0, 0, 8/5, 0, 0, 0, 0],
                    [-1/560, 8/315, -1/5, 8/5, -205/36, 8/5, -1/5, 8/315, -1/560],
                    [0, 0, 0, 0, 8/5, 0, 0, 0, 0],
                    [0, 0, 0, 0, -1/5, 0, 0, 0, 0],
                    [0, 0, 0, 0, 8/315, 0, 0, 0, 0],
                    [0, 0, 0, 0, -1/560, 0, 0, 0, 0]
                ], dtype=torch.float32)
            elif taylor_order == 10:
                return torch.tensor([
                    [0, 0, 0, 0, 0, 1/3150, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, -5/1008, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 5/126, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, -5/21, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 5/3, 0, 0, 0, 0, 0],
                    [1/3150, -5/1008, 5/126, -5/21, 5/3, -5269/900, 5/3, -5/21, 5/126, -5/1008, 1/3150],
                    [0, 0, 0, 0, 0, 5/3, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, -5/21, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 5/126, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, -5/1008, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1/3150, 0, 0, 0, 0, 0]
                ], dtype=torch.float32)

    raise ValueError(f"Invalid stencil parameters: dims={dims}, deriv_order={deriv_order}, taylor_order={taylor_order}")


def get_compact_stencil(dims, deriv_order, taylor_order=4):
    """
    Get compact finite difference stencils (Padé schemes) for higher accuracy with smaller stencils.
    These require solving tridiagonal systems but offer better spectral properties.
    
    Args:
        dims (int): Spatial dimensions (1 or 2)
        deriv_order (int): Order of derivative (1 or 2)
        taylor_order (int): Order of accuracy (4, 6, 8)
    
    Returns:
        tuple: (LHS_stencil, RHS_stencil) for compact schemes
    """
    
    if dims == 1:
        if deriv_order == 1:
            if taylor_order == 4:
                # Fourth-order compact scheme: α*f'_{i-1} + f'_i + α*f'_{i+1} = a*(f_{i+1} - f_{i-1})/(2h)
                # α = 1/4, a = 3/2
                lhs = torch.tensor([
                    [0, 1/4, 0],
                    [0, 1, 0], 
                    [0, 1/4, 0]
                ], dtype=torch.float32)
                rhs = torch.tensor([
                    [0, -3/4, 0],
                    [0, 0, 0],
                    [0, 3/4, 0]
                ], dtype=torch.float32)
                return lhs, rhs
            elif taylor_order == 6:
                # Sixth-order compact scheme: α*f'_{i-1} + f'_i + α*f'_{i+1} = a*(f_{i+1} - f_{i-1})/(2h) + b*(f_{i+2} - f_{i-2})/(4h)
                # α = 1/3, a = 14/9, b = 1/9
                lhs = torch.tensor([
                    [0, 0, 0, 0, 0],
                    [0, 1/3, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 1/3, 0],
                    [0, 0, 0, 0, 0]
                ], dtype=torch.float32)
                rhs = torch.tensor([
                    [0, 0, 1/36, 0, 0],
                    [0, 0, -14/18, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 14/18, 0, 0],
                    [0, 0, -1/36, 0, 0]
                ], dtype=torch.float32)
                return lhs, rhs
                
        elif deriv_order == 2:
            if taylor_order == 4:
                # Fourth-order compact scheme for second derivative
                # α*f''_{i-1} + f''_i + α*f''_{i+1} = a*(f_{i+1} - 2f_i + f_{i-1})/h²
                # α = 1/10, a = 12/10
                lhs = torch.tensor([
                    [0, 1/10, 0],
                    [0, 1, 0],
                    [0, 1/10, 0]
                ], dtype=torch.float32)
                rhs = torch.tensor([
                    [0, 6/5, 0],
                    [0, -12/5, 0],
                    [0, 6/5, 0]
                ], dtype=torch.float32)
                return lhs, rhs
                
    raise ValueError(f"Compact stencil not implemented for dims={dims}, deriv_order={deriv_order}, taylor_order={taylor_order}")

# Additional utility functions for higher-order operations

def get_mixed_derivative_stencil(deriv_orders, taylor_order=2):
    """
    Get stencils for mixed derivatives like ∂²f/∂x∂y
    
    Args:
        deriv_orders (tuple): Orders of derivatives in each direction, e.g., (1, 1) for ∂²f/∂x∂y
        taylor_order (int): Order of accuracy
        
    Returns:
        torch.Tensor: Mixed derivative stencil
    """
    if deriv_orders == (1, 1):  # ∂²f/∂x∂y
        if taylor_order == 2:
            return torch.tensor([
                [1/4, 0, -1/4],
                [0, 0, 0],
                [-1/4, 0, 1/4]
            ], dtype=torch.float32)
        elif taylor_order == 4:
            return torch.tensor([
                [0, 0, 0, 0, 0],
                [0, -1/12, 0, 1/12, 0],
                [0, 0, 0, 0, 0],
                [0, 1/12, 0, -1/12, 0],
                [0, 0, 0, 0, 0]
            ], dtype=torch.float32)
            
    raise ValueError(f"Mixed derivative stencil not implemented for orders {deriv_orders}")

def get_biharmonic_stencil(taylor_order=2):
    """
    Get stencils for the biharmonic operator ∇⁴ = ∇²∇²
    
    Args:
        taylor_order (int): Order of accuracy
        
    Returns:
        torch.Tensor: Biharmonic stencil
    """
    if taylor_order == 2:
        return torch.tensor([
            [0, 0, 1, 0, 0],
            [0, 2, -8, 2, 0],
            [1, -8, 20, -8, 1],
            [0, 2, -8, 2, 0],
            [0, 0, 1, 0, 0]
        ], dtype=torch.float32)
    elif taylor_order == 4:
        # More complex 7x7 stencil for 4th order accuracy
        stencil = torch.zeros(7, 7, dtype=torch.float32)
        # Center
        stencil[3, 3] = 12.0
        # First ring
        stencil[3, 2] = stencil[3, 4] = stencil[2, 3] = stencil[4, 3] = -8.0
        # Diagonal first ring  
        stencil[2, 2] = stencil[2, 4] = stencil[4, 2] = stencil[4, 4] = 2.0
        # Second ring
        stencil[3, 1] = stencil[3, 5] = stencil[1, 3] = stencil[5, 3] = 1.0
        return stencil
        
    raise ValueError(f"Biharmonic stencil not implemented for taylor_order={taylor_order}")

def get_anisotropic_stencil(deriv_order, taylor_order, aspect_ratio=1.0):
    """
    Get anisotropic stencils for problems with different scaling in x and y directions
    
    Args:
        deriv_order (int): Order of derivative
        taylor_order (int): Order of accuracy
        aspect_ratio (float): Ratio of grid spacing dy/dx
        
    Returns:
        torch.Tensor: Anisotropic stencil
    """
    # Get isotropic stencil first
    stencil = get_stencil(2, deriv_order, taylor_order)
    
    # Scale the y-direction coefficients by aspect_ratio^deriv_order
    # This is a simplified approach - full anisotropic stencils are more complex
    modified_stencil = stencil.clone()
    center = stencil.shape[0] // 2
    
    # Scale vertical contributions
    for i in range(stencil.shape[0]):
        if i != center:
            modified_stencil[i, center] *= aspect_ratio**deriv_order
            
    return modified_stencil

def optimize_stencil_coefficients(target_order, stencil_size, derivative_order=1):
    """
    Optimize finite difference coefficients for a given stencil size and target accuracy order.
    This uses the method of undetermined coefficients.
    
    Args:
        target_order (int): Target order of accuracy
        stencil_size (int): Size of the stencil (should be odd)
        derivative_order (int): Order of derivative
        
    Returns:
        torch.Tensor: Optimized 1D stencil coefficients
    """
    from scipy.linalg import solve
    
    if stencil_size % 2 == 0:
        raise ValueError("Stencil size must be odd")
        
    center = stencil_size // 2
    points = torch.arange(-center, center + 1, dtype=torch.float64)
    
    # Set up the system of equations based on Taylor series
    # We want the coefficients to satisfy certain moment conditions
    n_equations = target_order + derivative_order
    A = torch.zeros(n_equations, stencil_size, dtype=torch.float64)
    b = torch.zeros(n_equations, dtype=torch.float64)
    
    for i in range(n_equations):
        if i == derivative_order:
            b[i] = 1.0  # This should give the derivative
        A[i] = points**i / torch.factorial(torch.tensor(i, dtype=torch.float64))
    
    # Solve the linear system
    coeffs = torch.linalg.solve(A.T @ A, A.T @ b)
    
    return coeffs.float()

# %% 
