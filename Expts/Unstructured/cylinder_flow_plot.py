# %% 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.tri import Triangulation

import os 
plot_loc = os.getcwd() + '/Plots'

def create_cylinder_flow_plot(x, y, test_values, pred_values, run, 
                               batch_idx=0, var_idx=0,
                               cylinder_center=(0, 0), cylinder_radius=0.5,
                               time_steps=None, cmap='RdYlGn_r', vmin=None, vmax=None,
                               title="Re = 307 (Cylinder Flow)", figsize=(14, 6),
                               triangulation=None):
    """
    Create a visualization of flow around a cylinder at different time steps.
    Works with unstructured grids.
    
    Parameters:
    -----------
    x : numpy array
        X coordinates of the mesh (1D array of length Nxy)
    y : numpy array
        Y coordinates of the mesh (1D array of length Nxy)
    test_values : numpy array
        Ground truth field values with shape [Batch size, variables, Nxy, t]
    pred_values : numpy array
        Predicted field values with shape [Batch size, variables, Nxy, t]
    run : str or None
        Run identifier for saving
    batch_idx : int
        Index of batch to visualize (default: 0)
    var_idx : int
        Index of variable to visualize (default: 0)
    cylinder_center : tuple
        (x, y) coordinates of cylinder center
    cylinder_radius : float
        Radius of the cylinder
    time_steps : list
        List of time step indices to plot (if None, uses evenly spaced steps)
    cmap : str
        Colormap name
    vmin, vmax : float
        Min and max values for color scale (if None, computed from data)
    title : str
        Overall title for the figure
    figsize : tuple
        Figure size (width, height)
    triangulation : matplotlib.tri.Triangulation object, optional
        Pre-computed triangulation. If None, will be computed from x, y
    
    Returns:
    --------
    fig, axes : matplotlib figure and axes objects
    """
    
    # Extract dimensions
    batch_size, n_vars, Nxy, n_time = test_values.shape
    
    # Create triangulation if not provided
    if triangulation is None:
        triangulation = Triangulation(x, y)
    
    # Select time steps to plot
    if time_steps is None:
        # Default: plot 4 evenly spaced time steps
        time_steps = [0, n_time//3, 2*n_time//3, n_time-1]
    
    n_times = len(time_steps)
    
    # Create figure with 2 rows and n_times columns
    fig, axes = plt.subplots(2, n_times, figsize=figsize)
    
    # Ensure axes is 2D array even if n_times=1
    if n_times == 1:
        axes = axes.reshape(2, 1)
    
    # Extract data for selected batch and variable
    test_data = test_values[batch_idx, var_idx, :, :]  # Shape: [Nxy, t]
    pred_data = pred_values[batch_idx, var_idx, :, :]  # Shape: [Nxy, t]
    
    # Calculate global vmin and vmax if not provided
    if vmin is None or vmax is None:
        all_values = np.concatenate([
            test_data[:, time_steps].flatten(),
            pred_data[:, time_steps].flatten()
        ])
        if vmin is None:
            vmin = np.min(all_values)
        if vmax is None:
            vmax = np.max(all_values)
    
    # Create each subplot
    for col, t_idx in enumerate(time_steps):
        
        for row, (label, data) in enumerate([('Sim.', test_data), ('Pred.', pred_data)]):
            ax = axes[row, col]
            
            # Get field data for this time step (1D array of length Nxy)
            field = data[:, t_idx]
            
            # Plot using tricontourf for unstructured grid
            im = ax.tricontourf(triangulation, field, levels=50, cmap=cmap, 
                               vmin=vmin, vmax=vmax)
            
            # Add cylinder
            circle = Circle(cylinder_center, cylinder_radius, color='white', 
                          ec='black', linewidth=1.5, zorder=10)
            ax.add_patch(circle)
            
            # Set aspect ratio and labels
            ax.set_aspect('equal')
            ax.set_xlim(x.min(), x.max())
            ax.set_ylim(y.min(), y.max())
            
            # Only add labels on leftmost column
            if col == 0:
                ax.set_ylabel(label, fontsize=14, fontweight='bold', rotation=0, 
                            ha='right', va='center')
            
            # Only add time labels on top row
            if row == 0:
                ax.set_title(f't={t_idx}', fontsize=12, fontweight='bold')
            
            # Remove ticks
            ax.set_xticks([])
            ax.set_yticks([])
    
    # Add colorbar
    fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.02)
    
    # Add overall title
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.98, 0.96])
    
    # # Save if run is provided
    # if run is not None:
    #     os.makedirs(plot_loc, exist_ok=True)
    #     plt.savefig(f'{plot_loc}/cylinder_flow_run_{run}_batch_{batch_idx}_var_{var_idx}.png', 
    #                dpi=300, bbox_inches='tight')
    
    return fig, axes


# # Example usage:
# if __name__ == "__main__":
#     # Create example data with unstructured grid
#     batch_size = 2
#     n_vars = 3  # e.g., u, v, pressure
#     Nxy = 5000  # Number of points in unstructured grid
#     n_time = 300
    
#     # Generate random unstructured grid points (avoiding cylinder interior)
#     np.random.seed(42)
#     x_points = []
#     y_points = []
#     cylinder_center = (0, 0)
#     cylinder_radius = 0.5
    
#     # Generate points in domain, excluding cylinder
#     while len(x_points) < Nxy:
#         x_candidate = np.random.uniform(-1, 4)
#         y_candidate = np.random.uniform(-1.5, 1.5)
        
#         # Check if point is outside cylinder
#         dist = np.sqrt((x_candidate - cylinder_center[0])**2 + 
#                       (y_candidate - cylinder_center[1])**2)
#         if dist > cylinder_radius:
#             x_points.append(x_candidate)
#             y_points.append(y_candidate)
    
#     x = np.array(x_points)
#     y = np.array(y_points)
    
#     # Create dummy field data [Batch size, variables, Nxy, t]
#     test_values = np.random.randn(batch_size, n_vars, Nxy, n_time)
#     pred_values = test_values + np.random.randn(batch_size, n_vars, Nxy, n_time) * 0.1
    
#     # Pre-compute triangulation (optional, for efficiency)
#     tri = Triangulation(x, y)
    
#     # Create the plot
#     fig, axes = create_cylinder_flow_plot(
#         x, y, test_values, pred_values, 
#         run='example_001',
#         batch_idx=0,
#         var_idx=0,
#         cylinder_center=cylinder_center,
#         cylinder_radius=cylinder_radius,
#         time_steps=[0, 100, 200, 299],
#         title="Re = 307 (Cylinder Flow) - Variable 0",
#         triangulation=tri
#     )
    
#     plt.show()
# %%