# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.tri import Triangulation

import os
plot_loc = os.getcwd() + '/Plots'


def create_field_comparison_plot(x, y, test_values, pred_values, run=None,
                                   batch_idx=0, var_idx=0,
                                   time_steps=None, cmap='RdYlGn_r', vmin=None, vmax=None,
                                   title="Field Comparison", figsize=(14, 6),
                                   triangulation=None, obstacles=None,
                                   xlabel='x', ylabel='y', test_label='Ground Truth',
                                   pred_label='Prediction', time_label='t',
                                   save_prefix='field_plot'):
    """
    Create a general visualization comparing test/ground truth and predicted field values
    at different time steps. Works with both structured and unstructured grids.
    
    Parameters:
    -----------
    x : numpy array
        X coordinates of the mesh (1D array of length N)
    y : numpy array
        Y coordinates of the mesh (1D array of length N)
    test_values : numpy array
        Ground truth field values with shape [Batch, Variables, N, Time] or [N, Time]
    pred_values : numpy array
        Predicted field values with shape [Batch, Variables, N, Time] or [N, Time]
    run : str or None
        Run identifier for saving
    batch_idx : int
        Index of batch to visualize (default: 0)
    var_idx : int
        Index of variable to visualize (default: 0)
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
    obstacles : list of dict, optional
        List of obstacles to draw. Each dict should have:
        - 'type': 'circle' or 'rectangle'
        - For circles: 'center': (x, y), 'radius': float
        - For rectangles: 'xy': (x, y), 'width': float, 'height': float
        - Optional: 'color', 'edgecolor', 'linewidth'
    xlabel, ylabel : str
        Axis labels
    test_label, pred_label : str
        Labels for test and prediction rows
    time_label : str
        Prefix for time step labels
    save_prefix : str
        Prefix for saved filename
    
    Returns:
    --------
    fig, axes : matplotlib figure and axes objects
    """
    
    # Handle different input shapes
    if test_values.ndim == 2:
        # Shape: [N, Time]
        test_data = test_values
        pred_data = pred_values
    elif test_values.ndim == 3:
        # Shape: [Variables, N, Time]
        test_data = test_values[var_idx, :, :]
        pred_data = pred_values[var_idx, :, :]
    elif test_values.ndim == 4:
        # Shape: [Batch, Variables, N, Time]
        test_data = test_values[batch_idx, var_idx, :, :]
        pred_data = pred_values[batch_idx, var_idx, :, :]
    else:
        raise ValueError(f"Unexpected test_values shape: {test_values.shape}")
    
    # Extract dimensions
    N, n_time = test_data.shape
    
    # Validate coordinate lengths
    if len(x) != N or len(y) != N:
        raise ValueError(f"Coordinate lengths ({len(x)}, {len(y)}) don't match data length ({N})")
    
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
        
        for row, (label, data) in enumerate([(test_label, test_data), 
                                              (pred_label, pred_data)]):
            ax = axes[row, col]
            
            # Get field data for this time step (1D array of length N)
            field = data[:, t_idx]
            
            # Plot using tricontourf for unstructured grid
            im = ax.tricontourf(triangulation, field, levels=50, cmap=cmap,
                               vmin=vmin, vmax=vmax)
            
            # Add obstacles if provided
            if obstacles is not None:
                for obstacle in obstacles:
                    if obstacle['type'] == 'circle':
                        circle = Circle(
                            obstacle['center'], 
                            obstacle['radius'],
                            color=obstacle.get('color', 'white'),
                            ec=obstacle.get('edgecolor', 'black'),
                            linewidth=obstacle.get('linewidth', 1.5),
                            zorder=10
                        )
                        ax.add_patch(circle)
                    elif obstacle['type'] == 'rectangle':
                        from matplotlib.patches import Rectangle
                        rect = Rectangle(
                            obstacle['xy'],
                            obstacle['width'],
                            obstacle['height'],
                            color=obstacle.get('color', 'white'),
                            ec=obstacle.get('edgecolor', 'black'),
                            linewidth=obstacle.get('linewidth', 1.5),
                            zorder=10
                        )
                        ax.add_patch(rect)
            
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
                ax.set_title(f'{time_label}={t_idx}', fontsize=12, fontweight='bold')
            
            # Remove ticks
            ax.set_xticks([])
            ax.set_yticks([])
    
    # Add colorbar
    fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.02)
    
    # Add overall title
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.98, 0.96])
    
    # Save if run is provided
    if run is not None:
        os.makedirs(plot_loc, exist_ok=True)
        filename = f'{plot_loc}/{save_prefix}_run_{run}_batch_{batch_idx}_var_{var_idx}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {filename}")
    
    return fig, axes


def create_single_field_plot(x, y, field_values, time_steps=None,
                             cmap='RdYlGn_r', vmin=None, vmax=None,
                             title="Field Evolution", figsize=(14, 4),
                             triangulation=None, obstacles=None,
                             xlabel='x', ylabel='y', time_label='t',
                             save_path=None):
    """
    Create a visualization of a single field at different time steps.
    
    Parameters:
    -----------
    x : numpy array
        X coordinates of the mesh (1D array of length N)
    y : numpy array
        Y coordinates of the mesh (1D array of length N)
    field_values : numpy array
        Field values with shape [N, Time]
    time_steps : list
        List of time step indices to plot (if None, uses evenly spaced steps)
    cmap : str
        Colormap name
    vmin, vmax : float
        Min and max values for color scale
    title : str
        Overall title for the figure
    figsize : tuple
        Figure size (width, height)
    triangulation : matplotlib.tri.Triangulation object, optional
        Pre-computed triangulation
    obstacles : list of dict, optional
        List of obstacles to draw (same format as create_field_comparison_plot)
    xlabel, ylabel : str
        Axis labels
    time_label : str
        Prefix for time step labels
    save_path : str, optional
        Full path to save the figure
    
    Returns:
    --------
    fig, axes : matplotlib figure and axes objects
    """
    
    # Extract dimensions
    N, n_time = field_values.shape
    
    # Validate coordinate lengths
    if len(x) != N or len(y) != N:
        raise ValueError(f"Coordinate lengths ({len(x)}, {len(y)}) don't match data length ({N})")
    
    # Create triangulation if not provided
    if triangulation is None:
        triangulation = Triangulation(x, y)
    
    # Select time steps to plot
    if time_steps is None:
        # Default: plot 4 evenly spaced time steps
        time_steps = [0, n_time//3, 2*n_time//3, n_time-1]
    
    n_times = len(time_steps)
    
    # Create figure with 1 row and n_times columns
    fig, axes = plt.subplots(1, n_times, figsize=figsize)
    
    # Ensure axes is array even if n_times=1
    if n_times == 1:
        axes = [axes]
    
    # Calculate global vmin and vmax if not provided
    if vmin is None or vmax is None:
        all_values = field_values[:, time_steps].flatten()
        if vmin is None:
            vmin = np.min(all_values)
        if vmax is None:
            vmax = np.max(all_values)
    
    # Create each subplot
    for col, t_idx in enumerate(time_steps):
        ax = axes[col]
        
        # Get field data for this time step
        field = field_values[:, t_idx]
        
        # Plot using tricontourf
        im = ax.tricontourf(triangulation, field, levels=50, cmap=cmap,
                           vmin=vmin, vmax=vmax)
        
        # Add obstacles if provided
        if obstacles is not None:
            for obstacle in obstacles:
                if obstacle['type'] == 'circle':
                    circle = Circle(
                        obstacle['center'],
                        obstacle['radius'],
                        color=obstacle.get('color', 'white'),
                        ec=obstacle.get('edgecolor', 'black'),
                        linewidth=obstacle.get('linewidth', 1.5),
                        zorder=10
                    )
                    ax.add_patch(circle)
                elif obstacle['type'] == 'rectangle':
                    from matplotlib.patches import Rectangle
                    rect = Rectangle(
                        obstacle['xy'],
                        obstacle['width'],
                        obstacle['height'],
                        color=obstacle.get('color', 'white'),
                        ec=obstacle.get('edgecolor', 'black'),
                        linewidth=obstacle.get('linewidth', 1.5),
                        zorder=10
                    )
                    ax.add_patch(rect)
        
        # Set aspect ratio and limits
        ax.set_aspect('equal')
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min(), y.max())
        
        # Add time label
        ax.set_title(f'{time_label}={t_idx}', fontsize=12, fontweight='bold')
        
        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Add colorbar
    fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.02)
    
    # Add overall title
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.98, 0.96])
    
    # # Save if path is provided
    # if save_path is not None:
    #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
    #     print(f"Saved plot to: {save_path}")
    
    return fig, axes


# Example usage for backward compatibility with cylinder flow
def create_cylinder_flow_plot(x, y, test_values, pred_values, run,
                               batch_idx=0, var_idx=0,
                               cylinder_center=(0, 0), cylinder_radius=0.5,
                               time_steps=None, cmap='RdYlGn_r', vmin=None, vmax=None,
                               title="Re = 307 (Cylinder Flow)", figsize=(14, 6),
                               triangulation=None):
    """
    Wrapper function for backward compatibility with cylinder flow plotting.
    """
    obstacles = [{
        'type': 'circle',
        'center': cylinder_center,
        'radius': cylinder_radius,
        'color': 'white',
        'edgecolor': 'black',
        'linewidth': 1.5
    }]
    
    return create_field_comparison_plot(
        x, y, test_values, pred_values, run=run,
        batch_idx=batch_idx, var_idx=var_idx,
        time_steps=time_steps, cmap=cmap, vmin=vmin, vmax=vmax,
        title=title, figsize=figsize,
        triangulation=triangulation, obstacles=obstacles,
        test_label='Sim.', pred_label='Pred.',
        save_prefix='cylinder_flow'
    )


# # Example usage:
# if __name__ == "__main__":
#     # Example 1: Cylinder flow (original use case)
#     batch_size = 2
#     n_vars = 3
#     Nxy = 5000
#     n_time = 300
    
#     # Generate random unstructured grid points
#     np.random.seed(42)
#     x_points = []
#     y_points = []
#     cylinder_center = (0, 0)
#     cylinder_radius = 0.5
    
#     while len(x_points) < Nxy:
#         x_candidate = np.random.uniform(-1, 4)
#         y_candidate = np.random.uniform(-1.5, 1.5)
#         dist = np.sqrt((x_candidate - cylinder_center[0])**2 + 
#                       (y_candidate - cylinder_center[1])**2)
#         if dist > cylinder_radius:
#             x_points.append(x_candidate)
#             y_points.append(y_candidate)
    
#     x = np.array(x_points)
#     y = np.array(y_points)
    
#     # Create dummy field data
#     test_values = np.random.randn(batch_size, n_vars, Nxy, n_time)
#     pred_values = test_values + np.random.randn(batch_size, n_vars, Nxy, n_time) * 0.1
    
#     # Using the general function
#     obstacles = [{
#         'type': 'circle',
#         'center': cylinder_center,
#         'radius': cylinder_radius
#     }]
    
#     fig, axes = create_field_comparison_plot(
#         x, y, test_values, pred_values,
#         run='general_001',
#         batch_idx=0, var_idx=0,
#         time_steps=[0, 100, 200, 299],
#         title="Cylinder Flow - Velocity Field",
#         obstacles=obstacles,
#         test_label='Simulation',
#         pred_label='Neural Net'
#     )
    
#     # Example 2: Simple rectangular domain with no obstacles
#     N = 1000
#     x_simple = np.random.uniform(0, 10, N)
#     y_simple = np.random.uniform(0, 5, N)
#     field_simple = np.random.randn(N, 100)
    
#     fig2, axes2 = create_single_field_plot(
#         x_simple, y_simple, field_simple,
#         time_steps=[0, 30, 60, 99],
#         title="Temperature Distribution",
#         save_path='./Plots/temperature_example.png'
#     )
    
#     plt.show()
# # %%