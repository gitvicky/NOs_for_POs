import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.tri import Triangulation
import os

# ... (Imports and path setup remain the same)

def create_field_comparison_plot(x, y, test_values, pred_values, run=None,
                                   batch_idx=0, var_idx=0,
                                   time_steps=None, cmap='RdYlGn_r', 
                                   title="Field Comparison", 
                                   triangulation=None, obstacles=None,
                                   xlabel='x', ylabel='y', test_label='Solution',
                                   pred_label='Prediction', time_label='t',
                                   plot_loc='./tmp',
                                   save_prefix='field_plot'):
    """
    Updated visualization to match plots.py dimensions and scaling logic.
    """
    
    # Handle different input shapes
    if test_values.ndim == 2:
        test_data = test_values
        pred_data = pred_values
    elif test_values.ndim == 3:
        test_data = test_values[var_idx, :, :]
        pred_data = pred_values[var_idx, :, :]
    elif test_values.ndim == 4:
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
    
    # Select time steps to plot (Match plots.py: Initial, Middle, Final)
    if time_steps is None:
        time_steps = [0, n_time // 2, n_time - 1]
    
    n_times = len(time_steps)
    
    # Create figure with dimensions matching plots.py (aspect 0.5)
    fig, axes = plt.subplots(2, n_times, figsize=plt.figaspect(0.5))
    
    # Ensure axes is 2D array even if n_times=1
    if n_times == 1:
        axes = axes.reshape(2, 1)
        
    # Create each subplot
    for col, t_idx in enumerate(time_steps):
        
        # --- SCALING LOGIC CHANGED HERE ---
        # Calculate vmin/vmax specifically for this time step
        # strictly from the TEST (Ground Truth) data
        field_at_t = test_data[:, t_idx]
        vmin = np.min(field_at_t)
        vmax = np.max(field_at_t)
        
        # Plot Test (Row 0) and Pred (Row 1)
        for row, (label, data) in enumerate([(test_label, test_data), 
                                              (pred_label, pred_data)]):
            ax = axes[row, col]
            
            # Get field data for this time step
            field = data[:, t_idx]
            
            # Plot using tricontourf with the TEST-derived limits
            im = ax.tricontourf(triangulation, field, levels=50, cmap=cmap,
                               vmin=vmin, vmax=vmax)
            
            # Add obstacles
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
            
            # Set aspect and limits
            ax.set_aspect('equal')
            ax.set_xlim(x.min(), x.max())
            ax.set_ylim(y.min(), y.max())
            
            # Labels logic to match plots.py style
            # Row labels only on the left
            if col == 0:
                ax.set_ylabel(label, fontsize=12)
            else:
                # Hide y-ticks for inner plots
                ax.set_yticks([])
                
            # Time titles only on the top row
            if row == 0:
                ax.set_title(f'{time_label}={t_idx}', fontsize=12)
            
            # Hide x-ticks for all (clean look like plots.py)
            ax.set_xticks([])
            if col > 0:
                ax.set_yticks([])

            # Add colorbar to every plot individually (matching plots.py behavior)
            fig.colorbar(im, ax=ax, pad=0.05)

    # Add overall title
    # fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if run is provided
    if run is not None:
        os.makedirs(plot_loc, exist_ok=True)
        filename = f'{plot_loc}/{save_prefix}_run_{run}_batch_{batch_idx}_var_{var_idx}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        # Try saving to run object if it has the method
        try:
            run.save_file(filename, 'output')
        except:
            pass
            
    return fig, axes