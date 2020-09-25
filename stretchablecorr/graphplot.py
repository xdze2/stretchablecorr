import numpy as np
import matplotlib.pylab as plt



def imshow_color_diff(I, J):
    """display I and J images superimposed on different color channels
    """
    def norm_01(I):
        return (I - I.min())/I.ptp()
    Ic = np.dstack([norm_01(I)**0.5,
                    norm_01(J)**0.5,
                    0.5*np.ones_like(J)])
    plt.imshow(Ic)


def plot_vector_field(points, displacements,
                      view_factor=None, color='white'):
    amplitudes = np.sqrt(np.nansum(displacements**2, axis=1))

    mask = ~np.any(np.isnan(displacements), axis=1, keepdims=False)

    plt.quiver(*points[mask, :].T, *displacements[mask, :].T,
               angles='xy', color='white',
               scale_units='xy',
               units='dots',
               width=1,
               headwidth=3,
               headlength=4, headaxislength=3,
               scale=1/view_factor if view_factor else None,
               minlength=1e-4)

    plt.text(10., 10.,
             f'max(|u|)={np.nanmax(amplitudes):.2f}px  mean(|u|)={np.nanmean(amplitudes):.2f}px',
             fontsize=12, color=color,
             verticalalignment='top')

    # plot NaN points
    plt.plot(points[np.logical_not(mask), 0],
             points[np.logical_not(mask), 1],
             's', markersize=1, color='yellow', alpha=0.7)


def plot_grid_points(grid, background=None,
                     color='white', markersize=3,
                     show_pts_number=False,
                     window_half_size=None):
    """Plot grid points

    Parameters
    ----------
    grid : tuple (grid_x, grid_y)
        grid points arrays
    background : 2D array, by default None
        image to draw as a background
    color : str, by default 'white'
    markersize : int, by default 3
    show_pts_number : bool, by default False
    window_half_size : int, by default None
        if not None, draw one corresponding ROI box
    """
    if background is not None:
        plt.imshow(background)

    plt.plot(*grid, 'o', color=color, markersize=markersize)

    if show_pts_number:
        points = np.stack((grid[0].flatten(), grid[1].flatten()),
                          axis=-1)
        for k, (x, y) in enumerate(points):
            if len(points) > 10 and k % 5 != 0:
                continue
            text_offset = 10.0
            plt.text(x+text_offset, y+text_offset,
                     str(k), fontsize=8, color=color)

    # Graph one of the ROI
    if window_half_size:
        box = np.array([[-1, 1, 1, -1, -1],
                        [-1, -1, 1, 1, -1]])*(window_half_size + 1)
        middle_point = tuple(np.array(grid[0].shape) // 2 - 1)
        plt.plot(box[0]+grid[0][middle_point], box[1]+grid[1][middle_point],
                 color=color, linewidth=1)


from matplotlib.lines import Line2D

def plot_trajectories(trajectories, background=None, gaps=None,
                      color='black'):
    if background is not None:
        plt.imshow(background, alpha=.4)

    for k, xy in enumerate(np.swapaxes(trajectories, 0, 1)):
        plt.plot(*xy[0], 's', color=color, markersize=1)
        plt.plot(*xy.T, '-', linewidth=.5, markersize=2, color=color)
        if k % 5 == 0:
            plt.text(*xy[0], str(k), fontsize=6)

        if gaps is not None:
            g = gaps[:, k]    
            mask = ~np.isnan(g)
            mask[mask] &= g[mask] > 5
            plt.plot(*xy[1:-1, :][mask, :].T, 'o', markersize=2,
                     color='red')
            plt.legend([Line2D([0], [0], linestyle='', marker='o', color='red', markersize=2),], ['gap > 5px', ])
        #plt.axis('equal')


def plot_deformed_mesh(grid, displ_field,
                       color_values=None,
                       view_factor=10,
                       displ_threshold=True, cmap='Spectral'):
    
    if color_values is None:
        color_values = np.zeros_like(grid[0])
    
    # Scale displacements using view_factor
    points = np.stack( (grid[0].flatten(), grid[1].flatten()), axis=-1 )
    positions_amplified = displ_field*view_factor + points
    x_amplified = positions_amplified[:, 0].reshape(grid[0].shape)
    y_amplified = positions_amplified[:, 1].reshape(grid[0].shape)

    displ_field_amplified = view_factor * displ_field

    # Remove points where displacement > threshold
    if displ_threshold:
        diff_x = np.diff(x_amplified, axis=1, prepend=np.min(x_amplified)-10)
        diff_y = np.diff(y_amplified, axis=0, prepend=np.min(y_amplified)-10)
        displ_x_mask = np.less(diff_x, 0,
                               where=~np.isnan(diff_x))
        displ_y_mask = np.less(diff_y, 0,
                               where=~np.isnan(diff_y))
        displ_mask = np.logical_or(displ_x_mask, displ_y_mask)

        x_amplified[displ_mask] = np.NaN
        y_amplified[displ_mask] = np.NaN

    # Background Reference grid:
    moved_out = np.any(np.isnan(displ_field), axis=1).reshape(grid[0].shape)
    ref_colors = np.zeros_like(moved_out)
    ref_colors[moved_out] = 1
    
    # note : pcolormesh doesn't work with NaN
    plt.pcolor(*grid, ref_colors,  
               edgecolors='black', linewidth=1, antialiased=True,
               cmap='Reds', alpha=0.1)
    

    # Front mesh:
    cs = plt.pcolor(x_amplified, y_amplified, color_values,
                    edgecolors='#2b2b2b',
                    linewidth=1,
                    antialiased=True,
                    cmap=cmap)
    cs.cmap.set_over('gray')
    cs.cmap.set_under('gray')


    plt.annotate(f"×{view_factor}", (1, 1), xytext=(-5, -5),
                 xycoords='axes fraction', textcoords='offset points',
                 fontsize=14, fontweight='bold', ha='right', va='top')

    plt.axis('equal')
    plt.xlabel('x [pixel]'); plt.ylabel('y [pixel]');   