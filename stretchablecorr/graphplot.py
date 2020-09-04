import numpy as np
import matplotlib.pylab as plt


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