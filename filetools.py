# +
from glob import glob
import os, re
import imghdr
import doctest

import numpy as np
import matplotlib.pylab as plt
from skimage import io
from skimage.color import rgb2gray

from collections import defaultdict
import pickle

# -

def extract_digits(string):
    '''Extract all digits from given string
       used to sort images by name

       >>> extract_digits("a1b2cd34")
       1234
    '''
    return int(''.join(filter(str.isdigit, string)))


def print_numbered_list(name_list):
    """Pretty print id: name"""
    print('\n'.join([f'{k} - {n}' for k, n in enumerate(name_list)]))


def load_image(path):
    """Load the image at the given path
    using scikit-image `io.imread`
    convert to grayscale if needed

    Returns
    -------
    2D array (float)
        grayscale image
    """
    try:
        image = io.imread(path)
        # Convert to grayscale if needed:
        image = rgb2gray(image) if image.ndim == 3 else image
        image = image.astype(np.float)

    except FileNotFoundError:
        print(f"File {path} Not Found")
        image = None

    return image


def list_images(path):
    """list and sort image present in the directory,
    returns full path
    """
    images = sorted(os.listdir(path))
    # remove non-image file :
    images = [os.path.join(path, filename) for filename in images]
    images = [filename for filename in images
              if not os.path.isdir(filename) and imghdr.what(filename)]
    return images


def load_image_sequence(directory, verbose=True):
    """Load all images in directoy

    Returns
    -------
    3D array of shape (nbr of images, height, width)
    """

    images = list_images(directory)

    cube = [load_image(img_path)
            for img_path in images]

    cube = np.stack(cube, axis=0)

    if verbose:
        print('Image sequence:')
        print(f' {cube.shape[0]} frames')
        print(f' {cube.shape[2]}*{cube.shape[1]} pixels')
        print(f' memory size: {cube.nbytes // 1024**2} Mo')

    return cube, [os.path.basename(p) for p in images]


def create_dir(path, verbose=True):
    """Create the directory if doesn't exist"""
    if not os.path.isdir(path):
        os.makedirs(path)
        print("make dir", path)
    elif verbose:
        print('dir:', path)


def save_fig(fig_name, *subdirs,
             image_ext='svg',
             output_dir='./output/',
             close=False):
    """Save the current figure (using matplotlib savefig)
    construct the needed directories

    Parameters
    ----------
    fig_name : str
        name of the file (without extension)
    *subdirs : str
        possible sub-directories, for instance sample name
    image_ext : str, optional
        wanted figure format, by default 'svg'
    output_dir : str, optional
        base directory to output results, by default './output/'
    close : bool, optional
        close the figure after saving, by default False
        to use when saving many figures in a loop
    """
    path = os.path.join(output_dir, *subdirs)
    create_dir(path, verbose=False)
    filename = f"{fig_name}.{image_ext.strip('. ')}"
    path = os.path.join(path, filename)
    plt.savefig(path);
    print(f'figure saved at {path}', end='\r')
    if close:
        plt.close()


def save_data(data, array_name, *subdirs,
             output_dir='./output/'):
    """Save data using pickle

        see save_fig options
    """
    path = os.path.join(output_dir, *subdirs)
    create_dir(path, verbose=False)
    array_name = array_name.replace('.npy', '')
    filename = f"{array_name}.pck"
    path = os.path.join(path, filename)
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    print(f'data saved at {path}')


def plot_vector_field(points, displacements,
                      view_factor=None, color='white'):
    amplitudes = np.sqrt(np.nansum( displacements**2, axis=1 )) # disp. amplitude

    mask = ~np.any( np.isnan(displacements), axis=1, keepdims=False )

    plt.quiver(*points[mask, :].T, *displacements[mask, :].T,
               angles='xy', color='white',
               scale_units='xy',
               units='dots',
               width=1,
               headwidth=3,
               headlength=4, headaxislength=3,
               scale=1/view_factor if view_factor else None,
               minlength=1e-4);

    plt.text(10., 10.,
             f'max(|u|)={np.nanmax(amplitudes):.2f}px  mean(|u|)={np.nanmean(amplitudes):.2f}px',
             fontsize=12, color=color,
             verticalalignment='top')

    # plot NaN points
    plt.plot(points[np.logical_not(mask), 0], points[np.logical_not(mask), 1], 's',
         markersize=1, color='yellow', alpha=0.7);


def plot_grid_points(grid, background=None,
                     color='white', markersize=3,
                     show_pts_number=False,
                     window_half_size=None):

    if background is not None:
        plt.imshow(background)

    plt.plot(*grid, 'o', color=color, markersize=markersize);

    if show_pts_number:
        points = np.stack((grid[0].flatten(), grid[1].flatten()),
                          axis=-1)
        for k, (x, y) in enumerate(points):
            if len(points) > 10 and k % 5 != 0:
                continue
            text_offset = 10.0
            plt.text(x+text_offset, y+text_offset,
                     str(k), fontsize=8, color=color)

    if window_half_size:
        # graph one of the ROI
        box = np.array([[-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1]])*(window_half_size + 1)
        middle_point = tuple(np.array(grid[0].shape) // 2 - 1)
        plt.plot(box[0]+grid[0][middle_point], box[1]+grid[1][middle_point],
                 color=color, linewidth=1)

#
# =================================
#  Specific for nested dir struc.
# =================================
def parse_path(img_path, DATA_DIR, IMAGE_EXT, verbose=False):
    info = defaultdict(str)
    info['path'] = img_path
    # pre - processing
    img_path = img_path.replace(DATA_DIR, '')
    img_path = img_path.replace(IMAGE_EXT, '').strip('.')
    img_path = img_path.replace("\\", "/") # for windows
    img_path = img_path.strip("/")

    parts = img_path.split("/")
    if len(parts) != 3:
        if verbose:
            print("dir structure error:", parts)
        info['msg'] += "dir structure error "
        return info

    parts = [s.lower() for s in parts]
    sample_name, step_name, image_name = parts

    info['sample'] = sample_name
    info['step_name'] = step_name

    if not image_name.startswith(sample_name) or step_name not in image_name:
        if verbose:
            print('warning: no in prefix in filename', parts)
        info['msg'] += "warning: no in prefix in filename "

    if 'u' in step_name:
        info['direction'] = "unloading"
    else:
        info['direction'] = 'loading'

    strain = step_name.replace('u', '').replace('p', '.')
    try:
        info['applied_strain'] = float(strain)
    except ValueError:
        info['applied_strain'] = None

    image_name = image_name.replace(sample_name, '')
    image_name = image_name.replace(step_name, '')

    image_name = image_name.replace('_', '')

    img_pattern = re.compile( r'(u?)(\D*)(\d*)' )
    matchs = re.findall(img_pattern, image_name)

    if matchs:
        m = matchs[0]
        #info['direction'] = "unloading" if m[0] == 'u' else 'loading'
        info['tag'] = m[1]
        info['file_idx'] = int(m[-1])
    else:
        if verbose:
            print("filename error:", parts)
        info['msg'] += "filename error: " + parts

    info['label'] = f'{sample_name} {step_name}'#' {image_name}'
    return info


def parse_step_dir(stepname):
    """Parse the step name

    >>> parse_step_dir('7p1u')
    defaultdict(<class 'str'>, {'stepname': '7p1u', 'strain': 7.1, 'direction': 'unloading', 'tag': ''})
    >>> parse_step_dir('10p5')
    defaultdict(<class 'str'>, {'stepname': '10p5', 'strain': 10.5, 'direction': 'loading', 'tag': ''})
    >>> parse_step_dir('1P1night')
    defaultdict(<class 'str'>, {'stepname': '1P1night', 'strain': 1.1, 'direction': 'loading', 'tag': 'night'})
    """
    step_pattern = re.compile( r"(\d{1,2})(?:p|P)(\d)(u?)(\w*)" )
    matchs = re.findall(step_pattern, stepname)
    info = defaultdict(str)
    info['stepname'] = stepname
    if matchs and len(matchs) == 1:
        m = matchs[0]
        info['strain'] = int(m[0]) + int(m[1])/10
        info['direction'] = 'unloading' if m[2].lower() == 'u' else 'loading'
        info['tag'] = m[3]
    else:
        info['msg'] = 'error parsing stepname'

    return info


doctest.testmod(verbose=False)



## Extract one of the images inside sub-directories
# from shutil import copyfile

# path = sample_input_dir
# dirs = sorted(os.listdir(path))
# # remove non-image file :
# dirs = [os.path.join(path, d) for d in dirs
#         if os.path.isdir(os.path.join(path, d))]

# for d in dirs:
#     images = sorted(os.listdir(d))
#     print(images[0])
#     copyfile(os.path.join(d, images[0]), os.path.join(path, images[0].replace('hs2', '')))