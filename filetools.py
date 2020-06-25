# +
from glob import glob
import os, re
import imghdr
import doctest

import numpy as np
from skimage import io

from collections import defaultdict


# -

def extract_digits(string):
    '''Extract all digits from given string
       used to sort images by name
       
       >>> extract_digits("a1b2cd34")
       1234
    '''
    return int(''.join(filter(str.isdigit, string)))


def print_numbered_list(name_list):
    """Pretty print id: name\n"""
    print('\n'.join([f'{k} - {n}' for k, n in enumerate(name_list)]))





def load_image(path):
    """ Load the image at the given path
         returns 2d array (float)
         convert to grayscale if needed
    """
    try:
        I = io.imread(path)
        # Convert to grayscale if needed:
        I = I.mean(axis=2) if I.ndim == 3 else I
        I = I.astype(np.float)
    except FileNotFoundError:
        print("File %s Not Found" % path)
        I = None

    return I



def list_images(path):
    """list and sort image present in the directory,
    returns full path
    """
    images = sorted( os.listdir(path) )
    # remove non-image file :
    images = [os.path.join(path, filename) for filename in images
              if imghdr.what(os.path.join(path, filename))]
    return images


def load_images(images, verbose=True):
    """Load given list of image path and return a 3d array
    """
    cube = [load_image(img_path)
            for img_path in images]

    cube = np.dstack(cube)

    if verbose:
        print('Image cube:' + ' '*20)
        print(f' {cube.shape[0]}*{cube.shape[1]} pixels - {cube.shape[2]} frames') 
        print(f' memory size: {cube.nbytes // 1024**2} Mo')

    return cube



def create_dir(path):
    """Create the dir if doesn't exist"""
    if not os.path.isdir(path):
        os.mkdir(path)
        print("make dir", path)
    else:
        print('dir:', path)


# =================================
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



