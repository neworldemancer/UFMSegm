"""Utility for image, image stacks, filters image IO.
"""
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os


def read_image_stack(template, num, start_idx=0):
    """
    Args:
        template (str) : template to load images, e.g. `/path/to/stacks/%03.png`
        num (int): number of images to load, lading to file list [template%i for i in range(num)]
        start_idx (int): index to be put for the first image

    Returns: 
        stack (np.ndarray): stack of images, DHWC


    """
    stack = []
    sh0 = None
    for idx in range(num):
        path = template % (idx + start_idx)
        # im = plt.imread(path)
        im = np.asarray(Image.open(path))
        sh = im.shape
        sh0 = sh0 or sh
        if sh != sh0:
            imt = np.zeros(sh0, dtype=im.dtype)
            sh_min = np.min([sh, sh0], axis=0)
            if len(sh)==2:
                imt[:sh_min[0], :sh_min[1]] = im[:sh_min[0], :sh_min[1]]
            elif len(sh)==3:
                imt[:sh_min[0], :sh_min[1], :sh_min[2]] = im[:sh_min[0], :sh_min[1], :sh_min[2]]
            else:
                raise ValueError('images of rank 2 and 3 only are supported')
            im = imt
        stack.append(im)
    stack = np.asarray(stack)
    return stack


def read_image(path):
    """
    Args:
        path (str) : path to the images, e.g. `/path/to/stacks/img.png`

    Returns:
        image (np.ndarray): image, HWC
    """
    im = np.asarray(Image.open(path))
    im = np.asarray(im)
    return im
    
def read_mp_tiff(path):
    """
    Args:
        path (str) : path to the images, e.g. `/path/to/stacks/img.png`

    Returns:
        image (np.ndarray): image, DHWC
    """
    img = Image.open(path)
    images = []
    for i in range(img.n_frames):
        img.seek(i)
        images.append(np.array(img))
    return np.array(images)


def save_image(im, path, quality=100):
    """
    Args:
        im (np.ndarray) : image pixel values. 2D, 3D or 4D (RGBA, png only)
        path (str) : path to the images, e.g. `/path/to/stacks/img.png`
        quality (int) : quality in range 0(bad)..100(awesome)

    Returns:
        image (np.ndarray): image, HWC
    """
    p_img = Image.fromarray(im)
    p_img.save(path, quality=quality)


def read_image_stack_from_dir(directory):
    """
    Reads all images from the dir with the same extension as first one, sorted by name. 
    Args:
        directory (str): directory to load all images from
        
    Returns:
        stack (np.ndarray): stack of images, DHWC
    """
    allfiles = os.listdir(directory)
    sfx = allfiles[0].split('.')[-1]
    selected_files = [
        os.path.join(directory, file) for file in allfiles if file.endswith(sfx)
    ]
    selected_files.sort()
    # print(selected_files)
    stack = []
    if len(selected_files) != len(allfiles):
        print('using only files with "%s" suffix, %d files in total' %
              (sfx, len(selected_files)))
    for path in selected_files:
        # im = plt.imread(path)
        im = np.asarray(Image.open(path))
        # print(np.min(im), np.max(im))
        stack.append(im)
    stack = np.asarray(stack)
    return stack


def montage(images, save_to=None, space=1):
    """
    Originally from Parag K. Mital, CADL
    Draw all images as a montage separated by `space` pixel borders.

    Also saves the file to the destination specified by `save_to`,
    if not None.

    Args:
        images (np.ndarray): Input array to create montage of.  Array should be:
            depth x height x width x channels or
            depth x height x width.
            channel number (C) can be 1 or 3
        save_to (str, optional): Location to save the resulting montage image.
        space (int, optional): Border size.

    Returns:
        m (np.ndarray): Montaged image.
    """
    if isinstance(images, list):
        images = np.array(images)
    img_h = images.shape[1]
    img_w = images.shape[2]
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))
    if len(images.shape) == 4 and images.shape[3] == 3:
        m = np.ones(
            (images.shape[1] * n_plots + (n_plots + 1) * space,
             images.shape[2] * n_plots + (n_plots + 1) * space, 3)) * 0.5
    elif len(images.shape) == 4 and images.shape[3] == 1:
        m = np.ones(
            (images.shape[1] * n_plots + (n_plots + 1) * space,
             images.shape[2] * n_plots + (n_plots + 1) * space, 1)) * 0.5
    elif len(images.shape) == 3:
        m = np.ones(
            (images.shape[1] * n_plots + (n_plots + 1) * space,
             images.shape[2] * n_plots + (n_plots + 1) * space)) * 0.5
    else:
        raise ValueError('Could not parse image shape of {}'.format(
            images.shape))
    for i in range(n_plots):
        for j in range(n_plots):
            img_idx = i * n_plots + j
            if img_idx < images.shape[0]:
                this_img = images[img_idx]
                m[(1 + i) * space + i * img_h:
                  (1 + i) * space + (i + 1) * img_h,
                  (1 + j) * space + j * img_w:
                  (1 + j) * space + (j + 1) * img_w] = this_img
    if save_to is not None:
        plt.imsave(arr=np.squeeze(m), fname=save_to)
    return m


def draw_stack(stack, name=None):
    """
    Draws an image stack with matplotlib.
    
    Args:
        stack (np.ndarray): image stack, DHW or DHWC, channel number (C) can be 1 or 3
        name (str, optional): path to save image
    """
    line_width = max(stack.shape[-2] // 20, 1)
    montaged = montage(stack, save_to=name, space=line_width)
    plt.figure(figsize=(10, 10))
    plt.imshow(montaged, interpolation='nearest', cmap='gray')
    plt.grid(False)


def draw_samples(imgs, titles=None, height=10, num_extra_slots=0, color_range=None, cmap='gray', cbar=-1, extents=None,
                 show=True, cbar_x_pos=None):
    """
    Draws images from tuple `imgs`.

    Args:
        imgs (np.array, np.array, ...): images.
        titles (str, str, ...): figure captions.
        height (int): figure height.
        num_extra_slots (int): number of free image slots to be filled later in
        extents (list): range for the image coordinates, (left, right, bottom, top)
        cbar (int): image index to be used for color scale bar
        cmap (string): color map
        color_range (tuple): min and max of colors to be displayed
        show (bool): whether to call plt.show() in the end
        cbar_x_pos (tuple): coordinates of the color scale bar

    Returns:
        ax[],fig (Axes, figure): list of empty slots
    """

    n = len(imgs)
    n_tot = n + num_extra_slots
    if titles and len(titles) != n_tot:
        titles = None

    fig, ax = plt.subplots(1, n_tot, figsize=(int(height * n_tot * 1.1), height))
    if n_tot == 1:
        ax = np.array((ax,))

    im = None
    cax = None
    if 0 <= cbar < n:
        cax = fig.add_axes(
            [cbar_x_pos or 0.90, 0.15, 0.02, 0.7])  # left, bottom, with, height. fraction of while figure
    for i in range(n_tot):
        if i < n:
            extent = None
            if extents:
                extent = extents[i]
            aspect = 'auto' if extent is not None else None

            imc = ax[i].imshow(imgs[i],
                               interpolation='nearest',
                               cmap=cmap,
                               vmin=color_range[0] if color_range else None,
                               vmax=color_range[1] if color_range else None,
                               extent=extent, aspect=aspect
                               )
            if cbar == i:
                im = imc

        if titles:
            ax[i].set_title(titles[i])
        ax[i].grid(False)

    if 0 <= cbar < n:
        fig.colorbar(im, cax=cax, orientation='vertical')

    if num_extra_slots == 0:
        if show:
            plt.show()
        return None, fig
    return ax[n:], fig
