"""Binarize (make it black and white) an image with Pyhton."""

import numpy as np
from PIL import Image

def bipolize_image(img_path: str, threshold: int = 150) -> np.ndarray:
    """Binarize an image and return it."""
    image_file = Image.open(img_path)
    image = image_file.convert('L')  # convert image to monochrome
    array = np.array(image)
    barray = bipolize_array(array, threshold)
    return barray


def bipolize_array(a: np.ndarray, threshold: int = 150) ->np.ndarray:
    """Binarize a numpy array."""
    b = np.ones(a.shape, dtype=int)
    for i in range(len(a)):
        for j in range(len(a[i])):
            if a[i][j] > threshold:
                b[i][j] = (-1)
    return b

def binarize_image(img_path: str, threshold: int = 150) -> np.ndarray:
    """Binarize an image and return it."""
    image_file = Image.open(img_path)
    image = image_file.convert('L')  # convert image to monochrome
    image = np.array(image)
    image = binarize_array(image, threshold)
    return image

def binarize_array(a: np.ndarray, threshold: int = 150) ->np.ndarray:
    """Binarize a numpy array."""
    b = np.zeros(a.shape, dtype=int)
    for i in range(len(a)):
        for j in range(len(a[i])):
            if a[i][j] < threshold:
                b[i][j] = 1
    return b

def ShowBinayImage(image: np.ndarray):
    im = Image.new("RGB", image.shape, 0 )
    for i in range(len(image)):
        for j in range(len(image[i])):
            im.putpixel((j,i), (0,0,0) if image[i][j] > 0 else (255,255,255) )

    Image._show(im)
    
    
def PrintBinayImage(image: np.ndarray):
    s = ''
    for i in range(len(image)):
        for j in range(len(image[i])):
            s += '#' if image[i][j] > 0 else ' '
        s += '\n'
    print(s)