#!/usr/bin/env python

import numpy as np
from PIL import Image as IMG

__author__ = "Janis Kühn"
__license__ = "Apache 2.0"
__email__ = "jk@stud.uni-frankfurt.de"
__status__ = "Prototype"
__pythonVersion__ = "3.6"


def generate_barpicture(bps: int, ppb: int) -> IMG:
    """generate a random Bars-Test Image, with bps possible Bars per dimension and ppb Pixels per bar"""
    b = np.random.randint(0, 2, (2, bps), dtype=int)

    l = bps*ppb
    return makeimg(l,b,ppb)

def generate_barpicture_dens(bps: int, ppb: int, density: float = 0.5) -> IMG:
    """generate a random Bars-Test Image, with bps possible Bars per dimension and ppb Pixels per bar"""
    c = np.random.randint(0, bps*2, (int(density*bps*2)) , dtype=int)
    b = np.zeros((bps*2), dtype=int)
    b[c] = 1
    b = b.reshape((2,bps))

    l = bps * ppb
    return makeimg(l,b,ppb)


def generate_barpicture_dens2(bps: int, ppb: int, densityX: float = 0.5, densityY: float = 0.5) -> IMG:
    """generate a random Bars-Test Image, with bps possible Bars per dimension and ppb Pixels per bar"""
    cX = np.random.randint(0, bps, (int(densityX * bps)), dtype=int)
    cY = np.random.randint(0, bps, (int(densityY * bps)), dtype=int)
    cX += bps
    b = np.zeros((bps * 2), dtype=int)
    b[cX] = 1
    b[cY] = 1
    b = b.reshape((2, bps))

    l = bps * ppb
    return makeimg(l, b, ppb)


def makeimg(l,b,ppb):
    # generate image
    im = IMG.new("RGB", (l, l))
    for i in range(l):
        for j in range(l):
            im.putpixel((j, i), (0, 0, 0) if (b[0][i // ppb] == 1 or b[1][j // ppb] == 1) else (255, 255, 255))
    return im

im = generate_barpicture_dens2(5, 10, 0.2, 0.2)
im.show()
