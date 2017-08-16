import numpy as np
from PIL import Image as img


class BarTest:
    DEFAULT_DENSITY = 0.5
    ppb: int = 1
    bps: int = 1
    size: int = 1
    bars_active: np.ndarray = None
    
    def __init__(self, bps: int, ppb: int = 1, density: float = DEFAULT_DENSITY,
                 density_x: float = DEFAULT_DENSITY, density_y: float = DEFAULT_DENSITY):
        """
        Generate a random Bars-Test with probabilities for a bar to be active.
        :param bps: "Bars per side" -> How many bars in each dimension.
        :param ppb: "Pixels per bar" -> Thickness of a bar.
        :param density: Probability for a bar to be active. Overrides density_x and density_y.
        :param density_x: Probability for a vertical bar to be active.
        :param density_y: Probability for a horizontal bar to be active.
        """
        if density != self.DEFAULT_DENSITY:
            density_x = density
            density_y = density
            
        self.ppb = ppb
        self.bps = bps
        self.size = ppb * bps
        c_x = np.random.randint(0, bps, (int(density_x * bps)), dtype=int)
        c_y = np.random.randint(0, bps, (int(density_y * bps)), dtype=int)
        c_x += bps
        b = np.zeros((bps * 2), dtype=int)
        b[c_x] = 1
        b[c_y] = 1
        self. bars_active = b.reshape((2, bps))
        
    def as_array(self) -> np.ndarray:
        """
        Create an Numpy array for this bar test
        :return: 2D Numpy array, 1 -> active, 0 -> inactive
        """
        l = self.size
        ret = np.zeros((l, l), dtype=int)
        for i in range(l):
            for j in range(l):
                if self.bars_active[0][i // self.ppb] == 1 or self.bars_active[1][j // self.ppb] == 1:
                    ret[i][j] = 1
                    
        return ret
    
    def as_image(self) -> img:
        """
        Create an rbg image for this bar test
        :return: Image, black -> active, white -> inactive
        """
        l = self.size
        im = img.new("RGB", (l, l))
        for i in range(l):
            for j in range(l):
                im.putpixel((j, i),
                            (0, 0, 0)
                            if (self.bars_active[0][i // self.ppb] == 1 or self.bars_active[1][j // self.ppb] == 1) else
                            (255, 255, 255)
                            )
        return im
