import numpy as np
from PIL import Image as img
from neural import NeuralState
import os


class BarTest:
    DEFAULT_DENSITY = 0.5
    ppb = 1
    bps = 1
    size = 1
    bars_active = None
    
    def __init__(self, bps: int, ppb: int = 1, density: float = DEFAULT_DENSITY,
                 density_x: float = DEFAULT_DENSITY, density_y: float = DEFAULT_DENSITY,
                 from_file: str = None):
        """
        Generate a random Bars-Test with probabilities for a bar to be active.
        :param bps: "Bars per side" -> How many bars in each dimension.
        :param ppb: "Pixels per bar" -> Thickness of a bar.
        :param density: Probability for a bar to be active. Overrides density_x and density_y.
        :param density_x: Probability for a vertical bar to be active.
        :param density_y: Probability for a horizontal bar to be active.
        """
        if from_file:
            """
            f = open(from_file, "r", encoding="ASCII")
            lines = f.readlines()
            f.close()
            
            self.ppb = lines[-1]
            self.bps = lines[-2]
            lines = lines[:-2]

            f = open(from_file, "2", encoding="ASCII")
            f.writelines(lines)
            f.close()
            """
            self.bars_active = np.load(from_file)
            
        else:
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
            self.bars_active = b.reshape((2, bps))
        
    def as_matrix(self) -> np.ndarray:
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
    
    def as_neuralstate(self) -> NeuralState:
        """
        Create a neural state with the generated vector as neural vector
        :return: NeuralState
        """
        v = self.as_matrix().flatten()
        n = NeuralState(self.size, self.size, False, -1, initial_vector=v)
        return n
    
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

    def to_file(self, fn: str):
        """
        Save bar test to file.
        :param fn: File path to save to
        :return: -
        """
        np.save(fn, self.bars_active)
        """
        f = open(fn, "a", encoding="ASCII")
        l = [str(self.bps), str(self.ppb)]
        f.writelines(l)
        f.close()
        """


def generate_all_distinct_lines(bps: int, ppb: int, to_path: str = None) -> list:
    if to_path:
        try:
            os.mkdir(to_path)
        except OSError:
            pass  # Folder exists
    b = BarTest(bps, ppb)
    shp = b.bars_active.shape
    b.bars_active = np.zeros(shp)
    res = []
    for z in [0, 1]:
        for xy in range(bps):
            zn = "x" if z == 1 else "y"
            b.bars_active[z][xy] = 1
            if to_path:
                b.to_file(to_path+"/bars_" + zn + str(xy) + ".npy")
                b.as_image().save(to_path+"/bars_" + zn + str(xy) + ".png", "PNG")
            res.append(b.bars_active.copy())
            b.bars_active[z][xy] = 0
    return res
