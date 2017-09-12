from math import sqrt
from math import exp
from math import log
import numpy as np
from pics import print_binay_image


class NeuralState:  # S μ
    """A neuron state vector"""
    w = 0
    h = 0
    N = 0
    vec = None
    periodic = False
    max_dis = -1
    
    def __init__(self, width: int, height: int, periodic: bool, max_synapse_length: float,
                 initial_vector: np.ndarray = None):
        """
        A neural state vector. Includes weight * height = N neurons with binary value.
        :param width: Width of the 2D neuron array.
        :param height: Height of the 2D neuron array.
        :param periodic: Set true for periodic boundaries. Important in distance calculation.
        :param max_synapse_length: Maximal range of a synapse from one neuron to another. (-1 means infinity)
        :param initial_vector: (optional) Given neural vector with size N.
        """
        self.w = width
        self.h = height
        self.N = width*height
        self.periodic = periodic
        self.max_dis = max_synapse_length
        if initial_vector is None:
            self.vec = np.zeros(self.N, dtype=int)
        else:
            if initial_vector.size != width * height:
                print("Initial vector size does not meet width and height")
                quit()
            self.vec = initial_vector.copy()

    def as_matrix(self) -> np.ndarray:
        """
        Return activity pattern as 2D matrix.
        """
        dim2 = self.vec.reshape(self.w, self.h)
        return dim2
        
    def print(self):
        """
        Print out neural activity pattern.
        """
        print_binay_image(self.as_matrix())

    def xy_i(self, i: int) -> (int, int):
        """
        Calculates position of neuron i in 2D-Plane
        :param i: Index of neuron.
        :return: Tuple like (x, y)
        """
        xi = (i % self.w)
        yi = (i // self.w)
        return xi, yi
    
    def active_neuron_count(self) -> int:
        """
        Returns how many neurons are active.
        :return: Integer - Number of active neurons.
        """
        return np.count_nonzero(self.vec)
        
    def binary_weights(self) -> np.ndarray:
        """
        Return a binary weight matrix for given state.
        :return: 2D weight matrix with 1 if both neurons are active, else 0.
        """
        x = np.outer(self.vec, self.vec)
        np.fill_diagonal(x, 0)
        return x

    def initial_weight(self, form: int = 0, normalize: bool = False) -> np.ndarray:
        """
        Calculates a weight matrix with linear decreasing weight by distance between neurons.
        :param form: 0 -> linear, 1-> exp
        :param normalize: If set true the sum of all connection weights for one neuron is 1.
        :return: 2D weight matrix with (i,j) = neuron_linear_distance_weight(i,j)
        """
        weight = np.zeros((self.N, self.N), dtype=float)
        for i in range(self.N):
            for j in range(self.N):
                if i == j:
                    weight[i][j] = 0
                    continue
    
                d = self.distance(i, j)
                
                r = self.max_dis
                not_norm = 0
                if r == -1:
                    r = self.distance(0, self.N - 1)+1
                    if form == 0:
                        # linear decrease
                        not_norm = (r - d) / (r - 1)
                    elif form == 1:
                        # exponential decrease
                        not_norm = exp(-log(10) / (r - 1) * d)
                    
                elif d < r:
                    if form == 0:
                        # linear decrease
                        not_norm = (r - d)/(r - 1)
                        
                    elif form == 1:
                        # exponential decrease
                        not_norm = exp(-log(10) / (r - 1) * d)
                       
                weight[i][j] = not_norm
        
        if normalize:
            for i in range(self.N):
                weight[i] = weight[i] / sum(weight[i])
            
        return weight

    def distance_matrix(self) -> np.ndarray:
        d = np.zeros((self.N, self.N), dtype=float)
        for i in range(self.N):
            for j in range(self.N):
                d[i][j] = self.distance(i, j)
        
        return d

    def distance(self, i: int, j: int):
        if i == j:
            return 0
        
        xi, yi = self.xy_i(i)
        xj, yj = self.xy_i(j)
    
        dis = sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
    
        if self.periodic:
            d_p = [dis]
            xj_p = [xj + self.w, xj - self.w, xj, xj]
            yj_p = [yj, yj, yj + self.h, yj - self.h]
            for k in range(4):
                d_p.append(sqrt((xi - xj_p[k]) ** 2 + (yi - yj_p[k]) ** 2))
            dis = min(d_p)
    
        return dis
