from math import sqrt
import numpy as np
from pics import print_binay_image


class NeuralState:  # S μ
    """A neuron state vector"""
    w = 0
    h = 0
    N = 0
    vec = None
    
    def __init__(self, width: int, height: int, initial_vector: np.ndarray = None):
        """
        A neural state vector. Includes weight * height = N neurons with binary value.
        :param width: Width of the 2D neuron array.
        :param height: Height of the 2D neuron array.
        :param initial_vector: (optional) Given neural vector with size N.
        """
        self.w = width
        self.h = height
        self.N = width*height
        if initial_vector is None:
            self.vec = np.zeros(self.N, dtype=int)
        else:
            if initial_vector.size != width * height:
                print("Initial vector size does not meet width and height")
                quit()
            self.vec = initial_vector.copy()

    def print(self):
        """
        Print out neural activity pattern.
        """
        dim2 = self.vec.reshape(self.w, self.h)
        print_binay_image(dim2)

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
        return np.outer(self.vec, self.vec)

    def initial_weight(self, normalize: bool = False) -> np.ndarray:
        """
        Calculates a weight matrix with linear decreasing weight by distance between neurons.
        :return: 2D weight matrix with (i,j) = neuron_linear_distance_weight(i,j)
        """
        weight = np.zeros((self.N, self.N), dtype=float)
        for i in range(self.N):
            for j in range(self.N):
                weight[i][j] = self.neuron_linear_distance_weight(i, j, normalize)
        return weight

    def neuron_linear_distance_weight(self, i: int, j: int, normalize: bool = False) -> float:
        """
        Calculates the distance relative distance between two neurons.
        :param i: Index of first neuron
        :param j: Index of second neuron
        :param normalize: If set true the Sum of all connection weights for one neuron is 1
        :return: Linear value from 0.1 to 1 (0.1 for the maximum distance between them, 1 for no distance between them)
                    or 0 for i = j
        """
        if i == j:
            return 0
    
        xi, yi = self.xy_i(i)
        xj, yj = self.xy_i(j)
    
        d = sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
        m = sqrt((self.h - 1) ** 2 + (self.w - 1) ** 2)
        not_norm = (m - 0.9 * d - 0.1) / (m - 1)
        norm = not_norm / (0.6 * (m-1))
        return norm if normalize else not_norm
