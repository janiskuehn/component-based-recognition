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
        if initial_vector:
            self.vec = initial_vector.copy()
        else:
            self.vec = np.zeros(self.N, dtype=int)

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

    def neuron_linear_distance_weight(self, i: int, j: int) -> float:
        """
        Calculates the distance relative distance between two neurons.
        :param i: Index of first neuron
        :param j: Index of second neuron
        :return: Linear value from 0.1 to 1 (0.1 for the maximum distance between them, 1 for no distance between them)
                    or 0 for i = j
        """
        if i == j:
            return 0
        
        xi, yi = self.xy_i(i)
        xj, yj = self.xy_i(j)
        
        d = sqrt((xi - xj)**2 + (yi - yj)**2)
        m = sqrt((self.h - 1)**2 + (self.w - 1)**2)
        return (m - 0.9 * d - 0.1) / (m - 1)
    
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

    def initial_weight(self) -> np.ndarray:
        """
        Calculates a weight matrix with linear decreasing weight by distance between neurons.
        :return: 2D weight matrix with (i,j) = neuron_linear_distance_weight(i,j)
        """
        weight = np.zeros((self.N, self.N), dtype=float)
        for i in range(self.N):
            for j in range(self.N):
                weight[i][j] = self.neuron_linear_distance_weight(i, j)
        return weight
