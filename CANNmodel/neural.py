from math import sqrt
from math import exp
from math import log
import numpy as np
from pics import print_binay_image


class NeuralState:  # S μ
    """A neuron state vector"""
    width = 0  # width of neural map
    height = 0  # height of neural map
    n = 0  # Count of neurons
    
    periodic = False  # Is Vectormap periodic?
    max_dis = -1  # Maximum reach of of a synapse
    
    activation = None  # Vector of neural activation: activation of neuron i (Size N)
    weights = None  # Synaptic connection weight: connection strength between neuron i and j (Size N x N)
    positions = None  # position matrix: position of neuron i in map (Size N x 2)
    distances = None  # distance matrix: distance between neuron i and j (Size N x N)
    
    def __init__(self, width: int, height: int, periodic: bool, max_synapse_length: float,
                 initial_activation: np.ndarray = None, initial_weights: np.ndarray = None):
        """
        A neural state vector. Includes weight * height = N neurons with binary value.
        :param width: Width of the 2D neuron array.
        :param height: Height of the 2D neuron array.
        :param periodic: Set true for periodic boundaries. Important in distance calculation.
        :param max_synapse_length: Maximal range of a synapse from one neuron to another. (-1 means infinity)
        :param initial_activation: (optional) Given neural vector with size N.
        """
        self.width = width
        self.height = height
        self.n = width * height
        self.periodic = periodic
        self.max_dis = max_synapse_length
        
        if initial_activation is None:
            self.activation = np.zeros(self.n, dtype=float)
        else:
            if initial_activation.size != self.n:
                print("Initial activation vector size does not meet width and height")
                quit()
            self.activation = initial_activation.copy()
            
        if initial_weights is None:
            self.weights = np.zeros((self.n, self.n), dtype=float)
        else:
            if initial_weights.shape != (self.n, self.n):
                print("Initial vector size does not meet width and height")
                quit()
            self.weights = initial_weights.copy()
        
        self.positions = self.position_matrix()
        self.distance = self.distance_matrix()
            
    def __copy__(self):
        """
        Internal copy function.
        :return: 1:1 Copy of this neural state
        """
        return self.copy()

    def copy(self):
        """
        :return: 1:1 Copy of this neural state
        """
        return NeuralState(self.width, self.height, self.periodic, self.max_dis, self.activation, self.weights)
    
    def as_matrix(self) -> np.ndarray:
        """
        Return activity pattern as 2D matrix.
        """
        dim2 = self.activation.reshape(self.width, self.height)
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
        xi = (i % self.width)
        yi = (i // self.width)
        return xi, yi
    
    def active_neuron_count(self) -> int:
        """
        Returns how many neurons are active.
        :return: Integer - Number of active neurons.
        """
        return np.count_nonzero(self.activation)
        
    def binary_weights(self) -> np.ndarray:
        """
        Return a binary weight matrix for given state.
        :return: 2D weight matrix with 1 if both neurons are active, else 0.
        """
        x = np.outer(self.activation, self.activation)
        np.fill_diagonal(x, 0)
        return x

    def initial_weight(self, form: int = 0, normalize: bool = False) -> np.ndarray:
        """
        Calculates a weight matrix with linear decreasing weight by distance between neurons.
        :param form: 0 -> linear, 1-> exp
        :param normalize: If set true the sum of all connection weights for one neuron is 1.
        :return: 2D weight matrix with (i,j) = neuron_linear_distance_weight(i,j)
        """
        weight = np.zeros((self.n, self.n), dtype=float)
        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    weight[i][j] = 0
                    continue
    
                d = self.distance[i][j]
                
                r = self.max_dis
                not_norm = 0
                if r == -1:
                    r = self.distance[0][self.n - 1]+1
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
            for i in range(self.n):
                weight[i] = weight[i] / sum(weight[i])
            
        return weight


    def __distance_matrix(self) -> np.ndarray:
        """
        Generates matrix where element i,j contains the physical distance between neurons i and j
        :return: 2D matrix
        """
        d = np.zeros((self.n, self.n), dtype=float)
        for i in range(self.n):
            for j in range(self.n):
                
                dis = 0
                if i != j:
                    xi, yi = self.xy_i(i)
                    xj, yj = self.xy_i(j)
                    d[i][j] = sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
                    if self.periodic:
                        d_p = [dis]
                        xj_p = [xj + self.width, xj - self.width, xj, xj]
                        yj_p = [yj, yj, yj + self.height, yj - self.height]
                        for k in range(4):
                            d_p.append(sqrt((xi - xj_p[k]) ** 2 + (yi - yj_p[k]) ** 2))
                        dis = min(d_p)
                d[i][j] = dis
                
        return d


    def __position_matrix(self) -> np.ndarray:
        """
        Generates matrix where element i contains [x,y] coordinates of neuron i
        :return: 2D matrix of size n x 2
        """
        return p
        