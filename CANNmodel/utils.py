import numpy as np
from datetime import datetime as dati
from math import ceil
from neural import NeuralState
import random


def print_matrix(matrix: np.ndarray):
    s = ''
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            s += ' ' + str(matrix[i][j])
        s += '\n'
    print(s)


def metadata(size: int, r: float, dt: float, steps: int, a: float) -> str:
    return "Mesh size = " + str(size) + " x " + str(size) + \
            "\nR = " + str(r) + \
            "\ndt = " + str(dt) + \
            "\nsteps = " + str(steps) + \
            "\nt_max = " + str(dt * steps) + \
            "\nalpha = " + str(a)


def fts(inp) -> str:  # Float to String
    return ('%.15f' % inp).rstrip('0').rstrip('.')


def index_clustering_by_count(m: np.ndarray, bc: int, min_bs: int) -> list:
    """
    Returns index-packages of size bs_i * bs_j
    :param m: matrix
    :param bc: blockcount
    :param min_bs: minimal blocksize in elements
    :return: list of tupels
    """
    ret = []
    
    l_i = m.shape[0]
    l_j = m.shape[1]
    bs_i = max(ceil(l_i / bc), min_bs // l_j)
    bc = int(ceil(l_i / bs_i))
    
    splits_i = []
    for i in range(bc):
        splits_i.append((i * bs_i, min(l_i, (i + 1) * bs_i)))
    
    splits_j = (0, l_j)
    
    for x in splits_i:
        ret.append((x, splits_j))
        
    return ret


def log_print(s: str):
    print(dati.now().strftime('%H:%M:%S,%f') + ' ' + s)


def disturb_random(s: NeuralState, strength: float) -> NeuralState:
    a = s.copy()
    
    n = int(a.N * strength)
    ind = random.sample(range(a.N), n)
    val = np.random.randint(0, 2, n)
    a.vec[ind] = val
    
    return a
    

def disturb_inverting(s: NeuralState, strength: float) -> NeuralState:
    a = s.copy()
    
    c = a.active_neuron_count()
    n = int(c * strength)

    ind_inactive = np.where(n == 0)
    ind_inactive = random.sample(ind_inactive, n)
    
    ind_active = a.vec.nonzero()
    ind_active = random.sample(ind_active, n)

    a.vec[ind_inactive] = np.zeros(n)
    a.vec[ind_active] = np.ones(n)
    
    return a