import numpy as np
from datetime import datetime as dati
from math import ceil


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
    bc = ceil(l_i / bs_i)
    
    splits_i = []
    for i in range(bc):
        splits_i.append((i * bs_i, min(l_i, (i + 1) * bs_i)))
    
    splits_j = (0, l_j)
    
    for x in splits_i:
        ret.append((x, splits_j))
        
    return ret


def log_print(s: str):
    print(dati.now().strftime('%H:%M:%S,%f') + ' ' + s)
