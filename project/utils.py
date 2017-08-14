import numpy as np


def print_matrix(matrix: np.ndarray):
    s = ''
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            s += ' ' + str(matrix[i][j])
        s += '\n'
    print(s)
