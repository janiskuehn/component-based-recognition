import numpy as np


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
