from neural import NeuralState
from utils import *
from random import random as rand


def recall(input: NeuralState, w: np.ndarray, threshold: float = 1.0, iterations:int = 5) -> list:
    """Let the network work"""
    n = [input.vec.copy()]
    for i in range(iterations):
        x = rand() * (input.N-1)
        x = int(round(x))
        activity = np.dot(n[-1], w[x])
        new = n[-1].copy()
        new[x] = 1 if activity >= threshold else 0
        n.append(new)

    return n
