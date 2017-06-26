import numpy as np
from pics import PrintBinayImage
import random
from utils import PrintMatrix

def NeuralMap(neurons: np.ndarray, n: int, m:int):
    """Print out neural activity"""
    L = neurons.reshape(n,m)
    PrintBinayImage(L)

def learn_pattern(pat: np.ndarray, n: int ,m: int, connections) -> np.ndarray:
    """Connections-Array connections learns pattern pat"""
    (max_i, max_j) = pat.shape
    if max_i > n or max_j > m:
        print("Pattern to big")
        return None
    
    print('Learning pattern of size '+str((max_i,max_j)))
    flatpat = pat.flatten()
    connectionpattern = np.outer(flatpat,flatpat)
    np.fill_diagonal(connectionpattern,0)
    connections += connectionpattern
    return connections

def flat_pattern(pat: np.ndarray) -> np.ndarray:
    return pat.flatten()

def Recognition(neurons: np.ndarray, n: int, m: int, connections: np.ndarray, threshold: float = 1.0, iterations:int = 5, silence:bool=False) -> np.ndarray:
    """Let the network work"""
    finalMap = neurons.copy()
    for i in range(iterations):
        x = random.random() * (n*m-1)
        x = int(np.round(x))
        activity = finalMap.dot(connections[x])
        if not silence:
            print(str(i)+'\'s iteration:'+' calculate neuron number '+str(x)+'. Ingoing activity = '+str(activity))
            finalMap[x] = 1 if activity >= threshold else -1
    return finalMap
