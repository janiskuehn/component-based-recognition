#!/usr/bin/env python

from pics import *
from neural import *
from utils import *

__author__ = "Janis Kühn"
__license__ = "Apache 2.0"
__email__ = "jk@stud.uni-frankfurt.de"
__status__ = "Prototype"

# initialize network:
# N Neurons, n*m pixel image
n = 24
m = 24
N = n*m
neurons = np.zeros(N, dtype=int)

# K connections between all neurons
K = N*N
connections = np.zeros((N, N), dtype=int)

# stored pictures:
# pictures = ['../Alphabet/A.jpeg','../Alphabet/B.jpeg','../Alphabet/C.jpeg','../Alphabet/D.jpeg','../Alphabet/E.jpeg']
pictures = ['../Alphabet/A.jpeg', '../Alphabet/B.jpeg']
# pictures = ['../Alphabet/test.jpeg']

for p in pictures:
    # Read Image to pattern:
    original = bipolize_image(p, 150)
    PrintMatrix(original)
    # learn pattern
    learn_pattern(original, n, m, connections)


modified = bipolize_image('../Alphabet/B2.jpeg', 150)
# modified = binarize_image('../Alphabet/test.jpeg',150)

# start  recognition
neurons += modified.flatten()
print('initial neuronal activity:')
NeuralMap(neurons, n, m)

# progressing recognition
f1 = Recognition(neurons, n, m, connections, 1, 10*N, silence=True)
f2 = Recognition(neurons, n, m, connections, 1, N, silence=True)
print('final neuronal activity:')
NeuralMap(f1, n, m)
NeuralMap(f2, n, m)
