#!/usr/bin/env python

from neural import NeuralState
import weight_evolution1 as we
from barGenerator import BarTest
from PIL import Image as img


__author__ = "Janis Kühn"
__license__ = "GPLv3"
__email__ = "jk@stud.uni-frankfurt.de"
__status__ = "Prototype"
__pythonVersion__ = "3.6"

b = BarTest(10, 3, density=0.2)

v = b.as_array()
print(v)

s = NeuralState(10, 10, v)

print(
    # a.neuron_linear_distance_weight(0, 1)
    # a.initial_weight()
)

w0 = s.initial_weight()
print(w0)

alpha = 1
dt = 0.1

w1 = we.euler_evolution(w0, s, alpha, 1, dt)[1]

print(w1)