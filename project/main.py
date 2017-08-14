#!/usr/bin/env python

from neural import NeuralState
import barGenerator as bar


__author__ = "Janis Kühn"
__license__ = "Apache 2.0"
__email__ = "jk@stud.uni-frankfurt.de"
__status__ = "Prototype"
__pythonVersion__ = "3.6"

a = NeuralState(10, 10)

print(
    #a.neuron_linear_distance_weight(0, 1)
    a.initial_weight()
)