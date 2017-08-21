#!/usr/bin/env python

from neural import NeuralState
import weight_evolution1
from barGenerator import BarTest
import numpy as np
import output
import plot

__author__ = "Janis Kühn"
__license__ = "GPLv3"
__email__ = "jk@stud.uni-frankfurt.de"
__status__ = "Prototype"
__pythonVersion__ = "3.6"

# b = BarTest(4, 1, density=0.4)

# v = b.as_array()
v = np.array([[0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [1, 1, 1, 1]])

s = NeuralState(4, 4, v)

w0 = s.initial_weight_localized(periodic=True)

a = 1
b = 1
dt = 0.0000005
steps = 1000
stepsize = 50
# plot.height_plot(weight_evolution1.delta_w(w0, s, a, b))
plot.height_plot(w0)
# output.stepwise_plot_to_file(s, w0, steps, stepsize, dt, a, b, 1, True)
# output.print_last_result(s, w0, steps, dt, a, b)


