#!/usr/bin/env python

from neural import NeuralState
import weight_evolution1
from barGenerator import BarTest
from barGenerator import generate_all_distinct_lines
import numpy as np
import output
import plot
import utils
from multiprocessing import Process

__author__ = "Janis Kühn"
__license__ = "GPLv3"
__email__ = "jk@stud.uni-frankfurt.de"
__status__ = "Prototype"
__pythonVersion__ = "3.6"


SIZE = 4        # Number of bars in one dimension
PPB = 2         # width of one bar
B = 1           # beta - free parameter multiplied to the second term of the differential equation
DT = 0.1
R = 4           # max interaction range of a neuron

# set of all possible single bar constellations
SET = generate_all_distinct_lines(SIZE, PPB)

A = 0.1             # alpha
IND = 3             # which index of the generated SET should be used by default
STEPS = 10          # number of steps to simulate
STEPSIZE = 1        # determines which steps will be shown in generated graphs (every nth)
PERIODIC = False    # Set true for periodic boundary conditions


def one_pattern_combi():
    global SET, SIZE, B, DT, PPB
    b = BarTest(SIZE, PPB)
    
    ind_1 = [3, 7]
    
    steps_1 = [10, 100]
    stepsizes_1 = [2, 20]
    
    a1 = [0.1, 0.5, 1, 5, 10, 50, 100]

    for a in a1:
        for i in range(len(steps_1)):
            steps = steps_1[i]
            stepsize = stepsizes_1[i]
            for ind in ind_1:
                b.bars_active = SET[ind]
                s = b.as_neuralstate()
                s.max_dis = R
                s.periodic = PERIODIC
    
                m = utils.metadata(SIZE, R, DT, steps, a)
                w0 = s.initial_weight(form=1, normalize=True)
                output.complete_set(s, w0, steps, stepsize, DT, a, B, dynamic_dt=False, meta_data=m, to_file=True)


def multi_pattern():
    global SET, SIZE, A, B, DT, PPB
    s_set = []
    
    for bb in SET:
        b = BarTest(SIZE, PPB)
        b.bars_active = bb
        s_set.append(b.as_neuralstate())
        s_set[-1].max_dis = R
        s_set[-1].periodic = PERIODIC
    
    spp = 1
    rot = 20
    
    w0 = s_set[0].initial_weight(form=1, normalize=True)
    output.learning1(s_set, w0, spp, rot, DT, A, B, to_file=True)


def test():
    global A, B, SIZE, SET, IND, STEPS, STEPSIZE, PPB
    b = BarTest(SIZE, PPB)
    b.bars_active = SET[IND]
    s = b.as_neuralstate()
    s.max_dis = R
    s.periodic = PERIODIC

    m = utils.metadata(SIZE, R, DT, STEPS, A)
    w0 = s.initial_weight(form=1, normalize=True)
    
    # output.complete_set(s, w0, STEPS, STEPSIZE, DT, A, B, dynamic_dt=False, meta_data=m, to_file=True)
    
    # w_n = weight_evolution1.euler_evolution(w0, s, A, B, STEPS, DT)

    # plot.height_plot(weight_evolution1.delta_w(w0, s, A, B))

    # plot.height_plot(w0)

    output.stepwise_plot_to_file(s, w0, STEPS, STEPSIZE, DT, A, B, p_type=1, dynamic_dt=False)

    # output.print_last_result(s, w0, STEPS, DT, A, b)


if __name__ == "__main__":
    # one_pattern_combi()
    multi_pattern()
    # test()
