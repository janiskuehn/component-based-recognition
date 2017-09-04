#!python3.6

import weight_evolution1
import sys
import barGenerator as bar
import numpy as np

print(len(sys.argv))
if len(sys.argv) != 8:
    print("Parameters: <Bars per side> <Width of a bar> <Alpha> <dt> <Steps per pattern> <Rotations>"
          "<Max synaptic range>")
    quit()

# Beta:
B = 1
# Set true for periodic boundary conditions
PERIODIC = False

# Bars per Dimension:
BPS = sys.argv[1]
# Width of one bar:
PPB = sys.argv[2]
# Alpha:
A = sys.argv[3]
# Delta T:
DT = sys.argv[4]
# Stepcount per Pattern per Rotation:
SPP = sys.argv[5]
# Rotations:
ROT = sys.argv[6]
# Max interaction range of a neuron:
R = sys.argv[7]


# set of all possible single bar constellations
SET = bar.generate_all_distinct_lines(BPS, PPB)

print('Arguments:')
print('  Bars per dimension: %i' % BPS)
print('  Pixels per bar: %i' % PPB)
print('  Steps per pattern: %f' % SPP)
print('  Rotations: %i' % ROT)
print('  Alpha = %f' % A)
print('  Delta t = %f' % DT)
print('  Maximal Synaptic range = %f' % R + (' (periodic)' if PERIODIC else ' (not periodic)'))

# Execution ##
s_set = []

for bb in SET:
    b = bar.BarTest(BPS, PPB)
    b.bars_active = bb
    s_set.append(b.as_neuralstate())
    s_set[-1].max_dis = R
    s_set[-1].periodic = PERIODIC

w0 = s_set[0].initial_weight(form=1, normalize=True)

(w_t, t, dw_t, neurons_t) = weight_evolution1.learn_multiple_pattern(w0, s_set, A, B, SPP, ROT, DT)

ax = np.array([w_t, t, dw_t, neurons_t])
