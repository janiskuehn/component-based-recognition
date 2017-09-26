#!python3.5

import weight_evolution1
import sys
import barGenerator as bar
import numpy as np
from utils import fts
from os import mkdir
from os import path
from datetime import datetime as dati

if 9 < len(sys.argv) or len(sys.argv) < 8:
    print("Parameters: <Bars per side> <Width of a bar> <Alpha> <dt> <Steps per pattern> <Rotations>"
          " <Max synaptic range> [-p for parallel processing]")
    quit()

# Beta:
B = 1
# Set true for periodic boundary conditions
PERIODIC = False

# Bars per Dimension:
BPS = int(sys.argv[1])
# Width of one bar:
PPB = int(sys.argv[2])
# Alpha:
A = float(sys.argv[3])
# Delta T:
DT = float(sys.argv[4])
# Stepcount per Pattern per Rotation:
SPP = int(sys.argv[5])
# Rotations:
ROT = int(sys.argv[6])
# Max interaction range of a neuron:
R = float(sys.argv[7])
# Use mutliprocessing:
PARALLEL = False
if len(sys.argv) == 9 and sys.argv[8] == "-p":
    PARALLEL = True

print('Arguments:')
print('  Bars per dimension: %i' % BPS)
print('  Pixels per bar: %i' % PPB)
print('  Steps per pattern: %i' % SPP)
print('  Rotations: %i' % ROT)
print('  Alpha = %f' % A)
print('  Delta t = %f' % DT)
print('  Maximal Synaptic range = %f' % R + (' (periodic)' if PERIODIC else ' (not periodic)'))
print('  ' + ('Parallel execution.' if PARALLEL else 'Linear execution.'))

start = dati.now()

# set of all possible single bar constellations
SET = bar.generate_all_distinct_lines(BPS, PPB)

# Execution ##
s_set = []

for bb in SET:
    b = bar.BarTest(BPS, PPB)
    b.bars_active = bb
    s_set.append(b.as_neuralstate())
    s_set[-1].max_dis = R
    s_set[-1].periodic = PERIODIC

w0 = s_set[0].initial_weight(form=1, normalize=True)

w_f = weight_evolution1.learn_multiple_pattern(w0, s_set, A, B, SPP, ROT, DT, quiet=True,
                                                                     parallise=PARALLEL, onlyWfinale=True)

ax = w_f
fol = 'results/'
try:
    mkdir(fol)
except OSError:
    pass  # Folder exists

end = dati.now()
runtime = end - start
print('Execution needed %f seconds.' % runtime.total_seconds())
print('One rotation needed in average %f seconds.' % (runtime.total_seconds() / ROT))
print('Calculating dW needed in average %f seconds.' % np.mean(weight_evolution1.dw_calc_times))
print('Calculating D needed in average %f seconds.' % np.mean(weight_evolution1.w_power_calc_times))
print('Calculating D\' needed in average %f seconds.' % np.mean(weight_evolution1.d_prime_calc_times))


# Saving
fn = 'bps='+fts(BPS)+'_ppb='+fts(PPB)+'_spp='+fts(SPP)+'_rot='+fts(ROT)+'_alpha='+fts(A)+'_dt='+fts(DT)+'_r='+fts(R)\
     + ('_per' if PERIODIC else '')

i = 0
while path.exists('{}_v{:d}.npy'.format(fol+fn, i)):
    i += 1
    
file = '{}_{:d}.png'.format(fol+fn, i)
print('Saving W_finale to '+file)
np.save(file, ax)
