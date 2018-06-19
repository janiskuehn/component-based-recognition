import plot
import numpy as np
import sys

if len(sys.argv) != 2:
    print("Parameters: <.npy filename>")
    quit()
    
fname = str(sys.argv[1])

a = np.load(fname)

print(a)

plot.height_plot(a)