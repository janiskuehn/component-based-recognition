import numpy as np
from neural import NeuralState
import os.path
import weight_evolution1 as we
import plot

res_fo = "results/"

np.set_printoptions(precision=2)


def stepwise_plot_to_file(s: NeuralState, w0: np.ndarray, stepcount: int, stepsize: int,
                          dt: float, alpha: float, beta: float,  p_type: int = 0, dynamic_dt: bool = False):
    """
    TODO
    :param s:
    :param w0:
    :param stepcount:
    :param stepsize:
    :param dt:
    :param alpha:
    :param beta:
    :param p_type:
    :param dynamic_dt:
    :return:
    """
    dt_s = str(dt) if not dynamic_dt else "dyn"
    sdir = res_fo+"w_dt=" + dt_s + "_a=" + str(alpha)
    try:
        os.mkdir(sdir)
    except OSError:
        pass  # Folder exists

    plot.hinton(s.vec, file=sdir + "/neurons.png", max_weight=1)
    
    plot.hinton(s.binary_weights(), file=sdir + "/binary.png", max_weight=1)
    
    print("Calculating evolution...")
    w = we.euler_evolution(w0, s, alpha, beta, stepcount, dt, dynamic_dt)
    
    print("Saving files to "+sdir)
    for i in range(0, len(w), stepsize):
        fn = sdir + "/step=" + str(i) + ".png"
        if p_type == 0:
            plot.hinton(w[i], file=fn)
        elif p_type == 1:
            plot.height_plot(w[i], file=fn)


def print_last_result(s: NeuralState, w0: np.ndarray, stepcount: int, dt: float,
                      alpha: float, beta: float):
    """
    TODO
    :param s:
    :param w0:
    :param stepcount:
    :param stepsize:
    :param dt:
    :param alpha:
    :param beta:
    :return:
    """
    print("Neuronal state:")
    print(s.print())
    
    print("Binary weights:")
    print(s.binary_weights())
    
    print("Calculating evolution...")
    w = we.euler_evolution(w0, s, alpha, beta, stepcount, dt)
    
    print(w[-1])
