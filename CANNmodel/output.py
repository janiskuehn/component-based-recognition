import numpy as np
from neural import NeuralState
import os.path
import weight_evolution1 as we
import plot
import datetime

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
    today = datetime.date.today()
    sdir = res_fo+"w_dt=" + dt_s + "_a=" + str(alpha) + "_date=" + str(today)
    try:
        os.mkdir(sdir)
    except OSError:
        pass  # Folder exists

    plot.height_plot(s.as_matrix(), file=sdir + "/neurons.png")
    
    plot.height_plot(s.binary_weights(), file=sdir + "/binary.png")
    
    print("Calculating evolution...")
    w = we.euler_evolution(w0, s, alpha, beta, stepcount, dt, dynamic_dt)
    
    print("Saving files to " + sdir)
    for i in range(0, len(w), stepsize):
        fn = sdir + "/step=" + str(i) + ".png"
        if p_type == 0:
            plot.hinton(w[i], file=fn)
        elif p_type == 1:
            plot.height_plot(w[i], file=fn)


def print_last_result(s: NeuralState, w0: np.ndarray, stepcount: int, dt: float,
                      alpha: float, beta: float):
    """

    :param s:
    :param w0:
    :param stepcount:
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


def complete_set(s: NeuralState, w0: np.ndarray, stepcount: int, stepsize: int, dt: float, alpha: float,
                 beta: float, dynamic_dt: bool = False, meta_data: str = "", to_file: bool = True):

    today = datetime.date.today()
    sdir = res_fo + "res_" + str(today) + "/"
    
    try:
        os.mkdir(sdir)
    except OSError:
        pass  # Folder exists
    
    dt_s = str(dt) if not dynamic_dt else "dyn"
    
    print("Calculating evolution...")
    w, t, dw = we.euler_evolution_moreinfo(w0, s, alpha, beta, stepcount, dt, dynamic_dt)
    
    neurons = s.as_matrix()
    hopfield = s.binary_weights()
    
    iden = "_dt=" + dt_s + "_a=" + str(alpha) + "_steps=" + str(stepcount)
    fn = (sdir + "weights" + iden) if to_file else None
    plot.combined_plot1(w, t, dw, stepsize, neurons, hopfield, file=fn, metadata=meta_data)


def learning1(s_set: list, w0: np.ndarray, steps_per_pattern: int, rotations: int, dt: float, alpha: float, beta: float,
              to_file: bool = True):
    today = datetime.date.today()
    sdir = res_fo + "res_" + str(today) + "/"
    try:
        os.mkdir(sdir)
    except OSError:
        pass  # Folder exists
    
    # w: list # list of weight matrices
    # t: list # list of times
    # dw: list # list of weight deviations
    # neurons_t: list # list of neural states
    
    print("Learning pattern set ...")
    w, t, dw, neurons_t = we.learn_multiple_pattern(w0, s_set, alpha, beta, steps_per_pattern, rotations, dt)
    
    iden = "_dt=" + str(dt) + "_a=" + str(alpha) + "_spp=" + str(steps_per_pattern) + "_rot=" + str(rotations)
    fn = (sdir + "learning" + iden) if to_file else None
    plot.combined_learning_plot_patternwise(w, t, dw, neurons_t, s_set, steps_per_pattern, rotations, file=fn)
    plot.height_plot(w[-1], fn+"_final.png")
    plot.height_plot(w[0], fn + "_init.png")
