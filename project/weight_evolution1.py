import numpy as np
import numpy.linalg as la
import neural
import multiprocessing as mp
from utils import *

worker_count = mp.cpu_count()
MIN_BLOCKSIZE = 10*1000
# Necessary for Overflow Error detection:
np.seterr(all='raise')
dw_calc_times = []
w_power_calc_times = []
d_prime_calc_times = []


def euler_evolution(w0: np.ndarray, s: neural.NeuralState, alpha: float, beta: float, steps: int,
                    dt: float = 1, dynamic_dt: bool = False, parallise: bool = False) -> list:
    """
    Applies the Euler differential equation method to delta_w.
    :param w0: Initial state.
    :param s: Initial neuron state.
    :param alpha: Free parameter Alpha.
    :param beta: Free parameter to weight second term.
    :param steps: How many steps will be done.
    :param dt: Step size.
    :param dynamic_dt: If set dt is replaced by a value depending von dw for each step. It is chosen such the maximum
            change happening in the step ist 1.
    :param parallise: (optional) Set true for multiprocessing
    :return: A list starting with w0 and ending with w_steps
    """
    ret = [w0]
    for i in range(1, steps+1):
        try:
            dw = delta_w(ret[i - 1], s, alpha, beta, parallise)
            
        except OverflowError:
            print("Overflow Error by step = "+str(i)+" for alpha = "+str(alpha)+" and dt = "+str(dt))
            break
        except FloatingPointError:
            print("Overflow Error by step = " + str(i) + " for alpha = " + str(alpha) + " and dt = " + str(dt))
            break
        dt = dt if not dynamic_dt else 0.1 / np.max(np.abs(dw))
        # log_print("dt="+str(dt)+"_dyn="+str(dynamic_dt))
        ret.append(ret[i-1] + dt * dw)
    return ret


def euler_evolution_moreinfo(w0: np.ndarray, s: neural.NeuralState, alpha: float, beta: float, steps: int,
                             dt: float = 1, dynamic_dt: bool = False, parallise: bool = False) -> tuple:
    """
    """
    w_a = [w0]
    t_a = [0]
    dw_a = []
    
    for i in range(1, steps+1):
        try:
            dw = delta_w(w_a[i - 1], s, alpha, beta, parallise)
            
        except OverflowError:
            print("Overflow Error by step = "+str(i)+" for alpha = "+str(alpha)+" and dt = "+str(dt))
            break
        except FloatingPointError:
            print("Overflow Error by step = " + str(i) + " for alpha = " + str(alpha) + " and dt = " + str(dt))
            break
        dt = dt if not dynamic_dt else 0.1 / np.max(np.abs(dw))
        w_a.append(w_a[i-1] + dt * dw)
        t_a.append(t_a[-1] + dt)
        dw_a.append(dw)
        
    return w_a, t_a, dw_a


def learn_multiple_pattern(w0: np.ndarray, s_set: list, alpha: float, beta: float, steps_per_pattern: int,
                           rotations: int, dt: float, parallise: bool = False, onlyWfinale: bool = False,
                           quiet: bool = False):
    """
    
    :param w0:
    :param s_set:
    :param alpha:
    :param beta:
    :param steps_per_pattern:
    :param rotations:
    :param dt:
    :param parallise: (optional) Set true for multiprocessing
    :return:
    """

    if onlyWfinale:
        w_a = w0
    else:
        w_a = [w0]
        t_a = [0.0]
        dw_a = []
        neurons_a = []
    
    for rot in range(rotations):
        # rotation
        
        p_c = 0
        for pat in s_set:
            p_c += 1
            # pattern
            
            for i in range(steps_per_pattern):
                if not quiet:
                    log_print("Rot: %i/%i, Pat: %i/%i, Step: %i/%i"
                              % (rot + 1, rotations, p_c, len(s_set), i+1, steps_per_pattern))
                # learning
                
                if onlyWfinale:
                    wl = w_a
                else:
                    wl = w_a[i - 1]
                    
                try:
                    dw = delta_w(wl, pat, alpha, beta, parallise)
                except OverflowError:
                    print("Overflow Error by step = " + str(i) + " for alpha = " + str(alpha) + " and dt = " + str(dt))
                    break
                except FloatingPointError:
                    print("Overflow Error by step = " + str(i) + " for alpha = " + str(alpha) + " and dt = " + str(dt))
                    break
            
                if onlyWfinale:
                    w_a = w_a[i - 1] + dt * dw
                else:
                    w_a.append(w_a[i - 1] + dt * dw)
                    t_a.append(t_a[-1] + dt)
                    dw_a.append(dw)
                    neurons_a.append(pat)
                    
    if onlyWfinale:
        return w_a
    else:
        return w_a, t_a, dw_a, neurons_a


def delta_w(w: np.ndarray, s: neural.NeuralState, alpha: float, beta: float, parallel: bool = False) -> np.ndarray:
    """
    Calculates
    :param w: Weight matrix w(t).
    :param s: Neuron state vector S_μ.
    :param alpha: The free parameter Alpha.
    :param beta: Free parameter to weight second term.
    :return: 2D matrix dw/dt.
    """
    global worker_count, MIN_BLOCKSIZE, dw_calc_times
    shp = np.shape(w)
    l = w.shape[0]
    
    ret = np.zeros(shp, dtype=float)
    
    d_mat = d_matrix(w)
    d_prime_mat = d_prime_matrix(d_mat, s, parallel)

    dw_calc_start = dati.now()
    
    v_m = v(s)
    if parallel:
        index_packs = index_clustering_by_count(ret, worker_count, MIN_BLOCKSIZE)
        
        out = mp.JoinableQueue()
        processes = [mp.Process(target=delta_w_didj, args=(pak, d_prime_mat, w, s, alpha, beta, v_m, ret, out, ))
                     for pak in index_packs]
        
        for p in processes:
            p.Daemon = True
            p.start()
            
        for p in processes:
            ret += out.get()
            
        for p in processes:
            p.join()

        while not out.empty():
            ret += out.get()
        
    else:
        for i in range(l):
            for j in range(l):
                ret[i][j] = delta_w_ij(i, j, d_prime_mat, w, s, alpha, beta, v_m)

    dw_calc_end = dati.now(); dw_runtime = dw_calc_end - dw_calc_start
    dw_calc_times.append(dw_runtime.total_seconds())
    return ret
    

def delta_w_didj(pak: tuple, d_prime_mat: np.ndarray, w: np.ndarray, s: neural.NeuralState,
                 alpha: float, beta: float, v_m: np.ndarray, dw_ret: np.ndarray, out: mp.Queue):
    for i in range(*pak[0]):
        for j in range(*pak[1]):
            delta_w_ij(i, j, d_prime_mat, w, s, alpha, beta, v_m, dw_ret)
            
    out.put(dw_ret)
    
    
def delta_w_ij(i: int, j: int, d_prime_mat: np.ndarray, w: np.ndarray, s: neural.NeuralState,
               alpha: float, beta: float, v_m: np.ndarray, dw_ret: np.ndarray = None) -> float:
    """
    Calculate Element i j of dw/dt.
    :param i: Index i.
    :param j: Index j.
    :param d_prime_mat: D_prime previously calculated by d_prime_matrix(d).
    :param w: Weight matrix w(t).
    :param s: Neuron state vector S_μ.
    :param alpha: The free parameter Alpha.
    :param beta: Free parameter to weight second term.
    :param v_m: Distance matrix. (see v(s).)
    :param dw_ret: (optional) If set the return value will be written to dw_ret[i][j]
    :return: Float Element i j of dw/dt
    """
    if i == j or s.vec[i] == 0 or s.vec[j] == 0:
        return 0
    first_term = v_m[i][j] * alpha * (1 - s.active_neuron_count() * w[i][j])
    
    l = w.shape[0]
    a = [(w[i][j_prime] * d_prime_mat[i][j_prime]) for j_prime in range(l)]
    
    second_term = beta * w[i][j] * (d_prime_mat[i][j] - sum(a))
    
    ret = first_term + second_term
    
    if dw_ret is not None:
        dw_ret[i][j] = ret
    else:
        return ret


def d_matrix(w: np.ndarray) -> np.ndarray:
    """
    Calculate matrix D, which solves equation c = D s.
    :param w: Given weight matrix w.
    :param parallel: (optional) Set true for multiprocessing
    :return: Matrix with identical shape. D = identity + w + w2 + w3
    """
    global w_power_calc_times
    w_power_calc_start = dati.now()
    
    l = w.shape[0]
    one = np.identity(l)
    
    w2 = la.matrix_power(w, 2)
    w3 = np.dot(w, w2)

    w_power_calc_end = dati.now(); w_power_runtime = w_power_calc_end - w_power_calc_start
    w_power_calc_times.append(w_power_runtime.total_seconds())
    return one + w + w2 + w3


def d_prime_matrix(d_mat: np.ndarray, s: neural.NeuralState, parallel: bool = False) -> np.ndarray:
    """
    Calculate matrix D'.
    :param d_mat: Matrix D (as calculated by d_matrix(w))
    :param s: Neural Vector (needed because sum only over active neurons)
    :param parallel: (optional) Set true for multiprocessing
    :return: Matrix D' with D'_ij = sum_k(D_ik * D_jk)
    """
    global worker_count, MIN_BLOCKSIZE, d_prime_calc_times
    d_prime_calc_start = dati.now()
    
    ret = np.zeros(d_mat.shape, dtype=float)
    l = d_mat.shape[0]
    
    if parallel:
        index_packs = index_clustering_by_count(ret, worker_count, MIN_BLOCKSIZE)
        
        out = mp.Queue()
        processes = [mp.Process(target=d_prime_matrix_par, args=(pak, d_mat, s, ret, out, ))
                     for pak in index_packs]
        
        for p in processes:
            p.Daemon = True
            p.start()
            
        for p in processes:
            ret += out.get()
            
        for p in processes:
            p.join()
        
    else:
        for i in range(l):
            for j in range(l):
                if s.vec[i] != 0 and s.vec[j] != 0:
                    ret[i][j] = np.dot(d_mat[i], d_mat[j])

    d_prime_calc_end = dati.now(); d_prime_runtime = d_prime_calc_end - d_prime_calc_start
    d_prime_calc_times.append(d_prime_runtime.total_seconds())
    return ret


def d_prime_matrix_par(indices: tuple, d_mat: np.ndarray, s: neural.NeuralState, ret: np.ndarray, out: mp.Queue):
    for i in range(*indices[0]):
        for j in range(*indices[1]):
            if s.vec[i] != 0 and s.vec[j] != 0:
                ret[i][j] = np.dot(d_mat[i], d_mat[j])
    out.put(ret)


def v(s: neural.NeuralState):
    """
    Returns the delta Matrix v with v_ij = 1 if d(i,j) <= max_dis else 0.
    Max_dis is the maximal synaptic range which is specified in the NeuralState class.
    And d(i,j) is the distance between neuron i and neuron j.
    :param s: The neural state
    :return: 2d Numpy Matrix
    """
    v_m = s.distance_matrix()
    l = v_m.shape[0]
    for i in range(l):
        for j in range(l):
            v_m[i][j] = 0 if v_m[i][j] > s.max_dis else 1
    return v_m
