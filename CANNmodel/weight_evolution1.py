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


# parallisation concept:
class ParallelOperator:
    def __init__(self):
        self.inQueue = mp.Queue()
        self.outQueue = mp.Queue()
        self.workers = [mp.Process(target=po_worker, args=(self.inQueue, self.outQueue,)) for i in range(worker_count)]
        self.job_id = -1
        
        for p in self.workers:
            p.Daemon = True
            p.start()
    
    def next_job_id(self) -> int:
        self.job_id += 1
        return self.job_id
    
    def calc_dw(self, d_prime_mat: np.ndarray, w: np.ndarray, s: neural.NeuralState, alpha: float, beta: float,
                v_m: np.ndarray) -> np.ndarray:
        ret = np.zeros((s.N, s.N), dtype=float)
        index_packs = index_clustering_by_count(ret, worker_count, MIN_BLOCKSIZE)
        # log_print("calc_dw indices: " + str(index_packs))

        run_ids = []
        for pak in index_packs:
            run_ids.append(self.next_job_id())
            self.inQueue.put((run_ids[-1], 2, [pak, d_prime_mat, w, s, alpha, beta, v_m, ret]))

        for i in range(len(run_ids)):
            (run_id, res) = self.outQueue.get()
            ret += res
            run_ids.remove(run_id)

        if len(run_ids) > 0:
            print("Something wrong 1")
        return ret
    
    def calc_d_prime(self, d_mat: np.ndarray, s: neural.NeuralState) -> np.ndarray:
        ret = np.zeros((s.N, s.N), dtype=float)
        index_packs = index_clustering_by_count(ret, worker_count, MIN_BLOCKSIZE)
        # log_print("calc_d_prime indices: " + str(index_packs))
        
        run_ids = []
        for pak in index_packs:
            run_ids.append(self.next_job_id())
            self.inQueue.put((run_ids[-1], 1, [pak, d_mat, s, ret]))

        for i in range(len(run_ids)):
            (run_id, res) = self.outQueue.get()
            ret += res
            run_ids.remove(run_id)
        
        if len(run_ids) > 0:
            print("Something wrong 2")
        return ret
    
    def close(self):
        for p in self.workers:
            self.inQueue.put((-1, None, None))
        
        for p in self.workers:
            self.inQueue.close()
            self.inQueue.join_thread()
            self.outQueue.close()
            self.outQueue.join_thread()
            p.join()
            self.workers = None


def po_worker(in_q: mp.Queue, out_q: mp.Queue):
    log_print("Worker started")
    while True:
        # log_print("Worker check for input")
        (r_id, todo, args) = in_q.get()
        if r_id == -1:  # End progress
            log_print("Worker stopped")
            break
        
        else:
            if todo == 1:  # Calculate D_prime
                # log_print("job %i calc D_prime index indices: " % id + str(args[0]))
                result = d_prime_matrix_par(*args)
            
            elif todo == 2:  # Calculate dW
                # log_print("job %i calc dW index indices: " % id + str(args[0]))
                result = delta_w_didj(*args)
                
            else:
                result = None

            out_q.put((r_id, result))


# Differetial equation stuff:
def euler_evolution(w0: np.ndarray, s: neural.NeuralState, alpha: float, beta: float, steps: int,
                    dt: float = 1, dynamic_dt: bool = False, parallise: bool = True) -> list:
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
    par_op = None
    if parallise:
        par_op = ParallelOperator()
        
    ret = [w0]
    for i in range(1, steps+1):
        try:
            dw = delta_w(ret[-1], s, alpha, beta, par_op)
            
        except OverflowError:
            print("Overflow Error by step = "+str(i)+" for alpha = "+str(alpha)+" and dt = "+str(dt))
            break
        except FloatingPointError:
            print("Overflow Error by step = " + str(i) + " for alpha = " + str(alpha) + " and dt = " + str(dt))
            break
        dt = dt if not dynamic_dt else 0.1 / np.max(np.abs(dw))
        # log_print("dt="+str(dt)+"_dyn="+str(dynamic_dt))
        ret.append(ret[-1] + dt * dw)
        
    if parallise:
        par_op.close()
    
    return ret


def euler_evolution_moreinfo(w0: np.ndarray, s: neural.NeuralState, alpha: float, beta: float, steps: int,
                             dt: float = 1, parallise: bool = True) -> tuple:
    """
    """
    
    (w_a, t_a, dw_a, neurons_a) = learn_multiple_pattern(w0, [s], alpha, beta, steps, 1, dt, parallise,
                                                         only_w_finale=False, quiet=True)
    return w_a, t_a, dw_a


def learn_multiple_pattern(w0: np.ndarray, s_set: list, alpha: float, beta: float, steps_per_pattern: int,
                           rotations: int, dt: float, parallise: bool = True, only_w_finale: bool = False,
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
    :param only_w_finale: If set only the W_finale is returned, otherwise (w_a, t_a, dw_a, neurons_a) is returned
    :param quiet: do not log progress
    :return:
    """

    par_op = None
    if parallise:
        par_op = ParallelOperator()
        
    t_a = [0.0]
    dw_a = []
    neurons_a = []

    if only_w_finale:
        w_a = w0
    else:
        w_a = [w0]
    
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
                
                if only_w_finale:
                    wl = w_a.copy()
                else:
                    wl = w_a[-1]
                    
                try:
                    dw = delta_w(wl, pat, alpha, beta, par_op)
                except OverflowError:
                    print("Overflow Error by step = " + str(i) + " for alpha = " + str(alpha) + " and dt = " + str(dt))
                    break
                except FloatingPointError:
                    print("Overflow Error by step = " + str(i) + " for alpha = " + str(alpha) + " and dt = " + str(dt))
                    break
            
                if only_w_finale:
                    w_a = (wl + dt * dw)
                else:
                    w_a.append(wl + dt * dw)
                    t_a.append(t_a[-1] + dt)
                    dw_a.append(dw)
                    neurons_a.append(pat)

    if parallise:
        par_op.close()
        
    if only_w_finale:
        return w_a
    else:
        return w_a, t_a, dw_a, neurons_a


def delta_w(w: np.ndarray, s: neural.NeuralState, alpha: float, beta: float,
            parallel_op: ParallelOperator = None) -> np.ndarray:
    """
    Calculates
    :param w: Weight matrix w(t).
    :param s: Neuron state vector S_μ.
    :param alpha: The free parameter Alpha.
    :param beta: Free parameter to weight second term.
    :param parallel_op: ParallelOperator to use for a asynchronous calculation, for synchron calculation set None
    :return: 2D matrix dw/dt.
    """
    global worker_count, MIN_BLOCKSIZE, dw_calc_times
    
    d_mat = d_matrix(w)
    d_prime_mat = d_prime_matrix(d_mat, s, parallel_op)

    dw_calc_start = dati.now()
    
    v_m = v(s)
    if parallel_op is not None:
        ret = parallel_op.calc_dw(d_prime_mat, w, s, alpha, beta, v_m)
        
    else:
        shp = np.shape(w)
        ret = np.zeros(shp, dtype=float)
        
        for i in range(s.N):
            for j in range(s.N):
                ret[i][j] = delta_w_ij(i, j, d_prime_mat, w, s, alpha, beta, v_m)

    dw_calc_end = dati.now()
    dw_runtime = dw_calc_end - dw_calc_start
    dw_calc_times.append(dw_runtime.total_seconds())
    return ret
    

def delta_w_didj(pak: tuple, d_prime_mat: np.ndarray, w: np.ndarray, s: neural.NeuralState,
                 alpha: float, beta: float, v_m: np.ndarray, dw_ret: np.ndarray):
    for i in range(*pak[0]):
        for j in range(*pak[1]):
            dw_ret[i][j] = delta_w_ij(i, j, d_prime_mat, w, s, alpha, beta, v_m)
            
    return dw_ret
    
    
def delta_w_ij(i: int, j: int, d_prime_mat: np.ndarray, w: np.ndarray, s: neural.NeuralState,
               alpha: float, beta: float, v_m: np.ndarray) -> float:
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
    :return: Float Element i j of dw/dt
    """
    
    if i == j or s.vec[i] == 0 or s.vec[j] == 0:
        return 0
    
    else:
        first_term = v_m[i][j] * alpha * (1 - s.active_neuron_count() * w[i][j])
        
        a = np.dot(w[i], d_prime_mat[i])
        second_term = beta * w[i][j] * (d_prime_mat[i][j] - a)
        
        return first_term + second_term


def d_matrix(w: np.ndarray) -> np.ndarray:
    """
    Calculate matrix D, which solves equation c = D s.
    :param w: Given weight matrix w.
    :return: Matrix with identical shape. D = identity + w + w2 + w3
    """
    global w_power_calc_times
    w_power_calc_start = dati.now()
    
    one = np.identity(w.shape[0])
    
    w2 = la.matrix_power(w, 2)
    w3 = np.dot(w, w2)

    w_power_calc_end = dati.now()
    w_power_runtime = w_power_calc_end - w_power_calc_start
    w_power_calc_times.append(w_power_runtime.total_seconds())
    return one + w + w2 + w3


def d_prime_matrix(d_mat: np.ndarray, s: neural.NeuralState, parallel_op: ParallelOperator = None) -> np.ndarray:
    """
    Calculate matrix D'.
    :param d_mat: Matrix D (as calculated by d_matrix(w))
    :param s: Neural Vector (needed because sum only over active neurons)
    :param parallel_op: (optional) Set a ParallelOperator for multiprocessing
    :return: Matrix D' with D'_ij = sum_k(D_ik * D_jk)
    """
    global worker_count, MIN_BLOCKSIZE, d_prime_calc_times
    d_prime_calc_start = dati.now()
    
    if parallel_op is not None:
        ret = parallel_op.calc_d_prime(d_mat, s)
        
    else:
        ret = np.zeros(d_mat.shape, dtype=float)
        
        for i in range(s.N):
            for j in range(s.N):
                if s.vec[i] != 0 and s.vec[j] != 0:
                    ret[i][j] = np.dot(d_mat[i], d_mat[j])

    d_prime_calc_end = dati.now()
    d_prime_runtime = d_prime_calc_end - d_prime_calc_start
    d_prime_calc_times.append(d_prime_runtime.total_seconds())
    return ret


def d_prime_matrix_par(indices: tuple, d_mat: np.ndarray, s: neural.NeuralState, ret: np.ndarray):
    for i in range(*indices[0]):
        for j in range(*indices[1]):
            if s.vec[i] != 0 and s.vec[j] != 0:
                ret[i][j] = np.dot(d_mat[i], d_mat[j])
    return ret


def v(s: neural.NeuralState):
    """
    Returns the delta Matrix v with v_ij = 1 if d(i,j) <= max_dis else 0.
    Max_dis is the maximal synaptic range which is specified in the NeuralState class.
    And d(i,j) is the distance between neuron i and neuron j.
    :param s: The neural state
    :return: 2d Numpy Matrix
    """
    v_m = s.distance_matrix()
    for i in range(s.N):
        for j in range(s.N):
            v_m[i][j] = 0 if v_m[i][j] > s.max_dis else 1
    return v_m
