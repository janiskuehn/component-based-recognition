import numpy as np
import numpy.linalg as la
import neural

# Necessary for Overflow Error detection:
np.seterr(all='raise')


def euler_evolution(w0: np.ndarray, s: neural.NeuralState, alpha: float, beta: float, steps: int,
                    dt: float = 1, dynamic_dt: bool = False) -> list:
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
    :return: A list starting with w0 and ending with w_steps
    """
    ret = [w0]
    for i in range(1, steps+1):
        try:
            dw = delta_w(ret[i - 1], s, alpha, beta)
        except OverflowError:
            print("Overflow Error by step = "+str(i)+" for alpha = "+str(alpha)+" and dt = "+str(dt))
            break
        except FloatingPointError:
            print("Overflow Error by step = " + str(i) + " for alpha = " + str(alpha) + " and dt = " + str(dt))
            break
        dt = dt if not dynamic_dt else 0.1 / np.max(np.abs(dw))
        # print("dt="+str(dt)+"_dyn="+str(dynamic_dt))
        ret.append(ret[i-1] + dt * dw)
    return ret


def euler_evolution_moreinfo(w0: np.ndarray, s: neural.NeuralState, alpha: float, beta: float, steps: int,
                             dt: float = 1, dynamic_dt: bool = False) -> tuple:
    """
    """
    w_a = [w0]
    t_a = [0]
    dw_a = []
    
    for i in range(1, steps+1):
        try:
            dw = delta_w(w_a[i - 1], s, alpha, beta)
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
                           rotations: int, dt: float) -> tuple:
    """
    
    :param w0:
    :param s_set:
    :param alpha:
    :param beta:
    :param steps_per_pattern:
    :param rotations:
    :param dt:
    :return:
    """

    w_a = [w0]
    t_a = [0.0]
    dw_a = []
    neurons_a = []
    
    for rot in range(rotations):
        # rotation
        print("Rotation " + str(rot+1) + "/" + str(rotations))
        
        p_c = 0
        for pat in s_set:
            p_c += 1
            print("Pattern " + str(p_c) + "/" + str(len(s_set)))
            # pattern
            
            for i in range(steps_per_pattern):
                # learning
                try:
                    dw = delta_w(w_a[i - 1], pat, alpha, beta)
                except OverflowError:
                    print("Overflow Error by step = " + str(i) + " for alpha = " + str(alpha) + " and dt = " + str(dt))
                    break
                except FloatingPointError:
                    print("Overflow Error by step = " + str(i) + " for alpha = " + str(alpha) + " and dt = " + str(dt))
                    break
                w_a.append(w_a[i - 1] + dt * dw)
                t_a.append(t_a[-1] + dt)
                dw_a.append(dw)
            
                neurons_a.append(pat)

    return w_a, t_a, dw_a, neurons_a
    

def delta_w(w: np.ndarray, s: neural.NeuralState, alpha: float, beta: float) -> np.ndarray:
    """
    Calculates
    :param w: Weight matrix w(t).
    :param s: Neuron state vector S_μ.
    :param alpha: The free parameter Alpha.
    :param beta: Free parameter to weight second term.
    :return: 2D matrix dw/dt.
    """
    shp = np.shape(w)
    l = w.shape[0]
    
    ret = np.zeros(shp, dtype=float)
    
    d_mat = d_matrix(w)
    d_prime_mat = d_prime_matrix(d_mat, s)
    
    v_m = v(s)
    
    for i in range(l):
        for j in range(l):
            ret[i][j] = delta_w_ij(i, j, d_prime_mat, w, s, alpha, beta, v_m)
    return ret
    

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
    first_term = v_m[i][j] * alpha * (1 - s.active_neuron_count() * w[i][j])
    
    l = w.shape[0]
    a = [(w[i][j_prime] * d_prime_mat[i][j_prime]) for j_prime in range(l)]
    
    second_term = beta * w[i][j] * (d_prime_mat[i][j] - sum(a))
    
    ret = first_term + second_term
    return ret


def d_matrix(w: np.ndarray) -> np.ndarray:
    """
    Calculate matrix D, which solves equation c = D s.
    :param w: Given weight matrix w.
    :return: Matrix with identical shape. D = identity + w + w2 + w3
    """
    l = w.shape[0]
    one = np.identity(l)
    w2 = la.matrix_power(w, 2)
    w3 = la.matrix_power(w, 3)
    return one + w + w2 + w3


def d_prime_matrix(d_mat: np.ndarray, s: neural.NeuralState) -> np.ndarray:
    """
    Calculate matrix D'.
    :param d_mat: Matrix D (as calculated by d_matrix(w))
    :param s: Neural Vector (needed because sum only over active neurons)
    :return: Matrix D' with D'_ij = sum_k(D_ik * D_jk)
    """
    ret = np.zeros(d_mat.shape, dtype=float)
    l = d_mat.shape[0]
    for i in range(l):
        for j in range(l):
            a = [0 if (s.vec[k] == 0) else (d_mat[i][k] * d_mat[j][k]) for k in range(l)]
            ret[i][j] = sum(a)
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
    l = v_m.shape[0]
    for i in range(l):
        for j in range(l):
            v_m[i][j] = 0 if v_m[i][j] > s.max_dis else 1
    return v_m
