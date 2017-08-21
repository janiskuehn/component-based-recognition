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
    d_prime_mat = d_prime_matrix(d_mat)
    for i in range(l):
        for j in range(l):
            ret[i][j] = delta_w_ij(i, j, d_prime_mat, w, s, alpha, beta)
    return ret
    

def delta_w_ij(i: int, j: int, d_prime_mat: np.ndarray, w: np.ndarray, s: neural.NeuralState,
               alpha: float, beta: float) -> float:
    """
    Calculate Element i j of dw/dt.
    :param i: Index i.
    :param j: Index j.
    :param d_prime_mat: D_prime previously calculated by d_prime_matrix(d).
    :param w: Weight matrix w(t).
    :param s: Neuron state vector S_μ.
    :param alpha: The free parameter Alpha.
    :param beta: Free parameter to weight second term.
    :return: Float Element i j of dw/dt
    """
    if i == j:
        return 0
    first_term = alpha * (1 - s.active_neuron_count() * w[i][j])

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


def d_prime_matrix(d_mat: np.ndarray) -> np.ndarray:
    """
    Calculate matrix D'.
    :param d_mat: Matrix D (as calculated by d_matrix(w))
    :return: Matrix D' with D'_ij = sum_k(D_ik * D_jk)
    """
    ret = np.zeros(d_mat.shape, dtype=float)
    l = d_mat.shape[0]
    for i in range(l):
        for j in range(l):
            a = [(d_mat[i][k] * d_mat[j][k]) for k in range(l)]
            ret[i][j] = sum(a)
    return ret
