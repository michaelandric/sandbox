import numpy as np
from .vis_graph_funcs import _natural_vis_graph
from .vis_graph_funcs import _vis_graph_degree_l1_dist


try:
    from numba import jit
except ImportError as e:
    print(e, '\nYou should install numba. It will make this run faster.')


@jit
def nvg(list_input_series):
    """
    Construct the natural visibility graph.
    See here:
    http://www.pnas.org/content/105/13/4972.full

    Takes a list of numbers or numpy array as input. 
    Returns N x N adjacency matrix with "1" marking
    an endge between time points that "see" each other.

    list_input_series:
        list of numbers or numpy array
    returns:
        N x N adjacency matrix
    """

    N = len(list_input_series)
    t = list(range(N))
    A = np.zeros((N, N))
    for i in range(N-2):
        for j in range(i+2, N):
            k = i + 1

            test = (list_input_series[j] - list_input_series[i]) / (t[j] - t[i])

            while (list_input_series[k] - list_input_series[i]) / (t[k] - t[i]) < test and k < j:
                k += 1

            if k == j:
                A[i, j] = A[j, i] = 1

    # Add trivial connections of subsequent observations in time series
    for i in range(N-1):
        A[i, i+1] = A[i+1, i] = 1
    
    return A


def natural_vis_graph(list_input_series):
    N = len(list_input_series)
    t = np.arange(N)
    A = np.zeros((N, N), dtype=np.int64)

    return _natural_vis_graph(list_input_series, N, t, A)


def l1_dist(vec_a, vec_b):
    return np.sum(np.abs(vec_a - vec_b))


def get_visgraph_degree_l1_dist(list_input_series):
    adj1_top = nvg(list_input_series)
    bot_series = -1*list_input_series
    adj1_bot = nvg(bot_series)

    adj1_top = np.sum(adj1_top, 0) 
    adj1_bot = np.sum(adj1_bot, 0)
    return l1_dist(adj1_top, adj1_bot)

def get_vis_graph_degree_l1_dist(list_input_series):
    return _vis_graph_degree_l1_dist(list_input_series)


def hvg(list_input_series):
    """
    Construct the horizontal visibility graph.
    See here:
    http://www.pnas.org/content/105/13/4972.full

    Takes a list of numbers or numpy array as input. 
    Returns N x N adjacency matrix with "1" marking
    an endge between time points that "see" each other.

    list_input_series:
        list of numbers or numpy array
    returns:
        N x N adjacency matrix
    """