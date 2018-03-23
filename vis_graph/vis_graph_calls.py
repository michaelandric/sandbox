import numpy as np
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

    x = list_input_series
    N = len(x)
    t = list(range(N))
    A = np.zeros((N, N))
    for i in range(N-2):
        for j in range(i+2, N):
            k = i + 1

            test = (x[j] - x[i]) / (t[j] - t[i])

            while (x[k] - x[i]) / (t[k] - t[i]) < test and k < j:
                k += 1

            if k == j:
                A[i, j] = A[j, i] = 1

    # Add trivial connections of subsequent observations in time series
    for i in range(N-1):
        A[i, i+1] = A[i+1, i] = 1
    
    return A