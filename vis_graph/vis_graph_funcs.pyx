# cython: profile=True
# cython: linetrace=True
# filename: vis_graph_funcs.pyx
import numpy as np
cimport numpy as np

def _natural_vis_graph(np.ndarray[np.float64_t, ndim=1] list_input_series,
                       int N, np.ndarray[np.int64_t, ndim=1] t,
                       np.ndarray[np.int64_t, ndim=2] A):

    cdef:
        int i, j, k
        float test
 
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

    return np.asarray(A)


def _l1_dist(np.ndarray[np.int64_t, ndim=1] vec_a, np.ndarray[np.int64_t, ndim=1] vec_b):
    return np.sum(np.abs(vec_a - vec_b))


def _vis_graph_degree_l1_dist(np.ndarray[np.float64_t, ndim=1] list_input_series):
    cdef int N = len(list_input_series)
    cdef np.ndarray[np.int64_t, ndim=1] t = np.arange(N, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim=2] A_top = np.zeros([N, N], dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim=2] A_bot = np.zeros([N, N], dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim=2] adj1_top = np.zeros([N, N], dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim=2] adj1_bot = np.zeros([N, N], dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim=1] adj1_top_deg
    cdef np.ndarray[np.int64_t, ndim=1] adj1_bot_deg

    adj1_top = _natural_vis_graph(list_input_series, N, t, A_top)
    adj1_top_deg = np.sum(adj1_top, 0)

    adj1_bot = _natural_vis_graph(-1*list_input_series, N, t, A_bot)
    adj1_bot_deg = np.sum(adj1_bot, 0)

    cdef int dist
    dist = _l1_dist(adj1_top_deg, adj1_bot_deg)
    return dist