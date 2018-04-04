# cython: profile=True
# cython: linetrace=True
import unittest
import numpy as np
cimport numpy as np
from ..vis_graph_funcs import _vis_graph_degree_l1_dist


class VgTests(unittest.TestCase):
    """
    Tests for type errors.
    """

    def test_types_vis_graph_degree_l1_dist(self):
        list_input_series = np.random.rand(100)
        self.assertTrue(list_input_series.dtype == 'float64')
        #self.assertTrue(_vis_graph_degree_l1_dist(list_input_series))

    def test_vis_graph_degree_l1_dist(np.ndarray[np.float64_t, ndim=1] list_input_series):
        _vis_graph_degree_l1_dist(list_input_series)

if __name__ == '__main__':
    unittest.main()