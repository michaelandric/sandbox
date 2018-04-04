#!/usr/bin/env python
# encoding: utf-8
# filename: profile.py

import pstats, cProfile
import numpy as np
import pyximport

pyximport.install(setup_args={"include_dirs":np.get_include()},
                  reload_support=True)

import .test_vis_graph_funcs as tvg

cProfile.runctx("tvg.VgTests().test_types_vis_graph_degree_l1_dist",
 globals(), locals(), "Profile.prof")

s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()