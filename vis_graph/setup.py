from distutils.core import setup
from distutils.extension import Extension
import numpy as np
import sys


if '--use-cython' in sys.argv:
    USE_CYTHON = True
    sys.argv.remove('--use-cython')
else:
    USE_CYTHON = False
ext = '.pyx' if USE_CYTHON else '.cpp'
extensions = [Extension('vis_graph_funcs',
                        ['vis_graph_funcs{}'.format(ext)],
                        language='c++',
                        include_dirs=[np.get_include()])]

if USE_CYTHON:
    from Cython.Build import cythonize
    extensions = cythonize(extensions, gdb_debug=True)

setup(
    ext_modules = extensions
)
