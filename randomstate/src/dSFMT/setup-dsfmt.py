import os
import sys
from distutils.core import setup
from distutils.extension import Extension
from os import getcwd, name
from os.path import join

import numpy
from Cython.Build import cythonize

pwd = getcwd()

sources = [join(pwd, 'dSFMT_wrapper.pyx'),
           join(pwd, 'dSFMT.c')]


defs = [('HAVE_SSE2', '1'),('DSFMT_MEXP','19937')]

include_dirs = [pwd] + [numpy.get_include()]

extra_link_args = ['Advapi32.lib'] if name == 'nt' else []
extra_compile_args = [] if name == 'nt' else ['-std=c99']

setup(ext_modules=cythonize([Extension("dSFMT_wrapper",
                                       sources=sources,
                                       include_dirs=include_dirs,
                                       define_macros=defs,
                                       extra_compile_args=extra_compile_args,
                                       extra_link_args=extra_link_args)
                             ])
      )
