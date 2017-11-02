"""
randomstate contains implements NumPy-like RandomState objects with an
enhanced feature set.

This modules includes a number of alternative random number generators
in addition to the MT19937 that is included in NumPy. The RNGs include:

-  `MT19937 <https://github.com/numpy/numpy/blob/master/numpy/random/mtrand/>`__,
   the NumPy rng
-  `dSFMT <http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/SFMT/>`__ a
   SSE2-aware version of the MT19937 generator that is especially fast
   at generating doubles
-  `SFMT <http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/SFMT/>`__ a
   SSE2-aware version of the MT19937 generator that is optimized for
   integer values
-  `xorshift128+ <http://xorshift.di.unimi.it/>`__,
   `xoroshiro128+ <http://xoroshiro.di.unimi.it/>`__ and
   `xorshift1024\* <http://xorshift.di.unimi.it/>`__
-  `PCG32 <http://www.pcg-random.org/>`__ and
   `PCG64 <http:w//www.pcg-random.org/>`__
-  `MRG32K3A <http://simul.iro.umontreal.ca/rng>`__
-  A multiplicative lagged fibonacci generator (LFG(63, 1279, 861, \*))

Help is available at `on github <https://bashtage.github.io/ng-numpy-randomstate>`_.

New Features
------------

-  ``standard_normal``, ``normal``, ``randn`` and
   ``multivariate_normal`` all support an additional ``method`` keyword
   argument which can be ``bm`` or ``zig`` where ``bm`` corresponds to
   the current method using the Box-Muller transformation and ``zig``
   uses the much faster (100%+) Ziggurat method.
-  ``standard_exponential`` and ``standard_gamma`` both support an
   additional ``method`` keyword argument which can be ``inv`` or
   ``zig`` where ``inv`` corresponds to the current method using the
   inverse CDF and ``zig`` uses the much faster (100%+) Ziggurat method.
-  Core random number generators can produce either single precision
   (``np.float32``) or double precision (``np.float64``, the default)
   using an the optional keyword argument ``dtype``
-  Core random number generators can fill existing arrays using the
   ``out`` keyword argument

New Functions
-------------

-  ``random_entropy`` - Read from the system entropy provider, which is
   commonly used in cryptographic applications
-  ``random_raw`` - Direct access to the values produced by the
   underlying PRNG. The range of the values returned depends on the
   specifics of the PRNG implementation.
-  ``random_uintegers`` - unsigned integers, either 32-
   (``[0, 2**32-1]``) or 64-bit (``[0, 2**64-1]``)
-  ``jump`` - Jumps RNGs that support it. ``jump`` moves the state a
   great distance. *Only available if supported by the RNG.*
-  ``advance`` - Advanced the core RNG 'as-if' a number of draws were
   made, without actually drawing the numbers. *Only available if
   supported by the RNG.*
"""
from __future__ import division, absolute_import, print_function

from randomstate.prng.mt19937 import *
from randomstate.entropy import random_entropy
import randomstate.prng

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
