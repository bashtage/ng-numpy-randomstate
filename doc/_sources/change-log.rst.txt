.. _change-log:

Change Log
==========

Version 1.13.2
--------------
* Add Ziggurat generation for standard gamma
  (:meth:`~randomstate.prng.mt19937.standard_gamma`) for both floats and
  doubles.  The gamma generator uses a rejection sampler that
  depends on random double, normal and/or exponential values.

.. ipython:: python

   import numpy as np
   import randomstate as rs
   rs.seed(23456)
   rs.standard_gamma(2, size=3, method='zig') # New method
   rs.standard_gamma(2, size=3, method='inv') # Old method

* Add Ziggurat generator for standard exponential
  (:meth:`~randomstate.prng.mt19937.standard_exponential`) for both floats and
  doubles

.. ipython:: python

   import numpy as np
   import randomstate as rs
   rs.seed(23456)
   rs.standard_exponential(3, method='zig') # New method
   rs.standard_exponential(3, method='inv') # Old method

* Add SIMD-oriented Fast Mersenne Twister
  (`SFMT <http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/SFMT/>`_) generator.
* Add complex normal (:meth:`~randomstate.prng.mt19937.complex_normal`)
* Added support for jumping the MT19937 generator
* Added support for jumping the SFMT generator

Version 1.13
------------
* Synchronize with NumPy 1.13 release

Version 1.12
------------
* Synchronize with NumPy 1.12 release
* Add ``out`` argument which allows filling existing arrays. This feature was
  added to facilitate multithreaded filling of arrays using parallel random
  number generators.

  * Uniforms (:meth:`~randomstate.prng.mt19937.random_sample`)
  * Normals (:meth:`~randomstate.prng.mt19937.standard_normal`)
  * Standard Exponentials (:meth:`~randomstate.prng.mt19937.standard_exponential`)
  * Standard Gammas (:meth:`~randomstate.prng.mt19937.standard_gamma`)

Version 1.11.4
--------------
* Fix for error in Ziggurat implementation of Normal

Version 1.11.3
--------------
* Extended 32-bit generation to

  * Uniforms (:meth:`~randomstate.prng.mt19937.random_sample` and
    :meth:`~randomstate.prng.mt19937.rand`)
  * Normals (:meth:`~randomstate.prng.mt19937.standard_normal` and
    :meth:`~randomstate.prng.mt19937.randn`)
  * Standard Gammas (:meth:`~randomstate.prng.mt19937.standard_gamma`)
  * Standard Exponentials (:meth:`~randomstate.prng.mt19937.standard_exponential`)

  using the ``dtype`` keyword.
* Removed ``random_uintegers`` since these are special cases of ``randint``
* Release to include files required for install from PyPi

Version 1.11.2
--------------
* Added keyword argument `dtype` to `random_sample` which allows for single
  precision as well as double precision uniforms to be generated.

.. ipython:: python

   import numpy as np
   import randomstate as rs
   rs.seed(23456)
   rs.random_sample(3, dtype=np.float64)
   rs.seed(23456)
   rs.random_sample(3, dtype=np.float32)


Version 1.11.1
--------------

* Added xoroshiro128+ PRNG.  This is an improved version of the xorshirt128+
  PRNG and should be used instead.  In the long run, xorshift128+ will likely
  be removed.
* Fixed DeprecationWarning when initializing a PRNG using a single element
  array.

Version 1.11
------------
* Update to recent changes in NumPy's RandomState
* Expose system entropy through :func:`randomstate.entropy.random_entropy`
* Add vector initialization for all PRNGs

Version 1.10.1
--------------
* Added support for jumping the MRG32K3A generator
* Added support for jumping the dSFMT generator
* Update to recent changes in NumPy's RandomState

Version 1.10
------------
* This is the initial release with compatibility with NumPy 1.10