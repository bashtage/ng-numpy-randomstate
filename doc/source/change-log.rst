.. _change-log:

Change Log
==========

Head (not released)
-------------------
* Extended 32-bit generation to

  * Uniforms (:meth:`~randomstate.prng.mt19937.random_sample` and :meth:`~randomstate.prng.mt19937.rand`)
  * Normals (:meth:`~randomstate.prng.mt19937.standard_normal` and :meth:`~randomstate.prng.mt19937.randn`)
  * Standard Gammas (:meth:`~randomstate.prng.mt19937.standard_gamma`)
  * Standard Exponentials (:meth:`~randomstate.prng.mt19937.standard_exponential`)

  using the ``dtype`` keyword.
* Removed ``random_uintegers`` since these are special cases of
  :meth:`~randomstate.entropy.randint`

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
* Expose system entropy through :meth:`randomstate.entropy.random_entropy`
* Add vector initialization for all PRNGs

Version 1.10.1
--------------

* Added support for jumping the MRG32K3A generator
* Added support for jumping the dSFMT generator
* Update to recent changes in NumPy's RandomState

Version 1.10
------------

* This is the initial release with compatibility with NumPy 1.10