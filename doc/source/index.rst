randomstate's documentation
===========================
This package contains drop-in replacements for the NumPy RandomState object
that change the core random number generator.

What's New or Different
-----------------------
* ``randomstate.entropy.random_entropy`` provides access to the system
  source of randomness that is used in cryptographic applications (e.g.,
  ``/dev/urandom`` on Unix).
* The normal generator supports a 256-step Ziggurat method which is 2-6 times
  faster than NumPy's ``standard_normal``.  This generator can be accessed
  by passing the keyword argument ``method='zig'``.

.. ipython:: python

  from randomstate import standard_normal
  standard_normal(100000, method='zig')

* Optional ``dtype`` argument that accepts ``np.float32`` or ``np.float64``
  to produce either single or double prevision uniform random variables for
  select distributions

  * Uniforms (:meth:`~randomstate.prng.mt19937.random_sample` and :meth:`~randomstate.prng.mt19937.rand`)
  * Normals (:meth:`~randomstate.prng.mt19937.standard_normal` and :meth:`~randomstate.prng.mt19937.randn`)
  * Standard Gammas (:meth:`~randomstate.prng.mt19937.standard_gamma`)
  * Standard Exponentials (:meth:`~randomstate.prng.mt19937.standard_exponential`)

.. ipython:: python

  import numpy as np
  import randomstate as rs
  rs.seed(0)
  rs.random_sample(3, dtype=np.float64)
  rs.seed(0)
  rs.random_sample(3, dtype=np.float32)



* For changes since the previous release, see the :ref:`change-log`

Parallel Generation
-------------------

The included generators can be used in parallel, distributed applications in
one of two ways:

* :ref:`independent-streams`
* :ref:`jump-and-advance`

Supported Generators
--------------------
The main innovation is the inclusion of a number of alternative pseudo-random number
generators, 'in addition' to the standard PRNG in NumPy.  The included PRNGs are:

* MT19937 - The standard NumPy generator.  Produces identical results to NumPy
  using the same seed/state. See `NumPy's documentation`_.
* dSFMT - A SSE2 enables version of the MT19937 generator.  Theoretically the
  same, but with a different state and so it is not possible to produce a
  sequence identical to MT19937. See the `dSFMT authors' page`_.
* XoroShiro128+ - Improved version of XorShift128+ with improved performance
  and better statistical quality. Like the XorShift generators, can be jumped
  to produce multiple streams in parallel applications. See
  :meth:`randomstate.prng.xoroshiro128plus.jump` for details. More information
  about this PRNG is available at the `xorshift and xoroshiro authors' page`_.
* XorShit128+ and XorShift1024* - Vast fast generators based on the XSadd
  generator. These generators can be rapidly 'jumped' and so can be used in
  parallel applications. See the documentation for
  :meth:`randomstate.prng.xorshift1024.jump` for details. More information
  about these PRNGs is available at the
  `xorshift and xoroshiro authors' page`_.
* PCG-32 and PCG-64 - Fast generators that support many parallel streams and
  can be advanced by an arbitrary amount. See the documentation for
  :meth:`randomstate.prng.pcg64.advance`.  PCG-32 only as a period of
  :math:`2^{64}` while PCG-64 has a period of :math:`2^{128}`. See the
  `PCG author's page`_ for more details about this class of PRNG.
* Multiplicative Lagged Fibonacci Generator MLFG(1279, 861, \*) - A directly
  implemented multiplicative lagged Fibonacci generator with a very large
  period and good performance. Future plans include multiple stream support.
  See the `wiki page on Fibonacci generators`_.
* MRG32K3A - a classic and popular generator by L'Ecuyer. Future plans
  include multiple stream support. See the `MRG32K3A author's page`_. Lower
  performance than more modern generators.

.. _`NumPy's documentation`: http://docs.scipy.org/doc/numpy/reference/routines.random.html
.. _`dSFMT authors' page`: http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/SFMT/
.. _`xorshift and xoroshiro authors' page`:  http://xoroshiro.di.unimi.it/
.. _`PCG author's page`: http://www.pcg-random.org/
.. _`wiki page on Fibonacci generators`: https://en.wikipedia.org/wiki/Lagged_Fibonacci_generator
.. _`MRG32K3A author's page`: http://simul.iro.umontreal.ca/

New Features
~~~~~~~~~~~~
.. toctree::
   :maxdepth: 2

   Using in Parallel Applications <parallel>
   Reading System Entropy <entropy>


Individual Pseudo Random Number Generators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 2

   MT19937 <mt19937>
   dSFMT <dsfmt>
   XorShift128+ <xorshift128>
   XoroShiro128+ <xoroshiro128plus>
   XorShift1024* <xorshift1024>
   PCG-32 <pcg32>
   PCG-64 <pcg64>
   MLFG <mlfg>
   MRG32K3A <mrg32k3a>

Changes
~~~~~~~
.. toctree::
   :maxdepth: 2

   Change Log <change-log>

Indices and tables
~~~~~~~~~~~~~~~~~~

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

