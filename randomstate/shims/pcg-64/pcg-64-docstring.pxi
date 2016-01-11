DEF CLASS_DOCSTRING = """
RandomState(seed=None)

Container for the PCG-64 pseudo random number generator.

PCG-64 is a 128-bit implementation of O'Neill's permutation congruential
generator ([1]_, [2]_). PCG-64 has a period of 2**128 and supports advancing
an arbitrary number of steps as well as 2**127 streams.

`pcg64.RandomState` exposes a number of methods for generating random
numbers drawn from a variety of probability distributions. In addition to the
distribution-specific arguments, each method takes a keyword argument
`size` that defaults to ``None``. If `size` is ``None``, then a single
value is generated and returned. If `size` is an integer, then a 1-D
array filled with generated values is returned. If `size` is a tuple,
then an array with that shape is filled and returned.

*No Compatibility Guarantee*
'pcg64.RandomState' does not make a guarantee that a fixed seed and a
fixed series of calls to 'pcg64.RandomState' methods using the same
parameters will always produce the same results. This is different from
'numpy.random.RandomState' guarantee. This is done to simplify improving
random number generators.  To ensure identical results, you must use the
same release version.

Parameters
----------
seed : {None, long}, optional
    Random seed initializing the pseudo-random number generator.
    Can be an integer in [0, 2**128] or ``None`` (the default).
    If `seed` is ``None``, then `xorshift1024.RandomState` will try to read data
    from ``/dev/urandom`` (or the Windows analogue) if available or seed from
    the clock otherwise.
inc : {None, int}, optional
    Stream to return.
    Can be an integer in [0, 2**128] or ``None`` (the default).  If `inc` is
    ``None``, then 1 is used.  Can be used with the same seed to
    produce multiple streams using other values of inc.

Notes
-----
Supports the method advance to advance the PRNG an arbitrary number of steps.
The state of the PCG-64 PRNG is represented by 2 128-bit integers.

See pcg32 for a similar implementation with a smaller period.

References
----------
.. [1] "PCG, A Family of Better Random Number Generators",
       http://www.pcg-random.org/
.. [2] O'Neill, Melissa E. "PCG: A Family of Simple Fast Space-Efficient
       Statistically Good Algorithms for Random Number Generation"
"""
