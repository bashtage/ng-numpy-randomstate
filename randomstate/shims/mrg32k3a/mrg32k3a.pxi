DEF RNG_ADVANCEABLE = 0
DEF RNG_SEED = 1
DEF RNG_NAME = 'mrg32k3a'
DEF RNG_STATE_LEN = 4
DEF RNG_JUMPABLE = 0
DEF NORMAL_METHOD = 'zig'

cdef extern from "distributions.h":

    cdef struct s_mrg32k3a_state:
        int64_t s10
        int64_t s11
        int64_t s12
        int64_t s20
        int64_t s21
        int64_t s22

    ctypedef s_mrg32k3a_state mrg32k3a_state

    cdef struct s_aug_state:
        mrg32k3a_state *rng
        binomial_t *binomial

        int has_gauss, shift_zig_random_int, has_uint32
        double gauss
        uint32_t uinteger
        uint64_t zig_random_int

    ctypedef s_aug_state aug_state

    cdef void set_seed(aug_state* state, uint64_t seed)

ctypedef mrg32k3a_state rng_t

ctypedef uint64_t rng_state_t

cdef object _get_state(aug_state state):
    return (state.rng.s10, state.rng.s11, state.rng.s12,
            state.rng.s20, state.rng.s21, state.rng.s22)

cdef object _set_state(aug_state *state, object state_info):
    state.rng.s10 = state_info[0]
    state.rng.s11 = state_info[1]
    state.rng.s12 = state_info[2]
    state.rng.s20 = state_info[3]
    state.rng.s21 = state_info[4]
    state.rng.s22 = state_info[5]

DEF CLASS_DOCSTRING = """
RandomState(seed=None)

Container for L'Ecuyer MRG32K3A pseudo random number generator.

MRG32K3A is a 32-bit implementation of L'Ecuyer's combined multiple
recursive generator ([1]_, [2]_). MRG32K3A has a period of 2**191 and
supports multiple streams (NOT IMPLEMENTED YET).

`mrg32k3a.RandomState` exposes a number of methods for generating random
numbers drawn from a variety of probability distributions. In addition to the
distribution-specific arguments, each method takes a keyword argument
`size` that defaults to ``None``. If `size` is ``None``, then a single
value is generated and returned. If `size` is an integer, then a 1-D
array filled with generated values is returned. If `size` is a tuple,
then an array with that shape is filled and returned.

*No Compatibility Guarantee*
'mrg32k3a.RandomState' does not make a guarantee that a fixed seed and a
fixed series of calls to 'mrg32k3a.RandomState' methods using the same
parameters will always produce the same results. This is different from
'numpy.random.RandomState' guarantee. This is done to simplify improving
random number generators.  To ensure identical results, you must use the
same release version.

Parameters
----------
seed : {None, int}, optional
    Random seed initializing the pseudo-random number generator.
    Can be an integer in [0, 2**64] or ``None`` (the default).
    If `seed` is ``None``, then `mrg32k3a.RandomState` will try to read data
    from ``/dev/urandom`` (or the Windows analogue) if available or seed from
    the clock otherwise.

Notes
-----
The state of the MRG32KA PRNG is represented by 6 64-bit integers.

This implementation is integer based and produces integers in the interval
[0, 2**32-209+1].  These are treated as if they 32-bit random integers.

References
----------
.. [1] "Software developed by the Canada Research Chair in Stochastic
       Simulation and Optimization", http://simul.iro.umontreal.ca/
.. [2] Pierre L'Ecuyer, (1999) "Good Parameters and Implementations for
       Combined Multiple Recursive Random Number Generators.", Operations
       Research 47(1):159-164
"""
