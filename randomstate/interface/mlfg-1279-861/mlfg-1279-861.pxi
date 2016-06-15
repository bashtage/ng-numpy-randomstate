DEF RS_RNG_NAME = u'mlfg-1279-861'
DEF MLFG_STATE_LEN = 1279

cdef extern from "distributions.h":
    cdef struct s_mlfg_state:
        uint64_t lags[1279]
        int pos
        int lag_pos

    ctypedef s_mlfg_state mlfg_state

    cdef struct s_aug_state:
        mlfg_state *rng
        binomial_t *binomial

        int has_gauss, shift_zig_random_int, has_uint32, has_gauss_float
        float gauss_float
        double gauss
        uint64_t zig_random_int
        uint32_t uinteger

    ctypedef s_aug_state aug_state

    cdef void set_seed(aug_state* state, uint64_t seed)

    cdef void set_seed_by_array(aug_state* state, uint64_t *seed, int count)

ctypedef uint64_t rng_state_t

ctypedef mlfg_state rng_t

cdef object _get_state(aug_state state):
    cdef uint64_t [:] key = np.zeros(MLFG_STATE_LEN, dtype=np.uint64)
    cdef Py_ssize_t i
    for i in range(MLFG_STATE_LEN):
        key[i] = state.rng.lags[i]
    return (np.asanyarray(key), state.rng.pos, state.rng.lag_pos)

cdef object _set_state(aug_state *state, object state_info):
    cdef uint64_t [:] key = state_info[0]
    cdef Py_ssize_t i
    for i in range(MLFG_STATE_LEN):
        state.rng.lags[i] = key[i]
    state.rng.pos = state_info[1]
    state.rng.lag_pos = state_info[2]

DEF CLASS_DOCSTRING = u"""
RandomState(seed=None)

Container for a Multiplicative Lagged Fibonacci Generator (MLFG).

LFG(1279, 861, \*) is a 64-bit implementation of an MLFG that uses lags 1279
and 861 where random numbers are determined by

.. math::

   x_n = (x_{n-k} * x_{n-l}) \mathrm{Mod} 2^M

where k=861, l=1279 and M=64. The period of the generator is
2**1340 - 2**61.  Even though the PRNG uses 64 bits, only 63 are random
since all numbers in x must be odd.

``mlfg_1279_861.RandomState`` exposes a number of methods for generating random
numbers drawn from a variety of probability distributions. In addition to the
distribution-specific arguments, each method takes a keyword argument
``size`` that defaults to ``None``. If ``size`` is ``None``, then a single
value is generated and returned. If `size` is an integer, then a 1-D
array filled with generated values is returned. If ``size`` is a tuple,
then an array with that shape is filled and returned.

**No Compatibility Guarantee**

``mlfg_1279_861.RandomState`` does not make a guarantee that a fixed seed and a
fixed series of calls to ``mlfg_1279_861.RandomState`` methods using the same
parameters will always produce the same results. This is different from
``numpy.random.RandomState`` guarantee. This is done to simplify improving
random number generators.  To ensure identical results, you must use the
same release version.

Parameters
----------
seed : {None, int, array_like}, optional
    Random seed initializing the pseudo-random number generator.
    Can be an integer in [0, 2**64-1], array of integers in
    [0, 2**64-1] or ``None`` (the default). If `seed` is ``None``,
    then ``mlfg_1279_861.RandomState`` will try to read entropy from
    ``/dev/urandom`` (or the Windows analog) if available to
    produce a 64-bit seed. If unavailable, a 64-bit hash of the time
    and process ID is used.

Notes
-----
The state of the MLFG(1279,861,*) PRNG is represented by 1279 64-bit unsigned
integers as well as a two 32-bit integers representing the location in the
state array of the current and lagged values.

**State and Seeding**

The ``mlfg_1279_861.RandomState`` state vector consists of a 1279 element array
of 64-bit unsigned integers plus a two integers value between 0 and 1278
indicating  the current position and the position of the lagged value within
the main array required to produce the next random. All 1279 elements of the
state array must be odd.

``mlfg_1279_861.RandomState`` is seeded using either a single 64-bit unsigned integer
or a vector of 64-bit unsigned integers.  In either case, the input seed is
used as an input (or inputs) for another simple random number generator,
Splitmix64, and the output of this PRNG function is used as the initial state.
Using a single 64-bit value for the seed can only initialize a small range of
the possible initial state values.  When using an array, the SplitMix64 state
for producing the ith component of the initial state is XORd with the ith
value of the seed array until the seed array is exhausted. When using an array
the initial state for the SplitMix64 state is 0 so that using a single element
array and using the same value as a scalar will produce the same initial state.
"""
