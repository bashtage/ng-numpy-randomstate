DEF RS_RNG_NAME = 'mlfg-1279-861'
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

        int has_gauss, shift_zig_random_int, has_uint32
        double gauss
        uint64_t zig_random_int
        uint32_t uinteger

    ctypedef s_aug_state aug_state

    cdef void set_seed(aug_state* state, uint64_t seed)

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

DEF CLASS_DOCSTRING = """
RandomState(seed=None)

Container for a Multiplicative Lagged Fibonacci Generator (MLFG).

LFG(1279, 861, \*) is a 64-bit implementation of a MLFG that uses lags 1279 and
861 where random numbers are determined by

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
seed : {None, int}, optional
    Random seed initializing the pseudo-random number generator.
    Can be an integer in [0, 2**64] or ``None`` (the default).
    If ``seed`` is ``None``, then ``mlfg_1279_861.RandomState`` will try to
    read entropy from ``/dev/urandom`` (or the Windows analogue) if available
    to produce a 64-bit seed. If unavailable, a 64-bit hash of the time
    and process ID is used.

Notes
-----
The state of the MLFG(1279,861,*) PRNG is represented by 1279 64-bit unsigned
integers as well as a two 32-bit integers representing the location in the
state array of the current and lagged values.
"""
