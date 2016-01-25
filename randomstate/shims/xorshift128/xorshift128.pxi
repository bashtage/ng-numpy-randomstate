DEF RS_RNG_NAME = "xorshift-128+"
DEF RS_RNG_JUMPABLE = 1

cdef extern from "distributions.h":

    cdef struct s_xorshift128_state:
        uint64_t s[2]

    ctypedef s_xorshift128_state xorshift128_state

    cdef struct s_aug_state:
        xorshift128_state *rng
        binomial_t *binomial

        int has_gauss, shift_zig_random_int, has_uint32
        double gauss
        uint64_t zig_random_int
        uint32_t uinteger

    ctypedef s_aug_state aug_state

    cdef void set_seed(aug_state* state, uint64_t seed)

    cdef void jump_state(aug_state* state)

    cdef void init_state(aug_state* state, uint64_t* state_vals)


ctypedef uint64_t rng_state_t

ctypedef xorshift128_state rng_t

cdef inline object _get_state(aug_state state):
    return (state.rng.s[0], state.rng.s[1])

cdef inline object _set_state(aug_state *state, object state_info):
    state.rng.s[0] = state_info[0]
    state.rng.s[1] = state_info[1]



DEF CLASS_DOCSTRING = """
RandomState(seed=None)

Container for the xorshift128+ pseudo random number generator.

xorshift128+ is a 64-bit implementation of Saito and Matsumoto's XSadd
generator [1]_. xorshift128+ has a period of :math:`2^{128} - 1` and
supports jumping the sequence in increments of :math:`2^{64}`, which allow multiple
non-overlapping sequences to be generated.

``xorshift128.RandomState`` exposes a number of methods for generating random
numbers drawn from a variety of probability distributions. In addition to the
distribution-specific arguments, each method takes a keyword argument
`size` that defaults to ``None``. If `size` is ``None``, then a single
value is generated and returned. If `size` is an integer, then a 1-D
array filled with generated values is returned. If `size` is a tuple,
then an array with that shape is filled and returned.

**No Compatibility Guarantee**

``xorshift128.RandomState`` does not make a guarantee that a fixed seed and a
fixed series of calls to ``xorshift128.RandomState`` methods using the same
parameters will always produce the same results. This is different from
``numpy.random.RandomState`` guarantee. This is done to simplify improving
random number generators.  To ensure identical results, you must use the
same release version.

Parameters
----------
seed : {None, int}, optional
    Random seed initializing the pseudo-random number generator.
    Can be an integer in [0, 2**64] or ``None`` (the default).
    If `seed` is ``None``, then ``xorshift128.RandomState`` will try to read
    data from ``/dev/urandom`` (or the Windows analogue) if available.  If
    unavailable, a 64-bit hash of the time and process ID is used.

Notes
-----
See xorshift1024 for an implementation with a larger period
(:math:`2^{1024} - 1`) and jump size (:math:`2^{512} - 1`).

**Parallel Features**

``xorshift128.RandomState`` can be used in parallel applications by
calling the method ``jump`` which advances the
the state as-if :math:`2^{64}` random numbers have been generated. This
allow the original sequence to be split so that distinct segments can be used
on each worker process. All generators should be initialized with the same
seed to ensure that the segments come from the same sequence.

>>> import randomstate.prng.xorshift128 as rnd
>>> rs = [rnd.RandomState(1234) for _ in range(10)]
# Advance rs[i] by i jumps
>>> for i in range(10):
        rs[i].jump(i)

References
----------
.. [1] "xorshift*/xorshift+ generators and the PRNG shootout",
       http://xorshift.di.unimi.it/
.. [2] Marsaglia, George. "Xorshift RNGs." Journal of Statistical Software
       [Online], 8.14, pp. 1 - 6, .2003.
.. [3] Sebastiano Vigna. "An experimental exploration of Marsaglia's xorshift
       generators, scrambled." CoRR, abs/1402.6246, 2014.
.. [4] Sebastiano Vigna. "Further scramblings of Marsaglia's xorshift
       generators." CoRR, abs/1403.0930, 2014.
"""
