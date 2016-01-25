DEF RS_RNG_NAME = "xorshift-1024*"
DEF RS_RNG_JUMPABLE = 1

cdef extern from "distributions.h":

    cdef struct s_xorshift1024_state:
      uint64_t s[16]
      int p

    ctypedef s_xorshift1024_state xorshift1024_state

    cdef struct s_aug_state:
        xorshift1024_state *rng
        binomial_t *binomial

        int has_gauss, shift_zig_random_int, has_uint32
        double gauss
        uint64_t zig_random_int
        uint32_t uinteger

    ctypedef s_aug_state aug_state

    cdef void set_seed(aug_state* state, uint64_t seed)

    cdef void jump_state(aug_state* state)

    cdef void init_state(aug_state* state, uint64_t* state_vals)

ctypedef object rng_state_t

ctypedef xorshift1024_state rng_t

cdef object _get_state(aug_state state):
    cdef uint64_t [:] key = np.zeros(16, dtype=np.uint64)
    for i in range(16):
        key[i] = state.rng.s[i]
    return (np.asanyarray(key), state.rng.p)

cdef object _set_state(aug_state *state, object state_info):
    cdef uint64_t [:] key = state_info[0]
    cdef Py_ssize_t i
    for i in range(16):
        state.rng.s[i] = key[i]
    state.rng.p = state_info[1]


DEF CLASS_DOCSTRING = """
RandomState(seed=None)

Container for the xorshift1024* pseudo random number generator.

xorshift1024* is a 64-bit implementation of Saito and Matsumoto's XSadd
generator [1]_. xorshift1024* has a period of :math:`2^{1024} - 1` and
supports jumping the sequence in increments of :math:`2^{512}`, which allow multiple
non-overlapping sequences to be generated.

``xorshift1024.RandomState`` exposes a number of methods for generating random
numbers drawn from a variety of probability distributions. In addition to the
distribution-specific arguments, each method takes a keyword argument
`size` that defaults to ``None``. If `size` is ``None``, then a single
value is generated and returned. If `size` is an integer, then a 1-D
array filled with generated values is returned. If `size` is a tuple,
then an array with that shape is filled and returned.

**No Compatibility Guarantee**

``xorshift1024.RandomState`` does not make a guarantee that a fixed seed and a
fixed series of calls to ``xorshift1024.RandomState`` methods using the same
parameters will always produce the same results. This is different from
``numpy.random.RandomState`` guarantee. This is done to simplify improving
random number generators.  To ensure identical results, you must use the
same release version.

Parameters
----------
seed : {None, int}, optional
    Random seed initializing the pseudo-random number generator.
    Can be an integer in [0, 2**64] or ``None`` (the default).
    If `seed` is ``None``, then ``xorshift1024.RandomState`` will try to read
    data from ``/dev/urandom`` (or the Windows analogue) if available.  If
    unavailable, a 64-bit hash of the time and process ID is used.

Notes
-----
See xorshift128 for a faster implementation that has a smaller period.

**Parallel Features**

``xorshift1024.RandomState`` can be used in parallel applications by
calling the method ``jump`` which advances the
the state as-if :math:`2^{512}` random numbers have been generated. This
allow the original sequence to be split so that distinct segments can be used
on each worker process. All generators should be initialized with the same
seed to ensure that the segments come from the same sequence.

>>> import randomstate.prng.xorshift1024 as rnd
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
