DEF RS_RNG_NAME = u"xorshift-1024*"
DEF RS_RNG_JUMPABLE = 1

cdef extern from "distributions.h":

    cdef struct s_xorshift1024_state:
      uint64_t s[16]
      int p

    ctypedef s_xorshift1024_state xorshift1024_state

    cdef struct s_aug_state:
        xorshift1024_state *rng
        binomial_t *binomial

        int has_gauss, shift_zig_random_int, has_uint32, has_gauss_float
        float gauss_float
        double gauss
        uint64_t zig_random_int
        uint32_t uinteger

    ctypedef s_aug_state aug_state

    cdef void set_seed(aug_state* state, uint64_t seed)

    cdef void set_seed_by_array(aug_state* state, uint64_t *seed, int count)

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


DEF CLASS_DOCSTRING = u"""
RandomState(seed=None)

Container for the xorshift1024* pseudo-random number generator.

xorshift1024* is a 64-bit implementation of Saito and Matsumoto's XSadd
generator [1]_. xorshift1024* has a period of :math:`2^{1024} - 1` and
supports jumping the sequence in increments of :math:`2^{512}`, which allows
multiple non-overlapping sequences to be generated.

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
seed : {None, int, array_like}, optional
    Random seed initializing the pseudo-random number generator.
    Can be an integer in [0, 2**64-1], array of integers in
    [0, 2**64-1] or ``None`` (the default). If `seed` is ``None``,
    then ``xorshift1024.RandomState`` will try to read data from
    ``/dev/urandom`` (or the Windows analog) if available.  If
    unavailable, a 64-bit hash of the time and process ID is used.

Notes
-----
See xorshift128 for a faster implementation that has a smaller period.

**Parallel Features**

``xorshift1024.RandomState`` can be used in parallel applications by
calling the method ``jump`` which advances the state as-if
:math:`2^{512}` random numbers have been generated. This
allows the original sequence to be split so that distinct segments can be used
in each worker process. All generators should be initialized with the same
seed to ensure that the segments come from the same sequence.

>>> import randomstate.prng.xorshift1024 as rnd
>>> rs = [rnd.RandomState(1234) for _ in range(10)]
# Advance rs[i] by i jumps
>>> for i in range(10):
        rs[i].jump(i)

**State and Seeding**

The ``xorshift1024.RandomState`` state vector consists of a 16 element array
of 64-bit unsigned integers.

``xorshift1024.RandomState`` is seeded using either a single 64-bit unsigned integer
or a vector of 64-bit unsigned integers.  In either case, the input seed is
used as an input (or inputs) for another simple random number generator,
Splitmix64, and the output of this PRNG function is used as the initial state.
Using a single 64-bit value for the seed can only initialize a small range of
the possible initial state values.  When using an array, the SplitMix64 state
for producing the ith component of the initial state is XORd with the ith
value of the seed array until the seed array is exhausted. When using an array
the initial state for the SplitMix64 state is 0 so that using a single element
array and using the same value as a scalar will produce the same initial state.

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

DEF JUMP_DOCSTRING = u"""
jump(iter = 1)

Jumps the state of the random number generator as-if 2**512 random numbers
have been generated.

Parameters
----------
iter : integer, positive
    Number of times to jump the state of the rng.

Returns
-------
out : None
    Returns 'None' on success.

Notes
-----
Jumping the rng state resets any pre-computed random numbers. This is required
to ensure exact reproducibility.
"""
