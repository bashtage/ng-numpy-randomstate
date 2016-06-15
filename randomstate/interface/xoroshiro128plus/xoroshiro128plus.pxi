DEF RS_RNG_NAME = u"xoroshiro-128+"
DEF RS_RNG_JUMPABLE = 1

cdef extern from "distributions.h":

    cdef struct s_xoroshiro128plus_state:
        uint64_t s[2]

    ctypedef s_xoroshiro128plus_state xoroshiro128plus_state

    cdef struct s_aug_state:
        xoroshiro128plus_state *rng
        binomial_t *binomial

        int has_gauss, shift_zig_random_int, has_uint32, has_gauss_float
        float gauss_float
        double gauss
        uint64_t zig_random_int
        uint32_t uinteger

    ctypedef s_aug_state aug_state

    cdef void set_seed(aug_state* state, uint64_t seed)

    cdef void jump_state(aug_state* state)

    cdef void set_seed_by_array(aug_state* state, uint64_t *seed, int count)

    cdef void init_state(aug_state* state, uint64_t* state_vals)


ctypedef uint64_t rng_state_t

ctypedef xoroshiro128plus_state rng_t

cdef inline object _get_state(aug_state state):
    return (state.rng.s[0], state.rng.s[1])

cdef inline object _set_state(aug_state *state, object state_info):
    state.rng.s[0] = state_info[0]
    state.rng.s[1] = state_info[1]



DEF CLASS_DOCSTRING = u"""
RandomState(seed=None)

Container for the xoroshiro128plus+ pseudo-random number generator.

xoroshiro128+ is the successor to xorshift128+ written by David Blackman and
Sebastiano Vigna.  It is a 64-bit PRNG that uses a carefully handcrafted
shift/rotate-based linear transformation.  This change both improves speed and
statistical quality of the PRNG [1]_. xoroshiro128plus+ has a period of
:math:`2^{128} - 1` and supports jumping the sequence in increments of
:math:`2^{64}`, which allows  multiple non-overlapping sequences to be
generated.

``xoroshiro128plus.RandomState`` exposes a number of methods for generating random
numbers drawn from a variety of probability distributions. In addition to the
distribution-specific arguments, each method takes a keyword argument
`size` that defaults to ``None``. If `size` is ``None``, then a single
value is generated and returned. If `size` is an integer, then a 1-D
array filled with generated values is returned. If `size` is a tuple,
then an array with that shape is filled and returned.

**No Compatibility Guarantee**

``xoroshiro128plus.RandomState`` does not make a guarantee that a fixed seed and a
fixed series of calls to ``xoroshiro128plus.RandomState`` methods using the same
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
    then ``xoroshiro128plus.RandomState`` will try to read data from
    ``/dev/urandom`` (or the Windows analog) if available.  If
    unavailable, a 64-bit hash of the time and process ID is used.

Notes
-----
See xorshift1024 for an related PRNG implementation with a larger
period  (:math:`2^{1024} - 1`) and jump size (:math:`2^{512} - 1`).

**Parallel Features**

``xoroshiro128plus.RandomState`` can be used in parallel applications by
calling the method ``jump`` which advances the state as-if
:math:`2^{64}` random numbers have been generated. This
allow the original sequence to be split so that distinct segments can be used
in each worker process. All generators should be initialized with the same
seed to ensure that the segments come from the same sequence.

>>> import randomstate.prng.xoroshiro128plus as rnd
>>> rs = [rnd.RandomState(1234) for _ in range(10)]
# Advance rs[i] by i jumps
>>> for i in range(10):
        rs[i].jump(i)

**State and Seeding**

The ``xoroshiro128plus.RandomState`` state vector consists of a 2 element array
of 64-bit unsigned integers.

``xoroshiro128plus.RandomState`` is seeded using either a single 64-bit unsigned integer
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
.. [1] "xoroshiro+ / xorshift* / xorshift+ generators and the PRNG shootout",
       http://xorshift.di.unimi.it/
"""

DEF JUMP_DOCSTRING = u"""
jump(iter = 1)

Jumps the state of the random number generator as-if 2**64 random numbers
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
