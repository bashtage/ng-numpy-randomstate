DEF RS_RNG_NAME = u'sfmt'
DEF RS_RNG_JUMPABLE = 1
DEF SFMT_MEXP = 19937
DEF SFMT_N = 156 # (SFMT_MEXP / 128 + 1)
DEF RS_SEED_NBYTES = 1
DEF RS_SEED_ARRAY_BITS = 32

ctypedef uint32_t rng_state_t

cdef extern from "distributions.h":

    cdef union W128_T:
        uint32_t u[4]
        uint64_t u64[2]

    ctypedef W128_T w128_t;

    cdef struct SFMT_T:
        w128_t state[SFMT_N];
        int idx;

    ctypedef SFMT_T sfmt_t;

    cdef struct s_aug_state:
        sfmt_t *rng
        binomial_t *binomial

        int has_gauss, shift_zig_random_int, has_uint32, has_gauss_float
        float gauss_float
        double gauss
        uint64_t zig_random_int
        uint32_t uinteger

        uint64_t *buffered_uint64
        int buffer_loc

    ctypedef s_aug_state aug_state

    cdef void set_seed(aug_state* state, uint32_t seed)

    cdef void set_seed_by_array(aug_state* state, uint32_t init_key[], int key_length)

    cdef void jump_state(aug_state* state)


ctypedef sfmt_t rng_t

cdef object _get_state(aug_state state):
    cdef uint32_t [::1] key = np.zeros(4 * SFMT_N, dtype=np.uint32)
    cdef uint64_t [::1] buf = np.zeros(2 * SFMT_N, dtype=np.uint64)
    cdef Py_ssize_t i, j, key_loc = 0
    cdef w128_t state_val
    for i in range(SFMT_N):
        state_val = state.rng.state[i]
        for j in range(4):
            key[key_loc] = state_val.u[j]
            key_loc += 1
    for i in range(2 * SFMT_N):
        buf[i] = state.buffered_uint64[i]

    return (np.asarray(key), state.rng.idx,
            np.asarray(buf), state.buffer_loc)

cdef object _set_state(aug_state *state, object state_info):
    cdef Py_ssize_t i, j, key_loc = 0
    cdef uint32_t [::1] key = state_info[0]
    state.rng.idx = state_info[1]

    for i in range(SFMT_N):
        for j in range(4):
            state.rng.state[i].u[j] = key[key_loc]
            key_loc += 1

    state.buffer_loc = <int>state_info[3]
    for i in range(2 * SFMT_N):
        state.buffered_uint64[i] = state_info[2][i]


DEF CLASS_DOCSTRING = u"""
RandomState(seed=None)

Container for the SIMD-based Mersenne Twister pseudo-random number generator.

``sfmt.RandomState`` exposes a number of methods for generating random
numbers drawn from a variety of probability distributions [1]_ . In addition
to the distribution-specific arguments, each method takes a keyword argument
`size` that defaults to ``None``. If `size` is ``None``, then a single
value is generated and returned. If `size` is an integer, then a 1-D
array filled with generated values is returned. If `size` is a tuple,
then an array with that shape is filled and returned.

**No Compatibility Guarantee**

``sfmt.RandomState`` does not make a guarantee that a fixed seed and a
fixed series of calls to ``sfmt.RandomState`` methods using the same
parameters will always produce the same results. This is different from
``numpy.random.RandomState`` guarantee. This is done to simplify improving
random number generators.  To ensure identical results, you must use the
same release version.

Parameters
----------
seed : {None, int, array_like}, optional
    Random seed initializing the pseudo-random number generator.
    Can be an integer in [0, 2**32-1], array of integers in
    [0, 2**32-1] or ``None`` (the default). If `seed` is ``None``,
    then ``sfmt.RandomState`` will try to read entropy from
    ``/dev/urandom`` (or the Windows analog) if available to
    produce a 64-bit seed. If unavailable, the a 64-bit hash of the time
    and process ID is used.

Notes
-----
The Python stdlib module "random" also contains a Mersenne Twister
pseudo-random number generator with a number of methods that are similar
to the ones available in ``RandomState``. The `RandomState` object, besides
being NumPy-aware, also has the advantage that it provides a much larger
number of probability distributions to choose from.

>>> from randomstate.entropy import random_entropy
>>> import randomstate.prng.sfmt as rnd
>>> seed = random_entropy()
>>> rs = [rnd.RandomState(seed) for _ in range(10)]

**Parallel Features**

``sfmt.RandomState`` can be used in parallel applications by
calling the method ``jump`` which advances the state as-if :math:`2^{128}`
random numbers have been generated [2]_. This allows the original sequence to
be split so that distinct segments can be used in each worker process.  All
generators should be initialized with the same seed to ensure that the
segments come from the same sequence.

>>> from randomstate.entropy import random_entropy
>>> import randomstate.prng.sfmt as rnd
>>> seed = random_entropy()
>>> rs = [rnd.RandomState(seed) for _ in range(10)]
# Advance rs[i] by i jumps
>>> for i in range(10):
        rs[i].jump(i)
        
**State and Seeding**

The ``sfmt.RandomState`` state vector consists of a 624 element array of
32-bit unsigned integers plus a single integer value between 0 and 624
indicating  the current position within the main array. The implementation
used here augments this with a array of 64-bit unsigned integers which are used
to efficiently access the random numbers produced by the SFMT generator.

``sfmt.RandomState`` is seeded using either a single 32-bit unsigned integer
or a vector of 32-bit unsigned integers.  In either case, the input seed is
used as an input (or inputs) for a hashing function, and the output of the
hashing function is used as the initial state. Using a single 32-bit value
for the seed can only initialize a small range of the possible initial
state values.

References
----------
.. [1] Mutsuo Saito and Makoto Matsumoto, "SIMD-oriented Fast Mersenne
       Twister: a 128-bit Pseudorandom Number Generator." Monte Carlo
       and Quasi-Monte Carlo Methods 2006, Springer, pp. 607 -- 622, 2008.
.. [2] Hiroshi Haramoto, Makoto Matsumoto, and Pierre L\'Ecuyer, "A Fast
       Jump Ahead Algorithm for Linear Recurrences in a Polynomial Space",
       Sequences and Their Applications - SETA, 290--298, 2008.
"""

DEF JUMP_DOCSTRING = u"""
jump(iter = 1)

Jumps the state of the random number generator as-if 2**128 random numbers
have been generated.

Parameters
----------
iter : integer, positive
    Number of times to jump the state of the prng.

Returns
-------
out : None
    Returns 'None' on success.

Notes
-----
Jumping the rng state resets any pre-computed random numbers. This is required
to ensure exact reproducibility.
"""
