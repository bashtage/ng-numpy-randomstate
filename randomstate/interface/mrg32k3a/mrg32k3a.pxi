DEF RS_RNG_NAME = u'mrg32k3a'
DEF RS_RNG_JUMPABLE = 1

cdef extern from "distributions.h":

    cdef struct s_mrg32k3a_state:
        int64_t s1[3]
        int64_t s2[3]
        int loc

    ctypedef s_mrg32k3a_state mrg32k3a_state

    cdef struct s_aug_state:
        mrg32k3a_state *rng
        binomial_t *binomial

        int has_gauss, shift_zig_random_int, has_uint32, has_gauss_float
        float gauss_float
        double gauss
        uint32_t uinteger
        uint64_t zig_random_int

    ctypedef s_aug_state aug_state

    cdef void set_seed(aug_state* state, uint64_t seed)

    cdef void set_seed_by_array(aug_state* state, uint64_t *seed, int count)

ctypedef mrg32k3a_state rng_t

ctypedef uint64_t rng_state_t

cdef object _get_state(aug_state state):
    return (state.rng.s1[0], state.rng.s1[1], state.rng.s1[2],
            state.rng.s2[0], state.rng.s2[1], state.rng.s2[2],
            state.rng.loc)

cdef object _set_state(aug_state *state, object state_info):
    state.rng.s1[0] = state_info[0]
    state.rng.s1[1] = state_info[1]
    state.rng.s1[2] = state_info[2]
    state.rng.s2[0] = state_info[3]
    state.rng.s2[1] = state_info[4]
    state.rng.s2[2] = state_info[5]
    state.rng.loc = state_info[6]

cdef object matrix_power_127(x, m):
    n = x.shape[0]
    # Start at power 1
    out = x.copy()
    current_pow = x.copy()
    for i in range(7):
        current_pow = np.mod(current_pow.dot(current_pow), m)
        out = np.mod(out.dot(current_pow), m)
    return out

m1 = np.int64(4294967087)
a12 = np.int64(1403580)
a13n = np.int64(810728)
A1 = np.array([[0, 1, 0], [0, 0, 1], [-a13n, a12, 0]], dtype=np.int64)
A1p = np.mod(A1, m1).astype(np.uint64)
A1_127 = matrix_power_127(A1p, m1)

a21 = np.int64(527612)
a23n = np.int64(1370589)
A2 = np.array([[0, 1, 0], [0, 0, 1], [-a23n, 0, a21]], dtype=np.int64)
m2 = np.int64(4294944443)
A2p = np.mod(A2, m2).astype(np.uint64)
A2_127 = matrix_power_127(A2p, m2)

cdef void jump_state(aug_state* state):
    # vectors s1 and s2
    loc = state.rng.loc

    if loc == 0:
        loc_m1 = 2
        loc_m2 = 1
    elif loc == 1:
        loc_m1 = 0
        loc_m2 = 2
    else:
        loc_m1 = 1
        loc_m2 = 0

    s1 = np.array([state.rng.s1[loc_m2],
                   state.rng.s1[loc_m1],
                   state.rng.s1[loc]], dtype=np.uint64)
    s2 = np.array([state.rng.s2[loc_m2],
                   state.rng.s2[loc_m1],
                   state.rng.s2[loc]], dtype=np.uint64)

    # Advance the state
    s1 = np.mod(A1_127.dot(s1), m1)
    s2 = np.mod(A1_127.dot(s2), m2)

    # Restore state
    state.rng.s1[0] = s1[0]
    state.rng.s1[1] = s1[1]
    state.rng.s1[2] = s1[2]

    state.rng.s2[0] = s2[0]
    state.rng.s2[1] = s2[1]
    state.rng.s2[2] = s2[2]

    state.rng.loc = 2

DEF CLASS_DOCSTRING = u"""
RandomState(seed=None)

Container for L'Ecuyer MRG32K3A pseudo-random number generator.

MRG32K3A is a 32-bit implementation of L'Ecuyer's combined multiple
recursive generator [1]_, [2]_. MRG32K3A has a period of :math:`2^{191}`,
supports jumping ahead and is suitable for parallel applications.

``mrg32k3a.RandomState`` exposes a number of methods for generating random
numbers drawn from a variety of probability distributions. In addition to the
distribution-specific arguments, each method takes a keyword argument
`size` that defaults to ``None``. If `size` is ``None``, then a single
value is generated and returned. If `size` is an integer, then a 1-D
array filled with generated values is returned. If `size` is a tuple,
then an array with that shape is filled and returned.

**No Compatibility Guarantee**

``mrg32k3a.RandomState`` does not make a guarantee that a fixed seed and a
fixed series of calls to ``mrg32k3a.RandomState`` methods using the same
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
    then ``mrg32k3a.RandomState`` will try to read data from
    ``/dev/urandom`` (or the Windows analog) if available. If
    unavailable, a 64-bit hash of the time and process ID is used.

Notes
-----
The state of the MRG32KA PRNG is represented by 6 64-bit integers.

This implementation is integer based and produces integers in the interval
:math:`[0, 2^{32}-209+1]`.  These are treated as if they 32-bit random
integers.

**Parallel Features**

``mrg32k3a.RandomState`` can be used in parallel applications by
calling the method ``jump`` which advances the state as-if
:math:`2^{127}` random numbers have been generated [3]_. This
allows the original sequence to be split so that distinct segments can be used
in each worker process. All generators should be initialized with the same
seed to ensure that the segments come from the same sequence.

>>> import randomstate.prng.mrg32k3a as rnd
>>> rs = [rnd.RandomState(12345) for _ in range(10)]
# Advance rs[i] by i jumps
>>> for i in range(10):
        rs[i].jump(i)

**State and Seeding**

The ``mrg32k3a.RandomState`` state vector consists of a 6 element array
of 64-bit signed integers plus a single integers value between 0 and 2
indicating  the current position within the state vector.  The first three
elements of the state vector are in [0, 4294967087) and the second 3 are
in [0, 4294944443).

``mrg32k3a.RandomState`` is seeded using either a single 64-bit unsigned integer
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
.. [1] "Software developed by the Canada Research Chair in Stochastic
       Simulation and Optimization", http://simul.iro.umontreal.ca/
.. [2] Pierre L'Ecuyer, (1999) "Good Parameters and Implementations for
       Combined Multiple Recursive Random Number Generators.", Operations
       Research 47(1):159-164
.. [3] L'ecuyer, Pierre, Richard Simard, E. Jack Chen, and W. David Kelton.
       "An object-oriented random-number package with many long streams
       and substreams." Operations research 50, no. 6, pp. 1073-1075, 2002.
"""

DEF JUMP_DOCSTRING = u"""
jump(iter = 1)

Jumps the state of the random number generator as-if 2**127 random numbers
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