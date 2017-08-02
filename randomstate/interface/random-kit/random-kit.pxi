DEF RS_RNG_NAME = u'mt19937'
DEF RS_RNG_JUMPABLE = 1
DEF RS_NORMAL_METHOD = u'bm'
DEF RK_STATE_LEN = 624
DEF RS_SEED_NBYTES = 1

ctypedef uint32_t rng_state_t

cdef extern from "distributions.h":

    cdef struct s_randomkit_state:
      uint32_t key[RK_STATE_LEN]
      int pos

    ctypedef s_randomkit_state randomkit_state

    cdef struct s_aug_state:
        randomkit_state *rng
        binomial_t *binomial

        int has_gauss, shift_zig_random_int, has_uint32, has_gauss_float
        float gauss_float
        double gauss
        uint64_t zig_random_int
        uint32_t uinteger

    ctypedef s_aug_state aug_state

    cdef void set_seed(aug_state* state, uint32_t seed)

    cdef void set_seed_by_array(aug_state* state, uint32_t *init_key, int key_length)

    cdef void jump_state(aug_state* state)

ctypedef randomkit_state rng_t

cdef object _get_state(aug_state state):
    cdef uint32_t [:] key = np.zeros(RK_STATE_LEN, dtype=np.uint32)
    cdef Py_ssize_t i
    for i in range(RK_STATE_LEN):
        key[i] = state.rng.key[i]
    return (np.asanyarray(key), state.rng.pos)

cdef object _set_state(aug_state *state, object state_info):
    cdef uint32_t [:] key = state_info[0]
    cdef Py_ssize_t i
    for i in range(RK_STATE_LEN):
        state.rng.key[i] = key[i]
    state.rng.pos = state_info[1]

DEF CLASS_DOCSTRING = u"""
RandomState(seed=None)

Container for the Mersenne Twister pseudo-random number generator.

``mt19937.RandomState`` exposes a number of methods for generating random
numbers drawn from a variety of probability distributions. In addition to the
distribution-specific arguments, each method takes a keyword argument
`size` that defaults to ``None``. If `size` is ``None``, then a single
value is generated and returned. If `size` is an integer, then a 1-D
array filled with generated values is returned. If `size` is a tuple,
then an array with that shape is filled and returned.

**Compatibility Guarantee**

``mt19937.RandomState`` is identical to ``numpy.random.RandomState`` and
makes the same compatibility guarantee. A fixed seed and a fixed series of
calls to ``mt19937.RandomState`` methods using the same parameters will always
produce the same results up to roundoff error except when the values were
incorrect. Incorrect values will be fixed and the version in which the fix
was made will be noted in the relevant docstring. Extension of existing
parameter ranges and the addition of new parameters is allowed as long the
previous behavior remains unchanged.

Parameters
----------
seed : {None, int, array_like}, optional
    Random seed used to initialize the pseudo-random number generator.  Can
    be any integer between 0 and 2**32 - 1 inclusive, an array (or other
    sequence) of such integers, or ``None`` (the default).  If `seed` is
    ``None``, then `RandomState` will try to read data from
    ``/dev/urandom`` (or the Windows analog) if available or seed from
    the clock otherwise.

Notes
-----
The Python stdlib module "random" also contains a Mersenne Twister
pseudo-random number generator with a number of methods that are similar
to the ones available in ``RandomState``. ``RandomState``, besides being
NumPy-aware, has the advantage that it provides a much larger number
of probability distributions to choose from.

**Parallel Features**

``mt19937.RandomState`` can be used in parallel applications by
calling the method ``jump`` which advances the state as-if :math:`2^{128}`
random numbers have been generated ([1]_, [2]_). This allows the original sequence to
be split so that distinct segments can be used in each worker process.  All
generators should be initialized with the same seed to ensure that the
segments come from the same sequence.

>>> from randomstate.entropy import random_entropy
>>> import randomstate.prng.mt19937 as rnd
>>> seed = random_entropy()
>>> rs = [rnd.RandomState(seed) for _ in range(10)]
# Advance rs[i] by i jumps
>>> for i in range(10):
        rs[i].jump(i)

References
----------
.. [1] Hiroshi Haramoto, Makoto Matsumoto, and Pierre L\'Ecuyer, "A Fast
       Jump Ahead Algorithm for Linear Recurrences in a Polynomial Space",
       Sequences and Their Applications - SETA, 290--298, 2008.        
.. [2] Hiroshi Haramoto, Makoto Matsumoto, Takuji Nishimura, François 
       Panneton, Pierre L\'Ecuyer, "Efficient Jump Ahead for F2-Linear
       Random Number Generators", INFORMS JOURNAL ON COMPUTING, Vol. 20, 
       No. 3, Summer 2008, pp. 385-390.

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