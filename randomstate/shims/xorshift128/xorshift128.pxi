DEF RNG_NAME = "xorshift-128+"
DEF RNG_ADVANCEABLE = 0
DEF RNG_JUMPABLE = 1
DEF RNG_STATE_LEN = 4
DEF RNG_SEED=1
DEF NORMAL_METHOD = 'zig'

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

    cdef void jump(aug_state* state)

    cdef void init_state(aug_state* state, uint64_t* state_vals)


ctypedef uint64_t rng_state_t

ctypedef xorshift128_state rng_t

cdef inline object _get_state(aug_state state):
    return (state.rng.s[0], state.rng.s[1])

cdef inline object _set_state(aug_state state, object state_info):
    state.rng.s[0] = state_info[0]
    state.rng.s[1] = state_info[1]



DEF CLASS_DOCSTRING = """
RandomState(seed=None)

Container for the xorshift128+ pseudo random number generator.

xorshift128+ is a 64-bit implementation of Saito and Matsumoto's XSadd
generator. xorshift128+ has a period of 2**128 _ 1 and supports jumping the
sequence in increments of 2**64, which allow multiple non-overlapping
sequences to be generated.

`xorshift128.RandomState` exposes a number of methods for generating random
numbers drawn from a variety of probability distributions. In addition to the
distribution-specific arguments, each method takes a keyword argument
`size` that defaults to ``None``. If `size` is ``None``, then a single
value is generated and returned. If `size` is an integer, then a 1-D
array filled with generated values is returned. If `size` is a tuple,
then an array with that shape is filled and returned.

*No Compatibility Guarantee*
'xorshift128.RandomState' does not make a guarantee that a fixed seed and a
fixed series of calls to 'xorshift128.RandomState' methods using the same
parameters will always produce the same results. This is different from
'numpy.random.RandomState' guarantee. This is done to simplify improving
random number generators.  To ensure identical results, you must use the
same release version.

Parameters
----------
seed : {None, int}, optional
    Random seed initializing the pseudo-random number generator.
    Can be an integer or ``None`` (the default).
    If `seed` is ``None``, then `xorshift128.RandomState` will try to read data
    from ``/dev/urandom`` (or the Windows analogue) if available or seed from
    the clock otherwise.

Notes
-----
See xorshift1024 for an implementation with a larger period and jump size.
"""
