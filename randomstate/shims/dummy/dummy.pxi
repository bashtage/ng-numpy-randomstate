DEF RNG_NAME = 'dummy'
DEF RNG_ADVANCEABLE = 0
DEF RNG_JUMPABLE = 0
DEF RNG_STATE_LEN = 4
DEF RNG_SEED = 1

cdef extern from "core-rng.h":

    cdef struct s_aug_state:
        uint32_t *rng
        binomial_t *binomial

        int has_gauss, shift_zig_random_int, has_uint32
        double gauss
        uint64_t zig_random_int
        uint32_t uinteger

    ctypedef s_aug_state aug_state

    cdef void seed(aug_state* state, uint32_t seed)

    cdef void advance(aug_state* state, uint32_t delta)

ctypedef uint32_t rng_t

ctypedef uint32_t rng_state_t

cdef object _get_state(aug_state state):
    return None

cdef object _set_state(aug_state state, object state_info):
    pass

DEF CLASS_DOCSTRING = """
This is the dummy  docstring.
"""