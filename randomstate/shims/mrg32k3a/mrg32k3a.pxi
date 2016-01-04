DEF RNG_ADVANCEABLE = 0
DEF RNG_SEED = 1
DEF RNG_NAME = 'mrg32k3a'
DEF RNG_STATE_LEN = 4
DEF RNG_JUMPABLE = 0

cdef extern from "distributions.h":

    cdef struct s_mrg32k3a_state:
        int64_t s10
        int64_t s11
        int64_t s12
        int64_t s20
        int64_t s21
        int64_t s22

    ctypedef s_mrg32k3a_state mrg32k3a_state

    cdef struct s_aug_state:
        mrg32k3a_state *rng
        binomial_t *binomial

        int has_gauss, shift_zig_random_int, has_uint32
        double gauss
        uint32_t uinteger
        uint64_t zig_random_int

    ctypedef s_aug_state aug_state

    cdef void set_seed(aug_state* state, uint64_t seed)

ctypedef mrg32k3a_state rng_t

ctypedef uint64_t rng_state_t

cdef object _get_state(aug_state state):
    return (state.rng.s10, state.rng.s11, state.rng.s12,
            state.rng.s20, state.rng.s21, state.rng.s22)

cdef object _set_state(aug_state state, object state_info):
    state.rng.s10 = state_info[0]
    state.rng.s11 = state_info[1]
    state.rng.s12 = state_info[2]
    state.rng.s20 = state_info[3]
    state.rng.s21 = state_info[4]
    state.rng.s22 = state_info[5]

DEF CLASS_DOCSTRING = """
This is the mrg32k3a docstring.
"""