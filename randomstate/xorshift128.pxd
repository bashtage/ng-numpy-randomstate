from libc.stdint cimport uint32_t, uint64_t
from binomial cimport *

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

