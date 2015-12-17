DEF RNG_NAME = "xorshift-128+"
DEF RNG_ADVANCEABLE = 0
DEF RNG_SEED = 2

cdef extern from "core-rng.h":

    cdef struct s_xorshift128_state:
      uint64_t s[2]

    ctypedef s_xorshift128_state xorshift128_state

    cdef struct s_aug_state:
        xorshift128_state *rng

        int has_gauss, shift_zig_random_int, has_uint32
        double gauss
        uint64_t zig_random_int
        uint32_t uinteger

    ctypedef s_aug_state aug_state

    cdef void seed(aug_state* state, uint64_t seed, uint64_t inc)

ctypedef uint64_t rng_state_t

ctypedef xorshift128_state rng_t

cdef object _get_state(aug_state state):
    return (state.rng.s[0], state.rng.s[1])

