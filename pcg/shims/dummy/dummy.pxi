cdef extern from "core-rng.h":

    cdef struct s_aug_state:
        uint32_t *rng
        uint64_t state, inc

        int has_gauss, shift_zig_random_int
        double gauss
        uint64_t zig_random_int

    ctypedef s_aug_state aug_state

    cdef void seed(aug_state* state, uint32_t seed)

ctypedef uint32_t rng_t

ctypedef uint32_t rng_state_t
