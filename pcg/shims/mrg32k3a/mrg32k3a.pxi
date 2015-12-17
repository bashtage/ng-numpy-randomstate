DEF RNG_ADVANCEABLE = 0

DEF RNG_SEED = 0

cdef extern from "core-rng.h":

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

        int has_gauss, shift_zig_random_int
        double gauss
        uint64_t zig_random_int

    ctypedef s_aug_state aug_state

    cdef void seed(aug_state* state, uint64_t* seed)

ctypedef mrg32k3a_state rng_t

ctypedef object rng_state_t