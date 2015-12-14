DEF RNG_ADVANCEABLE = 1

DEF RNG_SEED = 2

ctypedef uint64_t rng_state_t

cdef extern from "core-rng.h":
    cdef struct pcg_state_setseq_64:
        uint64_t state;
        uint64_t inc;

    ctypedef pcg_state_setseq_64 pcg32_random_t

    cdef struct s_aug_state:
        pcg32_random_t *rng
        uint64_t state, inc

        int has_gauss, shift_zig_random_int
        double gauss
        uint64_t zig_random_int

    ctypedef s_aug_state aug_state

    cdef void seed(aug_state* state, uint64_t seed, uint64_t inc)

    cdef void advance(aug_state* state, uint64_t delta)

ctypedef pcg32_random_t rng_t
