DEF RNG_ADVANCEABLE = 1

DEF RNG_SEED = 2

cdef extern from "inttypes.h":
    ctypedef unsigned long long __uint128_t

cdef extern from "core-rng.h":
    ctypedef __uint128_t pcg128_t

    cdef struct pcg_state_setseq_128:
        pcg128_t state
        pcg128_t inc

    ctypedef pcg_state_setseq_128 pcg64_random_t

    cdef struct s_aug_state:
        pcg64_random_t *rng
        pcg128_t state, inc

        int has_gauss, shift_zig_random_int
        double gauss
        uint64_t zig_random_int

    ctypedef s_aug_state aug_state

    cdef void seed(aug_state* state, pcg128_t seed, pcg128_t inc)

    cdef void advance(aug_state* state, pcg128_t delta)

ctypedef pcg128_t rng_state_t

ctypedef pcg64_random_t rng_t

