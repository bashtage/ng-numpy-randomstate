cdef extern from "pcg_variants.h":
    # PCG-64
    ctypedef __uint128_t pcg128_t;

    cdef struct pcg_state_setseq_128:
        pcg128_t state;
        pcg128_t inc;

    ctypedef pcg_state_setseq_128 pcg64_random_t

    cdef void pcg_setseq_128_srandom_r(pcg64_random_t *rng, pcg128_t initstate, pcg128_t initseq)

    cdef void pcg64_advance_r(pcg64_random_t *rng, pcg128_t delta)

    cdef void pcg64_srandom_r(pcg64_random_t *rng, pcg128_t initstate, pcg128_t initseq)

    cdef uint64_t pcg64_random_r(pcg64_random_t *rng)

    cdef uint64_t pcg64_boundedrand_r(pcg64_random_t *rng, uint64_t i)

cdef extern from "pcg_helper.c":
    cdef struct s_aug_state:
        pcg64_random_t *rng
        pcg128_t state, inc

        bint has_gauss
        double gauss

        int shift_zig_random_int
        uint64_t zig_random_int

    ctypedef s_aug_state aug_state

cdef extern from "pcg_helper.c":
    cdef double pcg_random_double(pcg64_random_t *rng)

    cdef double pcg_random_gauss(pcg64_random_t *rng, bint *has_gauss, double *gauss)

    cdef double pcg_standard_exponential(pcg64_random_t *rng)

    cdef double pcg_standard_gamma(pcg64_random_t *rng, double shape, bint *has_gauss, double *gauss)

    cdef double pcg_random_gauss_zig(pcg64_random_t *rng,
                                     int *shift_zig_random_int,
                                     uint64_t *zig_random_int)

    cdef double pcg_random_double_2(aug_state *state)
