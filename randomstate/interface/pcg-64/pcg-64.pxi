DEF RS_RNG_NAME = u'pcg64'
DEF RS_RNG_ADVANCEABLE = 1
DEF RS_RNG_SEED=2
DEF RS_PCG128_EMULATED = 0
DEF RS_SEED_NBYTES = 4

cdef extern from "inttypes.h":
    ctypedef unsigned long long __uint128_t

cdef extern from "distributions.h":
    ctypedef __uint128_t pcg128_t

    cdef struct pcg_state_setseq_128:
        pcg128_t state
        pcg128_t inc

    ctypedef pcg_state_setseq_128 pcg64_random_t

    cdef struct s_aug_state:
        pcg64_random_t *rng
        binomial_t *binomial

        int has_gauss, shift_zig_random_int, has_uint32, has_gauss_float
        float gauss_float
        double gauss
        uint64_t zig_random_int
        uint32_t uinteger

    ctypedef s_aug_state aug_state

    cdef void advance_state(aug_state* state, pcg128_t delta)

    cdef void set_seed(aug_state* state, pcg128_t seed, pcg128_t inc)

ctypedef pcg128_t rng_state_t

ctypedef pcg64_random_t rng_t

cdef object _get_state(aug_state state):
    return (state.rng.state, state.rng.inc)

cdef object _set_state(aug_state *state, object state_info):
    state.rng.state = state_info[0]
    state.rng.inc = state_info[1]

include "pcg-64-docstring.pxi"