DEF RS_RNG_NAME = u'pcg64'
DEF RS_RNG_ADVANCEABLE = 1
DEF RS_RNG_SEED=2
DEF RS_PCG128_EMULATED = 1
DEF RS_SEED_NBYTES = 4

from cpython cimport PyLong_FromUnsignedLongLong, PyLong_AsUnsignedLongLong

cdef extern from "distributions.h":

    ctypedef struct pcg128_t:
        uint64_t high
        uint64_t low

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

cdef object pcg128_to_pylong(pcg128_t x):
    return PyLong_FromUnsignedLongLong(x.high) * 2**64 + PyLong_FromUnsignedLongLong(x.low)

cdef pcg128_t pcg128_from_pylong(object x):
    cdef pcg128_t out
    out.high = PyLong_AsUnsignedLongLong(x // (2 ** 64))
    out.low = PyLong_AsUnsignedLongLong(x % (2 ** 64))
    return out

cdef object _get_state(aug_state state):
    return (pcg128_to_pylong(state.rng.state), pcg128_to_pylong(state.rng.inc))

cdef object _set_state(aug_state *state, object state_info):
    state.rng.state = pcg128_from_pylong(state_info[0])
    state.rng.inc = pcg128_from_pylong(state_info[1])

include "pcg-64-docstring.pxi"
