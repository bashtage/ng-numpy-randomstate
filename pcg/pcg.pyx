#!python
#cython: wraparound=False, nonecheck=False, boundscheck=False, cdivision=True
import numpy as np
cimport numpy as np
import cython
from libc.stdint cimport uint32_t, uint64_t

cdef extern from "inttypes.h":
    ctypedef unsigned long long __uint128_t

cdef extern from "pcg_variants.h":
    # struct pcg_state_setseq_64 {
    #     uint64_t state;
    #     uint64_t inc;
    # };
    cdef struct pcg_state_setseq_64:
        uint64_t state
        uint64_t inc

    # typedef struct pcg_state_setseq_64      pcg32_random_t;
    ctypedef pcg_state_setseq_64 pcg32_random_t

    cdef void pcg_setseq_64_srandom_r(pcg32_random_t *rng, uint64_t initstate, uint64_t initseq)

    cdef void pcg32_srandom_r(pcg32_random_t *rng, uint64_t initstate, uint64_t initseq)

    cdef uint32_t pcg_setseq_64_xsh_rr_32_random_r(pcg32_random_t *rng)

    cdef uint32_t pcg32_random_r(pcg32_random_t *rng)

    cdef uint32_t pcg32_boundedrand_r(pcg32_random_t *rng, uint32_t i)

    cdef void pcg32_advance_r(pcg32_random_t *rng, uint64_t delta)

    # PCG-64
    ctypedef __uint128_t pcg128_t;

    cdef struct pcg_state_setseq_128:
        pcg128_t state;
        pcg128_t inc;

    ctypedef pcg_state_setseq_128 pcg64_random_t

    cdef void pcg_setseq_128_srandom_r(pcg64_random_t *rng, pcg128_t initstate, pcg128_t initseq)




cdef extern from "pcg_helper.c":
    cdef double pcg_random_double(pcg32_random_t *rng)

    cdef double pcg_random_gauss(pcg32_random_t *rng, bint *has_gauss, double *gauss)

    cdef double pcg_standard_exponential(pcg32_random_t *rng)

    cdef double pcg_standard_gamma(pcg32_random_t *rng, double shape, bint *has_gauss, double *gauss)

    cdef double pcg_random_double_64(pcg64_random_t *rng)

    cdef double pcg_random_gauss_64(pcg64_random_t *rng, bint *has_gauss, double *gauss)

    cdef double pcg_random_gauss_zig(pcg32_random_t *rng,
                                     int *shift_zig_random_int,
                                     uint32_t *zig_random_int)


cdef class PCGRandomState:
    '''Test class'''
    cdef pcg32_random_t rng
    cdef uint64_t state, inc
    cdef bint has_gauss
    cdef double gauss

    cdef pcg64_random_t rng64
    cdef pcg128_t state64, inc64

    cdef int shift_zig_random_int
    cdef uint32_t zig_random_int

    def __init__(self, state=None, inc=None):
        if state is not None and inc is not None:
            self.state = state
            self.inc = inc
            self.state64 = state
            self.inc64 = inc
        else:
            # TODO: Add entropy method to get state and inc
            self.state = 42
            self.inc = 52
            self.state64 = 42
            self.inc64 = 52

        self.shift_zig_random_int = 0
        self.zig_random_int = 0

        self.seed(self.state, self.inc)
        self.has_gauss = 0
        self.gauss = 0.0

    def seed(self, uint64_t state, uint64_t inc):
        pcg_setseq_64_srandom_r(&self.rng, state, inc)
        pcg_setseq_128_srandom_r(&self.rng64, self.state64, self.inc64)

    def get_state(self):
        return 'pcg32', (self.rng.state, self.rng.inc), self.has_gauss, self.gauss

    def advance(self, uint64_t delta):
        pcg32_advance_r(&self.rng, delta)
        return None

    def set_state(self, tuple state):
        if state[0] != 'pcg32' or len(state) != 4:
            raise ValueError('Must be a pcg32 state')
        self.rng.state = state[1][0]
        self.rng.inc = state[1][1]
        self.has_gauss = state[2]
        self.gauss = state[3]

    def random_integers(self, Py_ssize_t n=1):
        cdef Py_ssize_t i
        cdef uint32_t [:] randoms = np.zeros(n, dtype=np.uint32)
        for i in range(n):
            randoms[i] = pcg32_random_r(&self.rng)
        return np.asanyarray(randoms)

    def random_sample(self, Py_ssize_t n):
        cdef Py_ssize_t i
        cdef double [:] randoms = np.empty(n, dtype=np.double)
        cdef double temp
        for i in range(n):
            randoms[i] = pcg_random_double(&self.rng)

        return np.asanyarray(randoms)

    def random_sample_64(self, Py_ssize_t n):
        cdef Py_ssize_t i
        cdef double [:] randoms = np.empty(n, dtype=np.double)
        cdef double temp
        for i in range(n):
            randoms[i] = pcg_random_double_64(&self.rng64)

        return np.asanyarray(randoms)

    def standard_normal(self, Py_ssize_t n):
        cdef Py_ssize_t i
        cdef double [:] randoms = np.empty(n, dtype=np.double)
        for i in range(n):
            randoms[i] = pcg_random_gauss(&self.rng, &self.has_gauss, &self.gauss)

        return np.asanyarray(randoms)

    def standard_normal_zig(self, Py_ssize_t n):
        cdef Py_ssize_t i
        cdef double [:] randoms = np.empty(n, dtype=np.double)
        for i in range(n):
            randoms[i] = pcg_random_gauss_zig(&self.rng,
                                              &self.shift_zig_random_int,
                                              &self.zig_random_int)

        return np.asanyarray(randoms)

    def standard_normal_64(self, Py_ssize_t n):
        cdef Py_ssize_t i
        cdef double [:] randoms = np.empty(n, dtype=np.double)
        for i in range(n):
            randoms[i] = pcg_random_gauss_64(&self.rng64, &self.has_gauss, &self.gauss)

        return np.asanyarray(randoms)

    def standard_gamma(self, double shape, Py_ssize_t n):
        cdef Py_ssize_t i
        cdef double [:] randoms = np.empty(n, dtype=np.double)
        for i in range(n):
            randoms[i] = pcg_standard_gamma(&self.rng, shape, &self.has_gauss, &self.gauss)

        return np.asanyarray(randoms)

    def standard_exponential(self, Py_ssize_t n):
        cdef Py_ssize_t i
        cdef double [:] randoms = np.empty(n, dtype=np.double)
        for i in range(n):
            randoms[i] = pcg_standard_exponential(&self.rng)

        return np.asanyarray(randoms)

