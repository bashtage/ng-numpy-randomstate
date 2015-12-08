#!python
#cython: wraparound=False, nonecheck=False, boundscheck=False, cdivision=True
import numpy as np
cimport numpy as np
import cython
from libc.stdint cimport uint64_t

cdef extern from "inttypes.h":
    ctypedef unsigned long long __uint128_t

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
    cdef double pcg_random_double(pcg64_random_t *rng)

    cdef double pcg_random_gauss(pcg64_random_t *rng, bint *has_gauss, double *gauss)

    cdef double pcg_standard_exponential(pcg64_random_t *rng)

    cdef double pcg_standard_gamma(pcg64_random_t *rng, double shape, bint *has_gauss, double *gauss)

    cdef double pcg_random_gauss_zig(pcg64_random_t *rng,
                                     int *shift_zig_random_int,
                                     uint64_t *zig_random_int)


cdef class PCGRandomState:
    '''Test class'''

    cdef pcg64_random_t rng
    cdef pcg128_t state, inc

    cdef bint has_gauss
    cdef double gauss

    cdef int shift_zig_random_int
    cdef uint64_t zig_random_int

    def __init__(self, state=None, inc=None):
        if state is not None and inc is not None:
            self.state = state
            self.inc = inc
        else:
            # TODO: Add entropy method to get state and inc
            self.state = 42
            self.inc = 52

        self.shift_zig_random_int = 0
        self.zig_random_int = 0

        self.seed(self.state, self.inc)
        self.has_gauss = 0
        self.gauss = 0.0

    def seed(self, pcg128_t state, pcg128_t inc):
        pcg_setseq_128_srandom_r(&self.rng, self.state, self.inc)

    def get_state(self):
        return 'pcg64', (self.rng.state, self.rng.inc), self.has_gauss, self.gauss

    def advance(self, pcg128_t delta):
        pcg64_advance_r(&self.rng, delta)
        return None

    def set_state(self, tuple state):
        if state[0] != 'pcg64' or len(state) != 4:
            raise ValueError('Must be a pcg64 state')
        self.rng.state = state[1][0]
        self.rng.inc = state[1][1]
        self.has_gauss = state[2]
        self.gauss = state[3]

    def random_integers(self, Py_ssize_t n=1):
        cdef Py_ssize_t i
        cdef uint64_t [:] randoms = np.zeros(n, dtype=np.uint64)
        for i in range(n):
            randoms[i] = pcg64_random_r(&self.rng)
        return np.asanyarray(randoms)

    def random_sample(self, Py_ssize_t n):
        cdef Py_ssize_t i
        cdef double [:] randoms = np.empty(n, dtype=np.double)
        for i in range(n):
            randoms[i] = pcg_random_double(&self.rng)

        return np.asanyarray(randoms)


    def standard_normal_zig(self, Py_ssize_t n):
        cdef Py_ssize_t i
        cdef double [:] randoms = np.empty(n, dtype=np.double)
        for i in range(n):
            randoms[i] = pcg_random_gauss_zig(&self.rng,
                                              &self.shift_zig_random_int,
                                              &self.zig_random_int)

        return np.asanyarray(randoms)

    def standard_normal(self, Py_ssize_t n):
        cdef Py_ssize_t i
        cdef double [:] randoms = np.empty(n, dtype=np.double)
        for i in range(n):
            randoms[i] = pcg_random_gauss(&self.rng, &self.has_gauss, &self.gauss)

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
