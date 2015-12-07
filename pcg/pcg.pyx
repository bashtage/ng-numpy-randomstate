#!python
#cython: wraparound=False, nonecheck=False, boundscheck=False, cdivision=True
import numpy as np
cimport numpy as np
import cython
from libc.stdint cimport uint32_t, uint64_t

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


cdef extern from "pcg_helper.c":
    cdef double pcg_random_double(pcg32_random_t *rng)

    cdef double pcg_random_gauss(pcg32_random_t *rng, bint *has_gauss, double *gauss)


cdef class PCGRandomState:
    '''Test class'''
    cdef pcg32_random_t rng
    cdef uint64_t state, inc
    cdef bint has_gauss
    cdef double gauss

    def __init__(self, state=None, inc=None):
        if state is not None and inc is not None:
            self.state = state
            self.inc = inc
        else:
            # TODO: Add other method to get state and inc
            self.state = 42
            self.inc = 52

        self.seed(self.state, self.inc)
        self.has_gauss = 0
        self.gauss = 0.0

    def seed(self, uint64_t state, uint64_t inc):
        pcg_setseq_64_srandom_r(&self.rng, state, inc)

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

    def standard_normal(self, Py_ssize_t n):
        cdef Py_ssize_t i
        cdef double [:] randoms = np.empty(n, dtype=np.double)
        cdef double temp
        for i in range(n):
            randoms[i] = pcg_random_gauss(&self.rng, &self.has_gauss, &self.gauss)

        return np.asanyarray(randoms)

