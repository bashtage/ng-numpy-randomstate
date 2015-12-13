#!python
#cython: wraparound=False, nonecheck=False, boundscheck=False, cdivision=True
import numpy as np
cimport numpy as np
from libc.stdint cimport uint32_t, uint64_t

include "config.pxi"

IF RNG_PCG_32:
    include "shims/pcg-32/pcg-32.pxi"
IF RNG_PCG_64:
    include "shims/pcg-64/pcg-64.pxi"
IF RNG_DUMMY:
    include "shims/dummy/dummy.pxi"
IF RNG_RANDOMKIT:
    include "shims/random-kit/random-kit.pxi"

cdef extern from "core-rng.h":

    cdef uint64_t random_uint64(aug_state* state)

    cdef double random_double(aug_state* state)


cdef class PCGRandomState:
    '''Test class'''

    cdef rng_t rng
    cdef aug_state rng_state

    def __init__(self, state=None, inc=None):
        self.rng_state.rng = &self.rng
        IF RNG_PCG_32 or RNG_PCG_64:
            self.rng_state.state = 42
            self.rng_state.inc = 52

        self.rng_state.has_gauss = 0
        self.rng_state.gauss = 0.0
        IF RNG_PCG_32 or RNG_PCG_64:
            self.seed(self.rng_state.state, self.rng_state.inc)
        ELIF RNG_DUMMY or RNG_RANDOMKIT:
            self.seed(0)

    IF RNG_PCG_32 or RNG_PCG_64:
        def seed(self, rng_state_t state, rng_state_t inc):
            seed(&self.rng_state, state, inc)
    ELIF RNG_DUMMY or RNG_RANDOMKIT:
        def seed(self, rng_state_t inc):
            seed(&self.rng_state, inc)

    def random_sample(self, Py_ssize_t n=1):
        if n == 1:
            return random_double(&self.rng_state)

        cdef Py_ssize_t i
        cdef double [:] randoms = np.zeros(n, dtype=np.double)
        for i in range(n):
            randoms[i] = random_double(&self.rng_state)
        return np.asanyarray(randoms)

    def random_integers(self, Py_ssize_t n=1):
        if n == 1:
            return random_uint64(&self.rng_state)

        cdef Py_ssize_t i
        cdef uint64_t [:] randoms = np.zeros(n, dtype=np.uint64)
        for i in range(n):
            randoms[i] = random_uint64(&self.rng_state)
        return np.asanyarray(randoms)
