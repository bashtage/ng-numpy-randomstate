#!python
#cython: wraparound=False, nonecheck=False, boundscheck=False, cdivision=True
import numpy as np
cimport numpy as np
from libc.stdint cimport uint32_t, uint64_t
from libc.string cimport memcpy

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


cdef class RandomState:
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


    IF RNG_SEED == 1:
        def seed(self, rng_state_t val):
            seed(&self.rng_state, val)
    ELIF RNG_SEED == 2:
        def seed(self, rng_state_t state, rng_state_t inc):
            seed(&self.rng_state, state, inc)

    if RNG_ADVANCEABLE:
        def advance(self, rng_state_t delta):
            advance(&self.rng_state, delta)
            return None

    IF RNG_DUMMY:
        def get_state(self):
            return ('dummy',
                    self.rng_state.rng[0],
                    self.rng_state.has_gauss,
                    self.rng_state.gauss)

        def set_state(self, state):
            if state[0] != 'dummy' or len(state) != 4:
                raise ValueError('Not a dummy RNG state')
            cdef uint32_t val = state[1]
            memcpy(self.rng_state.rng, &val, sizeof(val))
            self.rng_state.has_gauss = state[2]
            self.rng_state.gauss = state[3]

    ELIF RNG_PCG_32 or RNG_PCG_64:
        def get_state(self):
            return ('pcg',
                    (self.rng_state.rng.state, self.rng_state.rng.inc),
                    self.rng_state.has_gauss,
                    self.rng_state.gauss)

        def set_state(self, state):
            if state[0] != 'pcg' or len(state) != 4:
                raise ValueError('Not a PCG RNG state')
            self.rng_state.state = state[1][0]
            self.rng_state.inc = state[1][1]
            self.seed(self.rng_state.state, self.rng_state.inc)
            self.rng_state.has_gauss = state[2]
            self.rng_state.gauss = state[3]
    ELIF RNG_RANDOMKIT:

        def get_state(self):
            cdef uint32_t [:] key = np.zeros(RK_STATE_LEN, dtype=np.uint32)
            cdef Py_ssize_t i
            for i in range(RK_STATE_LEN):
                key[i] = self.rng_state.rng.key[i]
            return ('mt19937',
                    (np.asanyarray(key), self.rng_state.rng.pos),
                    self.rng_state.has_gauss,
                    self.rng_state.gauss)

        def set_state(self, state):
            if state[0] != 'mt19937' or len(state) != 4:
                raise ValueError('Not a mt19937 RNG state')

            cdef uint32_t [:] key = state[1][0]
            cdef Py_ssize_t i
            for i in range(RK_STATE_LEN):
                 self.rng_state.rng.key[i] = key[i]
            self.rng_state.rng.pos = state[1][1]
            self.rng_state.has_gauss = state[2]
            self.rng_state.gauss = state[3]




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
