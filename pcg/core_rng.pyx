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
    cdef double random_gauss(aug_state* state)
    cdef double random_standard_exponential(aug_state* state)
    cdef double random_standard_cauchy(aug_state* state)

    cdef double random_standard_gamma(aug_state* state, double shape)



ctypedef double (* random_double_0)(aug_state* state)
ctypedef double (* random_double_1)(aug_state* state, double a)
ctypedef double (* random_double_2)(aug_state* state, double a, double b)
ctypedef double (* random_double_3)(aug_state* state, double a, double b, double c)

ctypedef uint64_t (* random_uint_0)(aug_state* state)
ctypedef uint64_t (* random_uint_1)(aug_state* state, uint64_t a)

cdef object cont0(aug_state* state, random_double_0 func, size):
    if size is None:
        return func(state)
    cdef Py_ssize_t n = compute_numel(size)
    cdef double [:] randoms = np.empty(n, np.double)
    for i in range(n):
        randoms[i] = func(state)
    return np.asanyarray(randoms).reshape(size)

cdef object uint0(aug_state* state, random_uint_0 func, size):
    if size is None:
        return func(state)
    cdef Py_ssize_t n = compute_numel(size)
    cdef uint64_t [:] randoms = np.empty(n, np.uint64)
    for i in range(n):
        randoms[i] = func(state)
    return np.asanyarray(randoms).reshape(size)

cdef Py_ssize_t compute_numel(size):
    cdef Py_ssize_t i, n = 1
    if isinstance(size, tuple):
        for i in range(len(size)):
            n *= size[i]
    else:
        n = size
    return n


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
            self.rng_state.rng.state = state[1][0]
            self.rng_state.rng.inc = state[1][1]
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


    def random_sample(self, size=None):
        return cont0(&self.rng_state, &random_double, size)

    def random_integers(self, size=None):
        return uint0(&self.rng_state, &random_uint64, size)

    def standard_normal(self, size=None):
        return cont0(&self.rng_state, &random_gauss, size)

    def standard_gamma(self, double shape, size=None):
        if size is None:
            return random_standard_gamma(&self.rng_state, shape)

        cdef Py_ssize_t i, n = compute_numel(size)
        cdef double [:] randoms = np.zeros(n, dtype=np.double)
        for i in range(n):
            randoms[i] = random_standard_gamma(&self.rng_state, shape)
        return np.asanyarray(randoms).reshape(size)

    def standard_exponential(self, size=None):
        return cont0(&self.rng_state, &random_standard_exponential, size)

    def standard_cauchy(self, size=None):
        return cont0(&self.rng_state, &random_standard_cauchy, size)
