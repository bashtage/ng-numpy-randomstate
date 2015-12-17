#!python
#cython: wraparound=False, nonecheck=False, boundscheck=False, cdivision=True
import numpy as np
cimport numpy as np
from libc.stdint cimport uint32_t, uint64_t, int64_t
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
IF RNG_XORSHIFT128:
    include "shims/xorshift128/xorshift128.pxi"
IF RNG_XORSHIFT1024:
    include "shims/xorshift1024/xorshift1024.pxi"
IF RNG_MRG32K3A:
    include "shims/mrg32k3a/mrg32k3a.pxi"

cdef extern from "core-rng.h":

    cdef uint64_t random_uint64(aug_state* state)
    cdef void entropy_init(aug_state* state)

    cdef double random_double(aug_state* state)
    cdef double random_gauss(aug_state* state)
    cdef double random_gauss_zig(aug_state* state)
    cdef double random_standard_exponential(aug_state* state)
    cdef double random_standard_cauchy(aug_state* state)

    cdef double random_exponential(aug_state *state, double scale)
    cdef double random_standard_gamma(aug_state* state, double shape)
    cdef double random_pareto(aug_state *state, double a)
    cdef double random_weibull(aug_state *state, double a)
    cdef double random_power(aug_state *state, double a)
    cdef double random_rayleigh(aug_state *state, double mode)
    cdef double random_standard_t(aug_state *state, double df)
    cdef double random_chisquare(aug_state *state, double df)

    cdef double random_normal(aug_state *state, double loc, double scale)
    cdef double random_uniform(aug_state *state, double loc, double scale)
    cdef double random_gamma(aug_state *state, double shape, double scale)
    cdef double random_beta(aug_state *state, double a, double b)
    cdef double random_f(aug_state *state, double dfnum, double dfden)
    cdef double random_laplace(aug_state *state, double loc, double scale)
    cdef double random_gumbel(aug_state *state, double loc, double scale)
    cdef double random_logistic(aug_state *state, double loc, double scale)
    cdef double random_lognormal(aug_state *state, double mean, double sigma)

    cdef long random_poisson(aug_state *state, double lam)
    cdef long rk_negative_binomial(aug_state *state, double n, double p)


ctypedef double (* random_double_0)(aug_state* state)
ctypedef double (* random_double_1)(aug_state* state, double a)
ctypedef double (* random_double_2)(aug_state* state, double a, double b)
ctypedef double (* random_double_3)(aug_state* state, double a, double b, double c)

ctypedef uint64_t (* random_uint_0)(aug_state* state)
ctypedef uint64_t (* random_uint_1)(aug_state* state, double a)
ctypedef uint64_t (* random_uint_2)(aug_state* state, double a, double b)

include "wrappers.pxi"

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
        if state is None:
            entropy_init(&self.rng_state)
        else:
            IF RNG_PCG_32 or RNG_PCG_64 or RNG_XORSHIFT128:
                self.seed(state, inc)
            ELIF RNG_DUMMY or RNG_RANDOMKIT:
                self.seed(state)

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

    ELIF RNG_XORSHIFT128:
        def get_state(self):
            return ('xorshift128',
                    (self.rng_state.rng.s[0], self.rng_state.rng.s[1]),
                    self.rng_state.has_gauss,
                    self.rng_state.gauss)

        def set_state(self, state):
            if state[0] != 'xorshift128' or len(state) != 4:
                raise ValueError('Not a XorShift128 RNG state')
            self.rng_state.rng.s[0] = state[1][0]
            self.rng_state.rng.s[1] = state[1][1]
            self.rng_state.has_gauss = state[2]
            self.rng_state.gauss = state[3]


    def random_sample(self, size=None):
        return cont0(&self.rng_state, &random_double, size)

    def random_integers(self, size=None):
        return uint0(&self.rng_state, &random_uint64, size)

    def standard_normal(self, size=None, method='inv'):
        if method == 'inv':
            return cont0(&self.rng_state, &random_gauss, size)
        else:
            return cont0(&self.rng_state, &random_gauss_zig, size)

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

    def standard_gamma(self, double shape, size=None):
        return cont1(&self.rng_state, &random_standard_gamma, shape, size)

    def pareto(self, shape, size=None):
        return cont1(&self.rng_state, &random_pareto, shape, size)

    def weibull(self, shape, size=None):
        return cont1(&self.rng_state, &random_weibull, shape, size)

    def power(self, shape, size=None):
        return cont1(&self.rng_state, &random_power, shape, size)

    def rayleigh(self, shape, size=None):
        return cont1(&self.rng_state, &random_rayleigh, shape, size)

    def standard_t(self, shape, size=None):
        return cont1(&self.rng_state, &random_standard_t, shape, size)

    def chisquare(self, shape, size=None):
        return cont1(&self.rng_state, &random_chisquare, shape, size)

    def normal(self, loc, scale, size=None):
        return cont2(&self.rng_state, &random_normal, loc, scale, size)

    def uniform(self, loc, scale, size=None):
        return cont2(&self.rng_state, &random_uniform, loc, scale, size)

    def gamma(self, loc, scale, size=None):
        return cont2(&self.rng_state, &random_gamma, loc, scale, size)

    def beta(self, loc, scale, size=None):
        return cont2(&self.rng_state, &random_beta, loc, scale, size)

    def f(self, loc, scale, size=None):
        return cont2(&self.rng_state, &random_f, loc, scale, size)

    def laplace(self, loc, scale, size=None):
        return cont2(&self.rng_state, &random_laplace, loc, scale, size)

    def gumbel(self, loc, scale, size=None):
        return cont2(&self.rng_state, &random_gumbel, loc, scale, size)

    def lognormal(self, loc, scale, size=None):
        return cont2(&self.rng_state, &random_lognormal, loc, scale, size)
