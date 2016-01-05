from libc.stdint cimport uint32_t, uint64_t, int64_t, int32_t
from wrappers cimport aug_state

cdef extern from "distributions.h":

    cdef uint64_t random_uint64(aug_state* state) nogil
    cdef uint32_t random_uint32(aug_state* state) nogil
    cdef int64_t random_positive_int64(aug_state* state) nogil
    cdef int32_t random_positive_int32(aug_state* state) nogil

    cdef long random_positive_int(aug_state* state) nogil
    cdef unsigned long random_uint(aug_state* state) nogil
    cdef unsigned long random_interval(aug_state* state, unsigned long max) nogil

    cdef void entropy_init(aug_state* state) nogil

    cdef double random_standard_uniform(aug_state* state) nogil
    cdef double random_gauss(aug_state* state) nogil
    cdef double random_gauss_zig(aug_state* state) nogil
    cdef double random_gauss_zig_julia(aug_state* state) nogil
    cdef double random_standard_exponential(aug_state* state) nogil
    cdef double random_standard_cauchy(aug_state* state) nogil

    cdef double random_exponential(aug_state *state, double scale) nogil
    cdef double random_standard_gamma(aug_state* state, double shape) nogil
    cdef double random_pareto(aug_state *state, double a) nogil
    cdef double random_weibull(aug_state *state, double a) nogil
    cdef double random_power(aug_state *state, double a) nogil
    cdef double random_rayleigh(aug_state *state, double mode) nogil
    cdef double random_standard_t(aug_state *state, double df) nogil
    cdef double random_chisquare(aug_state *state, double df) nogil

    cdef double random_normal(aug_state *state, double loc, double scale) nogil
    cdef double random_normal_zig(aug_state *state, double loc, double scale) nogil
    cdef double random_uniform(aug_state *state, double loc, double scale) nogil
    cdef double random_gamma(aug_state *state, double shape, double scale) nogil
    cdef double random_beta(aug_state *state, double a, double b) nogil
    cdef double random_f(aug_state *state, double dfnum, double dfden) nogil
    cdef double random_laplace(aug_state *state, double loc, double scale) nogil
    cdef double random_gumbel(aug_state *state, double loc, double scale) nogil
    cdef double random_logistic(aug_state *state, double loc, double scale) nogil
    cdef double random_lognormal(aug_state *state, double mean, double sigma) nogil
    cdef double random_noncentral_chisquare(aug_state *state, double df, double nonc) nogil
    cdef double random_wald(aug_state *state, double mean, double scale) nogil
    cdef double random_vonmises(aug_state *state, double mu, double kappa) nogil

    cdef double random_noncentral_f(aug_state *state, double dfnum, double dfden, double nonc) nogil
    cdef double random_triangular(aug_state *state, double left, double mode, double right) nogil

    cdef long random_poisson(aug_state *state, double lam) nogil
    cdef long random_negative_binomial(aug_state *state, double n, double p) nogil
    cdef long random_binomial(aug_state *state, double p, long n) nogil
    cdef long random_logseries(aug_state *state, double p) nogil
    cdef long random_geometric(aug_state *state, double p) nogil
    cdef long random_zipf(aug_state *state, double a) nogil
    cdef long random_hypergeometric(aug_state *state, long good, long bad, long sample) nogil
