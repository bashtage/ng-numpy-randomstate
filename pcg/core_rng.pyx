#!python
#cython: wraparound=False, nonecheck=False, boundscheck=False, cdivision=True
import numpy as np
cimport numpy as np
from libc.stdint cimport uint32_t, uint64_t, int64_t, int32_t
from libc.string cimport memcpy

include "config.pxi"

IF RNG_PCG_32:
    include "shims/pcg-32/pcg-32.pxi"
IF RNG_PCG_64:
    include "shims/pcg-64/pcg-64.pxi"
IF RNG_MT19937:
    include "shims/random-kit/random-kit.pxi"
IF RNG_XORSHIFT128:
    include "shims/xorshift128/xorshift128.pxi"
IF RNG_XORSHIFT1024:
    include "shims/xorshift1024/xorshift1024.pxi"
IF RNG_MRG32K3A:
    include "shims/mrg32k3a/mrg32k3a.pxi"

cdef extern from "core-rng.h":

    cdef uint64_t random_uint64(aug_state* state)
    cdef uint32_t random_uint32(aug_state* state)
    cdef uint64_t random_bounded_uint64(aug_state* state, uint64_t bound)
    cdef uint32_t random_bounded_uint32(aug_state* state, uint32_t bound)
    cdef int64_t random_bounded_int64(aug_state* state, int64_t low, int64_t high)
    cdef int32_t random_bounded_int32(aug_state* state, int32_t low, int32_t high)

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

include "wrappers.pxi"

cdef class RandomState:
    '''Test class'''

    cdef rng_t rng
    cdef aug_state rng_state

    IF RNG_SEED==1:
        def __init__(self, seed=None):
            self.rng_state.rng = &self.rng
            self.rng_state.has_gauss = 0
            self.rng_state.gauss = 0.0
            if seed is not None:
                self.seed(seed)
            else:
                entropy_init(&self.rng_state)
    ELSE:
        def __init__(self, seed=None, inc=None):
            self.rng_state.rng = &self.rng
            self.rng_state.has_gauss = 0
            self.rng_state.gauss = 0.0
            if seed is not None and inc is not None:
                self.seed(seed, inc)
            else:
                entropy_init(&self.rng_state)

    IF RNG_SEED==1:
        def seed(self, rng_state_t val):
            seed(&self.rng_state, val)
    ELSE:
        def seed(self, rng_state_t val, rng_state_t inc):
            seed(&self.rng_state, val, inc)

    if RNG_ADVANCEABLE:
        def advance(self, rng_state_t delta):
            advance(&self.rng_state, delta)
            self.rng_state.has_gauss = 0
            self.rng_state.gauss = 0.0
            return None

    if RNG_JUMPABLE:
        def jump(self, uint32_t iter = 1):
            """
            jump(iter = 1)

            Jumps the random number generator by a pre-specified skip.  The size of the jump is
            rng-specific.

            Parameters
            ----------
            iter : integer, positive
                Number of times to jump the state of the rng.

            Notes
            -----
            Jumping the rng state resets any pre-computed random numbers. This is required to ensure
            exact reproducibility.
            """
            cdef Py_ssize_t i;
            for i in range(iter):
                jump(&self.rng_state)
            self.rng_state.has_gauss = 0
            self.rng_state.gauss = 0.0
            return None

    def get_state(self):
        return (RNG_NAME,
                _get_state(self.rng_state),
                (self.rng_state.has_gauss, self.rng_state.gauss),
                (self.rng_state.has_uint32, self.rng_state.uinteger)
                )

    def set_state(self, state):
        rng_name = RNG_NAME
        if state[0] != rng_name or len(state) != RNG_STATE_LEN:
            raise ValueError('Not a ' + rng_name + ' RNG state')
        _set_state(self.rng_state, state[1])
        self.rng_state.has_gauss = state[2][0]
        self.rng_state.gauss = state[2][1]
        self.rng_state.has_uint32 = state[3][0]
        self.rng_state.uinteger = state[3][1]

    def random_sample(self, size=None):
        """
        random_sample(size=None)

        Return random floats in the half-open interval [0.0, 1.0).

        Results are from the "continuous uniform" distribution over the
        stated interval.  To sample :math:`Unif[a, b), b > a` multiply
        the output of `random_sample` by `(b-a)` and add `a`::

          (b - a) * random_sample() + a

        Parameters
        ----------
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        Returns
        -------
        out : float or ndarray of floats
            Array of random floats of shape `size` (unless ``size=None``, in which
            case a single float is returned).

        Examples
        --------
        >>> np.random.random_sample()
        0.47108547995356098
        >>> type(np.random.random_sample())
        <type 'float'>
        >>> np.random.random_sample((5,))
        array([ 0.30220482,  0.86820401,  0.1654503 ,  0.11659149,  0.54323428])

        Three-by-two array of random numbers from [-5, 0):

        >>> 5 * np.random.random_sample((3, 2)) - 5
        array([[-3.99149989, -0.52338984],
               [-2.99091858, -0.79479508],
               [-1.23204345, -1.75224494]])

        """
        return cont0(&self.rng_state, &random_double, size)

    def random_integers(self, size=None, int bits=64):

        if bits == 64:
            return uint0(&self.rng_state, &random_uint64, size)
        elif bits == 32:
            return uint0_32(&self.rng_state, &random_uint32, size)
        else:
            raise ValueError('Unknown value of bits.  Must be either 32 or 64.')

    def random_bounded_uintegers(self, high, size=None):
        if high < 4294967295:
            return uint1_i_32(&self.rng_state, &random_bounded_uint32, <uint32_t>range, size)
        else:
            return uint1_i(&self.rng_state, &random_bounded_uint64, range, size)

    def random_bounded_integers(self, int64_t low, high=None, size=None):
        cdef int64_t _high
        if high is None:
            _high = low
            low = 0
        else:
            _high = high
        if _high < 4294967295:
            return int2_i_32(&self.rng_state, &random_bounded_int32, <int32_t>low, <int32_t>high, size)
        else:
            return int2_i(&self.rng_state, &random_bounded_int64, low, high, size)


    def standard_normal(self, size=None, method='inv'):
        """
        standard_normal(size=None, method='inv')

        Draw samples from a standard Normal distribution (mean=0, stdev=1).

        Parameters
        ----------
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.
        method : str
            Either 'inv' or 'zig'. 'inv' uses the default FIXME method.  'zig' uses
            the much faster ziggurat method of FIXME.

        Returns
        -------
        out : float or ndarray
            Drawn samples.

        Examples
        --------
        >>> s = np.random.standard_normal(8000)
        >>> s
        array([ 0.6888893 ,  0.78096262, -0.89086505, ...,  0.49876311, #random
               -0.38672696, -0.4685006 ])                               #random
        >>> s.shape
        (8000,)
        >>> s = np.random.standard_normal(size=(3, 4, 2))
        >>> s.shape
        (3, 4, 2)

        """
        if method == 'inv':
            return cont0(&self.rng_state, &random_gauss, size)
        else:
            return cont0(&self.rng_state, &random_gauss_zig, size)

    def standard_exponential(self, size=None):
        return cont0(&self.rng_state, &random_standard_exponential, size)

    def standard_cauchy(self, size=None):
        """
        standard_cauchy(size=None)

        Draw samples from a standard Cauchy distribution with mode = 0.

        Also known as the Lorentz distribution.

        Parameters
        ----------
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        Returns
        -------
        samples : ndarray or scalar
            The drawn samples.

        Notes
        -----
        The probability density function for the full Cauchy distribution is

        .. math:: P(x; x_0, \\gamma) = \\frac{1}{\\pi \\gamma \\bigl[ 1+
                  (\\frac{x-x_0}{\\gamma})^2 \\bigr] }

        and the Standard Cauchy distribution just sets :math:`x_0=0` and
        :math:`\\gamma=1`

        The Cauchy distribution arises in the solution to the driven harmonic
        oscillator problem, and also describes spectral line broadening. It
        also describes the distribution of values at which a line tilted at
        a random angle will cut the x axis.

        When studying hypothesis tests that assume normality, seeing how the
        tests perform on data from a Cauchy distribution is a good indicator of
        their sensitivity to a heavy-tailed distribution, since the Cauchy looks
        very much like a Gaussian distribution, but with heavier tails.

        References
        ----------
        .. [1] NIST/SEMATECH e-Handbook of Statistical Methods, "Cauchy
              Distribution",
              http://www.itl.nist.gov/div898/handbook/eda/section3/eda3663.htm
        .. [2] Weisstein, Eric W. "Cauchy Distribution." From MathWorld--A
              Wolfram Web Resource.
              http://mathworld.wolfram.com/CauchyDistribution.html
        .. [3] Wikipedia, "Cauchy distribution"
              http://en.wikipedia.org/wiki/Cauchy_distribution

        Examples
        --------
        Draw samples and plot the distribution:

        >>> s = np.random.standard_cauchy(1000000)
        >>> s = s[(s>-25) & (s<25)]  # truncate distribution so it plots well
        >>> plt.hist(s, bins=100)
        >>> plt.show()

        """
        return cont0(&self.rng_state, &random_standard_cauchy, size)

    def standard_gamma(self, double shape, size=None):
        return cont1(&self.rng_state, &random_standard_gamma, shape, size)
