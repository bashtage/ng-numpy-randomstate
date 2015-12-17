#!python
#cython: wraparound=False, nonecheck=False, boundscheck=False, cdivision=True
import numpy as np
cimport numpy as np
from libc.stdint cimport uint32_t, uint64_t, int64_t
from libc.string cimport memcpy

include "config.pxi"

IF RNG_PCG_32:
    include "shims/pcg-32/pcg-32.pxi"
IF RNG_MT19337:
    include "shims/random-kit/random-kit.pxi"
IF RNG_XORSHIFT128:
    include "shims/xorshift128/xorshift128.pxi"
IF RNG_XORSHIFT1024:
    include "shims/xorshift1024/xorshift1024.pxi"

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
        self.rng_state.has_gauss = 0
        self.rng_state.gauss = 0.0
        if state is None:
            entropy_init(&self.rng_state)
        else:
            IF RNG_PCG_32 or RNG_PCG_64 or RNG_XORSHIFT128:
                self.seed(state, inc)
            ELIF RNG_DUMMY or RNG_MT19337:
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

    def get_state(self):
        return (RNG_NAME,
                _get_state(self.rng_state),
                self.rng_state.has_gauss,
                self.rng_state.gauss)

    IF RNG_PCG_32:
        def set_state(self, state):
            if state[0] != 'pcg' or len(state) != 4:
                raise ValueError('Not a PCG RNG state')
            self.rng_state.rng.state = state[1][0]
            self.rng_state.rng.inc = state[1][1]
            self.rng_state.has_gauss = state[2]
            self.rng_state.gauss = state[3]

    IF RNG_MT19337:
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
        def set_state(self, state):
            if state[0] != 'xorshift128' or len(state) != 4:
                raise ValueError('Not a XorShift128 RNG state')
            self.rng_state.rng.s[0] = state[1][0]
            self.rng_state.rng.s[1] = state[1][1]
            self.rng_state.has_gauss = state[2]
            self.rng_state.gauss = state[3]


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

    def random_integers(self, size=None):
        return uint0(&self.rng_state, &random_uint64, size)

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
