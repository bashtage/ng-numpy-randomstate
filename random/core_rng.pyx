#!python
#cython: wraparound=False, nonecheck=False, boundscheck=False, cdivision=True
import numpy as np
cimport numpy as np
from libc.stdint cimport uint32_t, uint64_t, int64_t, int32_t
try:
    from threading import Lock
except:
    from dummy_threading import Lock


include "config.pxi"

IF RNG_PCG32:
    include "shims/pcg-32/pcg-32.pxi"
IF RNG_PCG64:
    include "shims/pcg-64/pcg-64.pxi"
IF RNG_MT19937:
    include "shims/random-kit/random-kit.pxi"
IF RNG_XORSHIFT128:
    include "shims/xorshift128/xorshift128.pxi"
IF RNG_XORSHIFT1024:
    include "shims/xorshift1024/xorshift1024.pxi"
IF RNG_MRG32K3A:
    include "shims/mrg32k3a/mrg32k3a.pxi"
IF RNG_MLFG_1279_861:
    include "shims/mlfg-1279-861/mlfg-1279-861.pxi"
IF RNG_DUMMY:
    include "shims/dummy/dummy.pxi"

cdef extern from "core-rng.h":

    cdef uint64_t random_uint64(aug_state* state) nogil
    cdef uint32_t random_uint32(aug_state* state) nogil
    cdef uint64_t random_bounded_uint64(aug_state* state, uint64_t bound) nogil
    cdef uint32_t random_bounded_uint32(aug_state* state, uint32_t bound) nogil
    cdef int64_t random_bounded_int64(aug_state* state, int64_t low, int64_t high) nogil
    cdef int32_t random_bounded_int32(aug_state* state, int32_t low, int32_t high) nogil

    cdef void entropy_init(aug_state* state) nogil

    cdef double random_uniform(aug_state* state) nogil
    cdef double random_gauss(aug_state* state) nogil
    cdef double random_gauss_zig(aug_state* state) nogil
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
    cdef double random_scaled_uniform(aug_state *state, double loc, double scale) nogil
    cdef double random_gamma(aug_state *state, double shape, double scale) nogil
    cdef double random_beta(aug_state *state, double a, double b) nogil
    cdef double random_f(aug_state *state, double dfnum, double dfden) nogil
    cdef double random_laplace(aug_state *state, double loc, double scale) nogil
    cdef double random_gumbel(aug_state *state, double loc, double scale) nogil
    cdef double random_logistic(aug_state *state, double loc, double scale) nogil
    cdef double random_lognormal(aug_state *state, double mean, double sigma) nogil

    cdef long random_poisson(aug_state *state, double lam) nogil
    cdef long rk_negative_binomial(aug_state *state, double n, double p) nogil

include "wrappers.pxi"

cdef class RandomState:
    CLASS_DOCSTRING

    cdef rng_t rng
    cdef aug_state rng_state
    cdef object lock

    IF RNG_SEED==1:
        def __init__(self, seed=None):
            self.rng_state.rng = &self.rng
            self.rng_state.has_gauss = 0
            self.rng_state.gauss = 0.0
            self.lock = Lock()
            if seed is not None:
                self.seed(seed)
            else:
                entropy_init(&self.rng_state)

    ELSE:
        def __init__(self, seed=None, inc=None):
            self.rng_state.rng = &self.rng
            self.rng_state.has_gauss = 0
            self.rng_state.gauss = 0.0
            self.lock = Lock()
            if seed is not None and inc is not None:
                self.seed(seed, inc)
            else:
                entropy_init(&self.rng_state)

    IF RNG_SEED==1:
        def seed(self, val=None):
            """
            seed(seed=None)

            Seed the generator.

            This method is called when `RandomState` is initialized. It can be
            called again to re-seed the generator. For details, see `RandomState`.

            Parameters
            ----------
            val : int, optional
                Seed for `RandomState`.

            Notes
            -----
            Acceptable range for seed depends on specifics of PRNG.  See
            class documentation for details.

            Seeds are hashed to produce the required number of bits.

            See Also
            --------
            RandomState

            """
            if val is not None:
                seed(&self.rng_state, val)
            else:
                entropy_init(&self.rng_state)
    ELSE:
        def seed(self, val=None, inc=None):
            """
            seed(val=None, inc=None)

            Seed the generator.

            This method is called when `RandomState` is initialized. It can be
            called again to re-seed the generator. For details, see `RandomState`.

            Parameters
            ----------
            val : int, optional
                Seed for `RandomState`.
            inc : int, optional
                Increment to use for producing multiple streams

            See Also
            --------
            RandomState
            """
            if val is not None and inc is not None:
                seed(&self.rng_state, val, inc)
            else:
                entropy_init(&self.rng_state)

    if RNG_ADVANCEABLE:
        def advance(self, rng_state_t delta):
            """
            advance(delta)

            Advance the PRNG as-if delta drawn have occurred.

            Parameters
            ----------
            delta : integer, positive
                Number of draws to advance the PRNG.

            Returns
            -------
            out : None
                Returns 'None' on success.

            Notes
            -----
            Advancing the prng state resets any pre-computed random numbers.
            This is required to ensure exact reproducibility.
            """
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

            Returns
            -------
            out : None
                Returns 'None' on success.

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
        """
        get_state()

        Return a tuple representing the internal state of the generator.

        For more details, see `set_state`.

        Returns
        -------
        out : tuple(str, tuple, tuple, tuple)
            The returned tuple has the following items:

            1. the string containing the PRNG type.
            2. a tuple containing the PRNG-specific state
            3. a tuple containing two values :``has_gauss`` and ``cached_gaussian``
            4. a tuple containing two values :``has_uint32`` and ``cached_uint32``

        See Also
        --------
        set_state

        Notes
        -----
        `set_state` and `get_state` are not needed to work with any of the
        random distributions in NumPy. If the internal state is manually altered,
        the user should know exactly what he/she is doing.

        For information about the specific structure of the PRNG-specific
        component, see the class documentation.
        """
        return (RNG_NAME,
                _get_state(self.rng_state),
                (self.rng_state.has_gauss, self.rng_state.gauss),
                (self.rng_state.has_uint32, self.rng_state.uinteger))

    def set_state(self, state):
        """
        set_state(state)

        Set the internal state of the generator from a tuple.

        For use if one has reason to manually (re-)set the internal state of the
        pseudo-random number generating algorithm.

        Parameters
        ----------
        state : tuple(str, tuple, tuple, tuple)
            The returned tuple has the following items:

            1. the string containing the PRNG type.
            2. a tuple containing the PRNG-specific state
            3. a tuple containing two values :``has_gauss`` and ``cached_gaussian``
            4. a tuple containing two values :``has_uint32`` and ``cached_uint32``

        Returns
        -------
        out : None
            Returns 'None' on success.

        See Also
        --------
        get_state

        Notes
        -----
        `set_state` and `get_state` are not needed to work with any of the
        random distributions in NumPy. If the internal state is manually altered,
        the user should know exactly what he/she is doing.

        For information about the specific structure of the PRNG-specific
        component, see the class documentation.
        """
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
        return cont0(&self.rng_state, &random_uniform, size, self.lock)

    def random_uintegers(self, size=None, int bits=64):
        """
        random_uintegers(size=None, bits=64)

        Return random unsigned integers

        Parameters
        ----------
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.
        bits : int {32, 64}
            Size of the unsigned integer to return, either 32 bit or 64 bit.

        Returns
        -------
        out : uint or ndarray
            Drawn samples.

        Notes
        -----
        This method effectively exposes access to the raw underlying
        pseudo-random number generator since these all produce unsigned
        integers. In practice these are most useful for generating other
        random numbers.

        These should not be used to produce bounded random numbers by
        simple truncation.  Instead see ``random_bounded_integers``.
        """
        if bits == 64:
            return uint0(&self.rng_state, &random_uint64, size, self.lock)
        elif bits == 32:
            return uint0_32(&self.rng_state, &random_uint32, size, self.lock)
        else:
            raise ValueError('Unknown value of bits.  Must be either 32 or 64.')

    def random_bounded_uintegers(self, uint64_t high, size=None):
        if high < 4294967295:
            return uint1_i_32(&self.rng_state, &random_bounded_uint32, high, size, self.lock)
        else:
            return uint1_i(&self.rng_state, &random_bounded_uint64, high, size, self.lock)

    def random_bounded_integers(self, int64_t low, high=None, size=None):
        cdef int64_t _low, _high
        if high is None:
            _high = low
            _low = 0
        else:
            _high = high
            _low = low

        if _low >= -2147483648 and _high <= 2147483647:
            return int2_i_32(&self.rng_state, &random_bounded_int32,
                             <int32_t>_low, <int32_t>_high, size, self.lock)
        else:
            return int2_i(&self.rng_state, &random_bounded_int64, _low, _high, size, self.lock)


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
            return cont0(&self.rng_state, &random_gauss, size, self.lock)
        else:
            return cont0(&self.rng_state, &random_gauss_zig, size, self.lock)

    def standard_exponential(self, size=None):
        """
        standard_exponential(size=None)

        Draw samples from the standard exponential distribution.

        `standard_exponential` is identical to the exponential distribution
        with a scale parameter of 1.

        Parameters
        ----------
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        Returns
        -------
        out : float or ndarray
            Drawn samples.

        Examples
        --------
        Output a 3x8000 array:

        >>> n = np.random.standard_exponential((3, 8000))

        """
        return cont0(&self.rng_state, &random_standard_exponential, size, self.lock)

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
        return cont0(&self.rng_state, &random_standard_cauchy, size, self.lock)

    def standard_gamma(self, double shape, size=None):
        """
        standard_gamma(shape, size=None)

        Draw samples from a standard Gamma distribution.

        Samples are drawn from a Gamma distribution with specified parameters,
        shape (sometimes designated "k") and scale=1.

        Parameters
        ----------
        shape : float
            Parameter, should be > 0.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        Returns
        -------
        samples : ndarray or scalar
            The drawn samples.

        See Also
        --------
        scipy.stats.distributions.gamma : probability density function,
            distribution or cumulative density function, etc.

        Notes
        -----
        The probability density for the Gamma distribution is

        .. math:: p(x) = x^{k-1}\\frac{e^{-x/\\theta}}{\\theta^k\\Gamma(k)},

        where :math:`k` is the shape and :math:`\\theta` the scale,
        and :math:`\\Gamma` is the Gamma function.

        The Gamma distribution is often used to model the times to failure of
        electronic components, and arises naturally in processes for which the
        waiting times between Poisson distributed events are relevant.

        References
        ----------
        .. [1] Weisstein, Eric W. "Gamma Distribution." From MathWorld--A
               Wolfram Web Resource.
               http://mathworld.wolfram.com/GammaDistribution.html
        .. [2] Wikipedia, "Gamma-distribution",
               http://en.wikipedia.org/wiki/Gamma-distribution

        Examples
        --------
        Draw samples from the distribution:

        >>> shape, scale = 2., 1. # mean and width
        >>> s = np.random.standard_gamma(shape, 1000000)

        Display the histogram of the samples, along with
        the probability density function:

        >>> import matplotlib.pyplot as plt
        >>> import scipy.special as sps
        >>> count, bins, ignored = plt.hist(s, 50, normed=True)
        >>> y = bins**(shape-1) * ((np.exp(-bins/scale))/ \\
        ...                       (sps.gamma(shape) * scale**shape))
        >>> plt.plot(bins, y, linewidth=2, color='r')
        >>> plt.show()

        """
        return cont1(&self.rng_state, &random_standard_gamma, shape, size, self.lock)
