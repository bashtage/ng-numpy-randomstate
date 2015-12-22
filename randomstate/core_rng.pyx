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
include "src/common/binomial.pxi"

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

    cdef double random_sample(aug_state* state) nogil
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
    cdef double random_uniform(aug_state *state, double loc, double scale) nogil
    cdef double random_gamma(aug_state *state, double shape, double scale) nogil
    cdef double random_beta(aug_state *state, double a, double b) nogil
    cdef double random_f(aug_state *state, double dfnum, double dfden) nogil
    cdef double random_laplace(aug_state *state, double loc, double scale) nogil
    cdef double random_gumbel(aug_state *state, double loc, double scale) nogil
    cdef double random_logistic(aug_state *state, double loc, double scale) nogil
    cdef double random_lognormal(aug_state *state, double mean, double sigma) nogil

    cdef long random_poisson(aug_state *state, double lam) nogil
    cdef long random_negative_binomial(aug_state *state, double n, double p) nogil
    cdef long random_binomial(aug_state *state, long n, double p) nogil

include "wrappers.pxi"

cdef class RandomState:
    CLASS_DOCSTRING

    cdef binomial_t binomial_info
    cdef rng_t rng
    cdef aug_state rng_state
    cdef object lock

    IF RNG_SEED==1:
        def __init__(self, seed=None):
            self.rng_state.rng = &self.rng
            self.rng_state.binomial = &self.binomial_info
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
            self.rng_state.binomial = &self.binomial_info
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
        return  {'name': RNG_NAME,
                 'state': _get_state(self.rng_state),
                 'gauss': {'has_gauss': self.rng_state.has_gauss, 'gauss': self.rng_state.gauss},
                 'uint32': {'has_uint32': self.rng_state.has_uint32, 'uint32': self.rng_state.uinteger}
                 }

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
        if RNG_MT19937:
            if isinstance(state, tuple):
                if state[0] != 'MT19937':
                    raise ValueError('Not a ' + rng_name + ' RNG state')
                _set_state(self.rng_state, (state[1], state[2]))
                self.rng_state.has_gauss = state[3]
                self.rng_state.gauss = state[4]
                self.rng_state.has_uint32 = 0
                self.rng_state.uinteger = 0
                return None

        if state['name'] != rng_name:
            raise ValueError('Not a ' + rng_name + ' RNG state')
        _set_state(self.rng_state, state['state'])
        self.rng_state.has_gauss = state['gauss']['has_gauss']
        self.rng_state.gauss = state['gauss']['gauss']
        self.rng_state.has_uint32 = state['uint32']['has_uint32']
        self.rng_state.uinteger = state['uint32']['uint32']

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
        return cont0(&self.rng_state, &random_sample, size, self.lock)

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


    def binomial(self, uint64_t n, double p, size=None):
        """
        binomial(n, p, size=None)
        Draw samples from a binomial distribution.
        Samples are drawn from a binomial distribution with specified
        parameters, n trials and p probability of success where
        n an integer >= 0 and p is in the interval [0,1]. (n may be
        input as a float, but it is truncated to an integer in use)
        Parameters
        ----------
        n : float (but truncated to an integer)
                parameter, >= 0.
        p : float
                parameter, >= 0 and <=1.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.
        Returns
        -------
        samples : ndarray or scalar
                  where the values are all integers in  [0, n].
        See Also
        --------
        scipy.stats.distributions.binom : probability density function,
            distribution or cumulative density function, etc.
        Notes
        -----
        The probability density for the binomial distribution is
        .. math:: P(N) = \\binom{n}{N}p^N(1-p)^{n-N},
        where :math:`n` is the number of trials, :math:`p` is the probability
        of success, and :math:`N` is the number of successes.
        When estimating the standard error of a proportion in a population by
        using a random sample, the normal distribution works well unless the
        product p*n <=5, where p = population proportion estimate, and n =
        number of samples, in which case the binomial distribution is used
        instead. For example, a sample of 15 people shows 4 who are left
        handed, and 11 who are right handed. Then p = 4/15 = 27%. 0.27*15 = 4,
        so the binomial distribution should be used in this case.
        References
        ----------
        .. [1] Dalgaard, Peter, "Introductory Statistics with R",
               Springer-Verlag, 2002.
        .. [2] Glantz, Stanton A. "Primer of Biostatistics.", McGraw-Hill,
               Fifth Edition, 2002.
        .. [3] Lentner, Marvin, "Elementary Applied Statistics", Bogden
               and Quigley, 1972.
        .. [4] Weisstein, Eric W. "Binomial Distribution." From MathWorld--A
               Wolfram Web Resource.
               http://mathworld.wolfram.com/BinomialDistribution.html
        .. [5] Wikipedia, "Binomial-distribution",
               http://en.wikipedia.org/wiki/Binomial_distribution
        Examples
        --------
        Draw samples from the distribution:
        >>> n, p = 10, .5  # number of trials, probability of each trial
        >>> s = np.random.binomial(n, p, 1000)
        # result of flipping a coin 10 times, tested 1000 times.
        A real world example. A company drills 9 wild-cat oil exploration
        wells, each with an estimated probability of success of 0.1. All nine
        wells fail. What is the probability of that happening?
        Let's do 20,000 trials of the model, and count the number that
        generate zero positive results.
        >>> sum(np.random.binomial(9, 0.1, 20000) == 0)/20000.
        # answer = 0.38885, or 38%.
        """

        if n < 0:
            raise ValueError("n < 0")
        if p < 0:
            raise ValueError("p < 0")
        elif p > 1:
            raise ValueError("p > 1")
        elif np.isnan(p):
            raise ValueError("p is nan")
        # return discnp_array_sc(self.internal_state, rk_binomial, size, ln, fp, self.lock)
        # TODO: this function is incomplete
        return random_binomial(&self.rng_state, <long>n, p)

    def standard_t(self, df, size=None):
        """
        standard_t(df, size=None)

        Draw samples from a standard Student's t distribution with `df` degrees
        of freedom.

        A special case of the hyperbolic distribution.  As `df` gets
        large, the result resembles that of the standard normal
        distribution (`standard_normal`).

        Parameters
        ----------
        df : int
            Degrees of freedom, should be > 0.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        Returns
        -------
        samples : ndarray or scalar
            Drawn samples.

        Notes
        -----
        The probability density function for the t distribution is

        .. math:: P(x, df) = \\frac{\\Gamma(\\frac{df+1}{2})}{\\sqrt{\\pi df}
                  \\Gamma(\\frac{df}{2})}\\Bigl( 1+\\frac{x^2}{df} \\Bigr)^{-(df+1)/2}

        The t test is based on an assumption that the data come from a
        Normal distribution. The t test provides a way to test whether
        the sample mean (that is the mean calculated from the data) is
        a good estimate of the true mean.

        The derivation of the t-distribution was first published in
        1908 by William Gisset while working for the Guinness Brewery
        in Dublin. Due to proprietary issues, he had to publish under
        a pseudonym, and so he used the name Student.

        References
        ----------
        .. [1] Dalgaard, Peter, "Introductory Statistics With R",
               Springer, 2002.
        .. [2] Wikipedia, "Student's t-distribution"
               http://en.wikipedia.org/wiki/Student's_t-distribution

        Examples
        --------
        From Dalgaard page 83 [1]_, suppose the daily energy intake for 11
        women in Kj is:

        >>> intake = np.array([5260., 5470, 5640, 6180, 6390, 6515, 6805, 7515, \\
        ...                    7515, 8230, 8770])

        Does their energy intake deviate systematically from the recommended
        value of 7725 kJ?

        We have 10 degrees of freedom, so is the sample mean within 95% of the
        recommended value?

        >>> s = np.random.standard_t(10, size=100000)
        >>> np.mean(intake)
        6753.636363636364
        >>> intake.std(ddof=1)
        1142.1232221373727

        Calculate the t statistic, setting the ddof parameter to the unbiased
        value so the divisor in the standard deviation will be degrees of
        freedom, N-1.

        >>> t = (np.mean(intake)-7725)/(intake.std(ddof=1)/np.sqrt(len(intake)))
        >>> import matplotlib.pyplot as plt
        >>> h = plt.hist(s, bins=100, normed=True)

        For a one-sided t-test, how far out in the distribution does the t
        statistic appear?

        >>> np.sum(s<t) / float(len(s))
        0.0090699999999999999  #random

        So the p-value is about 0.009, which says the null hypothesis has a
        probability of about 99% of being true.

        """

        if df <= 0:
            raise ValueError("df <= 0")

        return cont1(&self.rng_state, &random_standard_t, df, size, self.lock)

    def bytes(self, Py_ssize_t length):
        """
        bytes(length)

        Return random bytes.
        Parameters
        ----------
        length : int
            Number of random bytes.

        Returns
        -------
        out : str
            String of length `length`.

        Examples
        --------
        >>> np.random.bytes(10)
        ' eh\\x85\\x022SZ\\xbf\\xa4' #random
        """
        cdef Py_ssize_t n_uint32 = ((length - 1) // 4 + 1)
        return self.random_uintegers(n_uint32, bits=32).tobytes()[:length]

    def randn(self, *args, method='inv'):
        """
        randn(d0, d1, ..., dn)

        Return a sample (or samples) from the "standard normal" distribution.

        If positive, int_like or int-convertible arguments are provided,
        `randn` generates an array of shape ``(d0, d1, ..., dn)``, filled
        with random floats sampled from a univariate "normal" (Gaussian)
        distribution of mean 0 and variance 1 (if any of the :math:`d_i` are
        floats, they are first converted to integers by truncation). A single
        float randomly sampled from the distribution is returned if no
        argument is provided.

        This is a convenience function.  If you want an interface that takes a
        tuple as the first argument, use `numpy.random.standard_normal` instead.

        Parameters
        ----------
        d0, d1, ..., dn : int, optional
            The dimensions of the returned array, should be all positive.
            If no argument is given a single Python float is returned.

        Returns
        -------
        Z : ndarray or float
            A ``(d0, d1, ..., dn)``-shaped array of floating-point samples from
            the standard normal distribution, or a single such float if
            no parameters were supplied.

        See Also
        --------
        random.standard_normal : Similar, but takes a tuple as its argument.

        Notes
        -----
        For random samples from :math:`N(\mu, \sigma^2)`, use:

        ``sigma * np.random.randn(...) + mu``

        Examples
        --------
        >>> np.random.randn()
        2.1923875335537315 #random

        Two-by-four array of samples from N(3, 6.25):

        >>> 2.5 * np.random.randn(2, 4) + 3
        array([[-4.49401501,  4.00950034, -1.81814867,  7.29718677],  #random
               [ 0.39924804,  4.68456316,  4.99394529,  4.84057254]]) #random
        """
        return self.standard_normal(size=args, method=method)

    def rand(self, *args):
        """
        rand(d0, d1, ..., dn)

        Random values in a given shape.

        Create an array of the given shape and propagate it with
        random samples from a uniform distribution
        over ``[0, 1)``.

        Parameters
        ----------
        d0, d1, ..., dn : int, optional
            The dimensions of the returned array, should all be positive.
            If no argument is given a single Python float is returned.

        Returns
        -------
        out : ndarray, shape ``(d0, d1, ..., dn)``
            Random values.

        See Also
        --------
        random

        Notes
        -----
        This is a convenience function. If you want an interface that
        takes a shape-tuple as the first argument, refer to
        np.random.random_sample .

        Examples
        --------
        >>> np.random.rand(3,2)
        array([[ 0.14022471,  0.96360618],  #random
               [ 0.37601032,  0.25528411],  #random
               [ 0.49313049,  0.94909878]]) #random
        """
        return self.random_sample(size=args)

    def uniform(self, double low=0.0, double high=1.0, size=None):
        """
        uniform(low=0.0, high=1.0, size=None)

        Draw samples from a uniform distribution.

        Samples are uniformly distributed over the half-open interval
        ``[low, high)`` (includes low, but excludes high).  In other words,
        any value within the given interval is equally likely to be drawn
        by `uniform`.

        Parameters
        ----------
        low : float, optional
            Lower boundary of the output interval.  All values generated will be
            greater than or equal to low.  The default value is 0.
        high : float
            Upper boundary of the output interval.  All values generated will be
            less than high.  The default value is 1.0.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        Returns
        -------
        out : ndarray
            Drawn samples, with shape `size`.

        See Also
        --------
        randint : Discrete uniform distribution, yielding integers.
        random_integers : Discrete uniform distribution over the closed
                          interval ``[low, high]``.
        random_sample : Floats uniformly distributed over ``[0, 1)``.
        random : Alias for `random_sample`.
        rand : Convenience function that accepts dimensions as input, e.g.,
               ``rand(2,2)`` would generate a 2-by-2 array of floats,
               uniformly distributed over ``[0, 1)``.

        Notes
        -----
        The probability density function of the uniform distribution is

        .. math:: p(x) = \frac{1}{b - a}

        anywhere within the interval ``[a, b)``, and zero elsewhere.

        Examples
        --------
        Draw samples from the distribution:

        >>> s = np.random.uniform(-1,0,1000)

        All values are within the given interval:

        >>> np.all(s >= -1)
        True
        >>> np.all(s < 0)
        True

        Display the histogram of the samples, along with the
        probability density function:

        >>> import matplotlib.pyplot as plt
        >>> count, bins, ignored = plt.hist(s, 15, normed=True)
        >>> plt.plot(bins, np.ones_like(bins), linewidth=2, color='r')
        >>> plt.show()
        """
        return cont2(&self.rng_state, &random_uniform, low, high - low, size, self.lock)


