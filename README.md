# ng-numpy-randomstate

This is an early attempt at writing a generic interface that woould allow 
alternative random generators in Python and Numpy. The first attempt is 
to include [pcg random number generator](http://www.pcg-random.org/) 
in addition to the MT19937 that is included in NumPy (it also includes
a dummy RNG, which repeats the same sequence of 20 values, which is only
for testing).

It uses source from 
[pcg](http://www.pcg-random.org/), [numpy](http://www.numpy.org/) and 
[randomkit](https://github.com/numpy/numpy/blob/master/numpy/random/mtrand/).

## Rationale
The main reason for this project is to include other PRNGs 
that support important features when working in parallel such
as the ability to produce multiple independent streams, to 
quickly advance the generator, or to jump ahead.

## Status

* There is no documentation.  
* Only a small number of rngs are available: standard normal, standard gamma, 
standard exponential, standard uniform and random 64-bit integers. 
* Setting and restoring state works

## Plans
There are still many improvements needed before this is really usable. 

At a minimum this needs to support:

  * More critical RNGs
  * Entropy based initialization

## Requirements
Requires (Built Using):

  * Numpy (1.10)
  * Cython (0.23)
 
So far all development has been on Linux. It has been tested (in a limited 
manner, mostly against crashes and build failures) on Linux 32 and 64-bit, 
as well as OSX 10.10 and PC-BSD 10.2 (should alsl work on Free BSD).

Formal tests (unit) have not been implmented.

## Building
There are two options.  The first will build a library for PCG64 called
`core_rng`.  

```bash
cd randomstate
python setup-basic.py build_ext --inplace
```

The second will build a number files, one for each RNG.

```bash
cd randomstate
python setup-testing.py build_ext --inplace
```

## Using
If you use `setup-basic.py`, 

```python
import core_rng

rs = core_rng.RandomState()
rs.random_sample(100)
```

If you use `setup-testing.py`, 

```python
import mt19937, pcg32, xorshift128

rs = mt19937.RandomState()
rs.random_sample(100)

rs = pcg32.RandomState()
rs.random_sample(100)

rs = xorshift129.RandomState()
rs.random_sample(100)
```

## License
Standard NCSA, plus sub licenses for components.

## Performance
Performance is promising.  Some early numbers:

```
Time to produce 1,000,000 uniforms
************************************************************
mrg32k3a_random_sample        46.12 ms
mt19937_random_sample         14.50 ms
numpy.random_random_sample    15.69 ms
pcg32_random_sample           11.65 ms
pcg64_random_sample            8.79 ms
xorshift1024_random_sample     7.03 ms
xorshift128_random_sample      6.32 ms

uniforms per second
************************************************************
mrg32k3a_random_sample         21.68 million
mt19937_random_sample          68.98 million
numpy.random_random_sample     63.72 million
pcg32_random_sample            85.81 million
pcg64_random_sample           113.70 million
xorshift1024_random_sample    142.19 million
xorshift128_random_sample     158.11 million

Speed-up relative to NumPy
************************************************************
mrg32k3a_random_sample        -66.0%
mt19937_random_sample           8.2%
pcg32_random_sample            34.7%
pcg64_random_sample            78.4%
xorshift1024_random_sample    123.1%
xorshift128_random_sample     148.1%
dtype: object
```

## Differences from `numpy.random.RandomState`

### New

* `random_bounded_integers` - bounded integers `[lower, upper]` where `lower >= -2**63` and `upper < 2**63`
* `random_bounded_uintegers` - bounded unsigned integers `[0, upper]` where `upper < 2**64`
* `random_uintegers` - unsigned integers `[0, 2**64-1]` 
* `jump` - Jumps RNGs that support it.  `jump` moves the stata a great distinace.
* `advance` - Advanced the core RNG 'as-if' a number of draws were made, without actually drawing the numbers

### Diffeent

* `random_integers` - Not sure
* `standard_t`- No support for broadcasting
* `binomial`- No support for broadcasting

### Same

These have been implemented and are the same (or quanitatively similar)
```
seed bytes get_state standard_cauchy standard_exponential
standard_gamma standard_normal random_sample
```

### Missing

These have not been implemented.

```
beta, chisquare, choice, dirichlet, exponential, f, gamma, geometric,
gumbel, hypergeometric, laplace, logistic, lognormal, logseries,
multinomial, multivariate_normal, negative_binomial, noncentral_chisquare, 
noncentral_f, normal, pareto, permutation,
poisson, poisson_lam_max, power, rand, randint, randn, rayleigh,
shuffle, tomaxint, triangular, uniform, vonmises, wald, weibull, zipf
```
