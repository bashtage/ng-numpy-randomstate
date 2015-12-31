# ng-numpy-randomstate

This is an early attempt at writing a generic interface that would allow 
alternative random generators in Python and Numpy. The first attempt is 
to include alternative core random number generators in addition to the 
MT19937 that is included in NumPy. New RNGs include:

* [MT19937](https://github.com/numpy/numpy/blob/master/numpy/random/mtrand/),
 the NumPy rng
* [xorshift128+](http://xorshift.di.unimi.it/) and 
[xorshift1024*](http://xorshift.di.unimi.it/)
* [PCG32](http://www.pcg-random.org/) and [PCG64](http:w//www.pcg-random.org/)
* [MRG32K3A](http://simul.iro.umontreal.ca/rng)
* A multiplicative lagged fibonacci generator (LFG(31, 1279, 861, *))
* A dummy RNG  - repeats the same sequence of 20 values -- only for testing

## Rationale
The main reason for this project is to include other PRNGs 
that support important features when working in parallel such
as the ability to produce multiple independent streams, to 
quickly advance the generator, or to jump ahead.

## Status

* There is no documentation for the core RNGs.
* Mostly complete drop-in replacement for `numpy.random.RandomState` 
* Setting and restoring state works

## Plans
It is mostly complete.  There are a few rough edges that need to be smoothed.
  
  * Document core RNG classes
  * Complete implementation of all `numpy.random.RandomState` function
  * Pickling support
  * Verify entropy based initialization is missing for some RNGs
  * Integrate a patch for PCG-64 that allows 32-bit platforms to be supported
  * Build on Windows
  * Additional refactoring where possible
  * Check types for consistenct (e.g. `long` vs `uint64`) for discrete things 

## Requirements
Requires:

  * Numpy (1.10)
  * Cython (0.23)

**Note:** it might work with outher versions but only tested with these 
versions. 

So far all development has been on Linux. It has been tested (in a limited 
manner, mostly against crashes and build failures) on Linux 32 and 64-bit, 
as well as OSX 10.10 and PC-BSD 10.2 (should also work on Free BSD).

All tests implemeted are _smoke_ tests that only make sure that something is 
output from the expected inputs. Formal tests (unit) have not been implemented.

## Installing

```bash
python setup.py install
```

## Building for Testing Purposes

There are two options.  The first will build a library for xorshift128 called
`core_rng`.  

```bash
cd randomstate
python setup-basic.py build_ext --inplace
```

The second will build a number files, one for each RNG.

```bash
cd randomstate
python setup-all.py build_ext --inplace
```

## Using
If you installed,

```python
import randomstate.xorshift128
rs = randomstate.xorshift128.RandomState()
rs.random_sample(100)

import randomstate.xorshift128.pcg32
rs = randomstate.pcg32.RandomState()
rs.random_sample(100)
```

If you use `setup-basic.py`, 

```python
import core_rng

rs = core_rng.RandomState()
rs.random_sample(100)
```

If you use `setup-all.py`, 

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
mlfg_1279_861_random_sample     6.86 ms
mrg32k3a_random_sample         35.38 ms
mt19937_random_sample           9.88 ms
numpy.random_random_sample     11.35 ms
pcg32_random_sample             6.89 ms
pcg64_random_sample             5.55 ms
xorshift1024_random_sample      5.37 ms
xorshift128_random_sample       5.42 ms

uniforms per second
************************************************************
mlfg_1279_861_random_sample    145.83 million
mrg32k3a_random_sample          28.27 million
mt19937_random_sample          101.25 million
numpy.random_random_sample      88.07 million
pcg32_random_sample            145.14 million
pcg64_random_sample            180.17 million
xorshift1024_random_sample     186.17 million
xorshift128_random_sample      184.34 million

Speed-up relative to NumPy
************************************************************
mlfg_1279_861_random_sample     65.6%
mrg32k3a_random_sample         -67.9%
mt19937_random_sample           15.0%
pcg32_random_sample             64.8%
pcg64_random_sample            104.6%
xorshift1024_random_sample     111.4%
xorshift128_random_sample      109.3%
```

## Differences from `numpy.random.RandomState`

### New

* `random_bounded_integers` - bounded integers `[lower, upper]` where `lower >= -2**63` 
and `upper < 2**63`
* `random_bounded_uintegers` - bounded unsigned integers `[0, upper]` 
where `upper < 2**64`
* `random_uintegers` - unsigned integers `[0, 2**64-1]` 
* `jump` - Jumps RNGs that support it.  `jump` moves the state a great 
distance. _Only available if supported by the RNG._
* `advance` - Advanced the core RNG 'as-if' a number of draws were made, 
without actually drawing the numbers. _Only available if supported by the RNG._

### Diffeent

* `random_integers` - Not sure
* `tomaxint` - Might be different

### Same

These have been implemented and are the same (or quantitatively similar)

```
seed                    bytes                   get_state 
standard_cauchy         standard_exponential    standard_gamma 
standard_normal         random_sample           beta
chisquare               choice                  dirichlet
exponential             f                       gamma
geometric               gumbel                  hypergeometric
laplace                 logistic                lognormal
logseries               multinomial             multivariate_normal
negative_binomial       noncentral_chisquare    noncentral_f
normal                  pareto                  permutation
poisson                 poisson_lam_max         power
rand                    randint                 randn   
rayleigh                shuffle                 tomaxint    
triangular              uniform                 vonmises
wald                    weibull                 zipf
```

### Missing

These have not been implemented yet.

```
choice
```
