# ng-numpy-randomstate
[![Build Status](https://travis-ci.org/bashtage/ng-numpy-randomstate.svg?branch=master)](https://travis-ci.org/bashtage/ng-numpy-randomstate) 
[![Build status](https://ci.appveyor.com/api/projects/status/odc5c4ukhru5xicl/branch/master?svg=true)](https://ci.appveyor.com/project/bashtage/ng-numpy-randomstate/branch/master)

This is a library and generic interface for alternative random generators 
in Python and Numpy. This modules includes a number of alternative random 
number generators in addition to the MT19937 that is included in NumPy. 
The RNGs include:

* [MT19937](https://github.com/numpy/numpy/blob/master/numpy/random/mtrand/),
 the NumPy rng
* [xorshift128+](http://xorshift.di.unimi.it/) and 
[xorshift1024*](http://xorshift.di.unimi.it/)
* [PCG32](http://www.pcg-random.org/) and [PCG64](http:w//www.pcg-random.org/)
* [MRG32K3A](http://simul.iro.umontreal.ca/rng)
* A multiplicative lagged fibonacci generator (LFG(31, 1279, 861, *))

## Rationale
The main reason for this project is to include other PRNGs that support 
important features when working in parallel such as the ability to produce 
multiple independent streams, to quickly advance the generator, or to jump 
ahead.

## Status

* Complete drop-in replacement for `numpy.random.RandomState`. The `mt19937` 
generator is identical to `numpy.random.RandomState`, and will produce an 
identical sequence of random numbers for a given seed.   
* Builds and passes all tests on:
  * Linux 32/64 bit, Python 2.6, 2.7, 3.3, 3.4, 3.5
  * PC-BSD (FreeBSD) 64-bit, Python 2.7
  * OSX  64-bit, Python 2.7
  * Windows 32/64 bit (only tested on Python 2.7 and 3.5, but should work on 3.3/3.4)
* There is no documentation for the core RNGs.

## Plans
It is essentially complete.  There are a few rough edges that need to be smoothed.
  
  * Stream support for MLFG and MRG32K3A
  * Creation of additional streams from a RandomState where supported (i.e. 
  a `next_stream()` method)
  
## Requirements
Building requires:

  * Numpy (1.9, 1.10)
  * Cython (0.22, 0.23)
  * Python (2.6, 2.7, 3.3, 3.4, 3.5)

**Note:** it might work with other versions but only tested with these 
versions. 

All development has been on 64-bit Linux, and it is regularly tested on 
Travis-CI. The library is occasionally tested on Linux 32-bit,  OSX 10.10, 
PC-BSD 10.2 (should also work on Free BSD) and Windows (Python 2.7/3.5, 
both 32 and 64-bit).

Basic tests are in place for all RNGs. The MT19937 is tested against NumPy's 
implementation for identical results. It also passes NumPy's test suite.

## Installing

```bash
python setup.py install
```

## Building for Testing Purposes

This command will build a single module containining xorshift128 called
`interface`.  

```bash
cd randomstate
python setup-basic.py build_ext --inplace
```

## Using

The separate generators are importable from `randomstate.prng`.

```python
import randomstate
rs = randomstate.prng.xorshift128.RandomState()
rs.random_sample(100)

rs = randomstate.prng.pcg64.RandomState()
rs.random_sample(100)

# Identical to NumPy
rs = randomstate.prng.mt19937.RandomState()
rs.random_sample(100)
```

Like NumPy, `randomstate` also exposes a single instance of the `mt19937` 
generator directly at the moduel level so that commands like

```python
import randomstate
randomstate.standard_normal()
randomstate.exponential(1.0, 1.0, size=10)
```

will work.

If you use `setup-basic.py`, 

```python
import interface

rs = interface.RandomState()
rs.random_sample(100)
```

## License
Standard NCSA, plus sub licenses for components.

## Performance
Performance is promising, and even the mt19937 seems to be faster than NumPy's mt19937. 

```
Time to produce 1,000,000 Standard normals
************************************************************
numpy-random-standard_normal                      58.34 ms
randomstate.prng-mlfg_1279_861-standard_normal    46.20 ms
randomstate.prng-mrg32k3a-standard_normal         75.95 ms
randomstate.prng-mt19937-standard_normal          52.68 ms
randomstate.prng-pcg32-standard_normal            48.38 ms
randomstate.prng-pcg64-standard_normal            46.27 ms
randomstate.prng-xorshift1024-standard_normal     45.53 ms
randomstate.prng-xorshift128-standard_normal      45.57 ms

Standard normals per second
************************************************************
numpy-random-standard_normal                      17.14 million
randomstate.prng-mlfg_1279_861-standard_normal    21.65 million
randomstate.prng-mrg32k3a-standard_normal         13.17 million
randomstate.prng-mt19937-standard_normal          18.98 million
randomstate.prng-pcg32-standard_normal            20.67 million
randomstate.prng-pcg64-standard_normal            21.61 million
randomstate.prng-xorshift1024-standard_normal     21.96 million
randomstate.prng-xorshift128-standard_normal      21.94 million

Speed-up relative to NumPy
************************************************************
randomstate.prng-mlfg_1279_861-standard_normal     26.3%
randomstate.prng-mrg32k3a-standard_normal         -23.2%
randomstate.prng-mt19937-standard_normal           10.8%
randomstate.prng-pcg32-standard_normal             20.6%
randomstate.prng-pcg64-standard_normal             26.1%
randomstate.prng-xorshift1024-standard_normal      28.1%
randomstate.prng-xorshift128-standard_normal       28.0%

--------------------------------------------------------------------------------
```

## Differences from `numpy.random.RandomState`

### New Features
* `stanard_normal` and `normal` support an additional `method` keyword 
argument which can be `inv` or `zig` where `inv` corresponds to the 
current method and `zig` uses tha much faster (100%+) ziggurat method.

### New Functions

* `random_entropy` - Read from the system entropy provider, which is commonly 
used in cryptographic applications
* `random_uintegers` - unsigned integers `[0, 2**64-1]` 
* `jump` - Jumps RNGs that support it.  `jump` moves the state a great 
distance. _Only available if supported by the RNG._
* `advance` - Advanced the core RNG 'as-if' a number of draws were made, 
without actually drawing the numbers. _Only available if supported by the RNG._
