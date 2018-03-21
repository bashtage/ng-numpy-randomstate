# Future development in [randomgen](https://github.com/bashtage/randomgen)

**End-of-life notification**

This library was designed to bring alternative generators to the NumPy 
infrastructure. It as been successful in advancing the conversation 
for a future implementation of a new random number API in NumPy which 
will allow new algorithms and/or generators. The next step
in this process is to separate the basic (or core RNG) from the 
functions that transform random bits into useful random numbers.
This has been implemented in a successor project  **randomgen** 
available on [GitHub](https://github.com/bashtage/randomgen) or
[PyPi](https://pypi.org/project/randomgen/).

[randomgen](https://github.com/bashtage/randomgen) has a slightly different API, so please see the  [randomgen documentation](https://bashtage.github.io/randomgen).

-----------------------------------------------------------------------

# randomstate

[![Travis Build Status](https://travis-ci.org/bashtage/ng-numpy-randomstate.svg?branch=master)](https://travis-ci.org/bashtage/ng-numpy-randomstate) 
[![Appveyor Build Status](https://ci.appveyor.com/api/projects/status/odc5c4ukhru5xicl/branch/master?svg=true)](https://ci.appveyor.com/project/bashtage/ng-numpy-randomstate/branch/master)
[![PyPI version](https://badge.fury.io/py/randomstate.svg)](https://badge.fury.io/py/randomstate)

## Introduction

This is a library and generic interface for alternative random 
generators in Python and NumPy. 

## Features

* Immediate drop in replacement for NumPy's RandomState

```python
# import numpy.random as rnd
import randomstate as rnd
x = rnd.standard_normal(100)
y = rnd.random_sample(100)
z = rnd.randn(10,10)
```

* Default random generator is identical to NumPy's RandomState (i.e., 
same seed, same random numbers).
* Support for random number generators that support independent streams 
and jumping ahead so that sub-streams can be generated
* Faster random number generation, especially for normal, standard
  exponential and standard gamma using the Ziggurat method

```python
import randomstate as rnd
w = rnd.standard_normal(10000, method='zig')
x = rnd.standard_exponential(10000, method='zig')
y = rnd.standard_gamma(5.5, 10000, method='zig')
```

* Support for 32-bit floating randoms for core generators. 
  Currently supported:

    * Uniforms (`random_sample`)
    * Exponentials (`standard_exponential`, both Inverse CDF and Ziggurat)
    * Normals (`standard_normal`, both Box-Muller and Ziggurat)
    * Standard Gammas (via `standard_gamma`, both Inverse CDF and Ziggurat)
  
  **WARNING**: The 32-bit generators are **experimental** and subject 
  to change.
  
  **Note**: There are _no_ plans to extend the alternative precision 
  generation to all random number types.

* Support for filling existing arrays using `out` keyword argument. Currently
  supported in (both 32- and 64-bit outputs)

    * Uniforms (`random_sample`)
    * Exponentials (`standard_exponential`)
    * Normals (`standard_normal`)
    * Standard Gammas (via `standard_gamma`)

## Included Pseudo Random Number Generators

This modules includes a number of alternative random 
number generators in addition to the MT19937 that is included in NumPy. 
The RNGs include:

* [MT19937](https://github.com/numpy/numpy/blob/master/numpy/random/mtrand/),
 the NumPy rng
* [dSFMT](http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/SFMT/) a 
  SSE2-aware version of the MT19937 generator that is especially fast at 
  generating doubles
* [SFMT](http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/SFMT/) a
  SSE2-aware version of the MT19937 generator that is optimized for
  integer values
* [xorshift128+](http://xorshift.di.unimi.it/),
  [xoroshiro128+](http://xoroshiro.di.unimi.it/) and
  [xorshift1024*](http://xorshift.di.unimi.it/)
* [PCG32](http://www.pcg-random.org/) and 
  [PCG64](http:w//www.pcg-random.org/)
* [MRG32K3A](http://simul.iro.umontreal.ca/rng)
* A multiplicative lagged fibonacci generator (LFG(63, 1279, 861, *))

## Differences from `numpy.random.RandomState`

### New Features
* `standard_normal`, `normal`, `randn` and `multivariate_normal` all 
  support an additional `method` keyword argument which can be `bm` or
  `zig` where `bm` corresponds to the current method using the Box-Muller
  transformation and `zig` uses the much faster (100%+) Ziggurat method.
* `standard_exponential` and `standard_gamma` both support an additional
  `method` keyword argument which can be `inv` or
  `zig` where `inv` corresponds to the current method using the inverse
  CDF and `zig` uses the much faster (100%+) Ziggurat method.
* Core random number generators can produce either single precision
  (`np.float32`) or double precision (`np.float64`, the default) using
  an the optional keyword argument `dtype`
* Core random number generators can fill existing arrays using the
  `out` keyword argument


### New Functions

* `random_entropy` - Read from the system entropy provider, which is 
commonly used in cryptographic applications
* `random_raw` - Direct access to the values produced by the underlying 
PRNG. The range of the values returned depends on the specifics of the 
PRNG implementation.
* `random_uintegers` - unsigned integers, either 32- (`[0, 2**32-1]`)
or 64-bit (`[0, 2**64-1]`)
* `jump` - Jumps RNGs that support it.  `jump` moves the state a great 
distance. _Only available if supported by the RNG._
* `advance` - Advanced the core RNG 'as-if' a number of draws were made, 
without actually drawing the numbers. _Only available if supported by 
the RNG._

## Status

* Complete drop-in replacement for `numpy.random.RandomState`. The 
`mt19937` generator is identical to `numpy.random.RandomState`, and 
will produce an identical sequence of random numbers for a given seed.   
* Builds and passes all tests on:
  * Linux 32/64 bit, Python 2.7, 3.4, 3.5, 3.6 (probably works on 2.6 and 3.3)
  * PC-BSD (FreeBSD) 64-bit, Python 2.7
  * OSX 64-bit, Python 2.7
  * Windows 32/64 bit (only tested on Python 2.7, 3.5 and 3.6, but
    should work on 3.3/3.4)

## Version
The version matched the latest version of NumPy where 
`randomstate.prng.mt19937` passes all NumPy test.

## Documentation

An occasionally updated build of the documentation is available on
[my github pages](http://bashtage.github.io/ng-numpy-randomstate/).

## Plans
This module is essentially complete.  There are a few rough edges that 
need to be smoothed.
  
  * Stream support for MLFG
  * Creation of additional streams from a RandomState where supported 
  (i.e. a `next_stream()` method)
  
## Requirements
Building requires:

  * Python (2.7, 3.4, 3.5, 3.6)
  * NumPy (1.9, 1.10, 1.11, 1.12)
  * Cython (0.22, **not** 0.23, 0.24, 0.25)
  * tempita (0.5+), if not provided by Cython
 
Testing requires pytest (3.0+).

**Note:** it might work with other versions but only tested with these 
versions. 

## Development and Testing

All development has been on 64-bit Linux, and it is regularly tested on 
Travis-CI. The library is occasionally tested on Linux 32-bit,  
OSX 10.10, PC-BSD 10.2 (should also work on Free BSD) and Windows 
(Python 2.7/3.5, both 32 and 64-bit).

Basic tests are in place for all RNGs. The MT19937 is tested against 
NumPy's implementation for identical results. It also passes NumPy's 
test suite.

## Installing

```bash
python setup.py install
```

### SSE2
`dSFTM` makes use of SSE2 by default.  If you have a very old computer 
or are building on non-x86, you can install using:

```bash
python setup.py install --no-sse2
```

### Windows
Either use a binary installer, or if building from scratch, use 
Python 3.5 with Visual Studio 2015 Community Edition. It can also be 
build using Microsoft Visual C++ Compiler for Python 2.7 and Python 2.7, 
although some modifications may be needed to `distutils` to find the 
compiler.

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

Like NumPy, `randomstate` also exposes a single instance of the 
`mt19937` generator directly at the module level so that commands like

```python
import randomstate
randomstate.standard_normal()
randomstate.exponential(1.0, 1.0, size=10)
```

will work.

## License
Standard NCSA, plus sub licenses for components.

## Performance
Performance is promising, and even the mt19937 seems to be faster than 
NumPy's mt19937. 

```
Speed-up relative to NumPy (Uniform Doubles)
************************************************************
randomstate.prng-dsfmt-random_sample               313.5%
randomstate.prng-mlfg_1279_861-random_sample       459.4%
randomstate.prng-mrg32k3a-random_sample            -57.6%
randomstate.prng-mt19937-random_sample              72.5%
randomstate.prng-pcg32-random_sample               232.8%
randomstate.prng-pcg64-random_sample               330.6%
randomstate.prng-xoroshiro128plus-random_sample    609.9%
randomstate.prng-xorshift1024-random_sample        348.8%
randomstate.prng-xorshift128-random_sample         489.7%

Speed-up relative to NumPy (Normals using Box-Muller)
************************************************************
randomstate.prng-dsfmt-standard_normal                26.8%
randomstate.prng-mlfg_1279_861-standard_normal        30.9%
randomstate.prng-mrg32k3a-standard_normal            -14.8%
randomstate.prng-mt19937-standard_normal              17.7%
randomstate.prng-pcg32-standard_normal                24.5%
randomstate.prng-pcg64-standard_normal                26.2%
randomstate.prng-xoroshiro128plus-standard_normal     31.4%
randomstate.prng-xorshift1024-standard_normal         27.4%
randomstate.prng-xorshift128-standard_normal          30.3%

Speed-up relative to NumPy (Normals using Ziggurat)
************************************************************
randomstate.prng-dsfmt-standard_normal               491.7%
randomstate.prng-mlfg_1279_861-standard_normal       439.6%
randomstate.prng-mrg32k3a-standard_normal            101.2%
randomstate.prng-mt19937-standard_normal             354.4%
randomstate.prng-pcg32-standard_normal               531.0%
randomstate.prng-pcg64-standard_normal               517.9%
randomstate.prng-xoroshiro128plus-standard_normal    674.0%
randomstate.prng-xorshift1024-standard_normal        486.7%
randomstate.prng-xorshift128-standard_normal         617.0%
```