# randomstate

[![Travis Build Status](https://travis-ci.org/bashtage/ng-numpy-randomstate.svg?branch=master)](https://travis-ci.org/bashtage/ng-numpy-randomstate) 
[![Appveyor Build Status](https://ci.appveyor.com/api/projects/status/odc5c4ukhru5xicl/branch/master?svg=true)](https://ci.appveyor.com/project/bashtage/ng-numpy-randomstate/branch/master)

This is a library and generic interface for alternative random generators 
in Python and Numpy. 

Features

* Immediate drop in replacement for Numy's RandomState

```python
# import numpy.random as rnd
import randomstate as rnd
x = rnd.standard_normal(100)
y = rnd.random_sample(100)
z = rnd.randn(10,10)
```

* Default random generator is identical to NumPy's RandomState (i.e., same 
seed, same random numbers).
* Support for random number generators that support independent streams and 
jumping ahead so that substreams can be generated
* Faster ranomd number generations, especially for Normals using the Ziggurat 
method 

```python
import randomstate as rnd
w = rnd.standard_normal(10000, method='zig')
```

## Included Pseudo Random Number Generators

This modules includes a number of alternative random 
number generators in addition to the MT19937 that is included in NumPy. 
The RNGs include:

* [MT19937](https://github.com/numpy/numpy/blob/master/numpy/random/mtrand/),
 the NumPy rng
* [dSFMT](http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/SFMT/) a SSE2-aware 
version of the MT19937 generator that is especially fast at generating doubles
* [xorshift128+](http://xorshift.di.unimi.it/) and 
[xorshift1024*](http://xorshift.di.unimi.it/)
* [PCG32](http://www.pcg-random.org/) and [PCG64](http:w//www.pcg-random.org/)
* [MRG32K3A](http://simul.iro.umontreal.ca/rng)
* A multiplicative lagged fibonacci generator (LFG(31, 1279, 861, *))

## Differences from `numpy.random.RandomState`

### New Features
* `stanard_normal`, `normal`, `randn` and `multivariate_normal` all support 
an additional `method` keyword argument which can be `inv` or `zig` where 
`inv` corresponds to the current method and `zig` uses tha much faster 
(100%+) ziggurat method.

### New Functions

* `random_entropy` - Read from the system entropy provider, which is commonly 
used in cryptographic applications
* `random_uintegers` - unsigned integers `[0, 2**64-1]` 
* `jump` - Jumps RNGs that support it.  `jump` moves the state a great 
distance. _Only available if supported by the RNG._
* `advance` - Advanced the core RNG 'as-if' a number of draws were made, 
without actually drawing the numbers. _Only available if supported by the RNG._

## Status

* Complete drop-in replacement for `numpy.random.RandomState`. The `mt19937` 
generator is identical to `numpy.random.RandomState`, and will produce an 
identical sequence of random numbers for a given seed.   
* Builds and passes all tests on:
  * Linux 32/64 bit, Python 2.6, 2.7, 3.3, 3.4, 3.5
  * PC-BSD (FreeBSD) 64-bit, Python 2.7
  * OSX  64-bit, Python 2.7
  * Windows 32/64 bit (only tested on Python 2.7 and 3.5, but should work on 3.3/3.4)

## Version
The version matched the latest verion of NumPy where 
`randomstate.prng.mt19937` passes all NumPy test.

## Documentation

A occasionally updated build of the documentation is available on
[my github pages](http://bashtage.github.io/ng-numpy-randomstate/).

## Plans
This module is essentially complete.  There are a few rough edges that need to be smoothed.
  
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

### SSE2
`dSFTM` makes use of SSE2 by default.  If you have a very old computer or are 
building on non-x86, you can install using:

```bash
python setup.py install --no-sse2
```

### Windows
Either use a binary installer or if building from scratch using Python 3.5 and 
the free Visual Studio 2015 Community Edition. It can also be build using 
Microsoft Visual C++ Compiler for Python 2.7 and Python 2.7, although some
modifications are needed to distutils to find the compiler.

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
generator directly at the module level so that commands like

```python
import randomstate
randomstate.standard_normal()
randomstate.exponential(1.0, 1.0, size=10)
```

will work.

## License
Standard NCSA, plus sub licenses for components.

## Performance
Performance is promising, and even the mt19937 seems to be faster than NumPy's mt19937. 

```
Speed-up relative to NumPy (Slow Normals)
************************************************************
randomstate.prng-dsfmt-standard_normal            107.2%
randomstate.prng-mlfg_1279_861-standard_normal     51.2%
randomstate.prng-mrg32k3a-standard_normal         -11.8%
randomstate.prng-mt19937-standard_normal           44.0%
randomstate.prng-pcg32-standard_normal             51.2%
randomstate.prng-pcg64-standard_normal             51.1%
randomstate.prng-xorshift1024-standard_normal      50.5%
randomstate.prng-xorshift128-standard_normal       52.1%

Speed-up relative to NumPy (Ziggural Normals)
************************************************************
randomstate.prng-dsfmt-standard_normal            283.7%
randomstate.prng-mlfg_1279_861-standard_normal    217.4%
randomstate.prng-mrg32k3a-standard_normal          16.6%
randomstate.prng-mt19937-standard_normal          201.3%
randomstate.prng-pcg32-standard_normal            274.9%
randomstate.prng-pcg64-standard_normal            310.8%
randomstate.prng-xorshift1024-standard_normal     336.3%
randomstate.prng-xorshift128-standard_normal      425.1%
```
