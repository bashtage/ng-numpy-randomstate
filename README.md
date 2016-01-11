# ng-numpy-randomstate
[![Build Status](https://travis-ci.org/bashtage/ng-numpy-randomstate.svg?branch=master)](https://travis-ci.org/bashtage/ng-numpy-randomstate)

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
* A dummy RNG  - repeats the same sequence of 20 values -- only for testing

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
  
  * Document core RNG classes
  * Pickling support
  * Verify entropy based initialization is missing for some RNGs
  * Multiple stream support for MLFG and MRG32K3A
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

There are two options.  The first will build a library for xorshift128 called
`core_rng`.  

```bash
cd randomstate
python setup-basic.py build_ext --inplace
```

The second will build a numberfiles, one for each RNG.

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

import randomstate.pcg64
rs = randomstate.pcg64.RandomState()
rs.random_sample(100)

# Identical to NumPy
import randomstate.mt19937
rs = randomstate.mt19937.RandomState()
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
Time to produce 1,000,000 Uniforms
************************************************************
numpy-random-random_sample                 11.44 ms
randomstate-mlfg_1279_861-random_sample     7.18 ms
randomstate-mrg32k3a-random_sample         35.70 ms
randomstate-mt19937-random_sample           9.26 ms
randomstate-pcg32-random_sample             6.75 ms
randomstate-pcg64-random_sample             5.79 ms
randomstate-xorshift1024-random_sample      5.73 ms
randomstate-xorshift128-random_sample       5.38 ms

Uniforms per second
************************************************************
numpy-random-random_sample                  87.39 million
randomstate-mlfg_1279_861-random_sample    139.27 million
randomstate-mrg32k3a-random_sample          28.01 million
randomstate-mt19937-random_sample          107.94 million
randomstate-pcg32-random_sample            148.16 million
randomstate-pcg64-random_sample            172.63 million
randomstate-xorshift1024-random_sample     174.63 million
randomstate-xorshift128-random_sample      185.83 million
dtype: object

Speed-up relative to NumPy
************************************************************
randomstate-mlfg_1279_861-random_sample     59.4%
randomstate-mrg32k3a-random_sample         -67.9%
randomstate-mt19937-random_sample           23.5%
randomstate-pcg32-random_sample             69.5%
randomstate-pcg64-random_sample             97.5%
randomstate-xorshift1024-random_sample      99.8%
randomstate-xorshift128-random_sample      112.6%
--------------------------------------------------------------------------------
```

## Differences from `numpy.random.RandomState`

### New Functions

* `random_uintegers` - unsigned integers `[0, 2**64-1]` 
* `jump` - Jumps RNGs that support it.  `jump` moves the state a great 
distance. _Only available if supported by the RNG._
* `advance` - Advanced the core RNG 'as-if' a number of draws were made, 
without actually drawing the numbers. _Only available if supported by the RNG._
