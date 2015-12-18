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
 
So far all development has been on Linux, so other platforms might not work.


## Building
There are two options.  The first will build a library for PCG64 called
`core_rng`.  

```bash
cd pcg
python setup.py build_ext --inplace
```

The second will build a number files, one for each RNG.

```bash
cd pcg
python setup-2.py build_ext --inplace
```

## Using
If you use `setup.py`, 

```python
import core_rng

rs = core_rng.RandomState()
rs.random_sample(100)
```

If you use `setup-2.py`, 

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

