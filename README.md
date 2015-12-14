# pcg-python

This is an early attempt at writing a generic interface that woould allow 
alternative random generators in Python and Numpy. The first attempt is 
to include [pcg random number generator](http://www.pcg-random.org/) 
in addition to the MT19937 that is included in NumPy (it also includes
a dummy RNG, which repeats the same sequence of 20 values, which is only
for testing).

It uses source from 
[pcg](http://www.pcg-random.org/), [numpy](http://www.numpy.org/) and 
[randomkit](https://github.com/numpy/numpy/blob/master/numpy/random/mtrand/).

## Status

* There is no documentation.  
* Only double and integer RNGs are available. 
* Setting and restoring state works

## Plans
There are still many improvements needed before this is really usable. 

At a minimum this needs to support:

  * More critical RNGs
  * Ability to return other shapes
  * Default arguments
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

The second will build 4 files, one for each RNG.

```bash
cd pcg
python setup-2.py build_ext --inplace
```

## Using

```python
import core_rng

rs = core_rng.PCGRandomState()
rs.random_double(100)
```

## License
Standard NCSA, plus sub licenses for components.