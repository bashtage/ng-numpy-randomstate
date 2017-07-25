Performance
-----------

.. py:module:: randomstate

Recommendation
**************
The recommended generator for single use is `xoroshiro128+`
(:class:`~randomstate.prng.xoroshiro128plus`).  The recommended generator
for use in large-scale parallel applications is
`xorshift1024*` (:class:`~randomstate.prng.xorshift1024`)
where the `jump` method is used to advance the state.

Timings
*******

The timings below are ns/random value.  The fastest generator is the
raw generator (`random_raw`) which does not make any transformation
to the underlying random value.  `xoroshiro128+` is the fastest, followed by
`xorshift1024*` and the two SIMD aware MT generators.  The original MT19937
generator is much slower since it requires 2 32-bit values to equal the output
of the faster generators.

Integer performance has a similar ordering although `dSFMT` is slower since
it generates 53-bit floating point values rather than integer values. On the
other hand, it is very fast for uniforms, although slower than `xoroshiro128+`.

The patterm is similar for other, more complex generators. The normal
performance of NumPy's MT19937 is much lower than the other since it
uses the Box-Muller transformation rather than the Ziggurat generator.

.. csv-table::
    :header: ,NumPy MT19937,MT19937,SFMT,dSFMT,xoroshiro128+,xorshift1024,PCG64
    :widths: 14,14,14,14,14,14,14,14

    Raw,,5.07,1.98,1.91,1.08,1.49,3.16
    Random Integers,4.84,3.32,2.36,3.36,1.99,2.53,2.32
    Uniforms,10.41,6.53,2.7,2.7,1.58,2.31,2.47
    Normal,69.4,15.34,9.93,11.57,9.56,11.69,11.64
    Complex Normal,,40.57,28.05,31.91,25.8,31.9,31.86
    Binomial,77.46,71.08,65.77,65.71,66.18,67.23,69.51
    Gamma,107.42,97.66,86.5,86.09,82.31,87.42,86.41
    Exponential,112.54,104.16,96.23,95.22,93.98,98.15,98.36
    Laplace,116.09,114.2,102.35,100.21,100.5,105.9,104.82
    Poisson,151.65,135.11,105.1,103.89,102.43,116.29,114.34
    Neg. Binomial,476.34,453.07,429.23,426.31,423.45,429.81,430.05
    Multinomial,1166.63,1146.25,1111.39,1097.51,1095.43,1103.77,1109.79

The next table presents the performance relative to `xoroshiro128+`. The overall
performance was computed using a geometric mean.

.. csv-table::
    :header: ,NumPy MT19937,MT19937,SFMT,dSFMT,xoroshiro128+,xorshift1024,PCG64
    :widths: 14,14,14,14,14,14,14,14

    Raw,,4.69,1.83,1.77,1.0,1.38,2.93
    Random Integers,2.43,1.67,1.19,1.69,1.0,1.27,1.17
    Uniforms,6.59,4.13,1.71,1.71,1.0,1.46,1.56
    Normal,7.26,1.6,1.04,1.21,1.0,1.22,1.22
    Complex Normal,,1.57,1.09,1.24,1.0,1.24,1.23
    Binomial,1.17,1.07,0.99,0.99,1.0,1.02,1.05
    Gamma,1.31,1.19,1.05,1.05,1.0,1.06,1.05
    Exponential,1.2,1.11,1.02,1.01,1.0,1.04,1.05
    Laplace,1.16,1.14,1.02,1.0,1.0,1.05,1.04
    Poisson,1.48,1.32,1.03,1.01,1.0,1.14,1.12
    Neg. Binomial,1.12,1.07,1.01,1.01,1.0,1.02,1.02
    Multinomial,1.06,1.05,1.01,1.0,1.0,1.01,1.01
    Overall,1.84,1.55,1.14,1.19,1.0,1.15,1.22


.. note::

   All timings were taken using Linux and gcc 5.4 on a i7-5600U processor.