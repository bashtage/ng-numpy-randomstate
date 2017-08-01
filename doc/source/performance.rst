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

    Raw,,4.21,1.81,1.7,1.06,1.5,2.67
    Random Integers,4.56,3.11,2.09,2.96,1.93,2.34,2.22
    Uniforms,9.77,6.13,2.41,2.22,1.46,2.37,2.56
    Normal,62.47,13.77,9.11,10.89,7.8,10.48,10.67
    Exponential,98.35,9.85,5.72,7.22,5.56,6.32,10.88
    Complex Normal,,37.07,24.41,27.88,23.31,27.78,28.34
    Gamma,97.99,44.94,38.27,33.41,31.2,34.18,34.09
    Binomial,87.99,79.95,78.85,77.12,76.88,76.09,76.99
    Laplace,101.73,103.95,91.57,89.02,91.94,93.61,92.13
    Poisson,131.93,119.95,99.42,94.84,92.71,100.28,101.17
    Neg. Binomial,433.77,416.69,410.2,397.71,389.21,396.14,394.78
    Multinomial,1072.82,1043.98,1021.58,1019.22,1016.7,1013.15,1018.41


The next table presents the performance relative to `xoroshiro128+`. The overall
performance was computed using a geometric mean.

.. csv-table::
    :header: ,NumPy MT19937,MT19937,SFMT,dSFMT,xoroshiro128+,xorshift1024,PCG64
    :widths: 14,14,14,14,14,14,14,14
    
    Raw,,3.97,1.71,1.6,1.0,1.42,2.52
    Random Integers,2.36,1.61,1.08,1.53,1.0,1.21,1.15
    Uniforms,6.69,4.2,1.65,1.52,1.0,1.62,1.75
    Normal,8.01,1.77,1.17,1.4,1.0,1.34,1.37
    Exponential,17.69,1.77,1.03,1.3,1.0,1.14,1.96
    Complex Normal,,1.59,1.05,1.2,1.0,1.19,1.22
    Gamma,3.14,1.44,1.23,1.07,1.0,1.1,1.09
    Binomial,1.14,1.04,1.03,1.0,1.0,0.99,1.0
    Laplace,1.11,1.13,1.0,0.97,1.0,1.02,1.0
    Poisson,1.42,1.29,1.07,1.02,1.0,1.08,1.09
    Neg. Binomial,1.11,1.07,1.05,1.02,1.0,1.02,1.01
    Multinomial,1.06,1.03,1.0,1.0,1.0,1.0,1.0
    Overall,2.61,1.62,1.15,1.2,1.0,1.16,1.28


.. note::

   All timings were taken using Linux and gcc 5.4 on a i5-3570 processor.
