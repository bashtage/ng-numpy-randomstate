Multithreaded Generation
========================

The four core distributions all allow existing arrays to be filled using the
``out`` keyword argument.  Existing arrays need to be contiguous and
well-behaved (writable and algined).  Under normal circumstances, arrays
created using the common constructors such as ``numpy.empty`` will satisfy
these requirements.

This example makes use of Python 3 ``futures`` to fill an array using multiple
threads.  Threads are long-lived so that repeated calls do not require any
additional overheads from thread creation. The undelying PRNG is xorshift2014
which is fast, has a long period and supports using ``jump`` to advange the
state. The random numbers generated are reproducible in the sense that the
same seed will produce the same outputs.

::

    import randomstate
    import multiprocessing
    import concurrent.futures
    import numpy as np

    class MultithreadedRNG(object):
        def __init__(self, n, seed=None, threads=None):
            rs = randomstate.prng.xorshift1024.RandomState(seed)
            if threads is None:
                threads = multiprocessing.cpu_count()
            self.threads = threads

            self._random_states = [rs]
            for _ in range(1, threads):
                _rs = randomstate.prng.xorshift1024.RandomState()
                rs.jump()
                _rs.set_state(rs.get_state())
                self._random_states.append(_rs)

            self.n = n
            self.executor = concurrent.futures.ThreadPoolExecutor(threads)
            self.values = np.empty(n)
            self.step = np.ceil(n / threads).astype(np.int)

        def fill(self):
            def _fill(random_state, out, first, last):
                random_state.standard_normal(out=out[first:last])

            futures = {}
            for i in range(self.threads):
                args = (_fill, self._random_states[i], self.values, i * self.step, (i + 1) * self.step)
                futures[self.executor.submit(*args)] = i
            concurrent.futures.wait(futures)

        def __del__(self):
            self.executor.shutdown(False)


.. ipython:: python
   :suppress:

   In [1]: import randomstate
     ....: import multiprocessing
     ....: import concurrent.futures
     ....: import numpy as np
     ....:
     ....: class MultithreadedRNG(object):
     ....:     def __init__(self, n, seed=None, threads=None):
     ....:         rs = randomstate.prng.xorshift1024.RandomState(seed)
     ....:         if threads is None:
     ....:             threads = multiprocessing.cpu_count()
     ....:         self.threads = threads
     ....:         self._random_states = [rs]
     ....:         for _ in range(1, threads):
     ....:             _rs = randomstate.prng.xorshift1024.RandomState()
     ....:             rs.jump()
     ....:             _rs.set_state(rs.get_state())
     ....:             self._random_states.append(_rs)
     ....:         self.n = n
     ....:         self.executor = concurrent.futures.ThreadPoolExecutor(threads)
     ....:         self.values = np.empty(n)
     ....:         self.step = np.ceil(n / threads).astype(np.int)
     ....:     def fill(self):
     ....:         def _fill(random_state, out, first, last):
     ....:             random_state.standard_normal(out=out[first:last])
     ....:         futures = {}
     ....:         for i in range(self.threads):
     ....:             args = (_fill, self._random_states[i], self.values, i * self.step, (i + 1) * self.step)
     ....:             futures[self.executor.submit(*args)] = i
     ....:         concurrent.futures.wait(futures)
     ....:     def __del__(self):
     ....:         self.executor.shutdown(False)
     ....:

The multithreaded random number generator can be used to fill an array.
The ``values`` attributes shows the zero-value before the fill and the
random value after.

.. ipython:: python

   mrng = MultithreadedRNG(10000000, seed=0)
   print(mrng.values[-1])
   mrng.fill()
   print(mrng.values[-1])

The time required to produce using multiple threads can be compared to
the time required to generate using a single thread.

.. ipython:: python

   print(mrng.threads)
   %timeit mrng.fill()


The single threaded call directly uses the PRNG.

.. ipython:: python

   values = np.empty(10000000)
   %timeit randomstate.prng.xorshift1024.standard_normal(out=values)

The gains are substantial and the scaling is reasonable even for large that
are only moderately large.  The gains are even larger when compared to a call
that does not use an existing array due to array creation overhead.

.. ipython:: python

   %timeit randomstate.prng.xorshift1024.standard_normal(10000000)
