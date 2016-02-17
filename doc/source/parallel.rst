Parallel Random Number Generation
=================================

There are three strategies implemented that can be used to produce
repeatable pseudo-random numbers across multiple processes (local
or distributed).

.. _independent-streams:

Independent Streams
-------------------

Currently only ``pcg32`` and ``pcg64`` support independent streams, and,
due to the limited period of ``pcg32`` (:math:`2^{64}`), only ``pcg64``
should be used.  This example shows how many streams can be created by
passing in different index values in the second input while using the
same seed in the first.

::

  from randomstate.entropy import random_entropy
  import randomstate.prng.pcg64 as pcg64

  entropy = random_entropy(4)
  # 128-bit number as a seed
  seed = reduce(lambda x, y: x + y, [long(entropy[i]) * 2 ** (32 * i) for i in range(4)])
  streams = [pcg64.RandomState(seed, stream) for stream in range(10)]


.. _jump-and-advance:

Jump/Advance the PRNG state
---------------------------

``jump`` advances the state of the PRNG *as-if* a large number of random
numbers have been drawn.  The specific number of draws varies by PRNG, and
ranges from :math:`2^{64}` to :math:`2^{512}`.  Additionally, the *as-if*
draws also depend on the size of the default random number produced by the
specific PRNG.  The PRNGs that support ``jump``, along with the period of
the PRNG, the size of the jump and the bits in the default unsigned random
are listed below.

+-----------------+-------------------------+-------------------------+-------------------------+
| PRNG            | Period                  |  Jump Size              | Bits                    |
+=================+=========================+=========================+=========================+
| dsfmt           | :math:`2^{19937}`       | :math:`2^{128}`         | 53                      |
+-----------------+-------------------------+-------------------------+-------------------------+
| mrg32k3a        | :math:`\approx 2^{191}` | :math:`2^{127}`         | 32                      |
+-----------------+-------------------------+-------------------------+-------------------------+
| xorshift128     | :math:`2^{128}`         | :math:`2^{64}`          | 64                      |
+-----------------+-------------------------+-------------------------+-------------------------+
| xorshift1024    | :math:`2^{1024}`        | :math:`2^{512}`         | 64                      |
+-----------------+-------------------------+-------------------------+-------------------------+

``jump`` can be used to produce long blocks which should be long enough to not
overlap.

::

  from randomstate.entropy import random_entropy
  import randomstate.prng.xorshift1024 as xorshift1024

  entropy = random_entropy(2).astype(np.uint64)
  # 64-bit number as a seed
  seed = entropy[0] * 2**32 + entropy[1]
  blocks = []
  for i in range(10):
      block = xorshift1024.RandomState(seed)
      block.jump(i)
      blocks.append(block)


``advance`` can be used to jump the state an arbitrary number of steps, and so
is a more general approach than ``jump``.  Only ``pcg32`` and ``pcg64``
support ``advance``, and since these also support independent streams, it is
not usually necessary to use ``advance``.
