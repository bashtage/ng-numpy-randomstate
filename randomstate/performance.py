import os
import struct
import timeit

import pandas as pd
import numpy as np
from numpy.random import RandomState

rs = RandomState()

SETUP = '''
import numpy as np
import {mod}.{rng}
rs = {mod}.{rng}.RandomState()
rs.random_sample()
'''

scale_32 = scale_64 = 1
if struct.calcsize('P') == 8 and os.name != 'nt':
    # 64 bit
    scale_32 = 0.5
else:
    scale_64 = 2

RNGS = ['mlfg_1279_861', 'mrg32k3a', 'pcg64', 'pcg32', 'mt19937', 'xorshift128', 'xorshift1024',
        'xoroshiro128plus', 'dsfmt', 'random']


def timer(code, setup):
    return 1000 * min(timeit.Timer(code, setup=setup).repeat(10, 10)) / 10.0


def print_legend(legend):
    print('\n' + legend + '\n' + '*' * max(60, len(legend)))


def run_timer(dist, command, numpy_command=None, setup='', random_type=''):
    print('-' * 80)
    if numpy_command is None:
        numpy_command = command

    res = {}
    for rng in RNGS:
        mod = 'randomstate.prng' if rng != 'random' else 'numpy'
        key = '-'.join((mod, rng, dist)).replace('"', '')
        command = numpy_command if 'numpy' in mod else command
        res[key] = timer(command.format(dist=dist), setup=setup.format(mod=mod, rng=rng))

    s = pd.Series(res)
    t = s.apply(lambda x: '{0:0.2f} ms'.format(x))
    print_legend('Time to produce 1,000,000 ' + random_type)
    print(t.sort_index())

    p = 1000.0 / s
    p = p.apply(lambda x: '{0:0.2f} million'.format(x))
    print_legend(random_type + ' per second')
    print(p.sort_index())

    baseline = [k for k in p.index if 'numpy' in k][0]
    p = 1000.0 / s
    p = p / p[baseline] * 100 - 100
    p = p.drop(baseline, 0)
    p = p.apply(lambda x: '{0:0.1f}%'.format(x))
    print_legend('Speed-up relative to NumPy')
    print(p.sort_index())
    print('-' * 80)


def timer_uniform():
    dist = 'random_sample'
    command = 'rs.{dist}(1000000)'
    run_timer(dist, command, None, SETUP, 'Uniforms')


def timer_32bit():
    info = np.iinfo(np.uint32)
    min, max = info.min, info.max
    dist = 'randint'
    command = 'rs.{dist}({min}, {max}+1, 1000000, dtype=np.uint64)'
    command = command.format(dist='{dist}', min=min, max=max)
    command_numpy = command
    run_timer(dist, command, None, SETUP, '32-bit unsigned integers')


def timer_64bit():
    info = np.iinfo(np.uint64)
    min, max = info.min, info.max
    dist = 'randint'
    command = 'rs.{dist}({min}, {max}+1, 1000000, dtype=np.uint64)'
    command = command.format(dist='{dist}', min=min, max=max)
    command_numpy = command
    run_timer(dist, command, None, SETUP, '64-bit unsigned integers')


def timer_normal():
    command = 'rs.{dist}(1000000, method="bm")'
    command_numpy = 'rs.{dist}(1000000)'
    dist = 'standard_normal'
    run_timer(dist, command, command_numpy, SETUP, 'Box-Muller normals')


def timer_normal_zig():
    command = 'rs.{dist}(1000000, method="zig")'
    command_numpy = 'rs.{dist}(1000000)'
    dist = 'standard_normal'
    run_timer(dist, command, command_numpy, SETUP, 'Standard normals (Ziggurat)')


if __name__ == '__main__':
    timer_uniform()
    timer_32bit()
    timer_64bit()
    timer_normal()
    timer_normal_zig()
