from collections import OrderedDict

import timeit

import numpy as np
import pandas as pd
from randomstate.prng import (mt19937, sfmt, dsfmt, xoroshiro128plus,
                              xorshift1024, pcg64)

REPS = 3
SIZE = 100000
SETUP = """
import numpy
from numpy import array, random
from randomstate.prng import (mt19937, sfmt, dsfmt, xoroshiro128plus,
                              xorshift1024, pcg64)
import randomstate
import randomstate.prng

rs = {prng}.RandomState(123456)
f = rs.__getattribute__('{fn}')
args = {args}
"""
prngs = (np.random, mt19937, sfmt, dsfmt, xoroshiro128plus, xorshift1024, pcg64)
functions = {'randint': {'low': 2 ** 31, 'dtype': 'uint32'},
             'random_sample': {},
             'random_raw': {'output': False},
             'standard_exponential': {'method': 'zig'},
             'standard_gamma': {'shape': 2.4, 'method': 'zig'},
             'standard_normal': {'method': 'zig'},
             'multinomial': {'n': 20, 'pvals': [1.0 / 6.0] * np.ones(6)},
             'negative_binomial': {'n': 5, 'p': 0.16},
             'poisson': {'lam': 3.0},
             'complex_normal': {'gamma': 2 + 0j, 'relation': 1 + 0.5j, 'method': 'zig'},
             'laplace': {'loc': 1, 'scale': 3},
             'binomial': {'n': 35, 'p': 0.25}}


def timer(prng: str, fn: str, args: dict):
    if prng == 'random':
        # Differences with NumPy
        if fn in ('random_raw', 'complex_normal'):
            return np.nan
        if fn in ('standard_normal','standard_exponential', 'standard_gamma'):
            args = {k: v for k, v in args.items() if k != 'method'}
    elif prng == 'mt19937' and fn == 'random_raw':  # To make comparable
        args['size'] = 2 * args['size']
    setup = SETUP.format(prng=prng, fn=fn, args=args)
    return min(timeit.Timer('f(**args)', setup=setup).repeat(10, REPS)) / SIZE / REPS


results = OrderedDict()
for prng in prngs:
    name = prng.__name__.split('.')[-1]
    speeds = OrderedDict()
    for fn, args in functions.items():
        args['size'] = SIZE
        speeds[fn] = np.round(timer(name, fn, args) * 10 ** 9, 2)
    results[name] = pd.Series(speeds)
    print(name)
    print(results[name])

results = pd.DataFrame(results)
results = results.loc[results.mean(1).sort_values().index]

index = {'randint': 'Random Integers',
         'random_sample': 'Uniforms',
         'random_raw': 'Raw',
         'standard_exponential': 'Exponential',
         'standard_gamma': 'Gamma',
         'standard_normal': 'Normal',
         'multinomial': 'Multinomial',
         'negative_binomial': 'Neg. Binomial',
         'poisson': 'Poisson',
         'complex_normal': 'Complex Normal',
         'laplace': 'Laplace',
         'binomial': 'Binomial'}

cols = {'sfmt': 'SFMT', 'dsfmt': 'dSFMT',
        'xoroshiro128plus': 'xoroshiro128+',
        'xorshift1024': 'xorshift1024', 'pcg64': 'PCG64',
        'mt19937': 'MT19937', 'random': 'NumPy MT19937'}

results.columns = [cols[c] for c in results]
results.index = [index[i] for i in results.index]

print(results)

from io import StringIO

sio = StringIO()
results.to_csv(sio)
sio.seek(0)
lines = sio.readlines()
for i, line in enumerate(lines):
    if i == 0:
        line = '    :header: ' + line
    else:
        line = '    ' + line
    lines[i] = line

lines.insert(1, '    \n')
lines.insert(1, '    :widths: 14,14,14,14,14,14,14,14\n')
lines.insert(0, '.. csv-table::\n')
print(''.join(lines))


std_results = (results.T / results.iloc[:,-3]).T
overall = np.exp(np.mean(np.log(std_results)))
overall.name = 'Overall'
std_results = std_results.append(overall)
std_results = np.round(std_results, 2)

print('\n\n' + '*'*80)
print(std_results)
print('\n'*4)

sio = StringIO()
std_results.to_csv(sio)
sio.seek(0)
lines = sio.readlines()
for i, line in enumerate(lines):
    if i == 0:
        line = '    :header: ' + line
    else:
        line = '    ' + line
    lines[i] = line

lines.insert(1, '    \n')
lines.insert(1, '    :widths: 14,14,14,14,14,14,14,14\n')
lines.insert(0, '.. csv-table::\n')
print(''.join(lines))
