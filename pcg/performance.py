import timeit
import sys
import pandas as pd

scale_32 = scale_64 = 1
if sys.maxsize < 2**32:
    # 32 bit
    scale_64 = 2
else:
    scale_32 = 0.5


def timer(code, setup):
    return 1000 * min(timeit.Timer(code, setup=setup).repeat(3, 10)) / 10.0


def print_legend(legend):
    print ('\n' + legend + '\n' + '*' * 60)


SETUP = '''
import {rng}

rs = {rng}.RandomState()
rs.random_sample()
'''

COMMAND = '''
rs.{dist}(1000000, method={method})
'''

COMMAND_NUMPY = '''
rs.{dist}(1000000)
'''

dist = 'standard_normal'
res = {}
for rng in ('dummy', 'pcg32', 'pcg64', 'mt19937', 'xorshift128', 'xorshift1024',
            'mrg32k3a', 'numpy.random'):
    for method in ('"inv"', '"zig"'):
        try:
            key = '_'.join((rng, method, dist)).replace('"', '')
            command = COMMAND if 'numpy' not in rng else COMMAND_NUMPY
            if 'numpy' in rng and 'zig' in method:
                continue
            res[key] = timer(command.format(dist=dist, method=method), setup=SETUP.format(rng=rng))
        except:
            pass

s = pd.Series(res)
t = s.apply(lambda x: '{0:0.2f} ms'.format(x))
print_legend('Time to produce 1,000,000 normals')
print(t.sort_index())

p = 1000.0 / s
p = p.apply(lambda x: '{0:0.2f} million'.format(x))
print_legend('Normals per second')
print(p.sort_index())

baseline = 'numpy.random_inv_standard_normal'
p = 1000.0 / s
p = p / p[baseline] * 100 - 100
p = p.drop(baseline, 0)
p = p.apply(lambda x: '{0:0.1f}%'.format(x))
print_legend('Speed-up relative to NumPy')
print(p.sort_index())

print('\n\n')
print((('-' * 60) + '\n') * 2)
COMMAND = '''
rs.{dist}(1000000)
'''

dist = 'random_sample'
res = {}
for rng in ('mrg32k3a', 'pcg64', 'pcg32', 'mt19937', 'xorshift128', 'xorshift1024', 'numpy.random'):
    try:
        key = '_'.join((rng, dist)).replace('"', '')
        command = COMMAND
        res[key] = timer(command.format(dist=dist), setup=SETUP.format(rng=rng))
    except:
        pass

s = pd.Series(res)
t = s.apply(lambda x: '{0:0.2f} ms'.format(x))
print_legend('Time to produce 1,000,000 uniforms')
print(t.sort_index())

p = 1000.0 / s
p = p.apply(lambda x: '{0:0.2f} million'.format(x))
print_legend('uniforms per second')
print(p.sort_index())

baseline = [k for k in p.index if 'numpy' in k][0]
p = 1000.0 / s
p = p / p[baseline] * 100 - 100
p = p.drop(baseline, 0)
p = p.apply(lambda x: '{0:0.1f}%'.format(x))
print_legend('Speed-up relative to NumPy')
print(p.sort_index())


print('\n\n')
print((('-' * 60) + '\n') * 2)
COMMAND = '''
rs.{dist}(1000000, bits=32)
'''

COMMAND_NUMPY = '''
rs.tomaxint({scale} * 1000000)
'''.format(scale=scale_32)

dist = 'random_integers'
res = {}
for rng in ('mrg32k3a', 'pcg64', 'pcg32', 'mt19937', 'xorshift128', 'xorshift1024', 'numpy.random'):
    try:
        key = '_'.join((rng, dist)).replace('"', '')
        command = COMMAND if 'numpy' not in rng else COMMAND_NUMPY
        res[key] = timer(command.format(dist=dist), setup=SETUP.format(rng=rng))
    except:
        pass

s = pd.Series(res)
t = s.apply(lambda x: '{0:0.2f} ms'.format(x))
print_legend('Time to produce 1,000,000 32-bit uints')
print(t.sort_index())

p = 1000.0 / s
p = p.apply(lambda x: '{0:0.2f} million'.format(x))
print_legend('32-bit unsigned integers per second')
print(p.sort_index())

baseline = [k for k in p.index if 'numpy' in k][0]
p = 1000.0 / s
p = p / p[baseline] * 100 - 100
p = p.drop(baseline, 0)
p = p.apply(lambda x: '{0:0.1f}%'.format(x))
print_legend('Speed-up relative to NumPy')
print(p.sort_index())


print('\n\n')
print((('-' * 60) + '\n') * 2)
COMMAND = '''
rs.{dist}(1000000)
'''

COMMAND_NUMPY = '''
rs.tomaxint({scale} * 1000000)
'''.format(scale=scale_64)

dist = 'random_integers'
res = {}
for rng in ('mrg32k3a', 'pcg64', 'pcg32', 'mt19937', 'xorshift128', 'xorshift1024', 'numpy.random'):
    try:
        key = '_'.join((rng, dist)).replace('"', '')
        command = COMMAND if 'numpy' not in rng else COMMAND_NUMPY
        res[key] = timer(command.format(dist=dist), setup=SETUP.format(rng=rng))
    except:
        pass

s = pd.Series(res)
t = s.apply(lambda x: '{0:0.2f} ms'.format(x))
print_legend('Time to produce 1,000,000 64-bit uints')
print(t.sort_index())

p = 1000.0 / s
p = p.apply(lambda x: '{0:0.2f} million'.format(x))
print_legend('64-bit unsigned integers per second')
print(p.sort_index())

baseline = [k for k in p.index if 'numpy' in k][0]
p = 1000.0 / s
p = p / p[baseline] * 100 - 100
p = p.drop(baseline, 0)
p = p.apply(lambda x: '{0:0.1f}%'.format(x))
print_legend('Speed-up relative to NumPy')
print(p.sort_index())


