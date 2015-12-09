import timeit

import pandas as pd


def timer(code, setup):
    return 1000 * min(timeit.Timer(code, setup=setup).repeat(3, 10)) / 10.0


def print_legend(legend):
    print ('\n' + legend + '\n' + '*' * 40)


setup = '''
import pcg

prs = pcg.PCGRandomState()
'''

pcg_normal = timer('prs.standard_normal(1000000)', setup)
pcg_normal_zig = timer('prs.standard_normal_zig(1000000)', setup)

np_setup = '''
import numpy as np

rs = np.random.RandomState()
'''
np_normal = timer('rs.standard_normal(1000000)', np_setup)

s = pd.Series({'pcg zig-based normal': pcg_normal_zig,
               'pcg normal': pcg_normal,
               'NumPy normal': np_normal})
t = s.apply(lambda x: '{0:0.2f} ms'.format(x))
print_legend('Time to produce 1,000,000 normals')
print(t.sort_index())

p = 1000.0 / s
p = p.apply(lambda x: '{0:0.2f} million'.format(x))
print_legend('Normals per second')
print(p.sort_index())

p = 1000.0 / s
p = p / p['NumPy normal'] * 100 - 100
p = p.drop('NumPy normal', 0)
p = p.apply(lambda x: '{0:0.1f}%'.format(x))
print_legend('Speed-up relative to NumPy')
print(p.sort_index())
