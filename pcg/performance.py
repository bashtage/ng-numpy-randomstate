import timeit

import pandas as pd


def timer(code, setup):
    time = 1000 * min(timeit.Timer(code, setup=setup).repeat(3, 10)) / 10.0
    return '{0:0.2f}'.format(time) + ' ms'


setup = '''
import pcg

prs = pcg.PCGRandomState()
'''

pcg_32_normal = timer('prs.standard_normal(1000000)', setup)
pcg_64_normal = timer('prs.standard_normal_64(1000000)', setup)
pcg_32_normal_zig = timer('prs.standard_normal_zig(1000000)', setup)

np_setup = '''
import numpy as np

rs = np.random.RandomState()
'''
np_normal = timer('rs.standard_normal(1000000)', np_setup)

s = pd.Series({'pcg32 Normal': pcg_32_normal,
                 'pcg32 Zig-based Normal': pcg_32_normal_zig,
                 'pcg64 normal': pcg_64_normal,
                 'NumPy normal': np_normal})

print(s.sort_values())
