#import randomstate.mlfg_1279_861 as mlfg_1279_861
#import randomstate.mrg32k3a as mrg32k3a
import randomstate.mt19937 as __mt19937
#import randomstate.pcg32 as pcg32
#import randomstate.xorshift1024 as xorshift1024
#import randomstate.xorshift128 as xorshift128

#try:
#    import randomstate.pcg64
#except ImportError:
#    pass

__all__ = ['standard_normal', 'standard_gamma', 'get_state', 'set_state', 'seed', 'bytes',
           'standard_exponential', 'standard_cauchy', 'standard_t', 'rand', 'randn']

__rs = __mt19937.RandomState()
RandomState = __mt19937.RandomState
standard_normal = __rs.standard_normal
standard_gamma = __rs.standard_gamma
standard_exponential = __rs.standard_exponential
standard_cauchy = __rs.standard_cauchy
standard_t = __rs.standard_t
random_sample = __rs.random_sample
get_state = __rs.get_state
set_state = __rs.set_state
seed = __rs.seed
bytes = __rs.bytes
randn = __rs.randn
rand = __rs.rand
