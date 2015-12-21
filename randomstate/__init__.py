import randomstate.mt19937

__all__ = ['standard_normal', 'standard_gamma', 'get_state', 'set_state', 'seed', 'bytes',
           'standard_exponential', 'standard_cauchy', 'standard_t', 'rand', 'randn']

_rs = mt19937.RandomState()
standard_normal = _rs.standard_normal
standard_gamma = _rs.standard_gamma
standard_exponential = _rs.standard_exponential
standard_cauchy = _rs.standard_cauchy
standard_t = _rs.standard_t
random_sample = _rs.random_sample
get_state = _rs.get_state
set_state = _rs.set_state
seed = _rs.seed
bytes = _rs.bytes

def randn(*args):
    return standard_normal(size=args)

def rand(*args):
    return random_sample(size=args)


