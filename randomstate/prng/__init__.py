from randomstate.prng.mt19937 import mt19937
from randomstate.prng.mlfg_1279_861 import mlfg_1279_861
from randomstate.prng.mrg32k3a import mrg32k3a
from randomstate.prng.pcg32 import pcg32
from randomstate.prng.pcg64 import pcg64
from randomstate.prng.xorshift1024 import xorshift1024
from randomstate.prng.xorshift128 import xorshift128

def __generic_ctor(rng_name='mt19937'):
    print(rng_name)
    if rng_name == 'mt19937':
        mod = mt19937
    elif rng_name == 'mlfg_1279_861':
        mod = mlfg_1279_861
    elif rng_name == 'mrg32k3a':
        mod = mrg32k3a
    elif rng_name == 'pcg32':
        mod = pcg32
    elif rng_name == 'pcg64':
        mod = pcg64
    elif rng_name == 'pcg32':
        mod = pcg32
    elif rng_name == 'xorshift128+':
        mod = xorshift128
    elif rng_name == 'xorshift1024*':
        mod = xorshift1024
    return mod.RandomState(0)
