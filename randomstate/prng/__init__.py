from .xorshift128 import xorshift128
from .xoroshiro128plus import xoroshiro128plus
from .xorshift1024 import xorshift1024
from .mlfg_1279_861 import mlfg_1279_861
from .mt19937 import mt19937
from .mrg32k3a import mrg32k3a
from .pcg32 import pcg32
from .pcg64 import pcg64
from .dsfmt import dsfmt

def __generic_ctor(mod_name='mt19937'):
    """
    Pickling helper function that returns a mod_name.RandomState object

    Parameters
    ----------
    mod_name: str
        String containing the module name

    Returns
    -------
    rs: RandomState
        RandomState from the module randomstate.prng.mod_name
    """
    try:
        mod_name = mod_name.decode('ascii')
    except AttributeError:
        pass
    if mod_name == 'mt19937':
        mod = mt19937
    elif mod_name == 'mlfg_1279_861':
        mod = mlfg_1279_861
    elif mod_name == 'mrg32k3a':
        mod = mrg32k3a
    elif mod_name == 'pcg32':
        mod = pcg32
    elif mod_name == 'pcg64':
        mod = pcg64
    elif mod_name == 'pcg32':
        mod = pcg32
    elif mod_name == 'xorshift128':
        mod = xorshift128
    elif mod_name == 'xoroshiro128plus':
        mod = xoroshiro128plus
    elif mod_name == 'xorshift1024':
        mod = xorshift1024
    elif mod_name == 'dsfmt':
        mod = dsfmt
    else:
        raise ValueError(str(mod_name) + ' is not a known PRNG module.')

    return mod.RandomState(0)
