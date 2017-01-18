from __future__ import division, absolute_import, print_function

from randomstate.prng.mt19937 import *
from randomstate.entropy import random_entropy
import randomstate.prng

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
