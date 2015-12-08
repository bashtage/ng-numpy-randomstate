from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from os.path import join
from os import getcwd

pwd = getcwd()

sources = [join(pwd, 'pcg.pyx')] + ['pcg-advance-128.c','pcg-rngs-128.c']

defs = [('PCG_HAS_128BIT_OPS', '1'),
        ('__SIZEOF_INT128__','16')]
setup(
    ext_modules = cythonize([Extension("pcg",
                             sources=sources,
                             include_dirs=[pwd],
                             define_macros=defs,
                             extra_compile_args=['-std=c99'])])
)



