DEPRECATION_MESSAGE = """
**End-of-life notification**

This library was designed to bring alternative generators to the NumPy 
infrastructure. It as been successful in advancing the conversation 
for a future implementation of a new random number API in NumPy which 
will allow new algorithms and/or generators. The next step
in this process is to separate the basic (or core RNG) from the 
functions that transform random bits into useful random numbers.
This has been implemented in a successor project  **randomgen** 
available on GitHub

https://github.com/bashtage/randomgen

or PyPi

https://pypi.org/project/randomstate/.

randomgen has a slightly different API, so please see the randomgen documentation

https://bashtage.github.io/randomgen.
"""


class RandomStateDeprecationWarning(Warning):
    pass
