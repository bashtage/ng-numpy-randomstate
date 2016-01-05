DEF RNG_NAME = "xorshift-1024*"
DEF RNG_ADVANCEABLE = 0
DEF RNG_JUMPABLE = 1
DEF RNG_SEED = 1
DEF RNG_STATE_LEN = 4
DEF NORMAL_METHOD = 'zig'

cdef extern from "distributions.h":

    cdef struct s_xorshift1024_state:
      uint64_t s[16]
      int p

    ctypedef s_xorshift1024_state xorshift1024_state

    cdef struct s_aug_state:
        xorshift1024_state *rng
        binomial_t *binomial

        int has_gauss, shift_zig_random_int, has_uint32
        double gauss
        uint64_t zig_random_int
        uint32_t uinteger

    ctypedef s_aug_state aug_state

    cdef void set_seed(aug_state* state, uint64_t seed)

    cdef void jump(aug_state* state)

    cdef void init_state(aug_state* state, uint64_t* state_vals)

ctypedef object rng_state_t

ctypedef xorshift1024_state rng_t

cdef object _get_state(aug_state state):
    cdef uint64_t [:] key = np.zeros(16, dtype=np.uint64)
    for i in range(16):
        key[i] = state.rng.s[i]
    return (np.asanyarray(key), state.rng.p)

cdef object _set_state(aug_state state, object state_info):
    cdef uint64_t [:] key = state_info[0]
    cdef Py_ssize_t i
    for i in range(16):
        state.rng.s[i] = key[i]
    state.rng.p = state_info[1]

DEF CLASS_DOCSTRING = """
This is the xorshift1024 docstring.
"""
