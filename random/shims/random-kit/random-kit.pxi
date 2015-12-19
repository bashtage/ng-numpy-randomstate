DEF RNG_NAME = 'mt19937'
DEF RNG_ADVANCEABLE = 0
DEF RNG_SEED = 1
DEF RNG_JUMPABLE = 0
DEF RNG_STATE_LEN = 4

DEF RK_STATE_LEN = 624

ctypedef uint32_t rng_state_t

cdef extern from "core-rng.h":

    cdef struct s_rk_state:
      uint32_t key[RK_STATE_LEN]
      int pos

    ctypedef s_rk_state rk_state

    cdef struct s_aug_state:
        rk_state *rng
        uint64_t state, inc

        int has_gauss, shift_zig_random_int, has_uint32
        double gauss
        uint64_t zig_random_int
        uint32_t uinteger

    ctypedef s_aug_state aug_state

    cdef void seed(aug_state* state, uint32_t seed)

ctypedef rk_state rng_t

cdef object _get_state(aug_state state):
    cdef uint32_t [:] key = np.zeros(RK_STATE_LEN, dtype=np.uint32)
    cdef Py_ssize_t i
    for i in range(RK_STATE_LEN):
        key[i] = state.rng.key[i]
    return (np.asanyarray(key), state.rng.pos)

cdef object _set_state(aug_state state, object state_info):
    cdef uint32_t [:] key = state_info[0]
    cdef Py_ssize_t i
    for i in range(RK_STATE_LEN):
        state.rng.key[i] = key[i]
    state.rng.pos = state_info[1]

DEF CLASS_DOCSTRING = """
This is the mt19937 docstring.
"""
