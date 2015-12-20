DEF RNG_NAME = 'mlfg-1279-861'
DEF RNG_ADVANCEABLE = 0
DEF RNG_JUMPABLE = 0
DEF RNG_STATE_LEN = 4
DEF RNG_SEED = 1

DEF MLFG_STATE_LEN = 1279

cdef extern from "core-rng.h":
    cdef struct s_mlfg_state:
        uint64_t lags[1279]
        int pos
        int lag_pos

    ctypedef s_mlfg_state mlfg_state

    cdef struct s_aug_state:
        mlfg_state *rng
        binomial_t *binomial

        int has_gauss, shift_zig_random_int, has_uint32
        double gauss
        uint64_t zig_random_int
        uint32_t uinteger

    ctypedef s_aug_state aug_state

    cdef void seed(aug_state* state, uint64_t seed)

ctypedef uint64_t rng_state_t

ctypedef mlfg_state rng_t

cdef object _get_state(aug_state state):
    cdef uint32_t [:] key = np.zeros(MLFG_STATE_LEN, dtype=np.uint32)
    cdef Py_ssize_t i
    for i in range(MLFG_STATE_LEN):
        key[i] = state.rng.lags[i]
    return (np.asanyarray(key), state.rng.pos, state.rng.lag_pos)

cdef object _set_state(aug_state state, object state_info):
    cdef uint32_t [:] key = state_info[0]
    cdef Py_ssize_t i
    for i in range(MLFG_STATE_LEN):
        state.rng.lags[i] = key[i]
    state.rng.pos = state_info[1]
    state.rng.lag_pos = state_info[2]

DEF CLASS_DOCSTRING = """
This is the mlfg docstring.
"""