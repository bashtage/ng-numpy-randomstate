DEF RNG_NAME = "xorshift-1024*"
DEF RNG_ADVANCEABLE = 0
DEF RNG_SEED = 0

cdef extern from "core-rng.h":

    cdef struct s_xorshift1024_state:
      uint64_t s[16]
      int p

    ctypedef s_xorshift1024_state xorshift1024_state

    cdef struct s_aug_state:
        xorshift1024_state *rng

        int has_gauss, shift_zig_random_int, has_uint32
        double gauss
        uint64_t zig_random_int
        uint32_t uinteger

    ctypedef s_aug_state aug_state

    cdef void seed(aug_state* state, uint64_t* seed)

ctypedef object rng_state_t

ctypedef xorshift1024_state rng_t

cdef object _get_state(aug_state state):
    cdef uint64_t [:] key = np.zeros(16, dtype=np.uint64)
    for i in range(16):
        key[i] = state.rng.s[i]
    return (np.asanyarray(key), state.rng.p)

