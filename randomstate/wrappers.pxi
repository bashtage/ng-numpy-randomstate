ctypedef double (* random_double_0)(aug_state* state) nogil
ctypedef double (* random_double_1)(aug_state* state, double a) nogil
ctypedef double (* random_double_2)(aug_state* state, double a, double b) nogil
ctypedef double (* random_double_3)(aug_state* state, double a, double b, double c) nogil

ctypedef uint64_t (* random_uint_0)(aug_state* state) nogil
ctypedef uint64_t (* random_uint_d)(aug_state* state, double a) nogil
ctypedef uint64_t (* random_uint_dd)(aug_state* state, double a, double b) nogil
ctypedef uint64_t (* random_uint_di)(aug_state* state, double a, uint64_t b) nogil
ctypedef uint64_t (* random_uint_i)(aug_state* state, uint64_t a) nogil
ctypedef uint64_t (* random_uint_iii)(aug_state* state, uint64_t a, uint64_t b, uint64_t c) nogil

ctypedef uint32_t (* random_uint_0_32)(aug_state* state) nogil
ctypedef uint32_t (* random_uint_1_i_32)(aug_state* state, uint32_t a) nogil

ctypedef int32_t (* random_int_2_i_32)(aug_state* state, int32_t a, int32_t b) nogil
ctypedef int64_t (* random_int_2_i)(aug_state* state, int64_t a, int64_t b) nogil

cdef Py_ssize_t compute_numel(size):
    cdef Py_ssize_t i, n = 1
    if isinstance(size, tuple):
        for i in range(len(size)):
            n *= size[i]
    else:
        n = size
    return n

cdef object uint0_32(aug_state* state, random_uint_0_32 func, object size, object lock):
    if size is None:
        return func(state)
    cdef Py_ssize_t i, n = compute_numel(size)
    cdef uint32_t [:] randoms = np.empty(n, np.uint32)
    with lock, nogil:
        for i in range(n):
            randoms[i] = func(state)
    return np.asanyarray(randoms).reshape(size)

cdef object uint1_i_32(aug_state* state, random_uint_1_i_32 func, uint32_t a, object size, object lock):
    if size is None:
        return func(state, a)
    cdef Py_ssize_t i, n = compute_numel(size)
    cdef uint64_t [:] randoms = np.empty(n, np.uint64)
    with lock, nogil:
        for i in range(n):
            randoms[i] = func(state, a)
    return np.asanyarray(randoms).reshape(size)


cdef object int2_i(aug_state* state, random_int_2_i func, int64_t a, int32_t b, object size, object lock):
    if size is None:
        return func(state, a, b)
    cdef Py_ssize_t i, n = compute_numel(size)
    cdef int64_t [:] randoms = np.empty(n, np.int64)
    with lock, nogil:
        for i in range(n):
            randoms[i] = func(state, a, b)
    return np.asanyarray(randoms).reshape(size)

cdef object int2_i_32(aug_state* state, random_int_2_i_32 func, int32_t a, int32_t b, object size, object lock):
    if size is None:
        return func(state, a, b)
    cdef Py_ssize_t i, n = compute_numel(size)
    cdef int64_t [:] randoms = np.empty(n, np.int64)
    with lock, nogil:
        for i in range(n):
            randoms[i] = func(state, a, b)
    return np.asanyarray(randoms).reshape(size)
