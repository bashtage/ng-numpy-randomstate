ctypedef double (* random_double_0)(aug_state* state)
ctypedef double (* random_double_1)(aug_state* state, double a)
ctypedef double (* random_double_2)(aug_state* state, double a, double b)
ctypedef double (* random_double_3)(aug_state* state, double a, double b, double c)

ctypedef uint64_t (* random_uint_0)(aug_state* state)
ctypedef uint64_t (* random_uint_1)(aug_state* state, double a)
ctypedef uint64_t (* random_uint_2)(aug_state* state, double a, double b)
ctypedef uint64_t (* random_uint_1_i)(aug_state* state, uint64_t a)

ctypedef uint32_t (* random_uint_0_32)(aug_state* state)
ctypedef uint32_t (* random_uint_1_i_32)(aug_state* state, uint32_t a)

ctypedef int32_t (* random_int_2_i_32)(aug_state* state, int32_t a, int32_t b)
ctypedef int64_t (* random_int_2_i)(aug_state* state, int64_t a, int64_t b)

cdef Py_ssize_t compute_numel(size):
    cdef Py_ssize_t i, n = 1
    if isinstance(size, tuple):
        for i in range(len(size)):
            n *= size[i]
    else:
        n = size
    return n

cdef object cont0(aug_state* state, random_double_0 func, size):
    if size is None:
        return func(state)
    cdef Py_ssize_t n = compute_numel(size)
    cdef double [:] randoms = np.empty(n, np.double)
    for i in range(n):
        randoms[i] = func(state)
    return np.asanyarray(randoms).reshape(size)

cdef object cont1(aug_state* state, random_double_1 func, double a, size):
    if size is None:
        return func(state, a)
    cdef Py_ssize_t n = compute_numel(size)
    cdef double [:] randoms = np.empty(n, np.double)
    for i in range(n):
        randoms[i] = func(state, a)
    return np.asanyarray(randoms).reshape(size)

cdef object cont2(aug_state* state, random_double_2 func, double a, double b, size):
    if size is None:
        return func(state, a, b)
    cdef Py_ssize_t n = compute_numel(size)
    cdef double [:] randoms = np.empty(n, np.double)
    for i in range(n):
        randoms[i] = func(state, a, b)
    return np.asanyarray(randoms).reshape(size)

cdef object cont3(aug_state* state, random_double_3 func, double a, double b, double c, size):
    if size is None:
        return func(state, a, b, c)
    cdef Py_ssize_t n = compute_numel(size)
    cdef double [:] randoms = np.empty(n, np.double)
    for i in range(n):
        randoms[i] = func(state, a, b, c)
    return np.asanyarray(randoms).reshape(size)

cdef object uint0(aug_state* state, random_uint_0 func, size):
    if size is None:
        return func(state)
    cdef Py_ssize_t n = compute_numel(size)
    cdef uint64_t [:] randoms = np.empty(n, np.uint64)
    for i in range(n):
        randoms[i] = func(state)
    return np.asanyarray(randoms).reshape(size)

cdef object uint0_32(aug_state* state, random_uint_0_32 func, size):
    if size is None:
        return func(state)
    cdef Py_ssize_t n = compute_numel(size)
    cdef uint32_t [:] randoms = np.empty(n, np.uint32)
    for i in range(n):
        randoms[i] = func(state)
    return np.asanyarray(randoms).reshape(size)

cdef object uint1_i(aug_state* state, random_uint_1_i func, uint64_t a, size):
    if size is None:
        return func(state, a)
    cdef Py_ssize_t n = compute_numel(size)
    cdef uint64_t [:] randoms = np.empty(n, np.uint64)
    for i in range(n):
        randoms[i] = func(state, a)
    return np.asanyarray(randoms).reshape(size)

cdef object uint1_i_32(aug_state* state, random_uint_1_i_32 func, uint32_t a, size):
    if size is None:
        return func(state, a)
    cdef Py_ssize_t n = compute_numel(size)
    cdef uint64_t [:] randoms = np.empty(n, np.uint64)
    for i in range(n):
        randoms[i] = func(state, a)
    return np.asanyarray(randoms).reshape(size)


cdef object int2_i(aug_state* state, random_int_2_i func, int64_t a, int32_t b, size):
    if size is None:
        return func(state, a, b)
    cdef Py_ssize_t n = compute_numel(size)
    cdef int64_t [:] randoms = np.empty(n, np.int64)
    for i in range(n):
        randoms[i] = func(state, a, b)
    return np.asanyarray(randoms).reshape(size)

cdef object int2_i_32(aug_state* state, random_int_2_i_32 func, int32_t a, int32_t b, size):
    if size is None:
        return func(state, a, b)
    cdef Py_ssize_t n = compute_numel(size)
    cdef int64_t [:] randoms = np.empty(n, np.int64)
    for i in range(n):
        randoms[i] = func(state, a, b)
    return np.asanyarray(randoms).reshape(size)
