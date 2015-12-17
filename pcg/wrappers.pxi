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
