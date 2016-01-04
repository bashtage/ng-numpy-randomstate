cimport numpy as np

cdef inline double kahan_sum(double *darr, np.npy_intp n):
    cdef double c, y, t, sum
    cdef np.npy_intp i
    sum = darr[0]
    c = 0.0
    for i in range(1, n):
        y = darr[i] - c
        t = sum + y
        c = (t-sum) - y
        sum = t
    return sum
