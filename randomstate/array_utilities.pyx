#!python
#cython: wraparound=False, nonecheck=False, boundscheck=False, cdivision=True
import numpy as np
cimport numpy as np
from libc.stdint cimport uint32_t, uint64_t, int64_t, int32_t
from array_utilities cimport (aug_state, random_uint_0_32, constraint_type,
    random_double_1, random_double_2, random_double_3, random_uint_iii,
    random_uint_i, random_uint_di, random_uint_0, random_uint_d,
    random_uint_dd, random_double_0, POISSON_LAM_MAX)

from cython_overrides cimport PyErr_Occurred, PyFloat_AsDouble, PyErr_Clear, PyInt_AsLong

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

cdef int check_array_constraint(np.ndarray val, object name, constraint_type cons) except -1:
    if cons == CONS_NON_NEGATIVE:
        if np.any(np.less(val, 0)):
            raise ValueError(name + " < 0")
    elif cons == CONS_POSITIVE:
        if np.any(np.less_equal(val, 0)):
            raise ValueError(name + " <= 0")
    elif cons == CONS_BOUNDED_0_1 or cons == CONS_BOUNDED_0_1_NOTNAN:
        if np.any(np.less_equal(val, 0)) or np.any(np.greater_equal(val, 1)):
            raise ValueError(name + " <= 0 or " + name + " >= 1")
        if cons == CONS_BOUNDED_0_1_NOTNAN:
            if np.any(np.isnan(val)):
                raise ValueError(name + ' contains NaNs')
    elif cons == CONS_GT_1:
        if np.any(np.less_equal(val, 1)):
            raise ValueError(name + " <= 1")
    elif cons == CONS_GTE_1:
        if np.any(np.less(val, 1)):
            raise ValueError(name + " < 1")
    elif cons == CONS_POISSON:
        if np.any(np.greater(val, POISSON_LAM_MAX)):
            raise ValueError(name + " value too large")
        if np.any(np.less(val, 0.0)):
            raise ValueError(name + " < 0")

    return 0



cdef int check_constraint(double val, object name, constraint_type cons) except -1:
    if cons == CONS_NON_NEGATIVE:
        if val < 0:
            raise ValueError(name + " < 0")
    elif cons == CONS_POSITIVE:
        if val <= 0:
            raise ValueError(name + " <= 0")
    elif cons == CONS_BOUNDED_0_1 or cons == CONS_BOUNDED_0_1_NOTNAN:
        if val < 0 or val > 1:
            raise ValueError(name + " <= 0 or " + name + " >= 1")
        if cons == CONS_BOUNDED_0_1_NOTNAN:
            if np.isnan(val):
                raise ValueError(name + ' contains NaNs')
    elif cons == CONS_GT_1:
        if val <= 1:
            raise ValueError(name + " <= 1")
    elif cons == CONS_GTE_1:
        if val < 1:
            raise ValueError(name + " < 1")
    elif cons == CONS_POISSON:
        if val < 0:
            raise ValueError(name + " < 0")
        elif val > POISSON_LAM_MAX:
            raise ValueError(name + " value too large")

    return 0

cdef object cont_broadcast_1(aug_state* state, void* func, object size, object lock,
                             object a, object a_name, constraint_type a_constraint):

    cdef np.ndarray a_arr, randoms
    cdef np.broadcast it
    cdef random_double_1 f = (<random_double_1>func)

    a_arr = <np.ndarray>np.PyArray_FROM_OTF(a, np.NPY_DOUBLE, np.NPY_ALIGNED)
    if a_constraint != CONS_NONE:
        check_array_constraint(a_arr, a_name, a_constraint)

    if size is not None:
        randoms = np.empty(size, np.double)
    else:
        #randoms = np.empty(np.shape(a_arr), np.double)
        randoms = np.PyArray_SimpleNew(np.PyArray_NDIM(a_arr), np.PyArray_DIMS(a_arr), np.NPY_DOUBLE)


    it = np.broadcast(randoms, a_arr)
    with lock, nogil:
        while np.PyArray_MultiIter_NOTDONE(it):
            a_val = (<double*>np.PyArray_MultiIter_DATA(it, 1))[0]
            (<double*>np.PyArray_MultiIter_DATA(it, 0))[0] = f(state, a_val)

            np.PyArray_MultiIter_NEXT(it)

    return randoms

cdef object cont_broadcast_2(aug_state* state, void* func, object size, object lock,
                 object a, object a_name, constraint_type a_constraint,
                 object b, object b_name, constraint_type b_constraint):
    cdef np.ndarray a_arr, b_arr, randoms
    cdef np.broadcast it
    cdef random_double_2 f = (<random_double_2>func)

    a_arr = <np.ndarray>np.PyArray_FROM_OTF(a, np.NPY_DOUBLE, np.NPY_ALIGNED)
    if a_constraint != CONS_NONE:
        check_array_constraint(a_arr, a_name, a_constraint)

    b_arr = <np.ndarray>np.PyArray_FROM_OTF(b, np.NPY_DOUBLE, np.NPY_ALIGNED)
    if b_constraint != CONS_NONE:
        check_array_constraint(b_arr, b_name, b_constraint)

    if size is not None:
        randoms = np.empty(size, np.double)
    else:
        it = np.PyArray_MultiIterNew2(a_arr, b_arr)
        randoms = np.empty(it.shape, np.double)
        #randoms = np.PyArray_SimpleNew(it.nd, np.PyArray_DIMS(it), np.NPY_DOUBLE)

    it = np.PyArray_MultiIterNew3(randoms, a_arr, b_arr)
    with lock, nogil:
        while np.PyArray_MultiIter_NOTDONE(it):
            a_val = (<double*>np.PyArray_MultiIter_DATA(it, 1))[0]
            b_val = (<double*>np.PyArray_MultiIter_DATA(it, 2))[0]
            (<double*>np.PyArray_MultiIter_DATA(it, 0))[0] = f(state, a_val, b_val)

            np.PyArray_MultiIter_NEXT(it)

    return randoms

cdef object cont_broadcast_3(aug_state* state, void* func, object size, object lock,
                 object a, object a_name, constraint_type a_constraint,
                 object b, object b_name, constraint_type b_constraint,
                 object c, object c_name, constraint_type c_constraint):
    cdef np.ndarray a_arr, b_arr, c_arr, randoms
    cdef np.broadcast it
    cdef random_double_3 f = (<random_double_3>func)

    a_arr = <np.ndarray>np.PyArray_FROM_OTF(a, np.NPY_DOUBLE, np.NPY_ALIGNED)
    if a_constraint != CONS_NONE:
        check_array_constraint(a_arr, a_name, a_constraint)

    b_arr = <np.ndarray>np.PyArray_FROM_OTF(b, np.NPY_DOUBLE, np.NPY_ALIGNED)
    if b_constraint != CONS_NONE:
        check_array_constraint(b_arr, b_name, b_constraint)

    c_arr = <np.ndarray>np.PyArray_FROM_OTF(c, np.NPY_DOUBLE, np.NPY_ALIGNED)
    if c_constraint != CONS_NONE:
        check_array_constraint(c_arr, c_name, c_constraint)

    if size is not None:
        randoms = np.empty(size, np.double)
    else:
        it = np.PyArray_MultiIterNew3(a_arr, b_arr, c_arr)
        #randoms = np.PyArray_SimpleNew(it.nd, np.PyArray_DIMS(it), np.NPY_DOUBLE)
        randoms = np.empty(it.shape, np.double)

    it = it = np.PyArray_MultiIterNew4(randoms, a_arr, b_arr, c_arr)
    # np.broadcast(randoms, a_arr, b_arr, c_arr)
    with lock, nogil:
        while np.PyArray_MultiIter_NOTDONE(it):
            a_val = (<double*>np.PyArray_MultiIter_DATA(it, 1))[0]
            b_val = (<double*>np.PyArray_MultiIter_DATA(it, 2))[0]
            c_val = (<double*>np.PyArray_MultiIter_DATA(it, 3))[0]
            (<double*>np.PyArray_MultiIter_DATA(it, 0))[0] = f(state, a_val, b_val, c_val)

            np.PyArray_MultiIter_NEXT(it)

    return randoms

cdef object cont(aug_state* state, void* func, object size, object lock, int narg,
                 object a, object a_name, constraint_type a_constraint,
                 object b, object b_name, constraint_type b_constraint,
                 object c, object c_name, constraint_type c_constraint):

    cdef double _a = 0.0, _b = 0.0, _c = 0.0
    cdef bint is_scalar = True
    if narg > 0:
        _a = PyFloat_AsDouble(a)
        if _a == -1.0:
            if PyErr_Occurred():
                is_scalar = False
        if a_constraint != CONS_NONE and is_scalar:
            check_constraint(_a, a_name, a_constraint)
    if narg > 1 and is_scalar:
        _b = PyFloat_AsDouble(b)
        if _b == -1.0:
            if PyErr_Occurred():
                is_scalar = False
        if b_constraint != CONS_NONE and is_scalar:
            check_constraint(_b, b_name, b_constraint)
    if narg == 3 and is_scalar:
        _c = PyFloat_AsDouble(c)
        if _c == -1.0:
            if PyErr_Occurred():
                is_scalar = False
        if c_constraint != CONS_NONE and is_scalar:
            check_constraint(_c, c_name, c_constraint)
    if not is_scalar:
        PyErr_Clear()
        if narg == 1:
            return cont_broadcast_1(state, func, size, lock,
                                    a, a_name, a_constraint)
        elif narg == 2:
            return cont_broadcast_2(state, func, size, lock,
                                    a, a_name, a_constraint,
                                    b, b_name, b_constraint)
        else:
            return cont_broadcast_3(state, func, size, lock,
                                    a, a_name, a_constraint,
                                    b, b_name, b_constraint,
                                    c, c_name, c_constraint)

    if size is None:
        if narg == 0:
            return (<random_double_0>func)(state)
        elif narg == 1:
            return (<random_double_1>func)(state, _a)
        elif narg == 2:
            return (<random_double_2>func)(state, _a, _b)
        elif narg == 3:
            return (<random_double_3>func)(state, _a, _b, _c)

    cdef Py_ssize_t i, n = compute_numel(size)
    cdef double [::1] randoms = np.empty(n, np.double)
    cdef random_double_0 f0;
    cdef random_double_1 f1;
    cdef random_double_2 f2;
    cdef random_double_3 f3;

    with lock, nogil:
        if narg == 0:
            f0 = (<random_double_0>func)
            for i in range(n):
                randoms[i] = f0(state)
        elif narg == 1:
            f1 = (<random_double_1>func)
            for i in range(n):
                randoms[i] = f1(state, _a)
        elif narg == 2:
            f2 = (<random_double_2>func)
            for i in range(n):
                randoms[i] = f2(state, _a, _b)
        elif narg == 3:
            f3 = (<random_double_3>func)
            for i in range(n):
                randoms[i] = f3(state, _a, _b, _c)

    return np.asarray(randoms).reshape(size)

cdef object discrete_broadcast_d(aug_state* state, void* func, object size, object lock,
                                 object a, object a_name, constraint_type a_constraint):

    cdef np.ndarray a_arr, randoms
    cdef np.broadcast it
    cdef random_uint_d f = (<random_uint_d>func)

    a_arr = <np.ndarray>np.PyArray_FROM_OTF(a, np.NPY_DOUBLE, np.NPY_ALIGNED)
    if a_constraint != CONS_NONE:
        check_array_constraint(a_arr, a_name, a_constraint)

    if size is not None:
        randoms = np.empty(size, np.int)
    else:
        #randoms = np.empty(np.shape(a_arr), np.double)
        randoms = np.PyArray_SimpleNew(np.PyArray_NDIM(a_arr), np.PyArray_DIMS(a_arr), np.NPY_LONG)

    it = np.broadcast(randoms, a_arr)
    with lock, nogil:
        while np.PyArray_MultiIter_NOTDONE(it):
            a_val = (<double*>np.PyArray_MultiIter_DATA(it, 1))[0]
            (<long*>np.PyArray_MultiIter_DATA(it, 0))[0] = f(state, a_val)

            np.PyArray_MultiIter_NEXT(it)

    return randoms

cdef object discrete_broadcast_dd(aug_state* state, void* func, object size, object lock,
                                  object a, object a_name, constraint_type a_constraint,
                                  object b, object b_name, constraint_type b_constraint):
    cdef np.ndarray a_arr, b_arr, randoms
    cdef np.broadcast it
    cdef random_uint_dd f = (<random_uint_dd>func)

    a_arr = <np.ndarray>np.PyArray_FROM_OTF(a, np.NPY_DOUBLE, np.NPY_ALIGNED)
    if a_constraint != CONS_NONE:
        check_array_constraint(a_arr, a_name, a_constraint)
    b_arr = <np.ndarray>np.PyArray_FROM_OTF(b, np.NPY_DOUBLE, np.NPY_ALIGNED)
    if b_constraint != CONS_NONE:
        check_array_constraint(b_arr, b_name, b_constraint)

    if size is not None:
        randoms = np.empty(size, np.int)
    else:
        it = np.PyArray_MultiIterNew2(a_arr, b_arr)
        randoms = np.empty(it.shape, np.int)
        # randoms = np.PyArray_SimpleNew(it.nd, np.PyArray_DIMS(it), np.NPY_LONG)

    it = np.PyArray_MultiIterNew3(randoms, a_arr, b_arr)
    with lock, nogil:
        while np.PyArray_MultiIter_NOTDONE(it):
            a_val = (<double*>np.PyArray_MultiIter_DATA(it, 1))[0]
            b_val = (<double*>np.PyArray_MultiIter_DATA(it, 2))[0]

            (<long*>np.PyArray_MultiIter_DATA(it, 0))[0] = f(state, a_val, b_val)

            np.PyArray_MultiIter_NEXT(it)

    return randoms

cdef object discrete_broadcast_di(aug_state* state, void* func, object size, object lock,
                                  object a, object a_name, constraint_type a_constraint,
                                  object b, object b_name, constraint_type b_constraint):
    cdef np.ndarray a_arr, b_arr, randoms
    cdef np.broadcast it
    cdef random_uint_di f = (<random_uint_di>func)

    a_arr = <np.ndarray>np.PyArray_FROM_OTF(a, np.NPY_DOUBLE, np.NPY_ALIGNED)
    if a_constraint != CONS_NONE:
        check_array_constraint(a_arr, a_name, a_constraint)

    b_arr = <np.ndarray>np.PyArray_FROM_OTF(b, np.NPY_LONG, np.NPY_ALIGNED)
    if b_constraint != CONS_NONE:
        check_array_constraint(b_arr, b_name, b_constraint)

    if size is not None:
        randoms = np.empty(size, np.int)
    else:
        it = np.PyArray_MultiIterNew2(a_arr, b_arr)
        randoms = np.empty(it.shape, np.int)
        #randoms = np.PyArray_SimpleNew(it.nd, np.PyArray_DIMS(it), np.NPY_LONG)

    it = np.PyArray_MultiIterNew3(randoms, a_arr, b_arr)
    with lock, nogil:
        while np.PyArray_MultiIter_NOTDONE(it):
            a_val = (<double*>np.PyArray_MultiIter_DATA(it, 1))[0]
            b_val = (<long*>np.PyArray_MultiIter_DATA(it, 2))[0]
            (<long*>np.PyArray_MultiIter_DATA(it, 0))[0] = f(state, a_val, b_val)

            np.PyArray_MultiIter_NEXT(it)

    return randoms

cdef object discrete_broadcast_iii(aug_state* state, void* func, object size, object lock,
                                  object a, object a_name, constraint_type a_constraint,
                                  object b, object b_name, constraint_type b_constraint,
                                  object c, object c_name, constraint_type c_constraint):
    cdef np.ndarray a_arr, b_arr, c_arr, randoms
    cdef np.broadcast it
    cdef random_uint_iii f = (<random_uint_iii>func)

    a_arr = <np.ndarray>np.PyArray_FROM_OTF(a, np.NPY_LONG, np.NPY_ALIGNED)
    if a_constraint != CONS_NONE:
        check_array_constraint(a_arr, a_name, a_constraint)

    b_arr = <np.ndarray>np.PyArray_FROM_OTF(b, np.NPY_LONG, np.NPY_ALIGNED)
    if b_constraint != CONS_NONE:
        check_array_constraint(b_arr, b_name, b_constraint)

    c_arr = <np.ndarray>np.PyArray_FROM_OTF(c, np.NPY_LONG, np.NPY_ALIGNED)
    if c_constraint != CONS_NONE:
        check_array_constraint(c_arr, c_name, c_constraint)

    if size is not None:
        randoms = np.empty(size, np.int)
    else:
        it = np.PyArray_MultiIterNew3(a_arr, b_arr, c_arr)
        randoms = np.empty(it.shape, np.int)
        #randoms = np.PyArray_SimpleNew(it.nd, np.PyArray_DIMS(it), np.NPY_LONG)

    it = np.PyArray_MultiIterNew4(randoms, a_arr, b_arr, c_arr)
    with lock, nogil:
        while np.PyArray_MultiIter_NOTDONE(it):
            a_val = (<long*>np.PyArray_MultiIter_DATA(it, 1))[0]
            b_val = (<long*>np.PyArray_MultiIter_DATA(it, 2))[0]
            c_val = (<long*>np.PyArray_MultiIter_DATA(it, 3))[0]
            (<long*>np.PyArray_MultiIter_DATA(it, 0))[0] = f(state, a_val, b_val, c_val)

            np.PyArray_MultiIter_NEXT(it)

    return randoms

cdef object discrete_broadcast_i(aug_state* state, void* func, object size, object lock,
                                  object a, object a_name, constraint_type a_constraint):
    cdef np.ndarray a_arr, randoms
    cdef np.broadcast it
    cdef random_uint_i f = (<random_uint_i>func)

    a_arr = <np.ndarray>np.PyArray_FROM_OTF(a, np.NPY_LONG, np.NPY_ALIGNED)
    if a_constraint != CONS_NONE:
        check_array_constraint(a_arr, a_name, a_constraint)

    if size is not None:
        randoms = np.empty(size, np.int)
    else:
        #randoms = np.empty(np.shape(a_arr), np.double)
        randoms = np.PyArray_SimpleNew(np.PyArray_NDIM(a_arr), np.PyArray_DIMS(a_arr), np.NPY_LONG)

    it = np.broadcast(randoms, a_arr)
    with lock, nogil:
        while np.PyArray_MultiIter_NOTDONE(it):
            a_val = (<long*>np.PyArray_MultiIter_DATA(it, 1))[0]
            (<long*>np.PyArray_MultiIter_DATA(it, 0))[0] = f(state, a_val)

            np.PyArray_MultiIter_NEXT(it)

    return randoms

# Needs double <vec>, double-double <vec>, double-long<vec>, long <vec>, long-long-long
cdef object disc(aug_state* state, void* func, object size, object lock,
                 int narg_double, int narg_long,
                 object a, object a_name, constraint_type a_constraint,
                 object b, object b_name, constraint_type b_constraint,
                 object c, object c_name, constraint_type c_constraint):

    cdef double _da = 0, _db = 0
    cdef long _ia = 0, _ib = 0 , _ic = 0
    cdef bint is_scalar = True
    if narg_double > 0:
        _da = PyFloat_AsDouble(a)
        if _da == -1.0:
            if PyErr_Occurred():
                is_scalar = False
        if a_constraint != CONS_NONE and is_scalar:
            check_constraint(_da, a_name, a_constraint)

        if narg_double > 1 and is_scalar:
            _db = PyFloat_AsDouble(b)
            if _db == -1.0:
                if PyErr_Occurred():
                    is_scalar = False
            if b_constraint != CONS_NONE and is_scalar:
                check_constraint(_db, b_name, b_constraint)
        if narg_long == 1 and is_scalar:
            _ib = PyInt_AsLong(b)
            if _ib == -1:
                if PyErr_Occurred():
                    is_scalar = False
            if b_constraint != CONS_NONE and is_scalar:
                check_constraint(<double>_ib, b_name, b_constraint)
    else:
        if narg_long > 0:
            _ia = PyInt_AsLong(a)
            if _ia == -1:
                if PyErr_Occurred():
                    is_scalar = False
            elif a_constraint != CONS_NONE and is_scalar:
                check_constraint(<double>_ia, a_name, a_constraint)
        if narg_long > 1 and is_scalar:
            _ib = PyInt_AsLong(b)
            if _ib == -1:
                if PyErr_Occurred():
                    is_scalar = False
            elif b_constraint != CONS_NONE and is_scalar:
                check_constraint(<double>_ib, b_name, b_constraint)
        if narg_long > 2 and is_scalar:
            _ic = PyInt_AsLong(c)
            if _ic == -1:
                if PyErr_Occurred():
                    is_scalar = False
            elif c_constraint != CONS_NONE and is_scalar:
                check_constraint(<double>_ic, c_name, c_constraint)

    if not is_scalar:
        PyErr_Clear()
        if narg_long == 0:
            if narg_double == 1:
                return discrete_broadcast_d(state, func, size, lock,
                                            a, a_name, a_constraint)
            elif narg_double == 2:
                return discrete_broadcast_dd(state, func, size, lock,
                                             a, a_name, a_constraint,
                                             b, b_name, b_constraint)
        elif narg_long == 1:
            if narg_double == 0:
                return discrete_broadcast_i(state, func, size, lock,
                                            a, a_name, a_constraint)
            elif narg_double == 1:
                return discrete_broadcast_di(state, func, size, lock,
                                             a, a_name, a_constraint,
                                             b, b_name, b_constraint)
        else:
            raise NotImplementedError("No vector path available")

    if size is None:
        if narg_long == 0:
            if narg_double == 0:
                return (<random_uint_0>func)(state)
            elif narg_double == 1:
                return (<random_uint_d>func)(state, _da)
            elif narg_double == 2:
                return (<random_uint_dd>func)(state, _da, _db)
        elif narg_long == 1:
            if narg_double == 0:
                return (<random_uint_i>func)(state, _ia)
            if narg_double == 1:
                return (<random_uint_di>func)(state, _da, _ib)
        else:
            return (<random_uint_iii>func)(state, _ia, _ib, _ic)

    cdef Py_ssize_t i, n = compute_numel(size)
    cdef np.int_t [::1] randoms = np.empty(n, np.int)
    cdef random_uint_0 f0;
    cdef random_uint_d fd;
    cdef random_uint_dd fdd;
    cdef random_uint_di fdi;
    cdef random_uint_i fi;
    cdef random_uint_iii fiii;


    with lock, nogil:
        if narg_long == 0:
            if narg_double == 0:
                f0 = (<random_uint_0>func)
                for i in range(n):
                    randoms[i] = f0(state)
            elif narg_double == 1:
                fd = (<random_uint_d>func)
                for i in range(n):
                    randoms[i] = fd(state, _da)
            elif narg_double == 2:
                fdd = (<random_uint_dd>func)
                for i in range(n):
                    randoms[i] = fdd(state, _da, _db)
        elif narg_long == 1:
            if narg_double == 0:
                fi = (<random_uint_i>func)
                for i in range(n):
                    randoms[i] = fi(state, _ia)
            if narg_double == 1:
                fdi = (<random_uint_di>func)
                for i in range(n):
                    randoms[i] = fdi(state, _da, _ib)
        else:
            fiii = (<random_uint_iii>func)
            for i in range(n):
                randoms[i] = fiii(state, _ia, _ib, _ic)

    return np.asarray(randoms).reshape(size)

