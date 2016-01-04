import numpy as np
cimport numpy as np
from libc.stdint cimport uint32_t, uint64_t, int64_t, int32_t

include "config.pxi"
include "src/common/binomial.pxi"

IF RNG_PCG32:
    include "shims/pcg-32/pcg-32.pxi"
IF RNG_PCG64:
    include "shims/pcg-64/pcg-64.pxi"
IF RNG_MT19937:
    include "shims/random-kit/random-kit.pxi"
IF RNG_XORSHIFT128:
    include "shims/xorshift128/xorshift128.pxi"
IF RNG_XORSHIFT1024:
    include "shims/xorshift1024/xorshift1024.pxi"
IF RNG_MRG32K3A:
    include "shims/mrg32k3a/mrg32k3a.pxi"
IF RNG_MLFG_1279_861:
    include "shims/mlfg-1279-861/mlfg-1279-861.pxi"
IF RNG_DUMMY:
    include "shims/dummy/dummy.pxi"

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

cdef double POISSON_LAM_MAX = <uint64_t>(np.iinfo('l').max - np.sqrt(np.iinfo('l').max)*10)

cdef enum ConstraintType:
    CONS_NONE
    CONS_NON_NEGATIVE
    CONS_POSITIVE
    CONS_BOUNDED_0_1
    CONS_BOUNDED_0_1_NOTNAN
    CONS_GT_1
    CONS_GTE_1
    CONS_POISSON


ctypedef ConstraintType constraint_type

cdef Py_ssize_t compute_numel(size)

cdef int check_array_constraint(np.ndarray val, object name, constraint_type cons) except -1

cdef int check_constraint(double val, object name, constraint_type cons) except -1

cdef object uint0_32(aug_state* state, random_uint_0_32 func, object size, object lock)

cdef object cont_broadcast_1(aug_state* state, void* func, object size, object lock,
                             object a, object a_name, constraint_type a_constraint)

cdef object cont_broadcast_2(aug_state* state, void* func, object size, object lock,
                 object a, object a_name, constraint_type a_constraint,
                 object b, object b_name, constraint_type b_constraint)

cdef object cont_broadcast_3(aug_state* state, void* func, object size, object lock,
                 object a, object a_name, constraint_type a_constraint,
                 object b, object b_name, constraint_type b_constraint,
                 object c, object c_name, constraint_type c_constraint)

cdef object cont(aug_state* state, void* func, object size, object lock, int narg,
                 object a, object a_name, constraint_type a_constraint,
                 object b, object b_name, constraint_type b_constraint,
                 object c, object c_name, constraint_type c_constraint)

cdef object discrete_broadcast_d(aug_state* state, void* func, object size, object lock,
                                 object a, object a_name, constraint_type a_constraint)

cdef object discrete_broadcast_dd(aug_state* state, void* func, object size, object lock,
                                  object a, object a_name, constraint_type a_constraint,
                                  object b, object b_name, constraint_type b_constraint)

cdef object discrete_broadcast_di(aug_state* state, void* func, object size, object lock,
                                  object a, object a_name, constraint_type a_constraint,
                                  object b, object b_name, constraint_type b_constraint)

cdef object discrete_broadcast_iii(aug_state* state, void* func, object size, object lock,
                                  object a, object a_name, constraint_type a_constraint,
                                  object b, object b_name, constraint_type b_constraint,
                                  object c, object c_name, constraint_type c_constraint)

cdef object discrete_broadcast_i(aug_state* state, void* func, object size, object lock,
                                  object a, object a_name, constraint_type a_constraint)

cdef object disc(aug_state* state, void* func, object size, object lock,
                 int narg_double, int narg_long,
                 object a, object a_name, constraint_type a_constraint,
                 object b, object b_name, constraint_type b_constraint,
                 object c, object c_name, constraint_type c_constraint)

