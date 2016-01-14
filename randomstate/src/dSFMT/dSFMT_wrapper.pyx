from libc.stdint cimport uint32_t, uint64_t
from libc.stdlib cimport malloc, free

DEF DSFMT_MEXP = 19937
DEF DSFMT_N = 191 #
DEF DSFMT_N_PLUS_1 = 192 #

cdef extern from "dSFMT.h":
    cdef union W128_T:
        uint64_t u[2];
        uint32_t u32[4];
        double d[2];

    ctypedef W128_T w128_t;

    cdef struct DSFMT_T:
        w128_t status[DSFMT_N_PLUS_1];
        int idx;

    ctypedef DSFMT_T dsfmt_t;

    cdef void dsfmt_init_gen_rand(dsfmt_t *dsfmt, uint32_t seed)
    cdef int dsfmt_get_min_array_size()
    cdef double dsfmt_genrand_close1_open2(dsfmt_t *dsfmt)
    cdef uint32_t dsfmt_genrand_uint32(dsfmt_t *dsfmt)
    cdef void dsfmt_fill_array_close1_open2(dsfmt_t *dsfmt, double array[], int size)
    cdef void dsfmt_init_by_array(dsfmt_t * dsfmt, uint32_t init_key[], int key_length)
    cdef void dsfmt_fill_array_close_open(dsfmt_t *dsfmt, double array[], int size)
    cdef double dsfmt_genrand_close_open(dsfmt_t *dsfmt)


cdef class Test:
    cdef:
        int idx
        dsfmt_t state

    def print_state(self):
        cdef w128_t val
        for i in range(DSFMT_N_PLUS_1):
            print(i)
            val = self.state.status[i]
            print(val.u[0])
            print(val.u[1])

    def init(self):
        dsfmt_init_gen_rand(&self.state, 0)

    def get_min_array_size(self):
        return dsfmt_get_min_array_size()

    def get_a_double(self):
        return dsfmt_genrand_close1_open2(&self.state)

