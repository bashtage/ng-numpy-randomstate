#define RNG_TYPE rk_state

#ifdef _WIN32
#include "../../src/common/stdint.h"
#define inline __inline
#else
#include <stdint.h>
#endif

#include "../../src/common/binomial.h"
#include "../../src/common/entropy.h"
#include "../../src/dSFMT/dSFMT.h"

typedef struct s_aug_state {
    dsfmt_t *rng;
    binomial_t *binomial;

    int has_gauss, shift_zig_random_int, has_uint32;
    double gauss;
    uint32_t uinteger;
    uint64_t zig_random_int;
} aug_state;

static inline uint32_t random_uint32(aug_state* state)
{
    double d = dsfmt_genrand_close1_open2(state->rng);
    uint64_t *out = (uint64_t *)&d;
    return (uint32_t)(*out & 0xffffffff);
}

static inline uint64_t random_uint64(aug_state* state)
{
    double d = dsfmt_genrand_close1_open2(state->rng);
    uint64_t out;
    uint64_t *tmp;
    tmp = (uint64_t *)&d;
    out = *tmp << 32;
    d = dsfmt_genrand_close1_open2(state->rng);
    tmp =  (uint64_t *)&d;
    out |= *tmp & 0xffffffff;
    return out;
}

static inline double random_double(aug_state* state)
{
    return dsfmt_genrand_close1_open2(state->rng) - 1.0;
}

extern void entropy_init(aug_state* state);

extern void set_seed_by_array(aug_state* state, uint32_t init_key[], int key_length);

extern void set_seed(aug_state* state, uint32_t seed);

