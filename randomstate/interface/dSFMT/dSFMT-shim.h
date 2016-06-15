#ifdef _WIN32
#include "../../src/common/stdint.h"
#define inline __forceinline
#else
#include <stdint.h>
#endif

#include "../../src/common/binomial.h"
#include "../../src/common/entropy.h"
#include "../../src/dSFMT/dSFMT-jump.h"
#include "../../src/dSFMT/dSFMT.h"


typedef struct s_aug_state {
    dsfmt_t *rng;
    binomial_t *binomial;

    int has_gauss, has_gauss_float, shift_zig_random_int, has_uint32;
    float gauss_float;
    double gauss;
    uint32_t uinteger;
    uint64_t zig_random_int;

    double *buffered_uniforms;
    int buffer_loc;
} aug_state;

static inline double random_double_from_buffer(aug_state *state)
{
    double out;
    if (state->buffer_loc >= (2 * DSFMT_N))
    {
        state->buffer_loc = 0;
        dsfmt_fill_array_close1_open2(state->rng, state->buffered_uniforms, 2 * DSFMT_N);
    }
    out = state->buffered_uniforms[state->buffer_loc];
    state->buffer_loc++;
    return  out;
}

static inline uint32_t random_uint32(aug_state* state)
{
    /* TODO: This can be improved to use upper bits */
    double d = random_double_from_buffer(state);//dsfmt_genrand_close1_open2(state->rng);
    uint64_t *out = (uint64_t *)&d;
    return (uint32_t)(*out & 0xffffffff);
}

static inline uint64_t random_uint64(aug_state* state)
{
    /* TODO: This can be improved to use upper bits */
    double d = random_double_from_buffer(state); //dsfmt_genrand_close1_open2(state->rng);
    uint64_t out;
    uint64_t *tmp;
    tmp = (uint64_t *)&d;
    out = *tmp << 32;
    d = random_double_from_buffer(state);//dsfmt_genrand_close1_open2(state->rng);
    tmp =  (uint64_t *)&d;
    out |= *tmp & 0xffffffff;
    return out;
}

static inline double random_double(aug_state* state)
{
    return random_double_from_buffer(state) - 1.0;
    // return dsfmt_genrand_close1_open2(state->rng) - 1.0;
}


static inline uint64_t random_raw_values(aug_state* state)
{
    double d = random_double_from_buffer(state);
    return *((uint64_t *)&d);
}


extern void entropy_init(aug_state* state);

extern void set_seed_by_array(aug_state* state, uint32_t init_key[], int key_length);

extern void set_seed(aug_state* state, uint32_t seed);

extern void jump_state(aug_state* state);