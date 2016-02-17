#include "dSFMT-shim.h"
#include "dSFMT-poly.h"

extern inline uint32_t random_uint32(aug_state* state);

extern inline uint64_t random_uint64(aug_state* state);

extern inline double random_double(aug_state* state);

extern inline uint64_t random_raw_values(aug_state* state);

void reset_buffer(aug_state* state)
{
    int i = 0;
    for (i = 0; i < (2 * DSFMT_N); i++)
    {
        state->buffered_uniforms[i] = 0.0;
    }
    state->buffer_loc = 2 * DSFMT_N;
}

extern void set_seed_by_array(aug_state* state, uint32_t init_key[], int key_length)
{
    reset_buffer(state);
    dsfmt_init_by_array(state->rng, init_key, key_length);
}

void set_seed(aug_state* state, uint32_t seed)
{
    reset_buffer(state);
    dsfmt_init_gen_rand(state->rng, seed);
}

void entropy_init(aug_state* state)
{
    uint32_t seeds[1];
    entropy_fill((void*) seeds, sizeof(seeds));
    set_seed(state,  seeds[0]);
}

void jump_state(aug_state* state)
{
    dSFMT_jump(state->rng, poly_128);
}