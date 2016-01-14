#include "dSFMT-shim.h"

extern inline uint32_t random_uint32(aug_state* state);

extern inline uint64_t random_uint64(aug_state* state);

extern inline double random_double(aug_state* state);

extern void set_seed_by_array(aug_state* state, uint32_t init_key[], int key_length)
{
    dsfmt_init_by_array(state->rng, init_key, key_length);
}

void set_seed(aug_state* state, uint32_t seed)
{
    dsfmt_init_gen_rand(state->rng, seed);
}

void entropy_init(aug_state* state)
{
    uint32_t seeds[1];
    entropy_fill((void*) seeds, sizeof(seeds));
    set_seed(state,  seeds[0]);
}