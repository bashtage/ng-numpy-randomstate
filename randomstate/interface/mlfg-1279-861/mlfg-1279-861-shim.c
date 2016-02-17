#include "mlfg-1279-861-shim.h"

extern inline uint32_t random_uint32(aug_state* state);

extern inline uint64_t random_uint64(aug_state* state);

extern inline double random_double(aug_state* state);

extern inline uint64_t random_raw_values(aug_state* state);

void set_seed(aug_state* state, uint64_t val)
{
    mlfg_seed(state->rng, val);
}

void set_seed_by_array(aug_state* state, uint64_t *vals, int count)
{
    mlfg_seed_by_array(state->rng, vals, count);
}

void entropy_init(aug_state* state)
{
    uint64_t seeds[1];
    entropy_fill((void*) seeds, sizeof(seeds));
    mlfg_seed(state->rng, seeds[0]);
}
