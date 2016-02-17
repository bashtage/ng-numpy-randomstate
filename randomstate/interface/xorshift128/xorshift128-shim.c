#include "xorshift128-shim.h"

extern inline uint32_t random_uint32(aug_state* state);

extern inline uint64_t random_uint64(aug_state* state);

extern inline double random_double(aug_state* state);

extern inline uint64_t random_raw_values(aug_state* state);

void set_seed(aug_state* state, uint64_t seed)
{
    xorshift128_seed(state->rng, seed);
}

void set_seed_by_array(aug_state* state, uint64_t *vals, int count)
{
    xorshift128_seed_by_array(state->rng, vals, count);
}


void entropy_init(aug_state* state)
{
    uint64_t seed[1];
    entropy_fill((void*) seed, sizeof(seed));
    xorshift128_seed(state->rng, seed[0]);
}

void jump_state(aug_state* state)
{
    xorshift128_jump(state->rng);
}

void init_state(aug_state* state, uint64_t* state_vals)
{
    xorshift128_init_state(state->rng, *(state_vals), *(state_vals + 1));
}

