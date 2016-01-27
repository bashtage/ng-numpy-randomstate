#include "mlfg-1279-861.h"
#include "../splitmix64/splitmix64.h"

extern inline uint64_t mlfg_next(mlfg_state* state);

void mlfg_seed(mlfg_state* state, uint64_t seed)
{
    uint64_t seeds[K];
    uint64_t seed_copy = seed;
    int i;
    for (i = 0; i < K; i++)
    {
        seeds[i] = splitmix64_next(&seed_copy);
        if ((seeds[i] % 2) != 1)
            seeds[i]++;
    }
    mlfg_init_state(state, seeds);
}

void mlfg_seed_by_array(mlfg_state* state, uint64_t *seed_array, int count)
{
    uint64_t seeds[K];
    uint64_t seed_copy = 0;
    int iter_bound = K >= count ? K : count;
    int i, loc = 0;
    for (i = 0; i < iter_bound ; i++)
    {
        if (i < count) {
            seed_copy ^= seed_array[i];
        }
        seeds[loc] = splitmix64_next(&seed_copy);
        if ((seeds[loc] % 2) != 1)
            seeds[loc]++;
        loc++;
        if (loc == K)
            loc = 0;
    }
    mlfg_init_state(state, seeds);
}


void mlfg_init_state(mlfg_state *state, uint64_t seeds[K])
{
    int i;
    for (i = 0; i < K; i++)
    {
        state->lags[i] = seeds[i];
    }
    state->pos = 0;
    state->lag_pos = K - J;
}