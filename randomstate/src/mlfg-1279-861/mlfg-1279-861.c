#include "mlfg-1279-861.h"
#include "../splitmix64/splitmix64.h"
#include <stdio.h>
#include <inttypes.h>

extern inline uint32_t mlfg_next(mlfg_state* state);

void mlfg_seed(mlfg_state* state, uint64_t seed)
{
    uint32_t seeds[K];
    uint64_t seed_copy = seed;
    int i;
    for (i = 0; i < K; i++)
    {
        seeds[i] = (uint32_t)(splitmix64_next(&seed_copy) >> 32);
        if ((seeds[i] % 2) != 1)
            seeds[i]++;
    }
    mlfg_init_state(state, seeds);
}

void mlfg_init_state(mlfg_state *state, uint32_t seeds[K])
{
    int i;
    for (i = 0; i < K; i++)
    {
        state->lags[i] = seeds[i];
    }
    state->pos = 0;
    state->lag_pos = K - J;
}

