#include "mrg32k3a-shim.h"

extern inline uint32_t random_uint32(aug_state* state);

extern inline uint64_t random_uint64(aug_state* state);

extern inline double random_double(aug_state* state);

extern inline uint64_t random_raw_values(aug_state* state);

void entropy_init(aug_state* state)
{
    uint32_t buf[6] = { 0 };
    int64_t seeds[6];
    uint32_t i, val;
    int all_zero = 0;
    while (!all_zero)
    {
        entropy_fill((void*) buf, sizeof(buf));
        for (i = 0; i<6; i++)
        {
            val = (i < 3) ? STATE_MAX_VALUE_1 : STATE_MAX_VALUE_2;
            seeds[i] = (int64_t)(buf[i] % val);
            all_zero = all_zero || (seeds[i] > 0);
        }
    }
    init_state(state, seeds);
}

void set_seed(aug_state* state, uint64_t val)
{
    mrg32k3a_seed(state->rng, val);
}

void set_seed_by_array(aug_state* state, uint64_t *vals, int count)
{
    mrg32k3a_seed_by_array(state->rng, vals, count);
}


void init_state(aug_state* state, int64_t vals[6])
{
    state->rng->s1[0] =  vals[0];
    state->rng->s1[1] =  vals[1];
    state->rng->s1[2] =  vals[2];
    state->rng->s2[0] =  vals[3];
    state->rng->s2[1] =  vals[4];
    state->rng->s2[2] =  vals[5];
    state->rng->loc = 2;
}

