#include "mrg32k3a-shim.h"

extern inline uint32_t random_uint32(aug_state* state);

extern inline uint64_t random_uint64(aug_state* state);

extern inline double random_double(aug_state* state);

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

void init_state(aug_state* state, int64_t vals[6])
{
    state->rng->s10 =  vals[0];
    state->rng->s11 =  vals[1];
    state->rng->s12 =  vals[2];
    state->rng->s20 =  vals[3];
    state->rng->s21 =  vals[4];
    state->rng->s22 =  vals[5];
}

