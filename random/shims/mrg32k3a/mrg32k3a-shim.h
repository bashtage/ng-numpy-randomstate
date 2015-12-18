#define RNG_TYPE pcg32_random_t

#define STATE_MAX_VALUE_1 4294967086
#define STATE_MAX_VALUE_2 4294944442

#include <stdint.h>
#include "../../src/entropy/entropy.h"
#include "../../src/mrg32k3a/mrg32k3a.h"

typedef struct s_aug_state {
    mrg32k3a_state *rng;

    int has_gauss, shift_zig_random_int, has_uint32;
    double gauss;
    uint64_t zig_random_int;
    uint32_t uinteger;
} aug_state;

inline uint32_t random_uint32(aug_state* state)
{
    return mrg32k3a_random(state->rng);
}

inline uint64_t random_uint64(aug_state* state)
{
    return (((uint64_t) mrg32k3a_random(state->rng) << 32) | mrg32k3a_random(state->rng));
}

inline void seed(aug_state* state, uint64_t val)
{
    mrg32k3a_seed(state->rng, val);
}

inline void init_state(aug_state* state, int64_t val[6])
{
    state->rng->s10 =  val[0];
    state->rng->s11 =  val[1];
    state->rng->s12 =  val[2];
    state->rng->s20 =  val[3];
    state->rng->s21 =  val[4];
    state->rng->s22 =  val[5];
}

inline void entropy_init(aug_state* state)
{
    uint32_t buf[6] = { 0 };
    int64_t seeds[6];
    int i, val, all_zero = 0;
    while (!all_zero)
    {
        entropy_fill((void*) buf, sizeof(buf));
        for (i=0; i<6; i++)
        {
            val = (i < 3) ? STATE_MAX_VALUE_1 : STATE_MAX_VALUE_2;
            seeds[i] = (int64_t)((buf[i] < val) ? buf[i] : val);
            all_zero = all_zero || (seeds[i] > 0);
        }
    }
    init_state(state, seeds);
}