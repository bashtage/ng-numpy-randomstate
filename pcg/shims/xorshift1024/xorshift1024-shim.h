#include <stdint.h>

#include "../../src/entropy/entropy.h"
#include "../../src/xorshift1024/xorshift1024.h"

typedef struct s_aug_state {
    xorshift1024_state *rng;

    int has_gauss, shift_zig_random_int, has_uint32;
    double gauss;
    uint32_t uinteger;
    uint64_t zig_random_int;

} aug_state;

inline uint32_t random_uint32(aug_state* state)
{
    if (state->has_uint32)
    {
        state->has_uint32 = 0;
        return state->uinteger;
    }
    state->has_uint32 = 1;
    uint64_t temp;
    temp = xorshift1024_next(state->rng);
    state->uinteger = (uint32_t)(temp >> 32);
    return (uint32_t)(temp & 0xFFFFFFFFLL);
}

inline uint64_t random_uint64(aug_state* state)
{
    return xorshift1024_next(state->rng);
}

inline void seed(aug_state* state, uint64_t* seed)
{
    xorshift1024_seed(state->rng, seed);
}

inline void entropy_init(aug_state* state)
{
    uint64_t seeds[16];
    entropy_fill((void*) seeds, sizeof(seeds));
    xorshift1024_seed(state->rng,  &seeds);
}

inline void jump(aug_state* state)
{
    xorshift1024_jump(state->rng);
}