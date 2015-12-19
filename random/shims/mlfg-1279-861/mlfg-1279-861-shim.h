#include <stdint.h>
#include "../../src/entropy/entropy.h"
#include "../../src/mlfg-1279-861/mlfg-1279-861.h"

typedef struct s_aug_state {
    mlfg_state *rng;

    int has_gauss, shift_zig_random_int, has_uint32;
    double gauss;
    uint32_t uinteger;
    uint64_t zig_random_int;
} aug_state;

inline uint32_t random_uint32(aug_state* state)
{
    return mlfg_next(state->rng) | (mlfg_next(state->rng) << 31);
}

inline uint64_t random_uint64(aug_state* state)
{
    return (((uint64_t)mlfg_next(state->rng)) << 33) | (((uint64_t)mlfg_next(state->rng)) << 2) | (mlfg_next(state->rng) & 0x03);
}

inline void seed(aug_state* state, uint64_t val)
{
    mlfg_seed(state->rng, val);
}

inline void entropy_init(aug_state* state)
{
    uint64_t seeds[1];
    entropy_fill((void*) seeds, sizeof(seeds));
    mlfg_seed(state->rng, seeds[0]);
}

inline double random_double(aug_state* state)
{
    int32_t a = mlfg_next(state->rng) >> 4, b = mlfg_next(state->rng) >> 5;
    return (a * 67108864.0 + b) / 9007199254740992.0;
}