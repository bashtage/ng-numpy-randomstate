#define RNG_TYPE rk_state

#include <stdint.h>

#include "../../src/entropy/entropy.h"
#include "../../src/random-kit/random-kit.h"

typedef struct s_aug_state {
    rk_state *rng;

    int has_gauss, shift_zig_random_int;
    double gauss;
    uint64_t zig_random_int;
} aug_state;

inline uint32_t random_uint32(aug_state* state)
{
    return rk_random(state->rng);
}

inline uint64_t random_uint64(aug_state* state)
{
    return (((uint64_t) rk_random(state->rng)) << 32) | rk_random(state->rng);
}

inline void seed(aug_state* state, uint32_t seed)
{
    rk_seed(state->rng, seed);
}

inline void entropy_init(aug_state* state)
{
    uint32_t seeds[1];
    entropy_fill((void*) seeds, sizeof(seeds));
    seed(state,  seeds[0]);
}