#include <stdint.h>

#include "../../src/entropy/entropy.h"
#include "../../src/dummy/dummy.h"

typedef struct s_aug_state {
    uint32_t *rng;

    int has_gauss, shift_zig_random_int;
    double gauss;
    uint64_t zig_random_int;
} aug_state;

inline uint32_t random_uint32(aug_state* state)
{
    return (uint32_t)(dummy_rng(state->rng) & 0xFFFFFFFFLL);
}

inline uint64_t random_uint64(aug_state* state)
{
    return dummy_rng(state->rng);
}

inline void seed(aug_state* state, uint32_t seed)
{
    *(state->rng) = seed % 20;
}

inline void advance(aug_state* state, uint32_t delta)
{
    *(state->rng) = (*(state->rng) + (delta % 20 )) % 20;
}

inline void entropy_init(aug_state* state)
{
    uint32_t seeds[1];
    entropy_fill((void*) seeds, sizeof(seeds));
    seed(state, seeds[0]);
}

inline double random_double(aug_state* state)
{
    uint64_t rn;
    int32_t a, b;
    rn = random_uint64(state);
    a = rn >> 37;
    b = (rn & 0xFFFFFFFFLL) >> 6;
    return (a * 67108864.0 + b) / 9007199254740992.0;
}