#define RNG_TYPE pcg32_random_t

#include <stdint.h>
#include "../../src/pcg/pcg_variants.h"

typedef struct s_aug_state {
    pcg32_random_t *rng;
    uint64_t state, inc;

    int has_gauss, shift_zig_random_int;
    double gauss;
    uint64_t zig_random_int;
} aug_state;

inline uint32_t random_uint32(aug_state* state)
{
    return pcg32_random_r(state->rng);
}

inline uint64_t random_uint64(aug_state* state)
{
    return (((uint64_t) pcg32_random_r((*state).rng) << 32) | pcg32_random_r((*state).rng));
}

inline void seed(aug_state* state, uint64_t seed, uint64_t inc)
{
    pcg_setseq_64_srandom_r(state->rng, seed, inc);
}

inline void advance(aug_state* state, uint64_t delta)
{
    pcg32_advance_r(state->rng, delta);
}
