#include <stdint.h>
#include "../../src/common/binomial.h"
#include "../../src/common/entropy.h"
/* #include "../../src/pcg/pcg_variants.h" */
#include "../../src/pcg/pcg32.h"


typedef struct s_aug_state {
    pcg32_random_t *rng;
    binomial_t *binomial;

    int has_gauss, has_gauss_float, shift_zig_random_int, has_uint32;
    float gauss_float;
    double gauss;
    uint32_t uinteger;
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

inline void set_seed(aug_state* state, uint64_t seed, uint64_t inc)
{
    pcg32_srandom_r(state->rng, seed, inc);
}

inline void advance_state(aug_state* state, uint64_t delta)
{
    pcg32_advance_r(state->rng, delta);
}

inline uint64_t random_raw_values(aug_state* state)
{
    return (uint64_t)random_uint32(state);
}

inline void entropy_init(aug_state* state)
{
    uint64_t seeds[2];
    entropy_fill((void*) seeds, sizeof(seeds));
    set_seed(state, seeds[0], seeds[1]);
}

inline double random_double(aug_state* state)
{
    int32_t a = random_uint32(state) >> 5, b = random_uint32(state) >> 6;
    return (a * 67108864.0 + b) / 9007199254740992.0;
}