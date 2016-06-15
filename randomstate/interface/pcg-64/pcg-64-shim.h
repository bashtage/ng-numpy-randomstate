#ifdef _WIN32
#include "../../src/common/inttypes.h"
#define inline __forceinline
#else
#include <inttypes.h>
#endif

#include "../../src/common/binomial.h"
#include "../../src/common/entropy.h"
#include "../../src/pcg64-compat/pcg64.h"

typedef struct s_aug_state {
    pcg64_random_t *rng;
    binomial_t *binomial;

    int has_gauss, has_gauss_float, shift_zig_random_int, has_uint32;
    float gauss_float;
    double gauss;
    uint32_t uinteger;
    uint64_t zig_random_int;
} aug_state;

inline uint32_t random_uint32(aug_state* state)
{
    uint64_t temp;
    if (state->has_uint32)
    {
        state->has_uint32 = 0;
        return state->uinteger;
    }
    state->has_uint32 = 1;
    temp = pcg64_random_r(state->rng);
    state->uinteger = (uint32_t)(temp >> 32);
    return (uint32_t)(temp & 0xFFFFFFFFLL);
}

inline uint64_t random_uint64(aug_state* state)
{
    return pcg64_random_r(state->rng);
}

inline uint64_t random_raw_values(aug_state* state)
{
    return random_uint64(state);
}

inline void set_seed(aug_state* state, pcg128_t seed, pcg128_t inc)
{
    pcg64_srandom_r(state->rng, seed, inc);
}

inline void advance_state(aug_state* state, pcg128_t delta)
{
    pcg64_advance_r(state->rng, delta);
}

inline void entropy_init(aug_state* state)
{
    pcg128_t seeds[2];
    entropy_fill((void*) seeds, sizeof(seeds));
    set_seed(state, seeds[0], seeds[1]);
}

inline double random_double(aug_state* state)
{
    uint64_t rn;
    rn = random_uint64(state);
    return (rn >> 11) * (1.0 / 9007199254740992.0);
}
