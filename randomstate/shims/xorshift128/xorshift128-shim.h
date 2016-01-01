#include <stdint.h>

#include "../../src/common/binomial.h"
#include "../../src/common/entropy.h"
#include "../../src/xorshift128/xorshift128.h"

typedef struct s_aug_state {
    xorshift128_state *rng;
    binomial_t *binomial;

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
    temp = xorshift128_next(state->rng);
    state->uinteger = (uint32_t)(temp >> 32);
    return (uint32_t)(temp & 0xFFFFFFFFLL);
}

inline uint64_t random_uint64(aug_state* state)
{
    return xorshift128_next(state->rng);
}

inline void set_seed(aug_state* state, uint64_t seed)
{
    xorshift128_seed(state->rng, seed);
}

inline void entropy_init(aug_state* state)
{
    uint64_t seeds[2];
    entropy_fill((void*) seeds, sizeof(seeds));
    xorshift128_init_state(state->rng,  seeds[0], seeds[1]);
}

inline void jump(aug_state* state)
{
    xorshift128_jump(state->rng);
}

inline void init_state(aug_state* state, uint64_t* state_vals)
{
    xorshift128_init_state(state->rng, *(state_vals), *(state_vals + 1));
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
