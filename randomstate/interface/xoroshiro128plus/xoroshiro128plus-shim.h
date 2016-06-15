#ifdef _WIN32
#include "../../src/common/stdint.h"
#define inline __forceinline
#else
#include <stdint.h>
#endif

#include "../../src/common/binomial.h"
#include "../../src/common/entropy.h"
#include "../../src/xoroshiro128plus/xoroshiro128plus.h"

typedef struct s_aug_state {
    xoroshiro128plus_state *rng;
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
    temp = xoroshiro128plus_next(state->rng);
    state->uinteger = (uint32_t)(temp >> 32);
    return (uint32_t)(temp & 0xFFFFFFFFLL);
}

inline uint64_t random_uint64(aug_state* state)
{
    return xoroshiro128plus_next(state->rng);
}

inline uint64_t random_raw_values(aug_state* state)
{
    return random_uint64(state);
}

inline double random_double(aug_state* state)
{
    uint64_t rn;
    rn = random_uint64(state);
    return (rn >> 11) * (1.0 / 9007199254740992.0);
}

extern void set_seed(aug_state* state, uint64_t seed);

extern void set_seed_by_array(aug_state* state, uint64_t *vals, int count);

extern void jump_state(aug_state* state);

extern void entropy_init(aug_state* state);
