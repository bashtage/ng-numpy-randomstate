
#ifdef _WIN32
#include "../../src/common/stdint.h"
#define inline __forceinline
#else
#include <stdint.h>
#endif

#include "../../src/common/entropy.h"
#include "../../src/common/binomial.h"
#include "../../src/mlfg-1279-861/mlfg-1279-861.h"

typedef struct s_aug_state {
    mlfg_state *rng;
    binomial_t *binomial;

    int has_gauss, has_gauss_float, shift_zig_random_int, has_uint32;
    float gauss_float;
    double gauss;
    uint32_t uinteger;
    uint64_t zig_random_int;
} aug_state;

inline uint32_t random_uint32(aug_state* state)
{
    return (uint32_t)(mlfg_next(state->rng) >> 32);
}

inline uint64_t random_uint64(aug_state* state)
{
    uint64_t out  = mlfg_next(state->rng) & 0xffffffff00000000ULL;
    out |= mlfg_next(state->rng) >> 32;
    return out;
}

inline uint64_t random_raw_values(aug_state* state)
{
    return  mlfg_next(state->rng) >> 1;
}

inline double random_double(aug_state* state)
{
    uint64_t rn;
    rn = mlfg_next(state->rng);
    return (rn >> 11) * (1.0 / 9007199254740992.0);
}

extern void set_seed(aug_state* state, uint64_t seed);

extern void set_seed_by_array(aug_state* state, uint64_t *vals, int count);

extern void entropy_init(aug_state* state);
