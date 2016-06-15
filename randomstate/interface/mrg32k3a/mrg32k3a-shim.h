#ifdef _WIN32
#include "../../src/common/stdint.h"
#define inline __forceinline
#else
#include <stdint.h>
#endif
#include "../../src/common/entropy.h"
#include "../../src/common/binomial.h"
#include "../../src/mrg32k3a/mrg32k3a.h"

#define STATE_MAX_VALUE_1 4294967086
#define STATE_MAX_VALUE_2 4294944442

typedef struct s_aug_state {
    mrg32k3a_state *rng;
    binomial_t *binomial;

    int has_gauss, has_gauss_float, shift_zig_random_int, has_uint32;
    float gauss_float;
    double gauss;
    uint64_t zig_random_int;
    uint32_t uinteger;
} aug_state;

inline uint32_t random_uint32(aug_state* state)
{
    return mrg32k3a_random(state->rng);
}

inline uint64_t random_uint64(aug_state* state)
{
    return (((uint64_t) mrg32k3a_random(state->rng) << 32) | mrg32k3a_random(state->rng));
}

inline uint64_t random_raw_values(aug_state* state)
{
    return (uint64_t)random_uint32(state);
}

inline double random_double(aug_state* state)
{
    int32_t a = random_uint32(state) >> 5, b = random_uint32(state) >> 6;
    return (a * 67108864.0 + b) / 9007199254740992.0;
}

extern void set_seed(aug_state* state, uint64_t val);

extern void set_seed_by_array(aug_state* state, uint64_t *vals, int count);

extern void init_state(aug_state* state, int64_t vals[6]);

extern void entropy_init(aug_state* state);

