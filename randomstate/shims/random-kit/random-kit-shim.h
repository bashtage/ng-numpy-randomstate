#define RNG_TYPE rk_state

#ifdef _WIN32
#include "../../src/common/stdint.h"
#define inline __inline
#else
#include <stdint.h>
#endif

#include "../../src/common/binomial.h"
#include "../../src/common/entropy.h"
#include "../../src/random-kit/random-kit.h"

typedef struct s_aug_state {
    rk_state *rng;
    binomial_t *binomial;

    int has_gauss, shift_zig_random_int, has_uint32;
    double gauss;
    uint32_t uinteger;
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

inline double random_double(aug_state* state)
{
    int32_t a = random_uint32(state) >> 5, b = random_uint32(state) >> 6;
    return (a * 67108864.0 + b) / 9007199254740992.0;
}

extern void entropy_init(aug_state* state);

extern void set_seed_by_array(aug_state* state, unsigned long init_key[], int key_length);

extern void set_seed(aug_state* state, uint32_t seed);



