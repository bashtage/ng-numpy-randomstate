#ifdef _WIN32
#include "../../src/common/stdint.h"
#define inline __inline
#else
#include <stdint.h>
#endif


#include "../../src/common/binomial.h"
#include "../../src/common/entropy.h"
#include "../../src/dummy/dummy.h"


typedef struct s_aug_state {
    uint32_t *rng;
    binomial_t *binomial;

    int has_gauss, shift_zig_random_int, has_uint32;
    double gauss;
    uint32_t uinteger;
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

inline void set_seed(aug_state* state, uint32_t seed)
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
    set_seed(state, seeds[0]);
}

inline double random_double(aug_state* state)
{
    uint64_t rn = random_uint64(state);
    return ((rn >> 37) * 67108864.0 + ((rn & 0xFFFFFFFFLL) >> 6)) / 9007199254740992.0;
}