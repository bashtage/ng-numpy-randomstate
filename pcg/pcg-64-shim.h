#define __SIZEOF_INT128__ 16

#include <stdint.h>
#include "pcg_variants.h"


typedef struct s_aug_state {
    pcg64_random_t *rng;
    pcg128_t state, inc;

    int has_gauss, shift_zig_random_int;
    double gauss;
    uint64_t zig_random_int;
} aug_state;

inline uint32_t random_uint32(aug_state* state)
{
    return (uint32_t)(pcg64_random_r((*state).rng) & 0xFFFFFFFFLL);
}

inline uint64_t random_uint64(aug_state* state)
{
    return pcg64_random_r((*state).rng);
}
