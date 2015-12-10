#include <stdint.h>
#include "dummy.h"

typedef struct s_aug_state {
    uint32_t *rng;

    int has_gauss, shift_zig_random_int;
    double gauss;
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
