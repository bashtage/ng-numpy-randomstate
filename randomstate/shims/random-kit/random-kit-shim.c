#include "random-kit-shim.h"

extern inline uint32_t random_uint32(aug_state* state);

extern inline uint64_t random_uint64(aug_state* state);

extern inline void set_seed(aug_state* state, uint32_t seed);

extern void set_seed_by_array(aug_state* state, unsigned long init_key[], int key_length)
{
    init_by_array(state->rng, init_key, key_length);
}

extern inline void entropy_init(aug_state* state);

extern inline double random_double(aug_state* state);