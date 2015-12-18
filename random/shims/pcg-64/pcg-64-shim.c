#include "pcg-64-shim.h"
#include "../../src/splitmix64/splitmix64.h"

extern inline uint32_t random_uint32(aug_state* state);

extern inline uint64_t random_uint64(aug_state* state);

extern inline void advance(aug_state* state, pcg128_t delta);

extern inline void entropy_init(aug_state* state);

extern inline void seed(aug_state* state, pcg128_t seed, pcg128_t inc);
