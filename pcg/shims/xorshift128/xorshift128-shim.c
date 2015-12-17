#include "xorshift128-shim.h"

extern inline uint32_t random_uint32(aug_state* state);

extern inline uint64_t random_uint64(aug_state* state);

extern inline void seed(aug_state* state, uint64_t seed, uint64_t inc);

extern inline void jump(aug_state* state);

extern inline void entropy_init(aug_state* state);