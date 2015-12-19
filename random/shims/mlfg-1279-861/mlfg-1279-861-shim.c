#include "mlfg-1279-861-shim.h"

extern inline uint32_t random_uint32(aug_state* state);

extern inline uint64_t random_uint64(aug_state* state);

extern inline void seed(aug_state* state, uint64_t seed);

extern inline void entropy_init(aug_state* state);

extern inline double random_double(aug_state* state);