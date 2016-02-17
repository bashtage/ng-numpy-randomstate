#include "pcg-32-shim.h"

extern inline uint32_t random_uint32(aug_state* state);

extern inline uint64_t random_uint64(aug_state* state);

extern inline double random_double(aug_state* state);

extern inline uint64_t random_raw_values(aug_state* state);

extern inline void set_seed(aug_state* state, uint64_t seed, uint64_t inc);

extern inline void advance_state(aug_state* state, uint64_t delta);

extern inline void entropy_init(aug_state* state);

