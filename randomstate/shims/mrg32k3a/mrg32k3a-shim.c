#include "mrg32k3a-shim.h"

extern inline uint32_t random_uint32(aug_state* state);

extern inline uint64_t random_uint64(aug_state* state);

extern inline void set_seed(aug_state* state, uint64_t seed);

extern inline void entropy_init(aug_state* state);

extern inline void init_state(aug_state* state, int64_t val[6]);

extern inline double random_double(aug_state* state);