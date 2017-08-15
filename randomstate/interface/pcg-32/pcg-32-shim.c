#include "pcg-32-shim.h"

extern NPY_INLINE uint32_t random_uint32(aug_state *state);

extern NPY_INLINE uint64_t random_uint64(aug_state *state);

extern NPY_INLINE double random_double(aug_state *state);

extern NPY_INLINE uint64_t random_raw_values(aug_state *state);

extern NPY_INLINE void set_seed(aug_state *state, uint64_t seed, uint64_t inc);

extern NPY_INLINE void advance_state(aug_state *state, uint64_t delta);

extern NPY_INLINE void entropy_init(aug_state *state);
