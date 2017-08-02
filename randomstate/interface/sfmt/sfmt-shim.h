#ifdef _WIN32
#include "../../src/common/stdint.h"
#define inline __forceinline
#else
#include <stdint.h>
#endif

#include "../../src/common/binomial.h"
#include "../../src/common/entropy.h"
#include "../../src/sfmt/sfmt.h"

typedef struct s_aug_state {
  sfmt_t *rng;
  binomial_t *binomial;

  int has_gauss, has_gauss_float, shift_zig_random_int, has_uint32;
  float gauss_float;
  double gauss;
  uint32_t uinteger;
  uint64_t zig_random_int;

  uint64_t *buffered_uint64;
  int buffer_loc;
} aug_state;

static inline uint64_t random_uint64_from_buffer(aug_state *state) {
  uint64_t out;
  if (state->buffer_loc >= (2 * SFMT_N)) {
    state->buffer_loc = 0;
    sfmt_fill_array64(state->rng, state->buffered_uint64, 2 * SFMT_N);
  }
  out = state->buffered_uint64[state->buffer_loc];
  state->buffer_loc++;
  return out;
}

static inline uint32_t random_uint32(aug_state *state) {
  uint64_t d;
  if (state->has_uint32) {
    state->has_uint32 = 0;
    return state->uinteger;
  }
  d = random_uint64_from_buffer(state);
  state->uinteger = (uint32_t)(d >> 32);
  state->has_uint32 = 1;
  return (uint32_t)(d & 0xFFFFFFFFUL);
}

static inline uint64_t random_uint64(aug_state *state) {
  return random_uint64_from_buffer(state);
}

static inline double random_double(aug_state *state) {
  return (random_uint64_from_buffer(state) >> 11) * (1.0 / 9007199254740992.0);
}

static inline uint64_t random_raw_values(aug_state *state) {
  return random_uint64_from_buffer(state);
}

extern void entropy_init(aug_state *state);

extern void set_seed_by_array(aug_state *state, uint32_t init_key[],
                              int key_length);

extern void set_seed(aug_state *state, uint32_t seed);

extern void jump_state(aug_state* state);