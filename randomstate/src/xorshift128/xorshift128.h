#include <stddef.h>
#include <math.h>
#ifdef _WIN32
#include "../common/stdint.h"
#define inline __forceinline
#else
#include <stdint.h>
#endif


typedef struct s_xorshift128_state
{
    uint64_t s[2];
} xorshift128_state;

void xorshift128_jump(xorshift128_state* state);

void xorshift128_seed(xorshift128_state* state, uint64_t seed);

void xorshift128_seed_by_array(xorshift128_state* state, uint64_t *seed_array, int count);

void xorshift128_init_state(xorshift128_state* state, uint64_t seed, uint64_t inc);

inline uint64_t xorshift128_next(xorshift128_state* state) {
    uint64_t s1 = state->s[0];
    const uint64_t s0 = state->s[1];
    state->s[0] = s0;
    s1 ^= s1 << 23; // a
    state->s[1] = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5); // b, c
    return state->s[1] + s0;
}
