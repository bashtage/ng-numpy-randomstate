#include <math.h>
#ifdef _WIN32
#include "../common/stdint.h"
#else
#include <stdint.h>
#endif

#ifdef _WIN32
#define inline __inline
#endif


#define RK_STATE_LEN 624

#define N 624
#define M 397
#define MATRIX_A 0x9908b0dfUL
#define UPPER_MASK 0x80000000UL
#define LOWER_MASK 0x7fffffffUL

typedef struct s_rk_state
{
    uint32_t key[RK_STATE_LEN];
    int pos;
}
rk_state;

extern void rk_seed(rk_state *state, uint32_t seed);

extern void rk_gen(rk_state *state);

/* Slightly optimized reference implementation of the Mersenne Twister */
inline uint32_t rk_random(rk_state *state)
{
    uint32_t y;

    if (state->pos == RK_STATE_LEN) {
        // Move to function to help inlining
        rk_gen(state);
    }
    y = state->key[state->pos++];

    /* Tempering */
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680UL;
    y ^= (y << 15) & 0xefc60000UL;
    y ^= (y >> 18);

    return y;
}

extern void init_by_array(rk_state *state, unsigned long init_key[], int key_length);
