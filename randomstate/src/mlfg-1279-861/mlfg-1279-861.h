#ifdef _WIN32
#include "../../src/common/stdint.h"
#define inline __forceinline
#else
#include <stdint.h>
#endif

#define K 1279
#define J 861

typedef struct s_mlfg_state
{
    uint64_t lags[K];
    int pos;
    int lag_pos;
} mlfg_state;

void mlfg_seed(mlfg_state* state, uint64_t seed);

void mlfg_seed_by_array(mlfg_state* state, uint64_t *seed_array, int count);

void mlfg_init_state(mlfg_state *state, uint64_t seeds[K]);

/*
*  Returns 64 bits, but the last bit is always 1.
*  Upstream functions are expected to understand this and
*  only use the upper 63 bits.  In most implementations,
*  fewer than 63 bits are needed, and it is thought to
*  be better to use the upper bits first.  For example,
*  when making a 64 bit unsigned int, take the two upper
*  32 bit segments.
*/
inline uint64_t mlfg_next(mlfg_state* state)
{
    state->pos++;
    state->lag_pos++;
    if (state->pos >= K)
        state->pos = 0;
    else if (state->lag_pos >= K)
        state->lag_pos = 0;
    state->lags[state->pos] = state->lags[state->lag_pos] * state->lags[state->pos];
    return state->lags[state->pos];
}

