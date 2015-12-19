#include <stdint.h>
#include <stdio.h>

#define K 1279
#define J 861

typedef struct s_mlfg_state
{
     uint32_t lags[K];
     int pos;
     int lag_pos;
} mlfg_state;

void mlfg_seed(mlfg_state* state, uint64_t seed);

inline uint32_t mlfg_next(mlfg_state* state)
{
   state->pos++;
   state->lag_pos++;
   if (state->pos >= K)
       state->pos = 0;
   else if (state->lag_pos >= K)
       state->lag_pos = 0;
   state->lags[state->pos] = state->lags[state->lag_pos] * state->lags[state->pos];
   return state->lags[state->pos] >> 1;
}
