#include <stdint.h>
#include <stdio.h>

#define K 607
#define J 273

typedef struct s_alfg_state
{
     uint32_t lags[K];
     int pos;
     int lag_pos;
} alfg_state;

inline uint32_t alfg_next(alfg_state* state)
{
   state->pos++;
   state->lag_pos++;
   if (state->pos >= K)
       state->pos = 0;
   else if (state->lag_pos >= K)
       state->lag_pos = 0;
   state->lags[state->pos] = state->lags[state->lag_pos] + state->lags[state->pos];
   return state->lags[state->pos];
}

inline void alfg_init(alfg_state* state)
{
    int i;
    for (i = 0; i < K; i++)
    {
        state->lags[i] = i + 1;
    }
    state->pos = 0;
    state->lag_pos = K - J;
}