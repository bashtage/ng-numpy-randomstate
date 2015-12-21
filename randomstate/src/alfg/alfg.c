#include "alfg.h"
#include "../splitmix64/splitmix64.h"
#include <stdio.h>
#include <inttypes.h>

extern inline uint32_t alfg_next(alfg_state* state);

extern inline void alfg_init(alfg_state* state);

void alfg_seed(alfg_state* state, uint64_t seed)
{
    uint64_t rn, seed_copy = seed;
    int any_odd = 0, i;
    while (!any_odd)
    {
        for (i = 0; i < K; i++)
        {
            rn = splitmix64_next(&seed_copy);
            state->lags[i] = (uint32_t)(rn & 0xffffffff);
            any_odd = any_odd |  (rn & 0x00000001);
        }
    }
    state->pos = 0;
    state->lag_pos = K - J;
}

int main(void)
{
   alfg_state state;
   alfg_seed(&state, 145968214);
   int i;
   for (i = 0; i < 100000000; i++)
   {
       alfg_next(&state);
   }
   for (i = 0; i < 100; i++)
       printf("%" PRIu32 "\n", alfg_next(&state));
}
