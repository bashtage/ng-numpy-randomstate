#include "mlfg-1279-861.h"
#include "../splitmix64/splitmix64.h"
#include <stdio.h>
#include <inttypes.h>

extern inline uint32_t mlfg_next(mlfg_state* state);

void mlfg_seed(mlfg_state* state, uint64_t seed)
{
    uint64_t rn, seed_copy = seed;
    int any_odd = 0, i;
    while (!any_odd)
    {
        for (i = 0; i < K; i++)
        {
            rn = splitmix64_next(&seed_copy);
            if ((rn % 2) != 1)
                rn++;
            state->lags[i] = (uint32_t)(rn & 0xffffffff);
            any_odd = any_odd |  (rn & 0x00000001);
        }
    }
    state->pos = 0;
    state->lag_pos = K - J;
}

int main(void)
{
   mlfg_state state;
   mlfg_seed(&state, 145968214);
   int i;
   for (i = 0; i < 100000000; i++)
   {
       mlfg_next(&state);
   }
   for (i = 0; i < 100; i++)
       printf("%" PRIu32 "\n", mlfg_next(&state));
}
