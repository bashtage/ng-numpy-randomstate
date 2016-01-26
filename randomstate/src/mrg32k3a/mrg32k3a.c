/*
   32-bits Random number generator U(0,1): MRG32k3a
   Author: Pierre L'Ecuyer,
   Source: Good Parameter Sets for Combined Multiple Recursive Random
           Number Generators,
           Shorter version in Operations Research,
           47, 1 (1999), 159--164.
   ---------------------------------------------------------
*/
#include "mrg32k3a.h"
#include "../splitmix64/splitmix64.h"

/***
The seeds for state->s10, state->s11, state->s12 must be integers in [0, m1 - 1] and not all 0.
The seeds for state->s20, state->s21, state->s22 must be integers in [0, m2 - 1] and not all 0.
***/

extern inline uint32_t mrg32k3a_random(mrg32k3a_state* state);

void mrg32k3a_seed(mrg32k3a_state* state, uint64_t seed)
{
    uint64_t seed_copy = seed;
    int64_t seeds[6];
    int64_t draw, upper;
    int i;
    for (i=0; i<6; i++)
    {
        if(i < 3)
            upper = m1;
        else
            upper = m2;

        draw = upper;
        while(draw >= upper)
        {
            draw = splitmix64_next(&seed_copy) >> 32;
        }
        seeds[i] = draw;
    }

    state->s1[0] = seeds[0];
    state->s1[1] = seeds[1];
    state->s1[2] = seeds[2];
    state->s2[0] = seeds[3];
    state->s2[1] = seeds[4];
    state->s2[2] = seeds[5];
    state->loc = 2;
}
