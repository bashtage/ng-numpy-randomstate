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

#define m1   4294967087LL
#define m2   4294944443LL
#define a12     1403580LL
#define a13n     810728LL
#define a21      527612LL
#define a23n    1370589LL

/***
The seeds for state->s10, state->s11, state->s12 must be integers in [0, m1 - 1] and not all 0.
The seeds for state->s20, state->s21, state->s22 must be integers in [0, m2 - 1] and not all 0.
***/
uint32_t mrg32k3a_random(mrg32k3a_state* state)
{
   int64_t k;
   int64_t p1, p2;
   /* Component 1 */
   p1 = a12 * state->s11 - a13n * state->s10;
   k = p1 / m1;
   p1 -= k * m1;
   if (p1 < 0.0)
      p1 += m1;
   state->s10 = state->s11;
   state->s11 = state->s12;
   state->s12 = p1;

   /* Component 2 */
   p2 = a21 * state->s22 - a23n * state->s20;
   k = p2 / m2;
   p2 -= k * m2;
   if (p2 < 0.0)
      p2 += m2;
   state->s20 = state->s21;
   state->s21 = state->s22;
   state->s22 = p2;

   /* Combination */
   if (p1 <= p2)
      return (p1 - p2 + m1);
   else
      return (p1 - p2);
}

void mrg32k3a_seed(mrg32k3a_state* state, int64_t seeds[6])
{
   state->s10 = seeds[0];
   state->s11 = seeds[1];
   state->s12 = seeds[2];
   state->s20 = seeds[3];
   state->s21 = seeds[4];
   state->s22 = seeds[5];
}