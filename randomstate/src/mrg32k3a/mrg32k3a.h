#ifdef _WIN32
#include "../../src/common/stdint.h"
#define inline __inline
#else
#include <stdint.h>
#endif

#define m1   4294967087LL
#define m2   4294944443LL
#define a12     1403580LL
#define a13n     810728LL
#define a21      527612LL
#define a23n    1370589LL

typedef struct s_mrg32k3a_state
{
    int64_t s10;
    int64_t s11;
    int64_t s12;
    int64_t s20;
    int64_t s21;
    int64_t s22;
} mrg32k3a_state;

inline uint32_t mrg32k3a_random(mrg32k3a_state* state)
{
    int64_t k;
    int64_t p1, p2;
    /* Component 1 */
    p1 = a12 * state->s11 - a13n * state->s10;
    k = p1 / m1;
    p1 -= k * m1;
    if (p1 < 0)
        p1 += m1;
    state->s10 = state->s11;
    state->s11 = state->s12;
    state->s12 = p1;

    /* Component 2 */
    p2 = a21 * state->s22 - a23n * state->s20;
    k = p2 / m2;
    p2 -= k * m2;
    if (p2 < 0)
        p2 += m2;
    state->s20 = state->s21;
    state->s21 = state->s22;
    state->s22 = p2;

    /* Combination */
    if (p1 <= p2)
        return (uint32_t)(p1 - p2 + m1);
    else
        return (uint32_t)(p1 - p2);
}

void mrg32k3a_seed(mrg32k3a_state* state, uint64_t seed);
