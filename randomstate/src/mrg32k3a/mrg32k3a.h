#ifdef _WIN32
#include "../../src/common/stdint.h"
#define inline __forceinline
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
    int64_t s1[3];
    int64_t s2[3];
    int loc;
} mrg32k3a_state;

inline uint32_t mrg32k3a_random(mrg32k3a_state* state)
{
    int64_t p1 = 0;
    int64_t p2 = 0;
    /* Component 1 */
    switch (state->loc) {
        case 0:
            p1 = a12 * state->s1[2] - a13n * state->s1[1];
            p2 = a21 * state->s2[0] - a23n * state->s2[1];
            state->loc = 1;
            break;
        case 1:
            p1 = a12 * state->s1[0] - a13n * state->s1[2];
            p2 = a21 * state->s2[1] - a23n * state->s2[2];
            state->loc = 2;
            break;
        case 2:
            p1 = a12 * state->s1[1] - a13n * state->s1[0];
            p2 = a21 * state->s2[2] - a23n * state->s2[0];
            state->loc = 0;
            break;
    }

    p1 -= (p1 >= 0) ?  (p1 / m1) * m1 : (p1 / m1) * m1 - m1;
    state->s1[state->loc] = p1;
    /* Component 2 */
    p2 -= (p2 >= 0) ? (p2 / m2) * m2 : (p2 / m2) * m2 - m2;
    state->s2[state->loc] = p2;

    /* Combination */
    return (uint32_t)((p1 <= p2) ? (p1 - p2 + m1) : (p1 - p2));
}

void mrg32k3a_seed(mrg32k3a_state* state, uint64_t seed);

void mrg32k3a_seed_by_array(mrg32k3a_state* state, uint64_t *seed, int count);