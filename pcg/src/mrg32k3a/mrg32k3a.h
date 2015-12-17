#include <stdint.h>

typedef struct s_mrg32k3a_state
{
    int64_t s10;
    int64_t s11;
    int64_t s12;
    int64_t s20;
    int64_t s21;
    int64_t s22;
} mrg32k3a_state;

uint32_t mrg32k3a_random(mrg32k3a_state* state);

void mrg32k3a_seed(mrg32k3a_state* state, int64_t seeds[6]);