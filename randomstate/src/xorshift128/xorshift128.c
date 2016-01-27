#include "xorshift128.h"
#include "../splitmix64/splitmix64.h"

extern inline uint64_t xorshift128_next(xorshift128_state* state);

void xorshift128_jump(xorshift128_state* state) {
    static const uint64_t JUMP[] = { 0x8a5cd789635d2dffULL, 0x121fd2155c472f96ULL };

    size_t i;
    uint64_t b;
    uint64_t s0 = 0;
    uint64_t s1 = 0;
    for(i = 0; i < sizeof JUMP / sizeof *JUMP; i++)
        for(b = 0; b < 64; b++) {
            if (JUMP[i] & 1ULL << b) {
                s0 ^= state->s[0];
                s1 ^= state->s[1];
            }
            xorshift128_next(state);
        }

    state->s[0] = s0;
    state->s[1] = s1;
}

void xorshift128_seed(xorshift128_state* state, uint64_t seed)
{

    uint64_t seed_copy = seed;
    uint64_t state1 = splitmix64_next(&seed_copy);
    uint64_t state2 = splitmix64_next(&seed_copy);
    xorshift128_init_state(state, state1, state2);
}

void xorshift128_seed_by_array(xorshift128_state* state, uint64_t *seed_array, int count)
{
    uint64_t initial_state[2] = {0};
    uint64_t seed_copy = 0;
    int iter_bound = 2>=count ? 2 : count;
    int i, loc = 0;
    for (i = 0; i < iter_bound; i++)
    {
        if (i < count)
            seed_copy ^= seed_array[i];
        initial_state[loc] = splitmix64_next(&seed_copy);
        loc ++;
        if (loc == 2)
            loc = 0;
    }
    xorshift128_init_state(state, initial_state[0], initial_state[1]);
}

void xorshift128_init_state(xorshift128_state* state, uint64_t seed, uint64_t inc)
{
    state->s[0] = seed;
    state->s[1] = inc;
}

