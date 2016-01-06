#include <inttypes.h>
#include <stdio.h>
#include "xorshift128.h"
#include "../splitmix64/splitmix64.h"

extern inline uint64_t xorshift128_next(xorshift128_state* state);

void xorshift128_jump(xorshift128_state* state) {
    static const uint64_t JUMP[] = { 0x8a5cd789635d2dffULL, 0x121fd2155c472f96ULL };

    uint64_t s0 = 0;
    uint64_t s1 = 0;
    for(int i = 0; i < sizeof JUMP / sizeof *JUMP; i++)
        for(int b = 0; b < 64; b++) {
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

void xorshift128_init_state(xorshift128_state* state, uint64_t seed, uint64_t inc)
{
    state->s[0] = seed;
    state->s[1] = inc;
}

int main(void)
{
    int i;
    uint64_t temp, seed = 1ULL;
    xorshift128_state state = { 0 };
    xorshift128_seed(&state, seed);

    FILE *fp;
    fp = fopen("xorshift128-testset-1.csv", "w");
    if(fp == NULL){
         printf("Couldn't open file\n");
         return -1;
    }
    fprintf(fp, "seed, %" PRIu64 "\n", seed);
    for (i=0; i < 1000; i++)
    {
        temp = xorshift128_next(&state);
        fprintf(fp, "%d, %" PRIu64 "\n", i, temp);
        printf("%d, %" PRIu64 "\n", i, temp);
    }
    fclose(fp);

    seed = 12345678910111ULL;
    xorshift128_seed(&state, seed);
    fp = fopen("xorshift128-testset-2.csv", "w");
    if(fp == NULL){
         printf("Couldn't open file\n");
         return -1;
    }
    fprintf(fp, "seed, %" PRIu64 "\n", seed);
    for (i=0; i < 1000; i++)
    {
        temp = xorshift128_next(&state);
        fprintf(fp, "%d, %" PRIu64 "\n", i, temp);
        printf("%d, %" PRIu64 "\n", i, temp);
    }
    fclose(fp);
}