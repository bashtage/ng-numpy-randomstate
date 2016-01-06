#include <stdio.h>
#include <inttypes.h>
#include "mlfg-1279-861.h"
#include "../splitmix64/splitmix64.h"

extern inline uint32_t mlfg_next(mlfg_state* state);

void mlfg_seed(mlfg_state* state, uint64_t seed)
{
    uint32_t seeds[K];
    uint64_t seed_copy = seed;
    int i;
    for (i = 0; i < K; i++)
    {
        seeds[i] = (uint32_t)(splitmix64_next(&seed_copy) >> 32);
        if ((seeds[i] % 2) != 1)
            seeds[i]++;
    }
    mlfg_init_state(state, seeds);
}

void mlfg_init_state(mlfg_state *state, uint32_t seeds[K])
{
    int i;
    for (i = 0; i < K; i++)
    {
        state->lags[i] = seeds[i];
    }
    state->pos = 0;
    state->lag_pos = K - J;
}




int main(void)
{
    int i;
    uint64_t seed = 1ULL;
    uint32_t temp;
    mlfg_state state = {{ 0 }};
    mlfg_seed(&state, seed);

    FILE *fp;
    fp = fopen("mlfg-testset-1.csv", "w");
    if(fp == NULL){
         printf("Couldn't open file\n");
         return -1;
    }
    fprintf(fp, "seed, %" PRIu64 "\n", seed);
    for (i=0; i < 1000; i++)
    {
        temp = mlfg_next(&state);
        fprintf(fp, "%d, %" PRIu32 "\n", i, temp);
        printf("%d, %" PRIu32 "\n", i, temp);
    }
    fclose(fp);

    seed = 12345678910111ULL;
    mlfg_seed(&state, seed);
    fp = fopen("mlfg-testset-2.csv", "w");
    if(fp == NULL){
         printf("Couldn't open file\n");
         return -1;
    }
    fprintf(fp, "seed, %" PRIu64 "\n", seed);
    for (i=0; i < 1000; i++)
    {
        temp = mlfg_next(&state);
        fprintf(fp, "%d, %" PRIu32 "\n", i, temp);
        printf("%d, %" PRIu32 "\n", i, temp);
    }
    fclose(fp);
}