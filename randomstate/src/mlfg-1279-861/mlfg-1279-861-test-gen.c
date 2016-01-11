/*
*  gcc -O2 ../splitmix64/splitmix64.c mlfg-1279-861.c mlfg-1279-861-test-gen.c -o mlfg -std=c99
*/

#include <stdio.h>
#include <inttypes.h>
#include "../splitmix64/splitmix64.h"
#include "mlfg-1279-861.h"

int main(void)
{
    int i;
    uint64_t seed = 1ULL;
    uint64_t temp;
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
        fprintf(fp, "%d, %" PRIu64 "\n", i, temp);
        printf("%d, %" PRIu64 "\n", i, temp);
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
        fprintf(fp, "%d, %" PRIu64 "\n", i, temp);
        printf("%d, %" PRIu64 "\n", i, temp);
    }
    fclose(fp);
}