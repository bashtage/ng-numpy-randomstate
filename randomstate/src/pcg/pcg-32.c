/*
* gcc pcg-32.c pcg-rngs-64.c pcg-output-64.c pcg-output-32.c pcg-advance-64.c -std=c99 -o pcg-32
*/

#include <stdio.h>
#include <inttypes.h>
#include "pcg_variants.h"

int main(void)
{
    int i;
    uint64_t seed1 = 42ULL, seed2 = 1ULL;
    uint32_t temp;
    pcg32_random_t state = {{ 0 }};
    pcg32_srandom_r(&state, seed1, seed2);
    FILE *fp;
    fp = fopen("pcg32-testset-1.csv", "w");
    if(fp == NULL){
         printf("Couldn't open file\n");
         return -1;
    }
    fprintf(fp, "seed, %" PRIu64 ", %" PRIu64 "\n", seed1, seed2);
    for (i=0; i < 1000; i++)
    {
        temp = pcg32_random_r(&state);
        fprintf(fp, "%d, %" PRIu32 "\n", i, temp);
        printf("%d, %" PRIu32 "\n", i, temp);
    }
    fclose(fp);

    seed1 = 12345678910111ULL;
    seed2 = 53ULL;
    pcg32_srandom_r(&state, seed1, seed2);
    fp = fopen("pcg32-testset-2.csv", "w");
    if(fp == NULL){
         printf("Couldn't open file\n");
         return -1;
    }
    fprintf(fp, "seed, %" PRIu64 ", %" PRIu64 "\n", seed1, seed2);
    for (i=0; i < 1000; i++)
    {
        temp = pcg32_random_r(&state);
        fprintf(fp, "%d, %" PRIu32 "\n", i, temp);
        printf("%d, %" PRIu32 "\n", i, temp);
    }
    fclose(fp);

}