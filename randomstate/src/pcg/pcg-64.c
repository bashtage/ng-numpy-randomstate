/*
*  gcc pcg-64.c pcg-rngs-128.c pcg-output-64.c pcg-output-128.c pcg-advance-128.c pcg-output-32.c -std=c99 -D__SIZEOF_INT128__=16 -o pcg-64
*/

#include <stdio.h>
#include <inttypes.h>
#include "pcg_variants.h"

int main(void)
{
    int i;
    pcg128_t seed1 = 42ULL, seed2 = 1ULL;
    uint64_t temp;
    pcg64_random_t state = {{ 0 ]};
    pcg64_srandom_r(&state, seed1, seed2);
    FILE *fp;
    fp = fopen("pcg64-testset-1.csv", "w");
    if(fp == NULL){
         printf("Couldn't open file\n");
         return -1;
    }
    fprintf(fp, "seed, %" PRIu64 ", %" PRIu64 "\n", (uint64_t)seed1, (uint64_t)seed2);
    for (i=0; i < 1000; i++)
    {
        temp = pcg64_random_r(&state);
        fprintf(fp, "%d, %" PRIu64 "\n", i, temp);
        printf("%d, %" PRIu64 "\n", i, temp);
    }
    fclose(fp);

    seed1 = 12345678910111ULL;
    seed2 = 53ULL;
    pcg64_srandom_r(&state, seed1, seed2);
    fp = fopen("pcg64-testset-2.csv", "w");
    if(fp == NULL){
         printf("Couldn't open file\n");
         return -1;
    }
    fprintf(fp, "seed, %" PRIu64 ", %" PRIu64 "\n", (uint64_t)seed1, (uint64_t)seed2);
    for (i=0; i < 1000; i++)
    {
        temp = pcg64_random_r(&state);
        fprintf(fp, "%d, %" PRIu64 "\n", i, temp);
        printf("%d, %" PRIu64 "\n", i, temp);
    }
    fclose(fp);

}