#include <stdio.h>
#include <inttypes.h>
#include "mlfg-1279-861.h"

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