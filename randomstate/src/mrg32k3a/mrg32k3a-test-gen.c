#include <stdio.h>
#include <inttypes.h>

int main(void)
{
    int i;
    uint64_t seed = 1ULL;
    uint32_t temp;
    mrg32k3a_state state = { 0 };
    mrg32k3a_seed(&state, seed);

    FILE *fp;
    fp = fopen("mrg32k3a-testset-1.csv", "w");
    if(fp == NULL){
         printf("Couldn't open file\n");
         return -1;
    }
    fprintf(fp, "seed, %" PRIu64 "\n", seed);
    for (i=0; i < 1000; i++)
    {
        temp = mrg32k3a_random(&state);
        fprintf(fp, "%d, %" PRIu32 "\n", i, temp);
        printf("%d, %" PRIu32 "\n", i, temp);
    }
    fclose(fp);

    seed = 12345678910111ULL;
    mrg32k3a_seed(&state, seed);
    fp = fopen("mrg32k3a-testset-2.csv", "w");
    if(fp == NULL){
         printf("Couldn't open file\n");
         return -1;
    }
    fprintf(fp, "seed, %" PRIu64 "\n", seed);
    for (i=0; i < 1000; i++)
    {
        temp = mrg32k3a_random(&state);
        fprintf(fp, "%d, %" PRIu32 "\n", i, temp);
        printf("%d, %" PRIu32 "\n", i, temp);
    }
    fclose(fp);
}
