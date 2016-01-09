#include <stdio.h>
#include <inttypes.h>

int main(void)
{
    int i;
    uint64_t temp, seed = 1ULL;
    xorshift1024_state state = {{ 0 }};
    xorshift1024_seed(&state, seed);

    FILE *fp;
    fp = fopen("xorshift1024-testset-1.csv", "w");
    if(fp == NULL){
         printf("Couldn't open file\n");
         return -1;
    }
    fprintf(fp, "seed, %" PRIu64 "\n", seed);
    for (i=0; i < 1000; i++)
    {
        temp = xorshift1024_next(&state);
        fprintf(fp, "%d, %" PRIu64 "\n", i, temp);
        printf("%d, %" PRIu64 "\n", i, temp);
    }
    fclose(fp);

    seed = 12345678910111ULL;
    xorshift1024_seed(&state, seed);
    fp = fopen("xorshift1024-testset-2.csv", "w");
    if(fp == NULL){
         printf("Couldn't open file\n");
         return -1;
    }
    fprintf(fp, "seed, %" PRIu64 "\n", seed);
    for (i=0; i < 1000; i++)
    {
        temp = xorshift1024_next(&state);
        fprintf(fp, "%d, %" PRIu64 "\n", i, temp);
        printf("%d, %" PRIu64 "\n", i, temp);
    }
    fclose(fp);
}