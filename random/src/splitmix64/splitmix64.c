#include "splitmix64.h"
#include <inttypes.h>
#include <stdio.h>
extern inline uint64_t splitmix64_next(uint64_t* x);

int main(void)
{
    uint64_t x = 1234;
    int i;
    uint64_t z;
    for (i=0; i<20; i++)
    {
        z = splitmix64_next(&x);
        printf("%" PRIu64 "\n",x);
        printf("%" PRIu64 "\n",z);
    }
}
