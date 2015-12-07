#include <math.h>
#include <stdbool.h>
#include "pcg_variants.h"

extern inline double pcg_random_double(pcg32_random_t* rng){
    uint32_t a, b;
    a = pcg32_random_r(rng) >> 5;
    b = pcg32_random_r(rng) >> 6;
    return (a * 67108864.0 + b) / 9007199254740992.0;
}

extern inline double pcg_random_gauss(pcg32_random_t* rng, int *has_gauss, double *gauss){
    if (*has_gauss)
    {
        const double temp = *gauss;
        *has_gauss = false;
        *gauss = 0.0;
        return temp;
    }
    else
    {
        double f, x1, x2, r2;

        do {
            x1 = 2.0*pcg_random_double(rng) - 1.0;
            x2 = 2.0*pcg_random_double(rng) - 1.0;
            r2 = x1*x1 + x2*x2;
        }
        while (r2 >= 1.0 || r2 == 0.0);

        /* Box-Muller transform */
        f = sqrt(-2.0*log(r2)/r2);
        /* Keep for next call */
        *gauss = f*x1;
        *has_gauss = true;
        return f*x2;
    }
}

