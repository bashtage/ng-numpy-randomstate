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


extern inline double pcg_standard_exponential(pcg32_random_t* rng)
{
    /* We use -log(1-U) since U is [0, 1) */
    return -log(1.0 - pcg_random_double(rng));
}


extern inline double pcg_standard_gamma(pcg32_random_t* rng, double shape, int *has_gauss, double *gauss)
{
    double b, c;
    double U, V, X, Y;

    if (shape == 1.0)
    {
        return pcg_standard_exponential(rng);
    }
    else if (shape < 1.0)
    {
        for (;;)
        {
            U = pcg_random_double(rng);
            V = pcg_standard_exponential(rng);
            if (U <= 1.0 - shape)
            {
                X = pow(U, 1./shape);
                if (X <= V)
                {
                    return X;
                }
            }
            else
            {
                Y = -log((1-U)/shape);
                X = pow(1.0 - shape + shape*Y, 1./shape);
                if (X <= (V + Y))
                {
                    return X;
                }
            }
        }
    }
    else
    {
        b = shape - 1./3.;
        c = 1./sqrt(9*b);
        for (;;)
        {
            do
            {
                X = pcg_random_gauss(rng, has_gauss, gauss);
                V = 1.0 + c*X;
            } while (V <= 0.0);

            V = V*V*V;
            U = pcg_random_double(rng);
            if (U < 1.0 - 0.0331*(X*X)*(X*X)) return (b*V);
            if (log(U) < 0.5*X*X + b*(1. - V + log(V))) return (b*V);
        }
    }
}


