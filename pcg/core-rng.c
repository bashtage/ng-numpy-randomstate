#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef PCG_32_RNG
    #include "shims/pcg-32/pcg-32-shim.h"
#else
    #ifdef PCG_64_RNG
        #include "shims/pcg-64/pcg-64-shim.h"
    #else
        #ifdef DUMMY_RNG
            #include "shims/dummy/dummy-shim.h"
        #else
            #error Unknown RNG!!!  Unknown RNG!!!  Unknown RNG!!!
        #endif
    #endif
#endif

extern inline double random_double(aug_state* state)
{
    uint64_t rn, a, b;
    rn = random_uint64(state);
    a = rn >> 37;
    b = (rn & 0xFFFFFFFFLL) >> 6;
    return (a * 67108864.0 + b) / 9007199254740992.0;
}

extern inline double random_standard_exponential(aug_state* state)
{
    /* We use -log(1-U) since U is [0, 1) */
    return -log(1.0 - random_double(state));
}


extern inline double random_gauss(aug_state* state){
    if (state->has_gauss)
    {
        const double temp = state->gauss;
        state->has_gauss = false;
        state->gauss = 0.0;
        return temp;
    }
    else
    {
        double f, x1, x2, r2;

        do {
            x1 = 2.0*random_double(state) - 1.0;
            x2 = 2.0*random_double(state) - 1.0;
            r2 = x1*x1 + x2*x2;
        }
        while (r2 >= 1.0 || r2 == 0.0);

        /* Box-Muller transform */
        f = sqrt(-2.0*log(r2)/r2);
        /* Keep for next call */
        state->gauss = f*x1;
        state->has_gauss = true;
        return f*x2;
    }
}

extern inline double random_standard_gamma(aug_state* state, double shape)
{
    double b, c;
    double U, V, X, Y;

    if (shape == 1.0)
    {
        return random_standard_exponential(state);
    }
    else if (shape < 1.0)
    {
        for (;;)
        {
            U = random_double(state);
            V = random_standard_exponential(state);
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
                X = random_gauss(state);
                V = 1.0 + c*X;
            } while (V <= 0.0);

            V = V*V*V;
            U = random_double(state);
            if (U < 1.0 - 0.0331*(X*X)*(X*X)) return (b*V);
            if (log(U) < 0.5*X*X + b*(1. - V + log(V))) return (b*V);
        }
    }
}


int main(void)
{
    aug_state state;
    #ifdef DUMMY_RNG
        uint32_t y = 0;
        state.rng = &y;
    #else
        RNG_TYPE *rng = malloc(sizeof *rng);
        state.rng = rng;
        seed(&state, 42u, 52u);
    #endif

    state.gauss = 0.0;
    state.has_gauss = 0;
    double x;
    for (int i=0; i<10000000; i++)
    {
        x = random_double(&state);
    }
    printf("%f\n", x);

    x = random_gauss(&state);
    printf("%f\n", x);

    x = random_standard_exponential(&state);
    printf("%f\n", x);

    x = random_standard_gamma(&state, 1.5);
    printf("%f\n", x);

    return 0;
}