#include <math.h>
#include <stdbool.h>
#include "pcg_variants.h"
#include "pcg-64.h"

extern inline double pcg_random_double_2(aug_state* state)
{
    uint64_t rn, a, b;
    rn = pcg64_random_r((*state).rng);
    a = rn >> 37;
    b = (rn & 0xFFFFFFFFLL) >> 6;
    return (a * 67108864.0 + b) / 9007199254740992.0;
}

extern inline double pcg_random_double(pcg64_random_t* rng)
{
    uint64_t rn, a, b;
    rn = pcg64_random_r(rng);
    a = rn >> 37;
    b = (rn & 0xFFFFFFFFLL) >> 6;
    return (a * 67108864.0 + b) / 9007199254740992.0;
}

extern inline double pcg_random_gauss(pcg64_random_t* rng, int *has_gauss, double *gauss){
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


extern inline double pcg_standard_exponential(pcg64_random_t* rng)
{
    /* We use -log(1-U) since U is [0, 1) */
    return -log(1.0 - pcg_random_double(rng));
}


extern inline double pcg_standard_gamma(pcg64_random_t* rng, double shape, int *has_gauss, double *gauss)
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


/* ------------------------------ Ziggurat -------------------------- */
/* From http://projects.scipy.org/numpy/attachment/ticket/1450/ziggurat.patch */
/* Untested */

#define ZIGNOR_C  128  /* number of blocks */
#define ZIGNOR_R  3.442619855899  /* start of the right tail */
#define ZIGNOR_V  9.91256303526217e-3

static inline double zig_NormalTail(pcg64_random_t* rng, int iNegative)
{
    double x, y;
    for (;;) {
        x = log(pcg_random_double(rng)) / ZIGNOR_R;
        y = log(pcg_random_double(rng));
        if (x * x < -2.0 * y)
        return iNegative ? x - ZIGNOR_R : ZIGNOR_R - x;
    }
}
static double s_adZigX[ZIGNOR_C + 1], s_adZigR[ZIGNOR_C];

static void zig_NorInit(void)
{
    int i;
    double f;
    f = exp(-0.5 * ZIGNOR_R * ZIGNOR_R);
    /* f(R) */
    s_adZigX[0] = ZIGNOR_V / f;
    /* [0] is bottom block: V / f(R) */
    s_adZigX[1] = ZIGNOR_R;
    s_adZigX[ZIGNOR_C] = 0;
    for (i = 2; i < ZIGNOR_C; i++) {
        s_adZigX[i] = sqrt(-2.0 * log(ZIGNOR_V / s_adZigX[i - 1] + f));
        f = exp(-0.5 * s_adZigX[i] * s_adZigX[i]);
    }
    for (i = 0; i < ZIGNOR_C; i++)
    s_adZigR[i] = s_adZigX[i + 1] / s_adZigX[i];
}


extern inline double pcg_random_gauss_zig(pcg64_random_t* rng,
                                          int *shift_zig_random_int,
                                          uint64_t *zig_random_int)
{
    static int initalized = 0;
    unsigned int i;
    double x, u, f0, f1;
    if (!initalized) {
        zig_NorInit();
        initalized = 1;
    }
    for (;;) {
        u = 2.0 * pcg_random_double(rng) - 1.0;
        /* Here we create an integer, i, which is between 0 and 127.
        Instead of calling to pcg64_random_r each time, we only do a call
        every 8th time, as the pcg64_random_r  will return 64-bits.
        state->shift_zig_random_int is a counter, which tells if the
        integer state->zig_random_int has to be shifted in the next call,
        or if state->zig_random_int needs to be re-generated.
        */
        if (shift_zig_random_int){
            *zig_random_int >>= 8;
        }
        else{
            *zig_random_int = pcg64_random_r(rng);
        }
        *shift_zig_random_int = (*shift_zig_random_int + 1) % 8;
        i = *zig_random_int & 0x7F;
        /* first try the rectangular boxes */
        if (fabs(u) < s_adZigR[i]){
            return u * s_adZigX[i];
        }
        /* bottom area: sample from the tail */
        if (i == 0){
            return zig_NormalTail(rng, u < 0);
        }
        /* is this a sample from the wedges? */
        x = u * s_adZigX[i];
        f0 = exp(-0.5 * (s_adZigX[i] * s_adZigX[i] - x * x));
        f1 = exp(-0.5 * (s_adZigX[i+1] * s_adZigX[i+1] - x * x));
        if (f1 + pcg_random_double(rng) * (f0 - f1) < 1.0){
            return x;
        }
    }
}
