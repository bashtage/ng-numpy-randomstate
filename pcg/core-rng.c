#include "core-rng.h"

uint64_t random_bounded_uint64(aug_state* state, uint64_t bound)
{
    uint64_t r, threshold = -bound % bound;
    for (;;) {
        r = random_uint64(state);
        if (r >= threshold)
            return r % bound;
    }
}

uint32_t random_bounded_uint32(aug_state* state, uint32_t bound)
{
    uint32_t r, threshold = -bound % bound;
    for (;;) {
        r = random_uint32(state);
        if (r >= threshold)
            return r % bound;
    }
}

int64_t random_bounded_int64(aug_state* state, int64_t low, int64_t high)
{
    uint64_t r = random_bounded_uint64(state, (uint64_t)(high - low));
    if(r >= -low)
    {
        return (int64_t)(r + low);
    }
    return (int64_t)r + low;
}

int32_t random_bounded_int32(aug_state* state, int32_t low, int32_t high)
{
    uint32_t r = random_bounded_uint32(state, (uint32_t)(high - low));
    if(r >= -low)
    {
        return (int32_t)(r + low);
    }
    return (int32_t)r + low;
}

double random_double(aug_state* state)
{
    uint64_t rn, a, b;
    rn = random_uint64(state);
    a = rn >> 37;
    b = (rn & 0xFFFFFFFFLL) >> 6;
    return (a * 67108864.0 + b) / 9007199254740992.0;
}

double random_standard_exponential(aug_state* state)
{
    /* We use -log(1-U) since U is [0, 1) */
    return -log(1.0 - random_double(state));
}


double random_gauss(aug_state* state) {
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

double random_standard_gamma(aug_state* state, double shape)
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


/*
 * log-gamma function to support some of these distributions. The
 * algorithm comes from SPECFUN by Shanjie Zhang and Jianming Jin and their
 * book "Computation of Special Functions", 1996, John Wiley & Sons, Inc.
 */
static double loggam(double x)
{
    double x0, x2, xp, gl, gl0;
    long k, n;

    static double a[10] = {8.333333333333333e-02,-2.777777777777778e-03,
                           7.936507936507937e-04,-5.952380952380952e-04,
                           8.417508417508418e-04,-1.917526917526918e-03,
                           6.410256410256410e-03,-2.955065359477124e-02,
                           1.796443723688307e-01,-1.39243221690590e+00
                          };
    x0 = x;
    n = 0;
    if ((x == 1.0) || (x == 2.0))
    {
        return 0.0;
    }
    else if (x <= 7.0)
    {
        n = (long)(7 - x);
        x0 = x + n;
    }
    x2 = 1.0/(x0*x0);
    xp = 2*M_PI;
    gl0 = a[9];
    for (k=8; k>=0; k--)
    {
        gl0 *= x2;
        gl0 += a[k];
    }
    gl = gl0/x0 + 0.5*log(xp) + (x0-0.5)*log(x0) - x0;
    if (x <= 7.0)
    {
        for (k=1; k<=n; k++)
        {
            gl -= log(x0-1.0);
            x0 -= 1.0;
        }
    }
    return gl;
}

double random_normal(aug_state *state, double loc, double scale)
{
    return loc + scale * random_gauss(state);
}

double random_exponential(aug_state *state, double scale)
{
    return scale * random_standard_exponential(state);
}

double random_uniform(aug_state *state, double loc, double scale)
{
    return loc + scale*random_double(state);
}

double random_gamma(aug_state *state, double shape, double scale)
{
    return scale * random_standard_gamma(state, shape);
}

double random_beta(aug_state *state, double a, double b)
{
    double Ga, Gb;

    if ((a <= 1.0) && (b <= 1.0))
    {
        double U, V, X, Y;
        /* Use Jonk's algorithm */

        while (1)
        {
            U = random_double(state);
            V = random_double(state);
            X = pow(U, 1.0/a);
            Y = pow(V, 1.0/b);

            if ((X + Y) <= 1.0)
            {
                if (X +Y > 0)
                {
                    return X / (X + Y);
                }
                else
                {
                    double logX = log(U) / a;
                    double logY = log(V) / b;
                    double logM = logX > logY ? logX : logY;
                    logX -= logM;
                    logY -= logM;

                    return exp(logX - log(exp(logX) + exp(logY)));
                }
            }
        }
    }
    else
    {
        Ga = random_standard_gamma(state, a);
        Gb = random_standard_gamma(state, b);
        return Ga/(Ga + Gb);
    }
}

double random_chisquare(aug_state *state, double df)
{
    return 2.0*random_standard_gamma(state, df/2.0);
}


double random_f(aug_state *state, double dfnum, double dfden)
{
    return ((random_chisquare(state, dfnum) * dfden) /
            (random_chisquare(state, dfden) * dfnum));
}

double random_standard_cauchy(aug_state *state)
{
    return random_gauss(state) / random_gauss(state);
}

double random_pareto(aug_state *state, double a)
{
    return exp(random_standard_exponential(state)/a) - 1;
}

double random_weibull(aug_state *state, double a)
{
    return pow(random_standard_exponential(state), 1./a);
}

double random_power(aug_state *state, double a)
{
    return pow(1 - exp(-random_standard_exponential(state)), 1./a);
}

double random_laplace(aug_state *state, double loc, double scale)
{
    double U;

    U = random_double(state);
    if (U < 0.5)
    {
        U = loc + scale * log(U + U);
    } else
    {
        U = loc - scale * log(2.0 - U - U);
    }
    return U;
}

double random_gumbel(aug_state *state, double loc, double scale)
{
    double U;

    U = 1.0 - random_double(state);
    return loc - scale * log(-log(U));
}

double random_logistic(aug_state *state, double loc, double scale)
{
    double U;

    U = random_double(state);
    return loc + scale * log(U/(1.0 - U));
}

double random_lognormal(aug_state *state, double mean, double sigma)
{
    return exp(random_normal(state, mean, sigma));
}

double random_rayleigh(aug_state *state, double mode)
{
    return mode*sqrt(-2.0 * log(1.0 - random_double(state)));
}

double random_standard_t(aug_state *state, double df)
{
    double num, denom;

    num = random_gauss(state);
    denom = random_standard_gamma(state, df/2);
    return sqrt(df/2)*num/sqrt(denom);
}

static long random_poisson_mult(aug_state *state, double lam)
{
    long X;
    double prod, U, enlam;

    enlam = exp(-lam);
    X = 0;
    prod = 1.0;
    while (1)
    {
        U = random_double(state);
        prod *= U;
        if (prod > enlam)
        {
            X += 1;
        }
        else
        {
            return X;
        }
    }
}


#define LS2PI 0.91893853320467267
#define TWELFTH 0.083333333333333333333333
static long random_poisson_ptrs(aug_state *state, double lam)
{
    long k;
    double U, V, slam, loglam, a, b, invalpha, vr, us;

    slam = sqrt(lam);
    loglam = log(lam);
    b = 0.931 + 2.53*slam;
    a = -0.059 + 0.02483*b;
    invalpha = 1.1239 + 1.1328/(b-3.4);
    vr = 0.9277 - 3.6224/(b-2);

    while (1)
    {
        U = random_double(state) - 0.5;
        V = random_double(state);
        us = 0.5 - fabs(U);
        k = (long)floor((2*a/us + b)*U + lam + 0.43);
        if ((us >= 0.07) && (V <= vr))
        {
            return k;
        }
        if ((k < 0) ||
                ((us < 0.013) && (V > us)))
        {
            continue;
        }
        if ((log(V) + log(invalpha) - log(a/(us*us)+b)) <=
                (-lam + k*loglam - loggam(k+1)))
        {
            return k;
        }


    }

}


long random_poisson(aug_state *state, double lam)
{
    if (lam >= 10)
    {
        return random_poisson_ptrs(state, lam);
    }
    else if (lam == 0)
    {
        return 0;
    }
    else
    {
        return random_poisson_mult(state, lam);
    }
}

long rk_negative_binomial(aug_state *state, double n, double p)
{
    double Y = random_gamma(state, n, (1-p)/p);
    return random_poisson(state, Y);
}


/* ------------------------------ Ziggurat -------------------------- */
/* From http://projects.scipy.org/numpy/attachment/ticket/1450/ziggurat.patch */
/* Untested */

#define ZIGNOR_C  128  /* number of blocks */
#define ZIGNOR_R  3.442619855899  /* start of the right tail */
#define ZIGNOR_V  9.91256303526217e-3

static double zig_NormalTail(aug_state* state, int iNegative)
{
    double x, y;
    for (;;) {
        x = log(random_double(state)) / ZIGNOR_R;
        y = log(random_double(state));
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


extern double random_gauss_zig(aug_state* state)
{
    static int initalized = 0;
    unsigned int i;
    double x, u, f0, f1;
    if (!initalized) {
        zig_NorInit();
        initalized = 1;
    }
    for (;;) {
        u = 2.0 * random_double(state) - 1.0;
        /* Here we create an integer, i, which is between 0 and 127.
        Instead of calling to random_uint64 each time, we only do a call
        every 8th time, as the random_uint64 will return 64-bits.
        state->shift_zig_random_int is a counter, which tells if the
        integer state->zig_random_int has to be shifted in the next call,
        or if state->zig_random_int needs to be re-generated.
        */
        if (state->shift_zig_random_int){
            state->zig_random_int >>= 8;
        }
        else{
            state->zig_random_int = random_uint64(state);
        }
        state->shift_zig_random_int = (state->shift_zig_random_int + 1) % 8;
        i = state->zig_random_int & 0x7F;
        /* first try the rectangular boxes */
        if (fabs(u) < s_adZigR[i]){
            return u * s_adZigX[i];
        }
        /* bottom area: sample from the tail */
        if (i == 0){
            return zig_NormalTail(state, u < 0);
        }
        /* is this a sample from the wedges? */
        x = u * s_adZigX[i];
        f0 = exp(-0.5 * (s_adZigX[i] * s_adZigX[i] - x * x));
        f1 = exp(-0.5 * (s_adZigX[i+1] * s_adZigX[i+1] - x * x));
        if (f1 + random_double(state) * (f0 - f1) < 1.0){
            return x;
        }
    }
}
