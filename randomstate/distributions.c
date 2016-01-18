#include "distributions.h"
#include <limits.h>

int64_t random_positive_int64(aug_state* state)
{
    return random_uint64(state) >> 1;
}

int32_t random_positive_int32(aug_state* state)
{
    return random_uint32(state) >> 1;
}

long random_positive_int(aug_state* state)
{
#if ULONG_MAX <= 0xffffffffUL
    return (long)(random_uint32(state) >> 1);
#else
    return (long)(random_uint64(state) >> 1);
#endif
}

unsigned long random_uint(aug_state* state)
{
#if ULONG_MAX <= 0xffffffffUL
    return random_uint32(state);
#else
    return random_uint64(state);
#endif
}


double random_standard_uniform(aug_state* state)
{
    return random_double(state);
}


void random_uniform_fill(aug_state* state, npy_intp count, double *out)
{
    int i;
    for (i=0; i < count; i++) {
        out[i] = random_double(state);
    }
}

double random_standard_exponential(aug_state* state)
{
    /* We use -log(1-U) since U is [0, 1) */
    return -log(1.0 - random_double(state));
}

void random_standard_exponential_fill(aug_state* state, npy_intp count, double *out)
{
    int i;
    for (i=0; i < count; i++) {
        out[i] = -log(1.0 - random_double(state));
    }
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

void random_gauss_fill(aug_state* state, npy_intp count, double *out) {
    int i;
    double f, x1, x2, r2;
    for (i = 0; i< count; i++) {
        if (state->has_gauss)
        {
            out[i] = state->gauss;
            state->has_gauss = false;
            state->gauss = 0.0;
        }
        else
        {
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
            out[i] = f*x2;
        }
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

double random_normal_zig(aug_state *state, double loc, double scale)
{
    return loc + scale * random_gauss_zig_julia(state);
}

double random_exponential(aug_state *state, double scale)
{
    return scale * random_standard_exponential(state);
}

double random_uniform(aug_state *state, double lower, double range)
{
    return lower + range*random_double(state);
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

long random_negative_binomial(aug_state *state, double n, double p)
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
        if (state->shift_zig_random_int) {
            state->zig_random_int >>= 8;
        }
        else {
            state->zig_random_int = random_uint64(state);
        }
        state->shift_zig_random_int = (state->shift_zig_random_int + 1) % 8;
        i = state->zig_random_int & 0x7F;
        /* first try the rectangular boxes */
        if (fabs(u) < s_adZigR[i]) {
            return u * s_adZigX[i];
        }
        /* bottom area: sample from the tail */
        if (i == 0) {
            return zig_NormalTail(state, u < 0);
        }
        /* is this a sample from the wedges? */
        x = u * s_adZigX[i];
        f0 = exp(-0.5 * (s_adZigX[i] * s_adZigX[i] - x * x));
        f1 = exp(-0.5 * (s_adZigX[i+1] * s_adZigX[i+1] - x * x));
        if (f1 + random_double(state) * (f0 - f1) < 1.0) {
            return x;
        }
    }
}



long random_binomial_btpe(aug_state *state, long n, double p)
{
    double r,q,fm,p1,xm,xl,xr,c,laml,lamr,p2,p3,p4;
    double a,u,v,s,F,rho,t,A,nrq,x1,x2,f1,f2,z,z2,w,w2,x;
    long m,y,k,i;

    if (!(state->binomial->has_binomial) ||
            (state->binomial->nsave != n) ||
            (state->binomial->psave != p))
    {
        /* initialize */
        state->binomial->nsave = n;
        state->binomial->psave = p;
        state->binomial->has_binomial = 1;
        state->binomial->r = r = min(p, 1.0-p);
        state->binomial->q = q = 1.0 - r;
        state->binomial->fm = fm = n*r+r;
        state->binomial->m = m = (long)floor(state->binomial->fm);
        state->binomial->p1 = p1 = floor(2.195*sqrt(n*r*q)-4.6*q) + 0.5;
        state->binomial->xm = xm = m + 0.5;
        state->binomial->xl = xl = xm - p1;
        state->binomial->xr = xr = xm + p1;
        state->binomial->c = c = 0.134 + 20.5/(15.3 + m);
        a = (fm - xl)/(fm-xl*r);
        state->binomial->laml = laml = a*(1.0 + a/2.0);
        a = (xr - fm)/(xr*q);
        state->binomial->lamr = lamr = a*(1.0 + a/2.0);
        state->binomial->p2 = p2 = p1*(1.0 + 2.0*c);
        state->binomial->p3 = p3 = p2 + c/laml;
        state->binomial->p4 = p4 = p3 + c/lamr;
    }
    else
    {
        r = state->binomial->r;
        q = state->binomial->q;
        fm = state->binomial->fm;
        m = state->binomial->m;
        p1 = state->binomial->p1;
        xm = state->binomial->xm;
        xl = state->binomial->xl;
        xr = state->binomial->xr;
        c = state->binomial->c;
        laml = state->binomial->laml;
        lamr = state->binomial->lamr;
        p2 = state->binomial->p2;
        p3 = state->binomial->p3;
        p4 = state->binomial->p4;
    }

    /* sigh ... */
Step10:
    nrq = n*r*q;
    u = random_double(state)*p4;
    v = random_double(state);
    if (u > p1) goto Step20;
    y = (long)floor(xm - p1*v + u);
    goto Step60;

Step20:
    if (u > p2) goto Step30;
    x = xl + (u - p1)/c;
    v = v*c + 1.0 - fabs(m - x + 0.5)/p1;
    if (v > 1.0) goto Step10;
    y = (long)floor(x);
    goto Step50;

Step30:
    if (u > p3) goto Step40;
    y = (long)floor(xl + log(v)/laml);
    if (y < 0) goto Step10;
    v = v*(u-p2)*laml;
    goto Step50;

Step40:
    y = (long)floor(xr - log(v)/lamr);
    if (y > n) goto Step10;
    v = v*(u-p3)*lamr;

Step50:
    k = labs(y - m);
    if ((k > 20) && (k < ((nrq)/2.0 - 1))) goto Step52;

    s = r/q;
    a = s*(n+1);
    F = 1.0;
    if (m < y)
    {
        for (i=m+1; i<=y; i++)
        {
            F *= (a/i - s);
        }
    }
    else if (m > y)
    {
        for (i=y+1; i<=m; i++)
        {
            F /= (a/i - s);
        }
    }
    if (v > F) goto Step10;
    goto Step60;

Step52:
    rho = (k/(nrq))*((k*(k/3.0 + 0.625) + 0.16666666666666666)/nrq + 0.5);
    t = -k*k/(2*nrq);
    A = log(v);
    if (A < (t - rho)) goto Step60;
    if (A > (t + rho)) goto Step10;

    x1 = y+1;
    f1 = m+1;
    z = n+1-m;
    w = n-y+1;
    x2 = x1*x1;
    f2 = f1*f1;
    z2 = z*z;
    w2 = w*w;
    if (A > (xm*log(f1/x1)
             + (n-m+0.5)*log(z/w)
             + (y-m)*log(w*r/(x1*q))
             + (13680.-(462.-(132.-(99.-140./f2)/f2)/f2)/f2)/f1/166320.
             + (13680.-(462.-(132.-(99.-140./z2)/z2)/z2)/z2)/z/166320.
             + (13680.-(462.-(132.-(99.-140./x2)/x2)/x2)/x2)/x1/166320.
             + (13680.-(462.-(132.-(99.-140./w2)/w2)/w2)/w2)/w/166320.))
    {
        goto Step10;
    }

Step60:
    if (p > 0.5)
    {
        y = n - y;
    }

    return y;
}

long random_binomial_inversion(aug_state *state, long n, double p)
{
    double q, qn, np, px, U;
    long X, bound;

    if (!(state->binomial->has_binomial) ||
            (state->binomial->nsave != n) ||
            (state->binomial->psave != p))
    {
        state->binomial->nsave = n;
        state->binomial->psave = p;
        state->binomial->has_binomial = 1;
        state->binomial->q = q = 1.0 - p;
        state->binomial->r = qn = exp(n * log(q));
        state->binomial->c = np = n*p;
        state->binomial->m = bound = (long)min(n, np + 10.0*sqrt(np*q + 1));
    } else
    {
        q = state->binomial->q;
        qn = state->binomial->r;
        np = state->binomial->c;
        bound = state->binomial->m;
    }
    X = 0;
    px = qn;
    U = random_double(state);
    while (U > px)
    {
        X++;
        if (X > bound)
        {
            X = 0;
            px = qn;
            U = random_double(state);
        } else
        {
            U -= px;
            px  = ((n-X+1) * p * px)/(X*q);
        }
    }
    return X;
}

long random_binomial(aug_state *state, double p, long n)
{
    double q;

    if (p <= 0.5)
    {
        if (p*n <= 30.0)
        {
            return random_binomial_inversion(state, n, p);
        }
        else
        {
            return random_binomial_btpe(state, n, p);
        }
    }
    else
    {
        q = 1.0-p;
        if (q*n <= 30.0)
        {
            return n - random_binomial_inversion(state, n, q);
        }
        else
        {
            return n - random_binomial_btpe(state, n, q);
        }
    }

}

double random_noncentral_chisquare(aug_state *state, double df, double nonc)
{
    if (nonc == 0) {
        return random_chisquare(state, df);
    }
    if(1 < df)
    {
        const double Chi2 = random_chisquare(state, df - 1);
        const double n = random_gauss(state) + sqrt(nonc);
        return Chi2 + n*n;
    }
    else
    {
        const long i = random_poisson(state, nonc / 2.0);
        return random_chisquare(state, df + 2 * i);
    }
}

double random_noncentral_f(aug_state *state, double dfnum, double dfden, double nonc)
{
    double t = random_noncentral_chisquare(state, dfnum, nonc) * dfden;
    return t / (random_chisquare(state, dfden) * dfnum);
}


double random_wald(aug_state *state, double mean, double scale)
{
    double U, X, Y;
    double mu_2l;

    mu_2l = mean / (2*scale);
    Y = random_gauss(state);
    Y = mean*Y*Y;
    X = mean + mu_2l*(Y - sqrt(4*scale*Y + Y*Y));
    U = random_double(state);
    if (U <= mean/(mean+X))
    {
        return X;
    } else
    {
        return mean*mean/X;
    }
}


double random_vonmises(aug_state *state, double mu, double kappa)
{
    double s;
    double U, V, W, Y, Z;
    double result, mod;
    int neg;

    if (kappa < 1e-8)
    {
        return M_PI * (2*random_double(state)-1);
    }
    else
    {
        /* with double precision rho is zero until 1.4e-8 */
        if (kappa < 1e-5) {
            /*
             * second order taylor expansion around kappa = 0
             * precise until relatively large kappas as second order is 0
             */
            s = (1./kappa + kappa);
        }
        else {
            double r = 1 + sqrt(1 + 4*kappa*kappa);
            double rho = (r - sqrt(2*r)) / (2*kappa);
            s = (1 + rho*rho)/(2*rho);
        }

        while (1)
        {
            U = random_double(state);
            Z = cos(M_PI*U);
            W = (1 + s*Z)/(s + Z);
            Y = kappa * (s - W);
            V = random_double(state);
            if ((Y*(2-Y) - V >= 0) || (log(Y/V)+1 - Y >= 0))
            {
                break;
            }
        }

        U = random_double(state);

        result = acos(W);
        if (U < 0.5)
        {
            result = -result;
        }
        result += mu;
        neg = (result < 0);
        mod = fabs(result);
        mod = (fmod(mod+M_PI, 2*M_PI)-M_PI);
        if (neg)
        {
            mod *= -1;
        }

        return mod;
    }
}


long random_logseries(aug_state *state, double p)
{
    double q, r, U, V;
    long result;

    r = log(1.0 - p);

    while (1) {
        V = random_double(state);
        if (V >= p) {
            return 1;
        }
        U = random_double(state);
        q = 1.0 - exp(r*U);
        if (V <= q*q) {
            result = (long)floor(1 + log(V)/log(q));
            if (result < 1) {
                continue;
            }
            else {
                return result;
            }
        }
        if (V >= q) {
            return 1;
        }
        return 2;
    }
}


long random_geometric_search(aug_state *state, double p)
{
    double U;
    long X;
    double sum, prod, q;

    X = 1;
    sum = prod = p;
    q = 1.0 - p;
    U = random_double(state);
    while (U > sum)
    {
        prod *= q;
        sum += prod;
        X++;
    }
    return X;
}

long random_geometric_inversion(aug_state *state, double p)
{
    return (long)ceil(log(1.0-random_double(state))/log(1.0-p));
}

long random_geometric(aug_state *state, double p)
{
    if (p >= 0.333333333333333333333333)
    {
        return random_geometric_search(state, p);
    } else
    {
        return random_geometric_inversion(state, p);
    }
}

long random_zipf(aug_state *state, double a)
{
    double T, U, V;
    long X;
    double am1, b;

    am1 = a - 1.0;
    b = pow(2.0, am1);
    do
    {
        U = 1.0-random_double(state);
        V = random_double(state);
        X = (long)floor(pow(U, -1.0/am1));
        /* The real result may be above what can be represented in a signed
         * long. It will get casted to -sys.maxint-1. Since this is
         * a straightforward rejection algorithm, we can just reject this value
         * in the rejection condition below. This function then models a Zipf
         * distribution truncated to sys.maxint.
         */
        T = pow(1.0 + 1.0/X, am1);
    } while (((V*X*(T-1.0)/(b-1.0)) > (T/b)) || X < 1);
    return X;
}


double random_triangular(aug_state *state, double left, double mode, double right)
{
    double base, leftbase, ratio, leftprod, rightprod;
    double U;

    base = right - left;
    leftbase = mode - left;
    ratio = leftbase / base;
    leftprod = leftbase*base;
    rightprod = (right - mode)*base;

    U = random_double(state);
    if (U <= ratio)
    {
        return left + sqrt(U*leftprod);
    } else
    {
        return right - sqrt((1.0 - U) * rightprod);
    }
}


long random_hypergeometric_hyp(aug_state *state, long good, long bad, long sample)
{
    long d1, k, z;
    double d2, u, y;

    d1 = bad + good - sample;
    d2 = (double)min(bad, good);

    y = d2;
    k = sample;
    while (y > 0.0)
    {
        u = random_double(state);
        y -= (long)floor(u + y/(d1 + k));
        k--;
        if (k == 0) break;
    }
    z = (long)(d2 - y);
    if (good > bad) z = sample - z;
    return z;
}

/* D1 = 2*sqrt(2/e) */
/* D2 = 3 - 2*sqrt(3/e) */
#define D1 1.7155277699214135
#define D2 0.8989161620588988
long random_hypergeometric_hrua(aug_state *state, long good, long bad, long sample)
{
    long mingoodbad, maxgoodbad, popsize, m, d9;
    double d4, d5, d6, d7, d8, d10, d11;
    long Z;
    double T, W, X, Y;

    mingoodbad = min(good, bad);
    popsize = good + bad;
    maxgoodbad = max(good, bad);
    m = min(sample, popsize - sample);
    d4 = ((double)mingoodbad) / popsize;
    d5 = 1.0 - d4;
    d6 = m*d4 + 0.5;
    d7 = sqrt((double)(popsize - m) * sample * d4 * d5 / (popsize - 1) + 0.5);
    d8 = D1*d7 + D2;
    d9 = (long)floor((double)(m + 1) * (mingoodbad + 1) / (popsize + 2));
    d10 = (loggam(d9+1) + loggam(mingoodbad-d9+1) + loggam(m-d9+1) +
           loggam(maxgoodbad-m+d9+1));
    d11 = min(min(m, mingoodbad)+1.0, floor(d6+16*d7));
    /* 16 for 16-decimal-digit precision in D1 and D2 */

    while (1)
    {
        X = random_double(state);
        Y = random_double(state);
        W = d6 + d8*(Y- 0.5)/X;

        /* fast rejection: */
        if ((W < 0.0) || (W >= d11)) continue;

        Z = (long)floor(W);
        T = d10 - (loggam(Z+1) + loggam(mingoodbad-Z+1) + loggam(m-Z+1) +
                   loggam(maxgoodbad-m+Z+1));

        /* fast acceptance: */
        if ((X*(4.0-X)-3.0) <= T) break;

        /* fast rejection: */
        if (X*(X-T) >= 1) continue;

        if (2.0*log(X) <= T) break;  /* acceptance */
    }

    /* this is a correction to HRUA* by Ivan Frohne in rv.py */
    if (good > bad) Z = m - Z;

    /* another fix from rv.py to allow sample to exceed popsize/2 */
    if (m < sample) Z = good - Z;

    return Z;
}
#undef D1
#undef D2

long random_hypergeometric(aug_state *state, long good, long bad, long sample)
{
    if (sample > 10)
    {
        return random_hypergeometric_hrua(state, good, bad, sample);
    } else
    {
        return random_hypergeometric_hyp(state, good, bad, sample);
    }
}


/*
*  Julia implementation of Ziggurat algo
*  MIT license
*/

static const uint64_t ki[] =
{   0x0007799ec012f7b2,0x0000000000000000,0x0006045f4c7de363,0x0006d1aa7d5ec0a5,
    0x000728fb3f60f777,0x0007592af4e9fbc0,0x000777a5c0bf655d,0x00078ca3857d2256,
    0x00079bf6b0ffe58b,0x0007a7a34ab092ad,0x0007b0d2f20dd1cb,0x0007b83d3aa9cb52,
    0x0007be597614224d,0x0007c3788631abe9,0x0007c7d32bc192ee,0x0007cb9263a6e86d,
    0x0007ced483edfa84,0x0007d1b07ac0fd39,0x0007d437ef2da5fc,0x0007d678b069aa6e,
    0x0007d87db38c5c87,0x0007da4fc6a9ba62,0x0007dbf611b37f3b,0x0007dd7674d0f286,
    0x0007ded5ce8205f6,0x0007e018307fb62b,0x0007e141081bd124,0x0007e2533d712de8,
    0x0007e3514bbd7718,0x0007e43d54944b52,0x0007e5192f25ef42,0x0007e5e67481118d,
    0x0007e6a6897c1ce2,0x0007e75aa6c7f64c,0x0007e803df8ee498,0x0007e8a326eb6272,
    0x0007e93954717a28,0x0007e9c727f8648f,0x0007ea4d4cc85a3c,0x0007eacc5c4907a9,
    0x0007eb44e0474cf6,0x0007ebb754e47419,0x0007ec242a3d8474,0x0007ec8bc5d69645,
    0x0007ecee83d3d6e9,0x0007ed4cb8082f45,0x0007eda6aee0170f,0x0007edfcae2dfe68,
    0x0007ee4ef5dccd3e,0x0007ee9dc08c394e,0x0007eee9441a17c7,0x0007ef31b21b4fb1,
    0x0007ef773846a8a7,0x0007efba00d35a17,0x0007effa32ccf69f,0x0007f037f25e1278,
    0x0007f0736112d12c,0x0007f0ac9e145c25,0x0007f0e3c65e1fcc,0x0007f118f4ed8e54,
    0x0007f14c42ed0dc8,0x0007f17dc7daa0c3,0x0007f1ad99aac6a5,0x0007f1dbcce80015,
    0x0007f20874cf56bf,0x0007f233a36a3b9a,0x0007f25d69a604ad,0x0007f285d7694a92,
    0x0007f2acfba75e3b,0x0007f2d2e4720909,0x0007f2f79f09c344,0x0007f31b37ec883b,
    0x0007f33dbae36abc,0x0007f35f330f08d5,0x0007f37faaf2fa79,0x0007f39f2c805380,
    0x0007f3bdc11f4f1c,0x0007f3db71b83850,0x0007f3f846bba121,0x0007f4144829f846,
    0x0007f42f7d9a8b9d,0x0007f449ee420432,0x0007f463a0f8675e,0x0007f47c9c3ea77b,
    0x0007f494e643cd8e,0x0007f4ac84e9c475,0x0007f4c37dc9cd50,0x0007f4d9d638a432,
    0x0007f4ef934a5b6a,0x0007f504b9d5f33d,0x0007f5194e78b352,0x0007f52d55994a96,
    0x0007f540d36aba0c,0x0007f553cbef0e77,0x0007f56642f9ec8f,0x0007f5783c32f31e,
    0x0007f589bb17f609,0x0007f59ac2ff1525,0x0007f5ab5718b15a,0x0007f5bb7a71427c,
    0x0007f5cb2ff31009,0x0007f5da7a67cebe,0x0007f5e95c7a24e7,0x0007f5f7d8b7171e,
    0x0007f605f18f5ef4,0x0007f613a958ad0a,0x0007f621024ed7e9,0x0007f62dfe94f8cb,
    0x0007f63aa036777a,0x0007f646e928065a,0x0007f652db488f88,0x0007f65e786213ff,
    0x0007f669c22a7d8a,0x0007f674ba446459,0x0007f67f623fc8db,0x0007f689bb9ac294,
    0x0007f693c7c22481,0x0007f69d881217a6,0x0007f6a6fdd6ac36,0x0007f6b02a4c61ee,
    0x0007f6b90ea0a7f4,0x0007f6c1abf254c0,0x0007f6ca03521664,0x0007f6d215c2db82,
    0x0007f6d9e43a3559,0x0007f6e16fa0b329,0x0007f6e8b8d23729,0x0007f6efc09e4569,
    0x0007f6f687c84cbf,0x0007f6fd0f07ea09,0x0007f703570925e2,0x0007f709606cad03,
    0x0007f70f2bc8036f,0x0007f714b9a5b292,0x0007f71a0a85725d,0x0007f71f1edc4d9e,
    0x0007f723f714c179,0x0007f728938ed843,0x0007f72cf4a03fa0,0x0007f7311a945a16,
    0x0007f73505ac4bf8,0x0007f738b61f03bd,0x0007f73c2c193dc0,0x0007f73f67bd835c,
    0x0007f74269242559,0x0007f745305b31a1,0x0007f747bd666428,0x0007f74a103f12ed,
    0x0007f74c28d414f5,0x0007f74e0709a42d,0x0007f74faab939f9,0x0007f75113b16657,
    0x0007f75241b5a155,0x0007f753347e16b8,0x0007f753ebb76b7c,0x0007f75467027d05,
    0x0007f754a5f4199d,0x0007f754a814b207,0x0007f7546ce003ae,0x0007f753f3c4bb29,
    0x0007f7533c240e92,0x0007f75245514f41,0x0007f7510e91726c,0x0007f74f971a9012,
    0x0007f74dde135797,0x0007f74be2927971,0x0007f749a39e051c,0x0007f747202aba8a,
    0x0007f744571b4e3c,0x0007f741473f9efe,0x0007f73def53dc43,0x0007f73a4dff9bff,
    0x0007f73661d4deaf,0x0007f732294f003f,0x0007f72da2d19444,0x0007f728cca72bda,
    0x0007f723a5000367,0x0007f71e29f09627,0x0007f7185970156b,0x0007f7123156c102,
    0x0007f70baf5c1e2c,0x0007f704d1150a23,0x0007f6fd93f1a4e5,0x0007f6f5f53b10b6,
    0x0007f6edf211023e,0x0007f6e587671ce9,0x0007f6dcb2021679,0x0007f6d36e749c64,
    0x0007f6c9b91bf4c6,0x0007f6bf8e1c541b,0x0007f6b4e95ce015,0x0007f6a9c68356ff,
    0x0007f69e20ef5211,0x0007f691f3b517eb,0x0007f6853997f321,0x0007f677ed03ff19,
    0x0007f66a08075bdc,0x0007f65b844ab75a,0x0007f64c5b091860,0x0007f63c8506d4bc,
    0x0007f62bfa8798fe,0x0007f61ab34364b0,0x0007f608a65a599a,0x0007f5f5ca4737e8,
    0x0007f5e214d05b48,0x0007f5cd7af7066e,0x0007f5b7f0e4c2a1,0x0007f5a169d68fcf,
    0x0007f589d80596a5,0x0007f5712c8d0174,0x0007f557574c912b,0x0007f53c46c77193,
    0x0007f51fe7feb9f2,0x0007f5022646ecfb,0x0007f4e2eb17ab1d,0x0007f4c21dd4a3d1,
    0x0007f49fa38ea394,0x0007f47b5ebb62eb,0x0007f4552ee27473,0x0007f42cf03d58f5,
    0x0007f4027b48549f,0x0007f3d5a44119df,0x0007f3a63a8fb552,0x0007f37408155100,
    0x0007f33ed05b55ec,0x0007f3064f9c183e,0x0007f2ca399c7ba1,0x0007f28a384bb940,
    0x0007f245ea1b7a2b,0x0007f1fcdffe8f1b,0x0007f1ae9af758cd,0x0007f15a8917f27e,
    0x0007f10001ccaaab,0x0007f09e413c418a,0x0007f034627733d7,0x0007efc15815b8d5,
    0x0007ef43e2bf7f55,0x0007eeba84e31dfe,0x0007ee237294df89,0x0007ed7c7c170141,
    0x0007ecc2f0d95d3a,0x0007ebf377a46782,0x0007eb09d6deb285,0x0007ea00a4f17808,
    0x0007e8d0d3da63d6,0x0007e771023b0fcf,0x0007e5d46c2f08d8,0x0007e3e937669691,
    0x0007e195978f1176,0x0007deb2c0e05c1c,0x0007db0362002a19,0x0007d6202c151439,
    0x0007cf4b8f00a2cb,0x0007c4fd24520efd,0x0007b362fbf81816,0x00078d2d25998e24
};

static const double wi[] =
{   1.7367254121602630e-15,9.5586603514556339e-17,1.2708704834810623e-16,
    1.4909740962495474e-16,1.6658733631586268e-16,1.8136120810119029e-16,
    1.9429720153135588e-16,2.0589500628482093e-16,2.1646860576895422e-16,
    2.2622940392218116e-16,2.3532718914045892e-16,2.4387234557428771e-16,
    2.5194879829274225e-16,2.5962199772528103e-16,2.6694407473648285e-16,
    2.7395729685142446e-16,2.8069646002484804e-16,2.8719058904113930e-16,
    2.9346417484728883e-16,2.9953809336782113e-16,3.0543030007192440e-16,
    3.1115636338921572e-16,3.1672988018581815e-16,3.2216280350549905e-16,
    3.2746570407939751e-16,3.3264798116841710e-16,3.3771803417353232e-16,
    3.4268340353119356e-16,3.4755088731729758e-16,3.5232663846002031e-16,
    3.5701624633953494e-16,3.6162480571598339e-16,3.6615697529653540e-16,
    3.7061702777236077e-16,3.7500889278747798e-16,3.7933619401549554e-16,
    3.8360228129677279e-16,3.8781025861250247e-16,3.9196300853257678e-16,
    3.9606321366256378e-16,4.0011337552546690e-16,4.0411583124143332e-16,
    4.0807276830960448e-16,4.1198623774807442e-16,4.1585816580828064e-16,
    4.1969036444740733e-16,4.2348454071520708e-16,4.2724230518899761e-16,
    4.3096517957162941e-16,4.3465460355128760e-16,4.3831194100854571e-16,
    4.4193848564470665e-16,4.4553546609579137e-16,4.4910405058828750e-16,
    4.5264535118571397e-16,4.5616042766900381e-16,4.5965029108849407e-16,
    4.6311590702081647e-16,4.6655819856008752e-16,4.6997804906941950e-16,
    4.7337630471583237e-16,4.7675377680908526e-16,4.8011124396270155e-16,
    4.8344945409350080e-16,4.8676912627422087e-16,4.9007095245229938e-16,
    4.9335559904654139e-16,4.9662370843221783e-16,4.9987590032409088e-16,
    5.0311277306593187e-16,5.0633490483427195e-16,5.0954285476338923e-16,
    5.1273716399787966e-16,5.1591835667857364e-16,5.1908694086703434e-16,
    5.2224340941340417e-16,5.2538824077194543e-16,5.2852189976823820e-16,
    5.3164483832166176e-16,5.3475749612647295e-16,5.3786030129452348e-16,
    5.4095367096239933e-16,5.4403801186554671e-16,5.4711372088173611e-16,
    5.5018118554603362e-16,5.5324078453927836e-16,5.5629288815190902e-16,
    5.5933785872484621e-16,5.6237605106900435e-16,5.6540781286489604e-16,
    5.6843348504368141e-16,5.7145340215092040e-16,5.7446789269419609e-16,
    5.7747727947569648e-16,5.8048187991076857e-16,5.8348200633338921e-16,
    5.8647796628943653e-16,5.8947006281858718e-16,5.9245859472561339e-16,
    5.9544385684180598e-16,5.9842614027720281e-16,6.0140573266426640e-16,
    6.0438291839361250e-16,6.0735797884236057e-16,6.1033119259564394e-16,
    6.1330283566179110e-16,6.1627318168165963e-16,6.1924250213258470e-16,
    6.2221106652737879e-16,6.2517914260879998e-16,6.2814699653988953e-16,
    6.3111489309056042e-16,6.3408309582080600e-16,6.3705186726088149e-16,
    6.4002146908880247e-16,6.4299216230548961e-16,6.4596420740788321e-16,
    6.4893786456033965e-16,6.5191339376461587e-16,6.5489105502874154e-16,
    6.5787110853507413e-16,6.6085381480782587e-16,6.6383943488035057e-16,
    6.6682823046247459e-16,6.6982046410815579e-16,6.7281639938375311e-16,
    6.7581630103719006e-16,6.7882043516829803e-16,6.8182906940062540e-16,
    6.8484247305500383e-16,6.8786091732516637e-16,6.9088467545571690e-16,
    6.9391402292275690e-16,6.9694923761748294e-16,6.9999060003307640e-16,
    7.0303839345521508e-16,7.0609290415654822e-16,7.0915442159548734e-16,
    7.1222323861967788e-16,7.1529965167453030e-16,7.1838396101720629e-16,
    7.2147647093647067e-16,7.2457748997883870e-16,7.2768733118146927e-16,
    7.3080631231227429e-16,7.3393475611774048e-16,7.3707299057898310e-16,
    7.4022134917657997e-16,7.4338017116476479e-16,7.4654980185558890e-16,
    7.4973059291369793e-16,7.5292290266240584e-16,7.5612709640179217e-16,
    7.5934354673958895e-16,7.6257263393567558e-16,7.6581474626104873e-16,
    7.6907028037219191e-16,7.7233964170182985e-16,7.7562324486711744e-16,
    7.7892151409638524e-16,7.8223488367564108e-16,7.8556379841610841e-16,
    7.8890871414417552e-16,7.9227009821522709e-16,7.9564843005293662e-16,
    7.9904420171571300e-16,8.0245791849212591e-16,8.0589009952726568e-16,
    8.0934127848215009e-16,8.1281200422845008e-16,8.1630284158098775e-16,
    8.1981437207065329e-16,8.2334719476060504e-16,8.2690192710884700e-16,
    8.3047920588053737e-16,8.3407968811366288e-16,8.3770405214202216e-16,
    8.4135299867980282e-16,8.4502725197240968e-16,8.4872756101861549e-16,
    8.5245470086955962e-16,8.5620947401062333e-16,8.5999271183276646e-16,
    8.6380527620052589e-16,8.6764806112455816e-16,8.7152199454736980e-16,
    8.7542804025171749e-16,8.7936719990210427e-16,8.8334051523084080e-16,
    8.8734907038131345e-16,8.9139399442240861e-16,8.9547646404950677e-16,
    8.9959770648910994e-16,9.0375900262601175e-16,9.0796169037400680e-16,
    9.1220716831348461e-16,9.1649689962191353e-16,9.2083241632623076e-16,
    9.2521532390956933e-16,9.2964730630864167e-16,9.3413013134252651e-16,
    9.3866565661866598e-16,9.4325583596767065e-16,9.4790272646517382e-16,
    9.5260849610662787e-16,9.5737543220974496e-16,9.6220595062948384e-16,
    9.6710260588230542e-16,9.7206810229016259e-16,9.7710530627072088e-16,
    9.8221725991905411e-16,9.8740719604806711e-16,9.9267855488079765e-16,
    9.9803500261836449e-16,1.0034804521436181e-15,1.0090190861637457e-15,
    1.0146553831467086e-15,1.0203941464683124e-15,1.0262405372613567e-15,
    1.0322001115486456e-15,1.0382788623515399e-15,1.0444832676000471e-15,
    1.0508203448355195e-15,1.0572977139009890e-15,1.0639236690676801e-15,
    1.0707072623632994e-15,1.0776584002668106e-15,1.0847879564403425e-15,
    1.0921079038149563e-15,1.0996314701785628e-15,1.1073733224935752e-15,
    1.1153497865853155e-15,1.1235791107110833e-15,1.1320817840164846e-15,
    1.1408809242582780e-15,1.1500027537839792e-15,1.1594771891449189e-15,
    1.1693385786910960e-15,1.1796266352955801e-15,1.1903876299282890e-15,
    1.2016759392543819e-15,1.2135560818666897e-15,1.2261054417450561e-15,
    1.2394179789163251e-15,1.2536093926602567e-15,1.2688244814255010e-15,
    1.2852479319096109e-15,1.3031206634689985e-15,1.3227655770195326e-15,
    1.3446300925011171e-15,1.3693606835128518e-15,1.3979436672775240e-15,
    1.4319989869661328e-15,1.4744848603597596e-15,1.5317872741611144e-15,
    1.6227698675312968e-15
};

static const double fi[] =
{   1.0000000000000000e+00,9.7710170126767082e-01,9.5987909180010600e-01,
    9.4519895344229909e-01,9.3206007595922991e-01,9.1999150503934646e-01,
    9.0872644005213032e-01,8.9809592189834297e-01,8.8798466075583282e-01,
    8.7830965580891684e-01,8.6900868803685649e-01,8.6003362119633109e-01,
    8.5134625845867751e-01,8.4291565311220373e-01,8.3471629298688299e-01,
    8.2672683394622093e-01,8.1892919160370192e-01,8.1130787431265572e-01,
    8.0384948317096383e-01,7.9654233042295841e-01,7.8937614356602404e-01,
    7.8234183265480195e-01,7.7543130498118662e-01,7.6863731579848571e-01,
    7.6195334683679483e-01,7.5537350650709567e-01,7.4889244721915638e-01,
    7.4250529634015061e-01,7.3620759812686210e-01,7.2999526456147568e-01,
    7.2386453346862967e-01,7.1781193263072152e-01,7.1183424887824798e-01,
    7.0592850133275376e-01,7.0009191813651117e-01,6.9432191612611627e-01,
    6.8861608300467136e-01,6.8297216164499430e-01,6.7738803621877308e-01,
    6.7186171989708166e-01,6.6639134390874977e-01,6.6097514777666277e-01,
    6.5561147057969693e-01,6.5029874311081637e-01,6.4503548082082196e-01,
    6.3982027745305614e-01,6.3465179928762327e-01,6.2952877992483625e-01,
    6.2445001554702606e-01,6.1941436060583399e-01,6.1442072388891344e-01,
    6.0946806492577310e-01,6.0455539069746733e-01,5.9968175261912482e-01,
    5.9484624376798689e-01,5.9004799633282545e-01,5.8528617926337090e-01,
    5.8055999610079034e-01,5.7586868297235316e-01,5.7121150673525267e-01,
    5.6658776325616389e-01,5.6199677581452390e-01,5.5743789361876550e-01,
    5.5291049042583185e-01,5.4841396325526537e-01,5.4394773119002582e-01,
    5.3951123425695158e-01,5.3510393238045717e-01,5.3072530440366150e-01,
    5.2637484717168403e-01,5.2205207467232140e-01,5.1775651722975591e-01,
    5.1348772074732651e-01,5.0924524599574761e-01,5.0502866794346790e-01,
    5.0083757512614835e-01,4.9667156905248933e-01,4.9253026364386815e-01,
    4.8841328470545758e-01,4.8432026942668288e-01,4.8025086590904642e-01,
    4.7620473271950547e-01,4.7218153846772976e-01,4.6818096140569321e-01,
    4.6420268904817391e-01,4.6024641781284248e-01,4.5631185267871610e-01,
    4.5239870686184824e-01,4.4850670150720273e-01,4.4463556539573912e-01,
    4.4078503466580377e-01,4.3695485254798533e-01,4.3314476911265209e-01,
    4.2935454102944126e-01,4.2558393133802180e-01,4.2183270922949573e-01,
    4.1810064983784795e-01,4.1438753404089090e-01,4.1069314827018799e-01,
    4.0701728432947315e-01,4.0335973922111429e-01,3.9972031498019700e-01,
    3.9609881851583223e-01,3.9249506145931540e-01,3.8890886001878855e-01,
    3.8534003484007706e-01,3.8178841087339344e-01,3.7825381724561896e-01,
    3.7473608713789086e-01,3.7123505766823922e-01,3.6775056977903225e-01,
    3.6428246812900372e-01,3.6083060098964775e-01,3.5739482014578022e-01,
    3.5397498080007656e-01,3.5057094148140588e-01,3.4718256395679348e-01,
    3.4380971314685055e-01,3.4045225704452164e-01,3.3711006663700588e-01,
    3.3378301583071823e-01,3.3047098137916342e-01,3.2717384281360129e-01,
    3.2389148237639104e-01,3.2062378495690530e-01,3.1737063802991350e-01,
    3.1413193159633707e-01,3.1090755812628634e-01,3.0769741250429189e-01,
    3.0450139197664983e-01,3.0131939610080288e-01,2.9815132669668531e-01,
    2.9499708779996164e-01,2.9185658561709499e-01,2.8872972848218270e-01,
    2.8561642681550159e-01,2.8251659308370741e-01,2.7943014176163772e-01,
    2.7635698929566810e-01,2.7329705406857691e-01,2.7025025636587519e-01,
    2.6721651834356114e-01,2.6419576399726080e-01,2.6118791913272082e-01,
    2.5819291133761890e-01,2.5521066995466168e-01,2.5224112605594190e-01,
    2.4928421241852824e-01,2.4633986350126363e-01,2.4340801542275012e-01,
    2.4048860594050039e-01,2.3758157443123795e-01,2.3468686187232990e-01,
    2.3180441082433859e-01,2.2893416541468023e-01,2.2607607132238020e-01,
    2.2323007576391746e-01,2.2039612748015194e-01,2.1757417672433113e-01,
    2.1476417525117358e-01,2.1196607630703015e-01,2.0917983462112499e-01,
    2.0640540639788071e-01,2.0364274931033485e-01,2.0089182249465656e-01,
    1.9815258654577511e-01,1.9542500351413428e-01,1.9270903690358912e-01,
    1.9000465167046496e-01,1.8731181422380025e-01,1.8463049242679927e-01,
    1.8196065559952254e-01,1.7930227452284767e-01,1.7665532144373500e-01,
    1.7401977008183875e-01,1.7139559563750595e-01,1.6878277480121151e-01,
    1.6618128576448205e-01,1.6359110823236570e-01,1.6101222343751107e-01,
    1.5844461415592431e-01,1.5588826472447920e-01,1.5334316106026283e-01,
    1.5080929068184568e-01,1.4828664273257453e-01,1.4577520800599403e-01,
    1.4327497897351341e-01,1.4078594981444470e-01,1.3830811644855071e-01,
    1.3584147657125373e-01,1.3338602969166913e-01,1.3094177717364430e-01,
    1.2850872227999952e-01,1.2608687022018586e-01,1.2367622820159654e-01,
    1.2127680548479021e-01,1.1888861344290998e-01,1.1651166562561080e-01,
    1.1414597782783835e-01,1.1179156816383801e-01,1.0944845714681163e-01,
    1.0711666777468364e-01,1.0479622562248690e-01,1.0248715894193508e-01,
    1.0018949876880981e-01,9.7903279038862284e-02,9.5628536713008819e-02,
    9.3365311912690860e-02,9.1113648066373634e-02,8.8873592068275789e-02,
    8.6645194450557961e-02,8.4428509570353374e-02,8.2223595813202863e-02,
    8.0030515814663056e-02,7.7849336702096039e-02,7.5680130358927067e-02,
    7.3522973713981268e-02,7.1377949058890375e-02,6.9245144397006769e-02,
    6.7124653827788497e-02,6.5016577971242842e-02,6.2921024437758113e-02,
    6.0838108349539864e-02,5.8767952920933758e-02,5.6710690106202902e-02,
    5.4666461324888914e-02,5.2635418276792176e-02,5.0617723860947761e-02,
    4.8613553215868521e-02,4.6623094901930368e-02,4.4646552251294443e-02,
    4.2684144916474431e-02,4.0736110655940933e-02,3.8802707404526113e-02,
    3.6884215688567284e-02,3.4980941461716084e-02,3.3093219458578522e-02,
    3.1221417191920245e-02,2.9365939758133314e-02,2.7527235669603082e-02,
    2.5705804008548896e-02,2.3902203305795882e-02,2.2117062707308864e-02,
    2.0351096230044517e-02,1.8605121275724643e-02,1.6880083152543166e-02,
    1.5177088307935325e-02,1.3497450601739880e-02,1.1842757857907888e-02,
    1.0214971439701471e-02,8.6165827693987316e-03,7.0508754713732268e-03,
    5.5224032992509968e-03,4.0379725933630305e-03,2.6090727461021627e-03,
    1.2602859304985975e-03
};

static const double ziggurat_nor_r      = 3.6541528853610087963519472518;
static const double ziggurat_nor_inv_r  = 0.27366123732975827203338247596; // 1.0 / ziggurat_nor_r;
static const double ziggurat_exp_r      = 7.6971174701310497140446280481;


double random_gauss_zig_julia(aug_state *state) {
    uint64_t r;
    int64_t rabs;
    int idx;
    double x, xx, yy;
    for (;;)
    {
        r = random_uint64(state) & 0x000fffffffffffff;
        rabs = (int64_t)(r >> 1);
        idx = rabs & 0xff;
        x = rabs * wi[idx];
        if (r & 0x1)
            x = -x;
        if (rabs < ki[idx])
            return x;  // # 99.3% of the time return here
        if (idx == 0)
        {
            for (;;)
            {
                xx = -ziggurat_nor_inv_r*log(random_double(state));
                yy = -log(random_double(state));
                if (yy+yy > xx*xx)
                    return ((rabs >> 8) & 0x1) ? -(ziggurat_nor_r+xx) : ziggurat_nor_r+xx;
            }
        }
        else
        {
            if (((fi[idx-1] - fi[idx])*random_double(state) + fi[idx]) < exp(-0.5*x*x))
                return x;
        }
    }
}


void random_gauss_zig_julia_fill(aug_state *state, npy_intp count, double *out) {
    uint64_t r;
    int64_t rabs;
    int idx, i;
    double x, xx, yy;
    for (i = 0; i < count; i++) {
        for (;;) {
            r = random_uint64(state) & 0x000fffffffffffff;
            rabs = (int64_t)(r >> 1);
            idx = rabs & 0xff;
            x = rabs * wi[idx];
            if (r & 0x1)
                x = -x;
            if (rabs < ki[idx]) {
                out[i] = x;  // # 99.3% of the time return here
                break;
            }
            if (idx == 0)
            {
                do {
                    xx = -ziggurat_nor_inv_r*log(random_double(state));
                    yy = -log(random_double(state));
                } while (yy+yy <= xx*xx);

                out[i] = ((rabs >> 8) & 0x1) ? -(ziggurat_nor_r+xx) : ziggurat_nor_r+xx;
                break;
            }
            else
            {
                if (((fi[idx-1] - fi[idx])*random_double(state) + fi[idx]) < exp(-0.5*x*x)) {
                    out[i] = x;
                    break;
                }
            }
        }
    }
}



unsigned long random_interval(aug_state *state, unsigned long max)
{
    unsigned long mask, value;
    if (max == 0) {
        return 0;
    }

    mask = max;

    /* Smallest bit mask >= max */
    mask |= mask >> 1;
    mask |= mask >> 2;
    mask |= mask >> 4;
    mask |= mask >> 8;
    mask |= mask >> 16;
#if ULONG_MAX > 0xffffffffUL
    mask |= mask >> 32;
#endif

    /* Search a random value in [0..mask] <= max */
#if ULONG_MAX > 0xffffffffUL
    if (max <= 0xffffffffUL) {
        while ((value = (random_uint32(state) & mask)) > max);
    }
    else {
        while ((value = (random_uint64(state) & mask)) > max);
    }
#else
    while ((value = (random_uint32(state) & mask)) > max);
#endif
    return value;
}



static inline uint64_t gen_mask(uint64_t max)
{
    uint64_t mask = max;
    mask |= mask >> 1;
    mask |= mask >> 2;
    mask |= mask >> 4;
    mask |= mask >> 8;
    mask |= mask >> 16;
    mask |= mask >> 32;
    return mask;
}
/*
 * Fills an array with cnt random npy_uint64 between off and off + rng
 * inclusive. The numbers wrap if rng is sufficiently large.
 */
void random_bounded_uint64_fill(aug_state *state, uint64_t off, uint64_t rng, npy_intp cnt, uint64_t *out)
{
    uint64_t val, mask;
    int i;

    if (rng == 0) {
        for (i = 0; i < cnt; i++) {
            out[i] = off;
        }
        return;
    }

    /* Smallest bit mask >= max */
    mask = gen_mask(rng);

    if (rng <= 0xffffffffUL) {
        for (i = 0; i < cnt; i++) {
            while ((val = (random_uint32(state) & mask)) > rng);
            out[i] =  off + val;
        }
    }
    else {
        for (i = 0; i < cnt; i++) {
            while ((val = (random_uint64(state) & mask)) > rng);
            out[i] =  off + val;
        }
    }
}


/*
 * Fills an array with cnt random npy_uint32 between off and off + rng
 * inclusive. The numbers wrap if rng is sufficiently large.
 */
void random_bounded_uint32_fill(aug_state *state, uint32_t off, uint32_t rng, npy_intp cnt, uint32_t *out)
{
    uint32_t val, mask = rng;
    int i;

    if (rng == 0) {
        for (i = 0; i < cnt; i++) {
            out[i] = off;
        }
        return;
    }

    /* Smallest bit mask >= max */
    mask = (uint32_t)gen_mask(rng);

    for (i = 0; i < cnt; i++) {
        while ((val = (random_uint32(state) & mask)) > rng);
        out[i] =  off + val;
    }
}


/*
 * Fills an array with cnt random npy_uint16 between off and off + rng
 * inclusive. The numbers wrap if rng is sufficiently large.
 */
void random_bounded_uint16_fill(aug_state *state, uint16_t off, uint16_t rng, npy_intp cnt, uint16_t *out)
{
    uint16_t val, mask;
    int i;
    uint32_t buf = 0;
    int bcnt = 0;

    if (rng == 0) {
        for (i = 0; i < cnt; i++) {
            out[i] = off;
        }
        return;
    }

    /* Smallest bit mask >= max */
    mask = (uint16_t)gen_mask(rng);

    for (i = 0; i < cnt; i++) {
        do {
            if (!bcnt) {
                buf = random_uint32(state);
                bcnt = 1;
            }
            else {
                buf >>= 16;
                bcnt--;
            }
            val = (uint16_t)buf & mask;
        } while (val > rng);
        out[i] =  off + val;
    }
}

/*
 * Fills an array with cnt random npy_uint8 between off and off + rng
 * inclusive. The numbers wrap if rng is sufficiently large.
 */
void random_bounded_uint8_fill(aug_state *state, uint8_t off, uint8_t rng, npy_intp cnt, uint8_t *out)
{
    uint8_t val, mask = rng;
    int i;
    uint32_t buf = 0;
    int bcnt = 0;

    if (rng == 0) {
        for (i = 0; i < cnt; i++) {
            out[i] = off;
        }
        return;
    }

    /* Smallest bit mask >= max */
    mask = (uint8_t)gen_mask(rng);

    for (i = 0; i < cnt; i++) {
        do {
            if (!bcnt) {
                buf = random_uint32(state);
                bcnt = 3;
            }
            else {
                buf >>= 8;
                bcnt--;
            }
            val = (uint8_t)buf & mask;
        } while (val > rng);
        out[i] =  off + val;
    }
}


/*
 * Fills an array with cnt random npy_bool between off and off + rng
 * inclusive.
 */
/* TODO: This needs to use bools. See original */
void random_bool_fill(aug_state *state, int8_t off, int8_t rng, npy_intp cnt, int8_t *out)
{
    int i;
    uint32_t buf = 0;
    int bcnt = 0;

    if (rng == 0) {
        for (i = 0; i < cnt; i++) {
            out[i] = off;
        }
        return;
    }

    for (i = 0; i < cnt; i++) {
        if (!bcnt) {
            buf = random_uint32(state);
            bcnt = 31;
        }
        else {
            buf >>= 1;
            bcnt--;
        }
        out[i] = (buf & 0x00000001) != 0;
    }
}
