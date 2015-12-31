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

double random_sample(aug_state* state)
{
    return random_double(state);
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
        state->binomial->m = bound = min(n, np + 10.0*sqrt(np*q + 1));
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

long random_binomial(aug_state *state, long n, double p)
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
    if (nonc == 0){
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
