#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef min
#define min(x,y) ((x<y)?x:y)
#define max(x,y) ((x>y)?x:y)
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846264338328
#endif

#if  defined(PCG_32_RNG)
#include "shims/pcg-32/pcg-32-shim.h"
#elif defined(PCG_64_RNG)
#include "shims/pcg-64/pcg-64-shim.h"
#elif defined(RANDOMKIT_RNG)
#include "shims/random-kit/random-kit-shim.h"
#elif defined(DUMMY_RNG)
#include "shims/dummy/dummy-shim.h"
#elif defined(XORSHIFT128_RNG)
#include "shims/xorshift128/xorshift128-shim.h"
#elif defined(XORSHIFT1024_RNG)
#include "shims/xorshift1024/xorshift1024-shim.h"
#elif defined(MRG32K3A_RNG)
#include "shims/mrg32k3a/mrg32k3a-shim.h"
#elif defined(MLFG_1279_861_RNG)
#include "shims/mlfg-1279-861/mlfg-1279-861-shim.h"
#else
#error Unknown RNG!!!  Unknown RNG!!!  Unknown RNG!!!
#endif

extern int64_t random_positive_int64(aug_state* state);

extern int32_t random_positive_int32(aug_state* state);

extern double random_standard_uniform(aug_state* state);

extern double random_standard_exponential(aug_state* state);

extern double random_gauss(aug_state* state);

extern double random_standard_gamma(aug_state* state, double shape);

extern double random_normal(aug_state *state, double loc, double scale);

extern double random_normal_zig(aug_state *state, double loc, double scale);

extern double random_exponential(aug_state *state, double scale);

extern double random_uniform(aug_state *state, double loc, double scale);

extern double random_gamma(aug_state *state, double shape, double scale);

extern double random_beta(aug_state *state, double a, double b);

extern double random_chisquare(aug_state *state, double df);

extern double random_f(aug_state *state, double dfnum, double dfden);

extern long random_negative_binomial(aug_state *state, double n, double p);

extern double random_standard_cauchy(aug_state *state);

extern double random_standard_t(aug_state *state, double df);

extern double random_pareto(aug_state *state, double a);

extern double random_weibull(aug_state *state, double a);

extern double random_power(aug_state *state, double a);

extern double random_laplace(aug_state *state, double loc, double scale);

extern double random_gumbel(aug_state *state, double loc, double scale);

extern double random_logistic(aug_state *state, double loc, double scale);

extern double random_lognormal(aug_state *state, double mean, double sigma);

extern double random_rayleigh(aug_state *state, double mode);

extern double random_gauss_zig(aug_state* state);

extern double random_gauss_zig_julia(aug_state* state);

extern double random_noncentral_chisquare(aug_state *state, double df, double nonc);

extern double random_noncentral_f(aug_state *state, double dfnum, double dfden, double nonc);

extern double random_wald(aug_state *state, double mean, double scale);

extern double random_vonmises(aug_state *state, double mu, double kappa);

extern double random_triangular(aug_state *state, double left, double mode, double right);

extern long random_poisson(aug_state *state, double lam);

extern long rk_negative_binomial(aug_state *state, double n, double p);

extern long random_binomial(aug_state *state, double p, long n);

extern long random_logseries(aug_state *state, double p);

extern long random_geometric(aug_state *state, double p);

extern long random_zipf(aug_state *state, double a);

extern long random_hypergeometric(aug_state *state, long good, long bad, long sample);

extern long random_positive_int(aug_state* state);

extern unsigned long random_uint(aug_state* state);

extern unsigned long random_interval(aug_state* state, unsigned long max);