extern inline double random_double(aug_state* state);

extern inline double random_standard_exponential(aug_state* state);

extern inline double random_gauss(aug_state* state);

extern inline double random_standard_gamma(aug_state* state, double shape);

extern inline double random_normal(aug_state *state, double loc, double scale);

extern inline double random_exponential(aug_state *state, double scale);

extern inline double random_uniform(aug_state *state, double loc, double scale);

extern inline double random_gamma(aug_state *state, double shape, double scale);

extern inline double random_beta(aug_state *state, double a, double b);

extern inline double random_chisquare(aug_state *state, double df);

extern inline double random_f(aug_state *state, double dfnum, double dfden);

extern inline long random_negative_binomial(aug_state *state, double n, double p);

extern inline double random_standard_cauchy(aug_state *state);

extern inline double random_standard_t(aug_state *state, double df);

extern inline double random_pareto(aug_state *state, double a);

extern inline double random_weibull(aug_state *state, double a);

extern inline double random_power(aug_state *state, double a);

extern inline double random_laplace(aug_state *state, double loc, double scale);

extern inline double random_gumbel(aug_state *state, double loc, double scale);

extern inline double random_logistic(aug_state *state, double loc, double scale);

extern inline double random_lognormal(aug_state *state, double mean, double sigma);

extern inline double random_rayleigh(aug_state *state, double mode);

long random_poisson(aug_state *state, double lam);

long rk_negative_binomial(aug_state *state, double n, double p);

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
