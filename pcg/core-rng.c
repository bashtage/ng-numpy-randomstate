extern inline double random_double(aug_state* state)

extern inline double random_standard_exponential(aug_state* state)

extern inline double random_gauss(aug_state* state){

extern inline double random_standard_gamma(aug_state* state, double shape)

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