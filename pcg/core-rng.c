#ifdef PCG_32_RNG
    #include "pcg-32-shim.h"
#else
    #ifdef PCG_64_RNG
        #include "pcg-64-shim.h"
    #else
        #ifdef DUMMY_RNG
            #include "dummy-shim.h"
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
