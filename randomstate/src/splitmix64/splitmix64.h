#ifdef _WIN32
#include "../common/inttypes.h"
#define inline __forceinline
#else
#include <inttypes.h>
#endif

inline uint64_t splitmix64_next(uint64_t* x) {
    uint64_t z = (*x += UINT64_C(0x9E3779B97F4A7C15));
    z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
    return z ^ (z >> 31);
}