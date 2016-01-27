#include "../splitmix64/splitmix64.h"
#include "xorshift1024.h"

#include <string.h>

extern inline uint64_t xorshift1024_next(xorshift1024_state* state);

void xorshift1024_jump(xorshift1024_state* state) {
    static const uint64_t JUMP[] = { 0x84242f96eca9c41dULL,
                                     0xa3c65b8776f96855ULL, 0x5b34a39f070b5837ULL, 0x4489affce4f31a1eULL,
                                     0x2ffeeb0a48316f40ULL, 0xdc2d9891fe68c022ULL, 0x3659132bb12fea70ULL,
                                     0xaac17d8efa43cab8ULL, 0xc4cb815590989b13ULL, 0x5ee975283d71c93bULL,
                                     0x691548c86c1bd540ULL, 0x7910c41d10a1e6a5ULL, 0x0b5fc64563b3e2a8ULL,
                                     0x047f7684e9fc949dULL, 0xb99181f2d8f685caULL, 0x284600e3f30e38c3ULL
                                   };
    size_t i, j;
    uint64_t b;
    uint64_t t[16] = { 0 };
    for(i = 0; i < sizeof JUMP / sizeof *JUMP; i++)
        for(b = 0; b < 64; b++) {
            if (JUMP[i] & 1ULL << b)
                for(j = 0; j < 16; j++)
                    t[j] ^= state->s[(j + state->p) & 15];
            xorshift1024_next(state);
        }

    memcpy(state->s, t, sizeof t);
}

void xorshift1024_seed(xorshift1024_state* state, uint64_t seed)
{
    uint64_t initial_state[16] = {0};
    uint64_t seed_copy = seed;
    int i;
    for (i = 0; i < 16; i++)
    {
        initial_state[i] = splitmix64_next(&seed_copy);
    }
    xorshift1024_init_state(state, initial_state);
}

void xorshift1024_init_state(xorshift1024_state* state, uint64_t* seeds)
{
    memcpy(&(state->s), seeds, sizeof(state->s));
    state->p = 0;
}


void xorshift1024_seed_by_array(xorshift1024_state* state, uint64_t *seed_array, int count)
{
    uint64_t initial_state[16] = {0};
    uint64_t seed_copy = 0;
    int iter_bound = 16>=count ? 16 : count;
    int i, loc = 0;
    for (i = 0; i < iter_bound; i++)
    {
        if (i < count)
            seed_copy ^= seed_array[i];
        initial_state[loc] = splitmix64_next(&seed_copy);
        loc ++;
        if (loc == 16)
            loc = 0;
    }
    xorshift1024_init_state(state, initial_state);
}