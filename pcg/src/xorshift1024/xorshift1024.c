#include "xorshift1024.h"
#include <stdint.h>
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

	uint64_t t[16] = { 0 };
	for(int i = 0; i < sizeof JUMP / sizeof *JUMP; i++)
		for(int b = 0; b < 64; b++) {
			if (JUMP[i] & 1ULL << b)
				for(int j = 0; j < 16; j++)
					t[j] ^= state->s[(j + state->p) & 15];
			xorshift1024_next(state);
		}

	memcpy(state->s, t, sizeof t);
}

void xorshift1024_seed(xorshift1024_state* state, uint64_t* seed)
{
    memcpy(&(state->s), seed, sizeof(state->s));
}