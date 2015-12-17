#include "xorshift128.h"

extern inline uint64_t xorshift128_next(xorshift128_state* state);

void xorshift128_jump(xorshift128_state* state) {
	static const uint64_t JUMP[] = { 0x8a5cd789635d2dffULL, 0x121fd2155c472f96ULL };

	uint64_t s0 = 0;
	uint64_t s1 = 0;
	for(int i = 0; i < sizeof JUMP / sizeof *JUMP; i++)
		for(int b = 0; b < 64; b++) {
			if (JUMP[i] & 1ULL << b) {
				s0 ^= state->s[0];
				s1 ^= state->s[1];
			}
			xorshift128_next(state);
		}

	state->s[0] = s0;
	state->s[1] = s1;
}

void xorshift128_seed(xorshift128_state* state, uint64_t seed, uint64_t inc)
{
    state->s[0] = seed;
    state->s[1] = inc;
}