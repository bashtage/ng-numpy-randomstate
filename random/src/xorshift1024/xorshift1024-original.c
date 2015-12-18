/*  Written in 2014-2015 by Sebastiano Vigna (vigna@acm.org)

To the extent possible under law, the author has dedicated all copyright
and related and neighboring rights to this software to the public domain
worldwide. This software is distributed without any warranty.

See <http://creativecommons.org/publicdomain/zero/1.0/>. */

#include <stdint.h>
#include <string.h>

/* This is a fast, top-quality generator. If 1024 bits of state are too
   much, try a xorshift128+ generator.

   The state must be seeded so that it is not everywhere zero. If you have
   a 64-bit seed, we suggest to seed a splitmix64 generator and use its
   output to fill s. */

uint64_t s[16];
int p;

uint64_t next(void) {
	const uint64_t s0 = s[p];
	uint64_t s1 = s[p = (p + 1) & 15];
	s1 ^= s1 << 31; // a
	s[p] = s1 ^ s0 ^ (s1 >> 11) ^ (s0 >> 30); // b,c
	return s[p] * UINT64_C(1181783497276652981);
}


/* This is the jump function for the generator. It is equivalent
   to 2^512 calls to next(); it can be used to generate 2^512
   non-overlapping subsequences for parallel computations. */

void jump() {
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
					t[j] ^= s[(j + p) & 15];
			next();
		}

	memcpy(s, t, sizeof t);
}
