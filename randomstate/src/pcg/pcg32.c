#include "pcg32.h"

extern inline void pcg_setseq_64_step_r(pcg_state_setseq_64* rng);
extern inline uint32_t pcg_output_xsl_rr_64_32(uint64_t state);
extern inline void pcg_setseq_64_srandom_r(pcg_state_setseq_64* rng,
        uint64_t initstate, uint64_t initseq);
extern inline uint32_t
pcg_setseq_64_xsl_rr_32_random_r(pcg_state_setseq_64* rng);
extern inline uint32_t
pcg_setseq_64_xsl_rr_32_boundedrand_r(pcg_state_setseq_64* rng,
                                      uint32_t bound);
extern inline void pcg_setseq_64_advance_r(pcg_state_setseq_64* rng,
        uint64_t delta);

uint64_t pcg_advance_lcg_64(uint64_t state, uint64_t delta, uint64_t cur_mult,
                            uint64_t cur_plus)
{
    uint64_t acc_mult = 1u;
    uint64_t acc_plus = 0u;
    while (delta > 0) {
        if (delta & 1) {
            acc_mult *= cur_mult;
            acc_plus = acc_plus * cur_mult + cur_plus;
        }
        cur_plus = (cur_mult + 1) * cur_plus;
        cur_mult *= cur_mult;
        delta /= 2;
    }
    return acc_mult * state + acc_plus;
}
