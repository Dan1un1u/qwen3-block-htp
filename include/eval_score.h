#ifndef QBH_EVAL_SCORE_H
#define QBH_EVAL_SCORE_H
#include <math.h>
#include <stdint.h>

/* Histogram bins preserve every emitted FP16 bit pattern / U8 output code.
 * This is quality-only reduction, not a replacement for the recipe's head. */
struct qbh_eval_score {
    float logsumexp;
    float target_logit;
    float nll;
    uint32_t count;
    uint32_t rank;
    uint32_t target_ties;
    uint32_t max_ties;
    uint32_t saturated;
    uint32_t nonfinite;
};

static float qbh_eval_value(uint32_t code, int u8, float scale, int32_t zero) {
    if (u8) return ((float)code - (float)zero) * scale;
    const uint32_t exponent = (code >> 10U) & 31U;
    const uint32_t mantissa = code & 1023U;
    float value;
    if (exponent == 31U) value = mantissa ? NAN : INFINITY;
    else value = exponent ? ldexpf(1.0f + mantissa / 1024.0f, (int)exponent - 15)
                          : ldexpf((float)mantissa, -24);
    return (code & 32768U) ? -value : value;
}

static struct qbh_eval_score qbh_eval_reduce(const uint32_t *hist,
    int u8, float scale, int32_t zero, uint32_t target_code) {
    struct qbh_eval_score out = {0};
    const uint32_t bins = u8 ? 256U : 65536U;
    float maximum = -INFINITY;
    double sum = 0.0;
    out.target_logit = qbh_eval_value(target_code, u8, scale, zero);
    out.rank = 1U;
    for (uint32_t i = 0; i < bins; ++i) {
        if (!hist[i]) continue;
        float value = qbh_eval_value(i, u8, scale, zero);
        out.count += hist[i];
        if (!isfinite(value)) { out.nonfinite += hist[i]; continue; }
        if (value > maximum) maximum = value;
        if (value > out.target_logit) out.rank += hist[i];
        if (value == out.target_logit) out.target_ties += hist[i];
    }
    for (uint32_t i = 0; i < bins; ++i) {
        if (!hist[i]) continue;
        float value = qbh_eval_value(i, u8, scale, zero);
        if (!isfinite(value)) continue;
        sum += (double)hist[i] * expf(value - maximum);
        if (value == maximum) out.max_ties += hist[i];
    }
    out.saturated = u8 ? hist[0] + hist[255] : 0;
    out.logsumexp = out.nonfinite ? NAN : maximum + logf((float)sum);
    out.nll = out.logsumexp - out.target_logit;
    return out;
}
#endif
