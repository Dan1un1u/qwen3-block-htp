#ifndef QWEN3_BLOCK_HTP_MLP_U8_H
#define QWEN3_BLOCK_HTP_MLP_U8_H

#include <stddef.h>
#include <stdint.h>

#define QBH_MLP_ACTIVATION_ZERO_POINT INT32_C(128)
#define QBH_MLP_SILU_KNEE INT32_C(64)
#define QBH_MLP_SILU_DENOMINATOR_SHIFT UINT32_C(7)
#define QBH_MLP_PRODUCT_SHIFT UINT32_C(5)
#define QBH_MLP_LUT_ENTRIES UINT32_C(65536)
#define QBH_MLP_LUT_BYTES (QBH_MLP_LUT_ENTRIES * sizeof(uint16_t))
#define QBH_MLP_PACKED_PAIR_LUT_ENTRIES UINT32_C(32768)
#define QBH_MLP_PACKED_PAIR_LUT_BYTES \
    (QBH_MLP_PACKED_PAIR_LUT_ENTRIES * sizeof(uint16_t))
#define QBH_MLP_GATHER_SCRATCH_BYTES UINT32_C(256)

static inline int32_t qbh_floor_div_pow2(int32_t value,
                                         uint32_t shift) {
    if (value >= 0) {
        return value >> shift;
    }
    return -(((-value) + ((INT32_C(1) << shift) - 1)) >> shift);
}

/*
 * Local fixed-point SiLU contract.  The clipped linear sigmoid is evaluated
 * in Q7, so this is an explicitly named approximation rather than a claim of
 * bit-exact floating-point SiLU.
 */
static inline int32_t qbh_mlp_silu_scalar(uint8_t gate) {
    int32_t x = (int32_t)gate - QBH_MLP_ACTIVATION_ZERO_POINT;
    int32_t sigmoid_numerator = x + QBH_MLP_SILU_KNEE;
    if (sigmoid_numerator < 0) {
        sigmoid_numerator = 0;
    } else if (sigmoid_numerator > 2 * QBH_MLP_SILU_KNEE) {
        sigmoid_numerator = 2 * QBH_MLP_SILU_KNEE;
    }
    return qbh_floor_div_pow2(
        x * sigmoid_numerator +
            (INT32_C(1) << (QBH_MLP_SILU_DENOMINATOR_SHIFT - 1U)),
        QBH_MLP_SILU_DENOMINATOR_SHIFT);
}

static inline uint8_t qbh_mlp_gate_up_scalar(uint8_t gate,
                                              uint8_t up) {
    int32_t up_centered =
        (int32_t)up - QBH_MLP_ACTIVATION_ZERO_POINT;
    int32_t product = qbh_mlp_silu_scalar(gate) * up_centered;
    int32_t centered = qbh_floor_div_pow2(
        product + (INT32_C(1) << (QBH_MLP_PRODUCT_SHIFT - 1U)),
        QBH_MLP_PRODUCT_SHIFT);
    int32_t encoded = centered + QBH_MLP_ACTIVATION_ZERO_POINT;
    if (encoded < 0) {
        return UINT8_C(0);
    }
    if (encoded > UINT8_MAX) {
        return UINT8_MAX;
    }
    return (uint8_t)encoded;
}

void qbh_mlp_gate_up_hvx(const uint8_t *gate, const uint8_t *up,
                         uint8_t *middle, size_t elements);

void qbh_mlp_gate_up_lut_hvx(const uint8_t *gate, const uint8_t *up,
                             uint8_t *middle, size_t elements,
                             const uint16_t *lut,
                             uint8_t *gather_scratch);

void qbh_mlp_gate_up_packed_pair_lut_hvx(
    const uint8_t *gate, const uint8_t *up, uint8_t *middle,
    size_t elements, const uint16_t *lut, uint8_t *gather_scratch);

void qbh_mlp_gate_up_requant_lut_hvx(
    const uint8_t *gate, const uint8_t *up, uint8_t *middle,
    size_t elements, const uint16_t *lut, uint8_t *gather_scratch,
    const uint8_t *gate_multipliers, const uint8_t *up_multipliers,
    int32_t gate_zero_point, int32_t up_zero_point);

void qbh_mlp_requant_u8_hvx(uint8_t *values, size_t elements,
                            const uint8_t *multipliers,
                            int32_t output_zero_point);

#endif
