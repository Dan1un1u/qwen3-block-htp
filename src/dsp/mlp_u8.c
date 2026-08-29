#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include <stddef.h>
#include <stdint.h>

#include "mlp_u8.h"

#define QBH_MLP_GATHER_HALF_BYTES UINT32_C(65536)
#define QBH_MLP_GATHER_REGION_MASK UINT32_C(65535)

static inline HVX_Vector qbh_mlp_gather_half(
    HVX_Vector gate, HVX_Vector up, const uint16_t *lut,
    HVX_Vector *scratch) {
    const HVX_Vector index_mask = Q6_Vh_vsplat_R(127);
    const HVX_Vector split = Q6_Vh_vsplat_R(127);
    HVX_Vector gate_low = Q6_V_vand_VV(gate, index_mask);
    HVX_Vector offsets = Q6_Vh_vadd_VhVh(
        Q6_Vh_vasl_VhR(gate_low, 9), Q6_Vh_vasl_VhR(up, 1));
    HVX_VectorPred high = Q6_Q_vcmp_gt_VuhVuh(gate, split);

    Q6_vgather_AQRMVh(
        scratch, Q6_Q_not_Q(high), (int32_t)(uintptr_t)lut,
        QBH_MLP_GATHER_REGION_MASK, offsets);
    Q6_vgather_AQRMVh(
        scratch, high,
        (int32_t)(uintptr_t)((const uint8_t *)lut +
                            QBH_MLP_GATHER_HALF_BYTES),
        QBH_MLP_GATHER_REGION_MASK, offsets);
    return *(volatile HVX_Vector *)scratch;
}

static inline HVX_Vector qbh_mlp_requant_vector(
    HVX_Vector input, HVX_Vector multipliers,
    HVX_Vector output_zero_point) {
    const HVX_Vector sign_flip = Q6_Vb_vsplat_R(0x80);
    const HVX_Vector centered = Q6_V_vxor_VV(input, sign_flip);
    const HVX_VectorPair product =
        Q6_Wh_vmpy_VbVb(centered, multipliers);
    const HVX_VectorPair interleaved = Q6_W_vshuff_VVR(
        Q6_V_hi_W(product), Q6_V_lo_W(product), -2);
    const HVX_Vector low = Q6_Vh_vadd_VhVh(
        Q6_V_lo_W(interleaved), output_zero_point);
    const HVX_Vector high = Q6_Vh_vadd_VhVh(
        Q6_V_hi_W(interleaved), output_zero_point);
    return Q6_Vub_vpack_VhVh_sat(high, low);
}

static HVX_Vector qbh_mlp_repeated_multipliers(
    const uint8_t *multipliers) {
    uint8_t repeated[sizeof(HVX_Vector)] __attribute__((aligned(128)));
    for (uint32_t row = 0; row < 4U; ++row) {
        for (uint32_t channel = 0; channel < 32U; ++channel) {
            repeated[row * 32U + channel] = multipliers[channel];
        }
    }
    return *(const HVX_Vector *)repeated;
}

static inline HVX_Vector qbh_mlp_half_vector(HVX_Vector x,
                                              HVX_Vector up) {
    const HVX_Vector zero_h = Q6_Vh_vsplat_R(0);
    const HVX_Vector knee_h = Q6_Vh_vsplat_R(QBH_MLP_SILU_KNEE);
    const HVX_Vector two_knee_h =
        Q6_Vh_vsplat_R(2 * QBH_MLP_SILU_KNEE);
    const HVX_Vector silu_round_w = Q6_V_vsplat_R(
        INT32_C(1) << (QBH_MLP_SILU_DENOMINATOR_SHIFT - 1U));
    const HVX_Vector product_round_w = Q6_V_vsplat_R(
        INT32_C(1) << (QBH_MLP_PRODUCT_SHIFT - 1U));
    HVX_Vector sigmoid_numerator = Q6_Vh_vadd_VhVh(x, knee_h);
    sigmoid_numerator = Q6_Vh_vmax_VhVh(sigmoid_numerator, zero_h);
    sigmoid_numerator =
        Q6_Vh_vmin_VhVh(sigmoid_numerator, two_knee_h);

    HVX_VectorPair silu_product =
        Q6_Ww_vmpy_VhVh(x, sigmoid_numerator);
    HVX_Vector silu_lo = Q6_Vw_vadd_VwVw(
        Q6_V_lo_W(silu_product), silu_round_w);
    HVX_Vector silu_hi = Q6_Vw_vadd_VwVw(
        Q6_V_hi_W(silu_product), silu_round_w);
    HVX_Vector silu = Q6_Vh_vasr_VwVwR(
        silu_hi, silu_lo, QBH_MLP_SILU_DENOMINATOR_SHIFT);

    HVX_VectorPair product = Q6_Ww_vmpy_VhVh(silu, up);
    HVX_Vector product_lo = Q6_Vw_vadd_VwVw(
        Q6_V_lo_W(product), product_round_w);
    HVX_Vector product_hi = Q6_Vw_vadd_VwVw(
        Q6_V_hi_W(product), product_round_w);
    return Q6_Vh_vasr_VwVwR(
        product_hi, product_lo, QBH_MLP_PRODUCT_SHIFT);
}

__attribute__((noinline)) void qbh_mlp_gate_up_hvx(
    const uint8_t *gate, const uint8_t *up, uint8_t *middle,
    size_t elements) {
    const HVX_Vector sign_flip = Q6_Vb_vsplat_R(0x80);
    const HVX_Vector output_zero_point =
        Q6_Vh_vsplat_R(QBH_MLP_ACTIVATION_ZERO_POINT);

    for (size_t offset = 0; offset < elements;
         offset += sizeof(HVX_Vector)) {
        HVX_Vector gate_u8 = *(const HVX_Vector *)(gate + offset);
        HVX_Vector up_u8 = *(const HVX_Vector *)(up + offset);
        HVX_Vector gate_s8 = Q6_V_vxor_VV(gate_u8, sign_flip);
        HVX_Vector up_s8 = Q6_V_vxor_VV(up_u8, sign_flip);
        HVX_VectorPair gate_h = Q6_Wh_vunpack_Vb(gate_s8);
        HVX_VectorPair up_h = Q6_Wh_vunpack_Vb(up_s8);
        HVX_Vector middle_lo = qbh_mlp_half_vector(
            Q6_V_lo_W(gate_h), Q6_V_lo_W(up_h));
        HVX_Vector middle_hi = qbh_mlp_half_vector(
            Q6_V_hi_W(gate_h), Q6_V_hi_W(up_h));
        middle_lo = Q6_Vh_vadd_VhVh(middle_lo, output_zero_point);
        middle_hi = Q6_Vh_vadd_VhVh(middle_hi, output_zero_point);
        *(HVX_Vector *)(middle + offset) =
            Q6_Vub_vpack_VhVh_sat(middle_hi, middle_lo);
    }
    asm volatile("barrier" : : : "memory");
}

__attribute__((noinline)) void qbh_mlp_gate_up_lut_hvx(
    const uint8_t *gate, const uint8_t *up, uint8_t *middle,
    size_t elements, const uint16_t *lut, uint8_t *gather_scratch) {
    HVX_Vector *scratch = (HVX_Vector *)gather_scratch;

    for (size_t offset = 0; offset < elements;
         offset += sizeof(HVX_Vector)) {
        HVX_Vector gate_u8 = *(const HVX_Vector *)(gate + offset);
        HVX_Vector up_u8 = *(const HVX_Vector *)(up + offset);
        HVX_VectorPair gate_h = Q6_Wuh_vunpack_Vub(gate_u8);
        HVX_VectorPair up_h = Q6_Wuh_vunpack_Vub(up_u8);
        HVX_Vector middle_lo = qbh_mlp_gather_half(
            Q6_V_lo_W(gate_h), Q6_V_lo_W(up_h), lut, scratch);
        HVX_Vector middle_hi = qbh_mlp_gather_half(
            Q6_V_hi_W(gate_h), Q6_V_hi_W(up_h), lut, scratch + 1);
        *(HVX_Vector *)(middle + offset) =
            Q6_Vub_vpack_VhVh_sat(middle_hi, middle_lo);
    }
    asm volatile("barrier" : : : "memory");
}

__attribute__((noinline)) void qbh_mlp_gate_up_requant_lut_hvx(
    const uint8_t *gate, const uint8_t *up, uint8_t *middle,
    size_t elements, const uint16_t *lut, uint8_t *gather_scratch,
    const uint8_t *gate_multipliers, const uint8_t *up_multipliers,
    int32_t gate_zero_point, int32_t up_zero_point) {
    HVX_Vector *scratch = (HVX_Vector *)gather_scratch;
    const HVX_Vector gate_scale =
        qbh_mlp_repeated_multipliers(gate_multipliers);
    const HVX_Vector up_scale =
        qbh_mlp_repeated_multipliers(up_multipliers);
    const HVX_Vector gate_zp = Q6_Vh_vsplat_R(gate_zero_point);
    const HVX_Vector up_zp = Q6_Vh_vsplat_R(up_zero_point);

    for (size_t offset = 0; offset < elements;
         offset += sizeof(HVX_Vector)) {
        HVX_Vector gate_u8 = qbh_mlp_requant_vector(
            *(const HVX_Vector *)(gate + offset), gate_scale, gate_zp);
        HVX_Vector up_u8 = qbh_mlp_requant_vector(
            *(const HVX_Vector *)(up + offset), up_scale, up_zp);
        HVX_VectorPair gate_h = Q6_Wuh_vunpack_Vub(gate_u8);
        HVX_VectorPair up_h = Q6_Wuh_vunpack_Vub(up_u8);
        HVX_Vector middle_lo = qbh_mlp_gather_half(
            Q6_V_lo_W(gate_h), Q6_V_lo_W(up_h), lut, scratch);
        HVX_Vector middle_hi = qbh_mlp_gather_half(
            Q6_V_hi_W(gate_h), Q6_V_hi_W(up_h), lut, scratch + 1);
        *(HVX_Vector *)(middle + offset) =
            Q6_Vub_vpack_VhVh_sat(middle_hi, middle_lo);
    }
    asm volatile("barrier" : : : "memory");
}

__attribute__((noinline)) void qbh_mlp_requant_u8_hvx(
    uint8_t *values, size_t elements, const uint8_t *multipliers,
    int32_t output_zero_point) {
    const HVX_Vector scale =
        qbh_mlp_repeated_multipliers(multipliers);
    const HVX_Vector output_zp = Q6_Vh_vsplat_R(output_zero_point);
    for (size_t offset = 0; offset < elements;
         offset += sizeof(HVX_Vector)) {
        HVX_Vector input = *(const HVX_Vector *)(values + offset);
        *(HVX_Vector *)(values + offset) =
            qbh_mlp_requant_vector(input, scale, output_zp);
    }
    asm volatile("barrier" : : : "memory");
}
