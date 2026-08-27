#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include <stddef.h>
#include <stdint.h>

#include "mlp_u8.h"

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
