#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include <stddef.h>
#include <stdint.h>

#include "mlp_u8.h"

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

__attribute__((noinline)) void qbh_mlp_gate_up_sp2_lut_hvx(
    const uint8_t *gate, const uint8_t *up, uint8_t *low, uint8_t *high,
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
        /* LUT contains v+32768: low byte l, high byte h+128. */
        *(HVX_Vector *)(low + offset) = Q6_Vb_vpacke_VhVh(middle_hi, middle_lo);
        *(HVX_Vector *)(high + offset) = Q6_Vb_vpacko_VhVh(middle_hi, middle_lo);
    }
    asm volatile("barrier" : : : "memory");
}

/* L32-0012: issue both independent half-vector gathers before consuming either. */
static inline void qbh_mlp_gather_half_issue(
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

}

__attribute__((noinline)) void qbh_mlp_gate_up_sp2_lut_pipelined_hvx(
    const uint8_t *gate, const uint8_t *up, uint8_t *low, uint8_t *high,
    size_t elements, const uint16_t *lut, uint8_t *gather_scratch) {
    HVX_Vector *scratch = (HVX_Vector *)gather_scratch;

    for (size_t offset = 0; offset < elements;
         offset += sizeof(HVX_Vector)) {
        HVX_Vector gate_u8 = *(const HVX_Vector *)(gate + offset);
        HVX_Vector up_u8 = *(const HVX_Vector *)(up + offset);
        HVX_VectorPair gate_h = Q6_Wuh_vunpack_Vub(gate_u8);
        HVX_VectorPair up_h = Q6_Wuh_vunpack_Vub(up_u8);
        qbh_mlp_gather_half_issue(Q6_V_lo_W(gate_h),Q6_V_lo_W(up_h),lut,scratch);
        qbh_mlp_gather_half_issue(Q6_V_hi_W(gate_h),Q6_V_hi_W(up_h),lut,scratch+1);
        HVX_Vector middle_lo=*(volatile HVX_Vector *)scratch;
        HVX_Vector middle_hi=*(volatile HVX_Vector *)(scratch+1);
        /* LUT contains v+32768: low byte l, high byte h+128. */
        *(HVX_Vector *)(low + offset) = Q6_Vb_vpacke_VhVh(middle_hi, middle_lo);
        *(HVX_Vector *)(high + offset) = Q6_Vb_vpacko_VhVh(middle_hi, middle_lo);
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

/* EXP0261: same LUT gather, retain the complete halfword rather than saturating
 * it to U8. LUT stores unquantized FP16 SwiGLU for the R4 input boundary. */
void qbh_mlp_gate_up_lut_f16_hvx(const uint8_t *gate,const uint8_t *up,
    uint16_t *middle,size_t elements,const uint16_t *lut,uint8_t *scratch_bytes) {
    HVX_Vector *scratch=(HVX_Vector *)scratch_bytes;
    for(size_t o=0;o<elements;o+=128U) {
        HVX_VectorPair g=Q6_Wuh_vunpack_Vub(*(const HVX_Vector *)(gate+o));
        HVX_VectorPair u=Q6_Wuh_vunpack_Vub(*(const HVX_Vector *)(up+o));
        ((HVX_Vector *)(middle+o))[0]=qbh_mlp_gather_half(Q6_V_lo_W(g),Q6_V_lo_W(u),lut,scratch);
        ((HVX_Vector *)(middle+o))[1]=qbh_mlp_gather_half(Q6_V_hi_W(g),Q6_V_hi_W(u),lut,scratch+1);
    }
    asm volatile("barrier" ::: "memory");
}

/* EXP0274: producer publishes compact reconstruction values. The consumer
 * separately packs native operands. Same LUT/gathers, no scalar fallback,
 * no native-to-generic roundtrip,4096B already-idle VTCM scratch per worker. */
__attribute__((noinline)) void qbh_mlp_gate_up_sp2_compact_hvx(
    const uint8_t *gate,const uint8_t *up,uint8_t *low,uint8_t *high,
    size_t elements,const uint16_t *lut,uint8_t *gather_scratch,uint8_t *compact) {
    HVX_Vector *scratch=(HVX_Vector *)gather_scratch;
    for(size_t off=0;off<elements;off+=128U) {
        HVX_VectorPair g=Q6_Wuh_vunpack_Vub(*(const HVX_Vector *)(gate+off));
        HVX_VectorPair u=Q6_Wuh_vunpack_Vub(*(const HVX_Vector *)(up+off));
        qbh_mlp_gather_half_issue(Q6_V_lo_W(g),Q6_V_lo_W(u),lut,scratch);
        qbh_mlp_gather_half_issue(Q6_V_hi_W(g),Q6_V_hi_W(u),lut,scratch+1);
        *(HVX_Vector *)(compact+2U*off)=*(volatile HVX_Vector *)scratch;
        *(HVX_Vector *)(compact+2U*off+128U)=*(volatile HVX_Vector *)(scratch+1);
    }
    asm volatile("barrier":::"memory");
    for(size_t off=0;off<elements;off+=128U) {
        HVX_Vector a=*(const HVX_Vector *)(compact+2U*off);
        HVX_Vector b=*(const HVX_Vector *)(compact+2U*off+128U);
        *(HVX_Vector *)(low+off)=Q6_Vb_vpacke_VhVh(b,a);
        *(HVX_Vector *)(high+off)=Q6_Vb_vpacko_VhVh(b,a);
    }
    asm volatile("barrier":::"memory");
}

__attribute__((noinline)) void qbh_mlp_gate_up_lut_pipelined_hvx(
    const uint8_t *gate, const uint8_t *up, uint8_t *low,
    size_t elements, const uint16_t *lut, uint8_t *gather_scratch) {
    HVX_Vector *scratch = (HVX_Vector *)gather_scratch;

    for (size_t offset = 0; offset < elements;
         offset += sizeof(HVX_Vector)) {
        HVX_Vector gate_u8 = *(const HVX_Vector *)(gate + offset);
        HVX_Vector up_u8 = *(const HVX_Vector *)(up + offset);
        HVX_VectorPair gate_h = Q6_Wuh_vunpack_Vub(gate_u8);
        HVX_VectorPair up_h = Q6_Wuh_vunpack_Vub(up_u8);
        qbh_mlp_gather_half_issue(Q6_V_lo_W(gate_h),Q6_V_lo_W(up_h),lut,scratch);
        qbh_mlp_gather_half_issue(Q6_V_hi_W(gate_h),Q6_V_hi_W(up_h),lut,scratch+1);
        HVX_Vector middle_lo=*(volatile HVX_Vector *)scratch;
        HVX_Vector middle_hi=*(volatile HVX_Vector *)(scratch+1);
        /* Ordinary LUT entries are U8 codes in halfwords; same pipelined gathers. */
        *(HVX_Vector *)(low + offset) = Q6_Vub_vpack_VhVh_sat(middle_hi, middle_lo);
    }
    asm volatile("barrier" : : : "memory");
}

/* L32-0035: decode has one live32-channel row per native tile. Query only
 * those LUT indices; the other three rows cannot affect the live HMX dot.
 * Initialize complete output vectors: padding represents reconstructed zero.
 * This removes dependence of gather cost on dead Gate/Up padding contents. */
__attribute__((noinline)) void qbh_mlp_gate_up_sp2_decode_row1_hvx(
    const uint8_t *gate,const uint8_t *up,uint8_t *low,uint8_t *high,
    const uint16_t *lut,uint8_t *gather_scratch) {
    HVX_Vector g=Q6_V_lo_W(Q6_Wuh_vunpack_Vub(*(const HVX_Vector *)gate));
    HVX_Vector u=Q6_V_lo_W(Q6_Wuh_vunpack_Vub(*(const HVX_Vector *)up));
    HVX_Vector mask=Q6_Vh_vsplat_R(127);
    HVX_Vector offsets=Q6_Vh_vadd_VhVh(Q6_Vh_vasl_VhR(Q6_V_vand_VV(g,mask),9),Q6_Vh_vasl_VhR(u,1));
    HVX_VectorPred upper=Q6_Q_vcmp_gt_VuhVuh(g,mask);
    HVX_VectorPred live_h=Q6_Q_vsetq_R(64U);
    Q6_vgather_AQRMVh((HVX_Vector *)gather_scratch,Q6_Q_and_QQn(live_h,upper),(int32_t)(uintptr_t)lut,QBH_MLP_GATHER_REGION_MASK,offsets);
    Q6_vgather_AQRMVh((HVX_Vector *)gather_scratch,Q6_Q_and_QQ(live_h,upper),(int32_t)(uintptr_t)((const uint8_t *)lut+QBH_MLP_GATHER_HALF_BYTES),QBH_MLP_GATHER_REGION_MASK,offsets);
    HVX_Vector value=*(volatile HVX_Vector *)gather_scratch;
    HVX_VectorPred live_b=Q6_Q_vsetq_R(32U);
    *(HVX_Vector *)low=Q6_V_vmux_QVV(live_b,Q6_Vb_vpacke_VhVh(Q6_V_vzero(),value),Q6_V_vzero());
    *(HVX_Vector *)high=Q6_V_vmux_QVV(live_b,Q6_Vb_vpacko_VhVh(Q6_V_vzero(),value),Q6_Vb_vsplat_R(128));
    asm volatile("barrier":::"memory");
}

/* Fair ablation: same live-row gather, explicit compact publication/pack. */
__attribute__((noinline)) void qbh_mlp_gate_up_sp2_decode_row1_compact_hvx(
    const uint8_t *gate,const uint8_t *up,uint8_t *low,uint8_t *high,
    const uint16_t *lut,uint8_t *gather_scratch,uint8_t *compact) {
    HVX_Vector g=Q6_V_lo_W(Q6_Wuh_vunpack_Vub(*(const HVX_Vector *)gate));
    HVX_Vector u=Q6_V_lo_W(Q6_Wuh_vunpack_Vub(*(const HVX_Vector *)up));
    HVX_Vector mask=Q6_Vh_vsplat_R(127);
    HVX_Vector offsets=Q6_Vh_vadd_VhVh(Q6_Vh_vasl_VhR(Q6_V_vand_VV(g,mask),9),Q6_Vh_vasl_VhR(u,1));
    HVX_VectorPred upper=Q6_Q_vcmp_gt_VuhVuh(g,mask);
    HVX_VectorPred live_h=Q6_Q_vsetq_R(64U);
    Q6_vgather_AQRMVh((HVX_Vector *)gather_scratch,Q6_Q_and_QQn(live_h,upper),(int32_t)(uintptr_t)lut,QBH_MLP_GATHER_REGION_MASK,offsets);
    Q6_vgather_AQRMVh((HVX_Vector *)gather_scratch,Q6_Q_and_QQ(live_h,upper),(int32_t)(uintptr_t)((const uint8_t *)lut+QBH_MLP_GATHER_HALF_BYTES),QBH_MLP_GATHER_REGION_MASK,offsets);
    HVX_Vector value=*(volatile HVX_Vector *)gather_scratch;
    /* Natural uint16 reconstructed-value producer; no native roundtrip.
     * Only row0 is live, just as in the direct native producer. */
    *(HVX_Vector *)compact=Q6_V_vmux_QVV(live_h,value,Q6_Vh_vsplat_R(32768));
    asm volatile("barrier" ::: "memory");
    value=*(volatile HVX_Vector *)compact;
    HVX_VectorPred live_b=Q6_Q_vsetq_R(32U);
    *(HVX_Vector *)low=Q6_V_vmux_QVV(live_b,Q6_Vb_vpacke_VhVh(Q6_V_vzero(),value),Q6_V_vzero());
    *(HVX_Vector *)high=Q6_V_vmux_QVV(live_b,Q6_Vb_vpacko_VhVh(Q6_V_vzero(),value),Q6_Vb_vsplat_R(128));
    asm volatile("barrier":::"memory");
}

/* Equal valid-row optimization for ordinary-A8 SwiGLU. Padding is dead. */
__attribute__((noinline)) void qbh_mlp_gate_up_decode_row1_hvx(
    const uint8_t *gate,const uint8_t *up,uint8_t *middle,
    const uint16_t *lut,uint8_t *gather_scratch) {
    HVX_Vector g=Q6_V_lo_W(Q6_Wuh_vunpack_Vub(*(const HVX_Vector *)gate));
    HVX_Vector u=Q6_V_lo_W(Q6_Wuh_vunpack_Vub(*(const HVX_Vector *)up));
    HVX_Vector mask=Q6_Vh_vsplat_R(127);
    HVX_Vector offsets=Q6_Vh_vadd_VhVh(Q6_Vh_vasl_VhR(Q6_V_vand_VV(g,mask),9),Q6_Vh_vasl_VhR(u,1));
    HVX_VectorPred upper=Q6_Q_vcmp_gt_VuhVuh(g,mask);
    HVX_VectorPred live_h=Q6_Q_vsetq_R(64U);
    Q6_vgather_AQRMVh((HVX_Vector *)gather_scratch,Q6_Q_and_QQn(live_h,upper),(int32_t)(uintptr_t)lut,QBH_MLP_GATHER_REGION_MASK,offsets);
    Q6_vgather_AQRMVh((HVX_Vector *)gather_scratch,Q6_Q_and_QQ(live_h,upper),(int32_t)(uintptr_t)((const uint8_t *)lut+QBH_MLP_GATHER_HALF_BYTES),QBH_MLP_GATHER_REGION_MASK,offsets);
    HVX_Vector value=*(volatile HVX_Vector *)gather_scratch;
    HVX_VectorPred live_b=Q6_Q_vsetq_R(32U);
    *(HVX_Vector *)middle=Q6_V_vmux_QVV(live_b,Q6_Vub_vpack_VhVh_sat(Q6_V_vzero(),value),Q6_V_vzero());
    asm volatile("barrier":::"memory");
}
