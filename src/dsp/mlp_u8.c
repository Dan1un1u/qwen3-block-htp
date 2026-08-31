#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include <stddef.h>
#include <stdint.h>

#include "mlp_u8.h"

#define QBH_MLP_GATHER_HALF_BYTES UINT32_C(65536)
#define QBH_MLP_GATHER_REGION_MASK UINT32_C(65535)
#define QBH_MLP_AFFINE_REGION_MASK UINT32_C(511)

/*
 * Exact layer-14 affine representation of silu_up_lut_u16.bin.  Entries
 * 0..121 are magnitudes in Q15 and entries 122..255 are in Q14.  The
 * coefficient sign is negative for gate codes below the formal zero point
 * 125.  Exhaustive generation verifies all 65,536 Gate/Up code pairs.
 */
static const uint16_t qbh_mlp_exact_affine_coefficients[256]
    __attribute__((aligned(128))) = {
    UINT16_C(46), UINT16_C(48), UINT16_C(51), UINT16_C(53), UINT16_C(56), UINT16_C(59), UINT16_C(62), UINT16_C(65),
    UINT16_C(69), UINT16_C(72), UINT16_C(76), UINT16_C(80), UINT16_C(84), UINT16_C(89), UINT16_C(93), UINT16_C(98),
    UINT16_C(103), UINT16_C(109), UINT16_C(114), UINT16_C(120), UINT16_C(127), UINT16_C(133), UINT16_C(140), UINT16_C(147),
    UINT16_C(154), UINT16_C(163), UINT16_C(170), UINT16_C(179), UINT16_C(188), UINT16_C(198), UINT16_C(208), UINT16_C(218),
    UINT16_C(229), UINT16_C(240), UINT16_C(253), UINT16_C(265), UINT16_C(278), UINT16_C(292), UINT16_C(306), UINT16_C(322),
    UINT16_C(337), UINT16_C(353), UINT16_C(371), UINT16_C(389), UINT16_C(407), UINT16_C(427), UINT16_C(447), UINT16_C(469),
    UINT16_C(491), UINT16_C(514), UINT16_C(538), UINT16_C(563), UINT16_C(590), UINT16_C(617), UINT16_C(646), UINT16_C(675),
    UINT16_C(706), UINT16_C(737), UINT16_C(771), UINT16_C(805), UINT16_C(841), UINT16_C(877), UINT16_C(916), UINT16_C(955),
    UINT16_C(997), UINT16_C(1039), UINT16_C(1083), UINT16_C(1128), UINT16_C(1175), UINT16_C(1223), UINT16_C(1273), UINT16_C(1324),
    UINT16_C(1376), UINT16_C(1430), UINT16_C(1485), UINT16_C(1541), UINT16_C(1598), UINT16_C(1657), UINT16_C(1717), UINT16_C(1778),
    UINT16_C(1840), UINT16_C(1902), UINT16_C(1965), UINT16_C(2029), UINT16_C(2093), UINT16_C(2157), UINT16_C(2222), UINT16_C(2285),
    UINT16_C(2349), UINT16_C(2412), UINT16_C(2473), UINT16_C(2533), UINT16_C(2592), UINT16_C(2649), UINT16_C(2703), UINT16_C(2754),
    UINT16_C(2801), UINT16_C(2845), UINT16_C(2884), UINT16_C(2919), UINT16_C(2948), UINT16_C(2971), UINT16_C(2987), UINT16_C(2996),
    UINT16_C(2997), UINT16_C(2989), UINT16_C(2972), UINT16_C(2945), UINT16_C(2908), UINT16_C(2858), UINT16_C(2797), UINT16_C(2723),
    UINT16_C(2636), UINT16_C(2533), UINT16_C(2416), UINT16_C(2284), UINT16_C(2136), UINT16_C(1971), UINT16_C(1788), UINT16_C(1589),
    UINT16_C(1371), UINT16_C(1135), UINT16_C(440), UINT16_C(303), UINT16_C(156), UINT16_C(0), UINT16_C(166), UINT16_C(341),
    UINT16_C(526), UINT16_C(720), UINT16_C(924), UINT16_C(1137), UINT16_C(1359), UINT16_C(1590), UINT16_C(1829), UINT16_C(2077),
    UINT16_C(2333), UINT16_C(2597), UINT16_C(2867), UINT16_C(3145), UINT16_C(3430), UINT16_C(3721), UINT16_C(4018), UINT16_C(4321),
    UINT16_C(4630), UINT16_C(4943), UINT16_C(5261), UINT16_C(5584), UINT16_C(5910), UINT16_C(6240), UINT16_C(6573), UINT16_C(6910),
    UINT16_C(7249), UINT16_C(7590), UINT16_C(7934), UINT16_C(8280), UINT16_C(8627), UINT16_C(8977), UINT16_C(9326), UINT16_C(9678),
    UINT16_C(10030), UINT16_C(10382), UINT16_C(10736), UINT16_C(11090), UINT16_C(11443), UINT16_C(11797), UINT16_C(12151), UINT16_C(12505),
    UINT16_C(12859), UINT16_C(13212), UINT16_C(13565), UINT16_C(13918), UINT16_C(14271), UINT16_C(14622), UINT16_C(14973), UINT16_C(15324),
    UINT16_C(15674), UINT16_C(16024), UINT16_C(16372), UINT16_C(16720), UINT16_C(17068), UINT16_C(17415), UINT16_C(17761), UINT16_C(18106),
    UINT16_C(18450), UINT16_C(18794), UINT16_C(19138), UINT16_C(19480), UINT16_C(19821), UINT16_C(20162), UINT16_C(20503), UINT16_C(20842),
    UINT16_C(21182), UINT16_C(21520), UINT16_C(21858), UINT16_C(22195), UINT16_C(22532), UINT16_C(22868), UINT16_C(23203), UINT16_C(23538),
    UINT16_C(23873), UINT16_C(24207), UINT16_C(24540), UINT16_C(24873), UINT16_C(25206), UINT16_C(25538), UINT16_C(25870), UINT16_C(26201),
    UINT16_C(26532), UINT16_C(26863), UINT16_C(27192), UINT16_C(27522), UINT16_C(27852), UINT16_C(28181), UINT16_C(28510), UINT16_C(28838),
    UINT16_C(29166), UINT16_C(29494), UINT16_C(29822), UINT16_C(30149), UINT16_C(30476), UINT16_C(30803), UINT16_C(31130), UINT16_C(31456),
    UINT16_C(31782), UINT16_C(32108), UINT16_C(32434), UINT16_C(32760), UINT16_C(33085), UINT16_C(33411), UINT16_C(33736), UINT16_C(34061),
    UINT16_C(34386), UINT16_C(34710), UINT16_C(35035), UINT16_C(35359), UINT16_C(35683), UINT16_C(36008), UINT16_C(36332), UINT16_C(36656),
    UINT16_C(36980), UINT16_C(37304), UINT16_C(37627), UINT16_C(37951), UINT16_C(38275), UINT16_C(38598), UINT16_C(38921), UINT16_C(39245),
    UINT16_C(39568), UINT16_C(39891), UINT16_C(40214), UINT16_C(40537), UINT16_C(40860), UINT16_C(41183), UINT16_C(41506), UINT16_C(41829),
};

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

void qbh_mlp_stage_exact_affine_coefficients(uint16_t *destination) {
    HVX_Vector *output = (HVX_Vector *)destination;
    const HVX_Vector *input =
        (const HVX_Vector *)qbh_mlp_exact_affine_coefficients;

    for (uint32_t index = 0;
         index < QBH_MLP_AFFINE_COEFFICIENT_BYTES / sizeof(HVX_Vector);
         ++index) {
        output[index] = input[index];
    }
    asm volatile("barrier" : : : "memory");
}

static inline HVX_Vector qbh_mlp_exact_affine_half(
    HVX_Vector gate, HVX_Vector up, const uint16_t *coefficients,
    HVX_Vector *scratch) {
    const HVX_Vector zero = Q6_V_vzero();
    const HVX_Vector up_zero_point = Q6_Vh_vsplat_R(110);
    const HVX_Vector gate_zero_point = Q6_Vh_vsplat_R(125);
    const HVX_Vector q15_boundary = Q6_Vh_vsplat_R(122);
    const HVX_Vector round_q14 = Q6_V_vsplat_R(INT32_C(1) << 13);
    const HVX_Vector round_q15 = Q6_V_vsplat_R(INT32_C(1) << 14);
    const HVX_Vector offsets = Q6_Vh_vasl_VhR(gate, 1);
    const HVX_Vector up_centered =
        Q6_Vh_vsub_VhVh(up, up_zero_point);
    const HVX_Vector up_negated = Q6_Vh_vsub_VhVh(zero, up_centered);
    const HVX_VectorPred negative =
        Q6_Q_vcmp_gt_VuhVuh(gate_zero_point, gate);
    const HVX_VectorPred use_q15 =
        Q6_Q_vcmp_gt_VuhVuh(q15_boundary, gate);
    const HVX_Vector signed_up =
        Q6_V_vmux_QVV(negative, up_negated, up_centered);
    HVX_VectorPair product;
    HVX_Vector q14;
    HVX_Vector q15;

    Q6_vgather_ARMVh(
        scratch, (int32_t)(uintptr_t)coefficients,
        QBH_MLP_AFFINE_REGION_MASK, offsets);
    product = Q6_Ww_vmpy_VhVuh(
        signed_up, *(volatile HVX_Vector *)scratch);
    q14 = Q6_Vh_vasr_VwVwR(
        Q6_Vw_vadd_VwVw(Q6_V_hi_W(product), round_q14),
        Q6_Vw_vadd_VwVw(Q6_V_lo_W(product), round_q14), 14);
    q15 = Q6_Vh_vasr_VwVwR(
        Q6_Vw_vadd_VwVw(Q6_V_hi_W(product), round_q15),
        Q6_Vw_vadd_VwVw(Q6_V_lo_W(product), round_q15), 15);
    return Q6_V_vmux_QVV(use_q15, q15, q14);
}

__attribute__((noinline)) void qbh_mlp_gate_up_exact_affine_hvx(
    const uint8_t *gate, const uint8_t *up, uint8_t *middle,
    size_t elements, const uint16_t *coefficients,
    uint8_t *gather_scratch) {
    HVX_Vector *scratch = (HVX_Vector *)gather_scratch;
    const HVX_Vector output_zero_point = Q6_Vh_vsplat_R(102);

    for (size_t offset = 0; offset < elements;
         offset += sizeof(HVX_Vector)) {
        const HVX_Vector gate_u8 =
            *(const HVX_Vector *)(gate + offset);
        const HVX_Vector up_u8 = *(const HVX_Vector *)(up + offset);
        const HVX_VectorPair gate_h = Q6_Wuh_vunpack_Vub(gate_u8);
        const HVX_VectorPair up_h = Q6_Wuh_vunpack_Vub(up_u8);
        HVX_Vector middle_lo = qbh_mlp_exact_affine_half(
            Q6_V_lo_W(gate_h), Q6_V_lo_W(up_h), coefficients,
            scratch);
        HVX_Vector middle_hi = qbh_mlp_exact_affine_half(
            Q6_V_hi_W(gate_h), Q6_V_hi_W(up_h), coefficients,
            scratch + 1);

        middle_lo = Q6_Vh_vadd_VhVh(middle_lo, output_zero_point);
        middle_hi = Q6_Vh_vadd_VhVh(middle_hi, output_zero_point);
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
