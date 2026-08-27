#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "probe_protocol.h"
#include "w4_u8_expand.h"

/*
 * vlut32 with Rt=0 selects byte 0 from each table halfword.  Therefore each
 * signed nibble value occupies the low byte of a 16-bit table entry rather
 * than a contiguous table byte.
 */
static const int8_t qbh_signed_w4_lut[128]
    __attribute__((aligned(128))) = {
        0,  0, 1,  0, 2,  0, 3,  0, 4,  0, 5,  0, 6,  0, 7,  0,
        -8, 0, -7, 0, -6, 0, -5, 0, -4, 0, -3, 0, -2, 0, -1, 0,
};

static HVX_Vector qbh_scale_w4_f16_lanes(
    HVX_Vector v_quant_f16, HVX_Vector v_scale) {
    const HVX_Vector v_one_f16 = Q6_Vh_vsplat_R(0x3c00);
    const HVX_VectorPair v_quant_qf32 =
        Q6_Wqf32_vmpy_VhfVhf(v_quant_f16, v_one_f16);
    const HVX_Vector v_quant_lo =
        Q6_Vsf_equals_Vqf32(Q6_V_lo_W(v_quant_qf32));
    const HVX_Vector v_quant_hi =
        Q6_Vsf_equals_Vqf32(Q6_V_hi_W(v_quant_qf32));
    const HVX_Vector v_product_lo =
        Q6_Vqf32_vmpy_VsfVsf(v_quant_lo, v_scale);
    const HVX_Vector v_product_hi =
        Q6_Vqf32_vmpy_VsfVsf(v_quant_hi, v_scale);
    return Q6_Vhf_equals_Wqf32(
        Q6_W_vcombine_VV(v_product_hi, v_product_lo));
}

static HVX_VectorPair qbh_scale_w4_f16_group(
    HVX_Vector v_signed_group, HVX_Vector v_scale) {
    /* Packed integer-HMX tiles group four K rows per output.  FP16 HMX groups
     * two K rows.  Deal the signed-byte vector as halfwords so the even
     * (row 0/1) pairs occupy the low half and the odd (row 2/3) pairs occupy
     * the high half before widening to FP16. */
    const HVX_Vector v_two_row_groups =
        Q6_Vh_vdeal_Vh(v_signed_group);
    const HVX_VectorPair v_quant_h =
        Q6_Wh_vunpack_Vb(v_two_row_groups);
    const HVX_Vector v_row01_f16 =
        Q6_Vhf_equals_Vh(Q6_V_lo_W(v_quant_h));
    const HVX_Vector v_row23_f16 =
        Q6_Vhf_equals_Vh(Q6_V_hi_W(v_quant_h));
    const HVX_Vector v_scaled01 = qbh_scale_w4_f16_lanes(
        v_row01_f16, v_scale);
    const HVX_Vector v_scaled23 = qbh_scale_w4_f16_lanes(
        v_row23_f16, v_scale);
    return Q6_W_vcombine_VV(v_scaled23, v_scaled01);
}

static HVX_VectorPair qbh_unpack_w4_f16_group(
    HVX_Vector v_signed_group) {
    const HVX_Vector v_two_row_groups =
        Q6_Vh_vdeal_Vh(v_signed_group);
    const HVX_VectorPair v_quant_h =
        Q6_Wh_vunpack_Vb(v_two_row_groups);
    const HVX_Vector v_row01_f16 =
        Q6_Vhf_equals_Vh(Q6_V_lo_W(v_quant_h));
    const HVX_Vector v_row23_f16 =
        Q6_Vhf_equals_Vh(Q6_V_hi_W(v_quant_h));
    return Q6_W_vcombine_VV(v_row23_f16, v_row01_f16);
}

__attribute__((noinline)) void qbh_expand_w4_to_f16_hvx(
    const uint8_t *packed_w4, const float *channel_scales,
    void *expanded_f16, uint32_t k_tiles) {
    const HVX_Vector v_lut = *(const HVX_Vector *)qbh_signed_w4_lut;
    const HVX_Vector v_nibble_mask = Q6_Vb_vsplat_R(0x0f);
    const HVX_Vector v_scale = *(const HVX_Vector *)channel_scales;
    HVX_Vector *destination = (HVX_Vector *)expanded_f16;

    /* The FP16 Crouton carrier stores two adjacent K rows per output.  Each
     * widening multiply separates the even and odd halfword lanes, so both
     * qf32 halves use the same naturally ordered 32-channel SF scale vector.
     * Keeping the source scale in SF preserves the scalar contract
     * half(q * scale_f32).  The vectorization pattern is adapted from
     * haozixu/htp-ops-lib@85eb88e, src/dsp/ops/mat_mul.c. */
    for (uint32_t packed_vector = 0;
         packed_vector < k_tiles * 4U; ++packed_vector) {
        const HVX_Vector v_packed = *(const HVX_Vector *)(
            packed_w4 + (size_t)packed_vector * sizeof(HVX_Vector));
        const HVX_Vector v_low_indices =
            Q6_V_vand_VV(v_packed, v_nibble_mask);
        const HVX_Vector v_high_indices =
            Q6_Vub_vlsr_VubR(v_packed, 4);
        const HVX_Vector v_low = Q6_Vb_vlut32_VbVbR_nomatch(
            v_low_indices, v_lut, 0);
        const HVX_Vector v_high = Q6_Vb_vlut32_VbVbR_nomatch(
            v_high_indices, v_lut, 0);
        const HVX_VectorPair v_unpacked =
            Q6_W_vshuff_VVR(v_high, v_low, -1);
        const HVX_VectorPair v_groups0 = qbh_scale_w4_f16_group(
            Q6_V_lo_W(v_unpacked), v_scale);
        const HVX_VectorPair v_groups1 = qbh_scale_w4_f16_group(
            Q6_V_hi_W(v_unpacked), v_scale);
        destination[packed_vector * 4U] = Q6_V_lo_W(v_groups0);
        destination[packed_vector * 4U + 1U] = Q6_V_hi_W(v_groups0);
        destination[packed_vector * 4U + 2U] = Q6_V_lo_W(v_groups1);
        destination[packed_vector * 4U + 3U] = Q6_V_hi_W(v_groups1);
    }
    asm volatile("barrier" : : : "memory");
}

__attribute__((noinline)) void qbh_unpack_w4_to_f16_hvx(
    const uint8_t *packed_w4, void *expanded_f16, uint32_t k_tiles) {
    const HVX_Vector v_lut = *(const HVX_Vector *)qbh_signed_w4_lut;
    const HVX_Vector v_nibble_mask = Q6_Vb_vsplat_R(0x0f);
    HVX_Vector *destination = (HVX_Vector *)expanded_f16;

    /* Per-output-channel scales are applied by the FP16 HMX output-scale
     * unit.  The hot expansion loop therefore only decodes signed W4 and
     * converts the integer carrier to FP16 Crouton. */
    for (uint32_t packed_vector = 0;
         packed_vector < k_tiles * 4U; ++packed_vector) {
        const HVX_Vector v_packed = *(const HVX_Vector *)(
            packed_w4 + (size_t)packed_vector * sizeof(HVX_Vector));
        const HVX_Vector v_low_indices =
            Q6_V_vand_VV(v_packed, v_nibble_mask);
        const HVX_Vector v_high_indices =
            Q6_Vub_vlsr_VubR(v_packed, 4);
        const HVX_Vector v_low = Q6_Vb_vlut32_VbVbR_nomatch(
            v_low_indices, v_lut, 0);
        const HVX_Vector v_high = Q6_Vb_vlut32_VbVbR_nomatch(
            v_high_indices, v_lut, 0);
        const HVX_VectorPair v_unpacked =
            Q6_W_vshuff_VVR(v_high, v_low, -1);
        const HVX_VectorPair v_groups0 = qbh_unpack_w4_f16_group(
            Q6_V_lo_W(v_unpacked));
        const HVX_VectorPair v_groups1 = qbh_unpack_w4_f16_group(
            Q6_V_hi_W(v_unpacked));
        destination[packed_vector * 4U] = Q6_V_lo_W(v_groups0);
        destination[packed_vector * 4U + 1U] = Q6_V_hi_W(v_groups0);
        destination[packed_vector * 4U + 2U] = Q6_V_lo_W(v_groups1);
        destination[packed_vector * 4U + 3U] = Q6_V_hi_W(v_groups1);
    }
    asm volatile("barrier" : : : "memory");
}

uint32_t qbh_audit_w4_to_f16_tile(
    const uint8_t *packed_w4, const float *channel_scales,
    const void *expanded_f16, uint32_t *first_logical_index,
    uint32_t *expected_half_bits, uint32_t *actual_half_bits) {
    const uint16_t *actual = (const uint16_t *)expanded_f16;
    uint32_t mismatches = 0U;

    for (uint32_t input = 0; input < 32U; ++input) {
        for (uint32_t output = 0; output < 32U; ++output) {
            uint32_t physical =
                ((input / 4U) * 32U + output) * 4U + input % 4U;
            uint8_t byte = packed_w4[physical / 2U];
            uint8_t nibble = (physical & 1U) != 0U
                                 ? (uint8_t)(byte >> 4U)
                                 : (uint8_t)(byte & UINT8_C(0x0f));
            int8_t quant = (nibble & UINT8_C(0x08)) != 0U
                               ? (int8_t)(nibble | UINT8_C(0xf0))
                               : (int8_t)nibble;
            __fp16 expected =
                (__fp16)((float)quant * channel_scales[output]);
            uint16_t expected_bits;
            uint32_t target =
                ((input / 2U) * 32U + output) * 2U + input % 2U;
            memcpy(&expected_bits, &expected, sizeof(expected_bits));
            if (actual[target] != expected_bits) {
                if (mismatches == 0U) {
                    *first_logical_index = input * 32U + output;
                    *expected_half_bits = expected_bits;
                    *actual_half_bits = actual[target];
                }
                ++mismatches;
            }
        }
    }
    return mismatches;
}

uint32_t qbh_audit_unscaled_w4_to_f16_tile(
    const uint8_t *packed_w4, const void *expanded_f16,
    uint32_t *first_logical_index, uint32_t *expected_half_bits,
    uint32_t *actual_half_bits) {
    const uint16_t *actual = (const uint16_t *)expanded_f16;
    uint32_t mismatches = 0U;

    for (uint32_t input = 0; input < 32U; ++input) {
        for (uint32_t output = 0; output < 32U; ++output) {
            uint32_t physical =
                ((input / 4U) * 32U + output) * 4U + input % 4U;
            uint8_t byte = packed_w4[physical / 2U];
            uint8_t nibble = (physical & 1U) != 0U
                                 ? (uint8_t)(byte >> 4U)
                                 : (uint8_t)(byte & UINT8_C(0x0f));
            int8_t quant = (nibble & UINT8_C(0x08)) != 0U
                               ? (int8_t)(nibble | UINT8_C(0xf0))
                               : (int8_t)nibble;
            __fp16 expected = (__fp16)quant;
            uint16_t expected_bits;
            uint32_t target =
                ((input / 2U) * 32U + output) * 2U + input % 2U;
            memcpy(&expected_bits, &expected, sizeof(expected_bits));
            if (actual[target] != expected_bits) {
                if (mismatches == 0U) {
                    *first_logical_index = input * 32U + output;
                    *expected_half_bits = expected_bits;
                    *actual_half_bits = actual[target];
                }
                ++mismatches;
            }
        }
    }
    return mismatches;
}

__attribute__((noinline)) void qbh_expand_w4_to_s8_hvx(
    const uint8_t *packed_w4, const uint8_t *channel_scales,
    int8_t *expanded_s8, uint32_t k_tiles) {
    int8_t repeated_scales[128] __attribute__((aligned(128)));
    const HVX_Vector v_lut = *(const HVX_Vector *)qbh_signed_w4_lut;
    const HVX_Vector v_nibble_mask = Q6_Vb_vsplat_R(0x0f);

    for (uint32_t output_channel = 0;
         output_channel < QBH_HMX_OUTPUT_CHANNELS; ++output_channel) {
        for (uint32_t lane = 0; lane < 4U; ++lane) {
            repeated_scales[output_channel * 4U + lane] =
                (int8_t)channel_scales[output_channel];
        }
    }
    const HVX_Vector v_scales = *(const HVX_Vector *)repeated_scales;

    for (uint32_t packed_vector = 0;
         packed_vector < k_tiles * 4U; ++packed_vector) {
        const HVX_Vector v_packed = *(const HVX_Vector *)(
            packed_w4 + (size_t)packed_vector * sizeof(HVX_Vector));
        const HVX_Vector v_low_indices =
            Q6_V_vand_VV(v_packed, v_nibble_mask);
        const HVX_Vector v_high_indices =
            Q6_Vub_vlsr_VubR(v_packed, 4);
        const HVX_Vector v_low = Q6_Vb_vlut32_VbVbR_nomatch(
            v_low_indices, v_lut, 0);
        const HVX_Vector v_high = Q6_Vb_vlut32_VbVbR_nomatch(
            v_high_indices, v_lut, 0);
        const HVX_VectorPair v_unpacked =
            Q6_W_vshuff_VVR(v_high, v_low, -1);
        const HVX_Vector v_q0 = Q6_V_lo_W(v_unpacked);
        const HVX_Vector v_q1 = Q6_V_hi_W(v_unpacked);
        const HVX_VectorPair v_product0 =
            Q6_Wh_vmpy_VbVb(v_q0, v_scales);
        const HVX_VectorPair v_product1 =
            Q6_Wh_vmpy_VbVb(v_q1, v_scales);
        /* Byte multiply separates even and odd inputs into two halfword
         * vectors.  Re-interleave halfwords before narrowing back to the
         * original HMX carrier order. */
        const HVX_VectorPair v_product0_interleaved =
            Q6_W_vshuff_VVR(Q6_V_hi_W(v_product0),
                            Q6_V_lo_W(v_product0), -2);
        const HVX_VectorPair v_product1_interleaved =
            Q6_W_vshuff_VVR(Q6_V_hi_W(v_product1),
                            Q6_V_lo_W(v_product1), -2);
        const HVX_Vector v_scaled0 = Q6_Vb_vpack_VhVh_sat(
            Q6_V_hi_W(v_product0_interleaved),
            Q6_V_lo_W(v_product0_interleaved));
        const HVX_Vector v_scaled1 = Q6_Vb_vpack_VhVh_sat(
            Q6_V_hi_W(v_product1_interleaved),
            Q6_V_lo_W(v_product1_interleaved));
        HVX_Vector *destination = (HVX_Vector *)(
            expanded_s8 + (size_t)packed_vector * 2U *
                              sizeof(HVX_Vector));
        destination[0] = v_scaled0;
        destination[1] = v_scaled1;
    }
    asm volatile("barrier" : : : "memory");
}

__attribute__((noinline)) void qbh_unpack_w4_to_s8_hvx(
    const uint8_t *packed_w4, int8_t *expanded_s8,
    uint32_t k_tiles) {
    const HVX_Vector v_lut = *(const HVX_Vector *)qbh_signed_w4_lut;
    const HVX_Vector v_nibble_mask = Q6_Vb_vsplat_R(0x0f);

    for (uint32_t packed_vector = 0;
         packed_vector < k_tiles * 4U; ++packed_vector) {
        const HVX_Vector v_packed = *(const HVX_Vector *)(
            packed_w4 + (size_t)packed_vector * sizeof(HVX_Vector));
        const HVX_Vector v_low_indices =
            Q6_V_vand_VV(v_packed, v_nibble_mask);
        const HVX_Vector v_high_indices =
            Q6_Vub_vlsr_VubR(v_packed, 4);
        const HVX_Vector v_low = Q6_Vb_vlut32_VbVbR_nomatch(
            v_low_indices, v_lut, 0);
        const HVX_Vector v_high = Q6_Vb_vlut32_VbVbR_nomatch(
            v_high_indices, v_lut, 0);
        const HVX_VectorPair v_unpacked =
            Q6_W_vshuff_VVR(v_high, v_low, -1);
        HVX_Vector *destination = (HVX_Vector *)(
            expanded_s8 + (size_t)packed_vector * 2U *
                              sizeof(HVX_Vector));
        destination[0] = Q6_V_lo_W(v_unpacked);
        destination[1] = Q6_V_hi_W(v_unpacked);
    }
    asm volatile("barrier" : : : "memory");
}

__attribute__((noinline)) void qbh_copy_hmx_bias_hvx(
    const uint8_t *source, uint8_t *destination) {
    const HVX_Vector *source_vectors = (const HVX_Vector *)source;
    HVX_Vector *destination_vectors = (HVX_Vector *)destination;

    destination_vectors[0] = source_vectors[0];
    destination_vectors[1] = source_vectors[1];
    asm volatile("barrier" : : : "memory");
}
