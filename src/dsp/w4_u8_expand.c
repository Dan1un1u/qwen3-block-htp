#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include <stddef.h>
#include <stdint.h>

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

__attribute__((noinline)) void qbh_copy_hmx_bias_hvx(
    const uint8_t *source, uint8_t *destination) {
    const HVX_Vector *source_vectors = (const HVX_Vector *)source;
    HVX_Vector *destination_vectors = (HVX_Vector *)destination;

    destination_vectors[0] = source_vectors[0];
    destination_vectors[1] = source_vectors[1];
    asm volatile("barrier" : : : "memory");
}
