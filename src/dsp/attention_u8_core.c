#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "attention_u8_core.h"
#include "hmx_int8_tile.h"

#define QBH_ATTN_U8_HVX_BYTES UINT32_C(128)
#define QBH_ATTN_U8_SCORE_ZP INT32_C(128)
#define QBH_ATTN_U8_EXP_FRAC_BITS UINT32_C(15)

static const uint8_t qbh_attention_u8_lane_index[QBH_ATTN_U8_HVX_BYTES]
    __attribute__((aligned(QBH_ATTN_U8_HVX_BYTES))) = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
        48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
        64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
        80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95,
        96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
        110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122,
        123, 124, 125, 126, 127,
    };

/* One word holds the four adjacent K lanes consumed by an integer-HMX
 * weight tile.  Consecutive words for one logical K row are 128 bytes apart
 * in the packed carrier. */
static const int32_t qbh_attention_u8_vscatter_offsets[32]
    __attribute__((aligned(QBH_ATTN_U8_HVX_BYTES))) = {
        0, 128, 256, 384, 512, 640, 768, 896,
        1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920,
        2048, 2176, 2304, 2432, 2560, 2688, 2816, 2944,
        3072, 3200, 3328, 3456, 3584, 3712, 3840, 3968,
    };

static uint16_t qbh_attention_u8_float_to_half_bits(float value) {
    __fp16 converted = (__fp16)value;
    uint16_t bits;
    memcpy(&bits, &converted, sizeof(bits));
    return bits;
}

static int32_t qbh_attention_u8_round_div_signed(
    int32_t numerator, int32_t denominator) {
    if (numerator >= 0) {
        return (numerator + denominator / 2) / denominator;
    }
    return -((-numerator + denominator / 2) / denominator);
}

static int32_t qbh_attention_u8_clip_s8(
    int32_t value, uint32_t *saturation_count) {
    if (value < INT8_MIN) {
        if (saturation_count != NULL) {
            ++*saturation_count;
        }
        return INT8_MIN;
    }
    if (value > INT8_MAX) {
        if (saturation_count != NULL) {
            ++*saturation_count;
        }
        return INT8_MAX;
    }
    return value;
}

static uint8_t qbh_attention_u8_clip_u8(int32_t value) {
    if (value < 0) {
        return 0U;
    }
    if (value > UINT8_MAX) {
        return UINT8_MAX;
    }
    return (uint8_t)value;
}

static HVX_Vector qbh_attention_u8_center_u8_to_s8(
    HVX_Vector value, int32_t zero_point) {
    const HVX_VectorPair unpacked = Q6_Wuh_vunpack_Vub(value);
    const HVX_Vector offset = Q6_Vh_vsplat_R(zero_point);
    const HVX_Vector low = Q6_Vh_vsub_VhVh(
        Q6_V_lo_W(unpacked), offset);
    const HVX_Vector high = Q6_Vh_vsub_VhVh(
        Q6_V_hi_W(unpacked), offset);
    return Q6_Vb_vpack_VhVh_sat(high, low);
}

static int32_t qbh_attention_u8_sum_signed_bytes(HVX_Vector value) {
    int32_t lanes[32] __attribute__((aligned(QBH_ATTN_U8_HVX_BYTES)));
    const HVX_VectorPair halves = Q6_Wh_vunpack_Vb(value);
    const HVX_VectorPair words0 =
        Q6_Ww_vunpack_Vh(Q6_V_lo_W(halves));
    const HVX_VectorPair words1 =
        Q6_Ww_vunpack_Vh(Q6_V_hi_W(halves));
    const HVX_Vector sum = Q6_Vw_vadd_VwVw(
        Q6_Vw_vadd_VwVw(
            Q6_V_lo_W(words0), Q6_V_hi_W(words0)),
        Q6_Vw_vadd_VwVw(
            Q6_V_lo_W(words1), Q6_V_hi_W(words1)));
    int32_t result = 0;

    *(HVX_Vector *)lanes = sum;
    for (uint32_t lane = 0U; lane < 32U; ++lane) {
        result += lanes[lane];
    }
    return result;
}

void qbh_attention_u8_pack_k_native(
    const uint8_t *k_head_tiles,
    const struct qbh_attention_config *config,
    int8_t *weight_tiles, uint32_t *bias_words) {
    const uint32_t divisor = UINT32_C(1) << config->score_shift;
    const int32_t rounding = config->score_shift == 0U
                                 ? 0
                                 : (int32_t)(divisor / 2U);
    const uint16_t conversion = qbh_attention_u8_float_to_half_bits(
        512.0f / (float)divisor);

    uint8_t row[QBH_ATTN_U8_HVX_BYTES]
        __attribute__((aligned(QBH_ATTN_U8_HVX_BYTES)));
    const HVX_Vector offsets_base =
        *(const HVX_Vector *)qbh_attention_u8_vscatter_offsets;

    for (uint32_t n_tile = 0U;
         n_tile < QBH_ATTENTION_SCORE_TILES; ++n_tile) {
        int32_t sums[QBH_HMX_OUTPUT_CHANNELS] = {0};
        int8_t *destination = weight_tiles +
            (size_t)n_tile * QBH_ATTENTION_HEAD_DIM_TILES *
                QBH_HMX_WEIGHT_BYTES;

        for (uint32_t output = 0U;
             output < QBH_HMX_OUTPUT_CHANNELS; ++output) {
            const uint32_t token =
                n_tile * QBH_HMX_OUTPUT_CHANNELS + output;
            const HVX_Vector offsets = Q6_Vw_vadd_VwVw(
                offsets_base, Q6_V_vsplat_R(output * 4U));
            for (uint32_t k_tile = 0U;
                 k_tile < QBH_ATTENTION_HEAD_DIM_TILES; ++k_tile) {
                memcpy(
                    row + k_tile * QBH_HMX_INPUT_CHANNELS,
                    k_head_tiles +
                        (size_t)k_tile * QBH_HMX_ACTIVATION_BYTES +
                        (size_t)token * QBH_HMX_INPUT_CHANNELS,
                    QBH_HMX_INPUT_CHANNELS);
            }
            {
                const HVX_Vector centered =
                    qbh_attention_u8_center_u8_to_s8(
                        *(const HVX_Vector *)row,
                        config->k_zero_point);
                sums[output] =
                    qbh_attention_u8_sum_signed_bytes(centered);
                Q6_vscatter_RMVwV(
                    (uint32_t)(uintptr_t)destination,
                    QBH_ATTENTION_HEAD_DIM_TILES *
                            QBH_HMX_WEIGHT_BYTES -
                        1U,
                    offsets, centered);
            }
        }
        {
            uint32_t *bias = bias_words +
                (size_t)n_tile *
                    (QBH_HMX_BIAS_BYTES / sizeof(uint32_t));
            for (uint32_t output = 0U;
                 output < QBH_HMX_OUTPUT_CHANNELS; ++output) {
                bias[output] = conversion;
                bias[QBH_HMX_OUTPUT_CHANNELS + output] =
                    (uint32_t)(-config->q_zero_point * sums[output] +
                               QBH_ATTN_U8_SCORE_ZP * (int32_t)divisor +
                               rounding);
            }
        }
    }
    asm volatile("barrier" ::: "memory");
}

void qbh_attention_u8_pack_v_native(
    const uint8_t *v_head_tiles,
    const struct qbh_attention_config *config,
    int8_t *weight_tiles, uint32_t *bias_words,
    uint32_t *saturation_count) {
    const uint32_t divisor = UINT32_C(1) << config->av_shift;
    const int32_t rounding = config->av_shift == 0U
                                 ? 0
                                 : (int32_t)(divisor / 2U);
    const uint16_t conversion = qbh_attention_u8_float_to_half_bits(
        512.0f / (float)divisor);
    const int32_t hmx_output_zero_point =
        config->av_multiplier == 1U
            ? config->output_zero_point
            : (int32_t)QBH_ATTENTION_HMX_CENTER;

    int16_t recenter_lut[256]
        __attribute__((aligned(QBH_ATTN_U8_HVX_BYTES)));
    uint8_t recentered[QBH_ATTN_U8_HVX_BYTES]
        __attribute__((aligned(QBH_ATTN_U8_HVX_BYTES)));

    for (uint32_t code = 0U; code <= UINT8_MAX; ++code) {
        const int32_t centered =
            (int32_t)code - config->v_zero_point;
        const int32_t requantized =
            qbh_attention_u8_round_div_signed(
                centered * (int32_t)config->v_recenter_numerator,
                (int32_t)config->v_recenter_denominator);
        recenter_lut[code] = (int16_t)qbh_attention_u8_clip_s8(
            requantized, NULL);
    }

    for (uint32_t n_tile = 0U;
         n_tile < QBH_ATTENTION_HEAD_DIM_TILES; ++n_tile) {
        const uint8_t *source = v_head_tiles +
            (size_t)n_tile * QBH_HMX_ACTIVATION_BYTES;
        for (uint32_t k_tile = 0U;
             k_tile < QBH_ATTENTION_SCORE_TILES; ++k_tile) {
            int8_t *destination = weight_tiles +
                ((size_t)n_tile * QBH_ATTENTION_SCORE_TILES + k_tile) *
                    QBH_HMX_WEIGHT_BYTES;
            for (uint32_t input_group = 0U;
                 input_group < QBH_HMX_INPUT_CHANNELS / 4U;
                 ++input_group) {
                const uint32_t token =
                    k_tile * QBH_HMX_INPUT_CHANNELS +
                    input_group * 4U;
                const HVX_Vector values = *(const HVX_Vector *)(
                    source +
                    (size_t)token * QBH_HMX_OUTPUT_CHANNELS);
                *(HVX_Vector *)recentered = values;
                for (uint32_t index = 0U;
                     index < QBH_ATTN_U8_HVX_BYTES; ++index) {
                    recentered[index] = (uint8_t)
                        recenter_lut[recentered[index]];
                }

                if (saturation_count != NULL) {
                    for (uint32_t index = 0U;
                         index < QBH_ATTN_U8_HVX_BYTES; ++index) {
                        const int32_t centered =
                            (int32_t)((const uint8_t *)&values)[index] -
                            config->v_zero_point;
                        const int32_t requantized =
                            qbh_attention_u8_round_div_signed(
                                centered * (int32_t)
                                    config->v_recenter_numerator,
                                (int32_t)
                                    config->v_recenter_denominator);
                        *saturation_count +=
                            requantized < INT8_MIN ||
                            requantized > INT8_MAX;
                    }
                }
                for (uint32_t output = 0U;
                     output < QBH_HMX_OUTPUT_CHANNELS; ++output) {
                    const uint8_t byte0 = recentered[output];
                    const uint8_t byte1 =
                        recentered[QBH_HMX_OUTPUT_CHANNELS + output];
                    const uint8_t byte2 =
                        recentered[2U * QBH_HMX_OUTPUT_CHANNELS + output];
                    const uint8_t byte3 =
                        recentered[3U * QBH_HMX_OUTPUT_CHANNELS + output];
                    ((uint32_t *)(destination +
                         (size_t)input_group * 128U))[output] =
                        (uint32_t)byte0 |
                        ((uint32_t)byte1 << 8U) |
                        ((uint32_t)byte2 << 16U) |
                        ((uint32_t)byte3 << 24U);
                }
            }
        }
        {
            uint32_t *bias = bias_words +
                (size_t)n_tile *
                    (QBH_HMX_BIAS_BYTES / sizeof(uint32_t));
            for (uint32_t output = 0U;
                 output < QBH_HMX_OUTPUT_CHANNELS; ++output) {
                bias[output] = conversion;
                bias[QBH_HMX_OUTPUT_CHANNELS + output] =
                    (uint32_t)(hmx_output_zero_point * (int32_t)divisor +
                               rounding);
            }
        }
    }
    asm volatile("barrier" ::: "memory");
}

void qbh_attention_u8_pack_v_native_vgather(
    const uint8_t *v_head_tiles,
    const struct qbh_attention_config *config,
    int8_t *weight_tiles, uint32_t *bias_words, uint8_t *scratch,
    uint32_t *saturation_count) {
    const uint32_t divisor = UINT32_C(1) << config->av_shift;
    const int32_t rounding = config->av_shift == 0U
                                 ? 0
                                 : (int32_t)(divisor / 2U);
    const uint16_t conversion = qbh_attention_u8_float_to_half_bits(
        512.0f / (float)divisor);
    const int32_t hmx_output_zero_point =
        config->av_multiplier == 1U
            ? config->output_zero_point
            : (int32_t)QBH_ATTENTION_HMX_CENTER;

    int16_t *recenter_lut = (int16_t *)(
        scratch + QBH_ATTN_U8_VGATHER_LUT_OFFSET);
    int16_t *gathered_low = (int16_t *)(
        scratch + QBH_ATTN_U8_VGATHER_SCRATCH_OFFSET);
    int16_t *gathered_high = gathered_low + 64;

    for (uint32_t code = 0U; code <= UINT8_MAX; ++code) {
        const int32_t centered =
            (int32_t)code - config->v_zero_point;
        const int32_t requantized =
            qbh_attention_u8_round_div_signed(
                centered * (int32_t)config->v_recenter_numerator,
                (int32_t)config->v_recenter_denominator);
        recenter_lut[code] = (int16_t)qbh_attention_u8_clip_s8(
            requantized, NULL);
    }

    for (uint32_t n_tile = 0U;
         n_tile < QBH_ATTENTION_HEAD_DIM_TILES; ++n_tile) {
        const uint8_t *source = v_head_tiles +
            (size_t)n_tile * QBH_HMX_ACTIVATION_BYTES;
        for (uint32_t k_tile = 0U;
             k_tile < QBH_ATTENTION_SCORE_TILES; ++k_tile) {
            int8_t *destination = weight_tiles +
                ((size_t)n_tile * QBH_ATTENTION_SCORE_TILES + k_tile) *
                    QBH_HMX_WEIGHT_BYTES;
            for (uint32_t input_group = 0U;
                 input_group < QBH_HMX_INPUT_CHANNELS / 4U;
                 ++input_group) {
                const uint32_t token =
                    k_tile * QBH_HMX_INPUT_CHANNELS +
                    input_group * 4U;
                const HVX_Vector values = *(const HVX_Vector *)(
                    source +
                    (size_t)token * QBH_HMX_OUTPUT_CHANNELS);
                const HVX_VectorPair value_h =
                    Q6_Wuh_vunpack_Vub(values);
                const HVX_Vector offsets_low = Q6_Vh_vasl_VhR(
                    Q6_V_lo_W(value_h), 1);
                const HVX_Vector offsets_high = Q6_Vh_vasl_VhR(
                    Q6_V_hi_W(value_h), 1);
                const HVX_VectorPred all_lanes = Q6_Q_vcmp_eq_VwVw(
                    offsets_low, offsets_low);
                HVX_Vector recentered;

                Q6_vgather_AQRMVh(
                    gathered_low, all_lanes,
                    (int32_t)(uintptr_t)recenter_lut,
                    QBH_ATTN_U8_VGATHER_LUT_BYTES - 1U,
                    offsets_low);
                Q6_vgather_AQRMVh(
                    gathered_high, all_lanes,
                    (int32_t)(uintptr_t)recenter_lut,
                    QBH_ATTN_U8_VGATHER_LUT_BYTES - 1U,
                    offsets_high);
                recentered = Q6_Vb_vpack_VhVh_sat(
                    *(volatile HVX_Vector *)gathered_high,
                    *(volatile HVX_Vector *)gathered_low);

                if (saturation_count != NULL) {
                    for (uint32_t index = 0U;
                         index < QBH_ATTN_U8_HVX_BYTES; ++index) {
                        const int32_t centered =
                            (int32_t)((const uint8_t *)&values)[index] -
                            config->v_zero_point;
                        const int32_t requantized =
                            qbh_attention_u8_round_div_signed(
                                centered * (int32_t)
                                    config->v_recenter_numerator,
                                (int32_t)
                                    config->v_recenter_denominator);
                        *saturation_count +=
                            requantized < INT8_MIN ||
                            requantized > INT8_MAX;
                    }
                }
                for (uint32_t output = 0U;
                     output < QBH_HMX_OUTPUT_CHANNELS; ++output) {
                    const uint8_t *bytes =
                        (const uint8_t *)&recentered;
                    const uint8_t byte0 = bytes[output];
                    const uint8_t byte1 =
                        bytes[QBH_HMX_OUTPUT_CHANNELS + output];
                    const uint8_t byte2 =
                        bytes[2U * QBH_HMX_OUTPUT_CHANNELS + output];
                    const uint8_t byte3 =
                        bytes[3U * QBH_HMX_OUTPUT_CHANNELS + output];
                    ((uint32_t *)(destination +
                         (size_t)input_group * 128U))[output] =
                        (uint32_t)byte0 |
                        ((uint32_t)byte1 << 8U) |
                        ((uint32_t)byte2 << 16U) |
                        ((uint32_t)byte3 << 24U);
                }
            }
        }
        {
            uint32_t *bias = bias_words +
                (size_t)n_tile *
                    (QBH_HMX_BIAS_BYTES / sizeof(uint32_t));
            for (uint32_t output = 0U;
                 output < QBH_HMX_OUTPUT_CHANNELS; ++output) {
                bias[output] = conversion;
                bias[QBH_HMX_OUTPUT_CHANNELS + output] =
                    (uint32_t)(hmx_output_zero_point *
                                   (int32_t)divisor +
                               rounding);
            }
        }
    }
    asm volatile("barrier" ::: "memory");
}

void qbh_attention_u8_pack_v_native_vgather_vdeal(
    const uint8_t *v_head_tiles,
    const struct qbh_attention_config *config,
    int8_t *weight_tiles, uint32_t *bias_words, uint8_t *scratch,
    uint32_t *saturation_count) {
    const uint32_t divisor = UINT32_C(1) << config->av_shift;
    const int32_t rounding = config->av_shift == 0U
                                 ? 0
                                 : (int32_t)(divisor / 2U);
    const uint16_t conversion = qbh_attention_u8_float_to_half_bits(
        512.0f / (float)divisor);
    const int32_t hmx_output_zero_point =
        config->av_multiplier == 1U
            ? config->output_zero_point
            : (int32_t)QBH_ATTENTION_HMX_CENTER;
    int16_t *recenter_lut = (int16_t *)(
        scratch + QBH_ATTN_U8_VGATHER_LUT_OFFSET);
    int16_t *gathered_low = (int16_t *)(
        scratch + QBH_ATTN_U8_VGATHER_SCRATCH_OFFSET);
    int16_t *gathered_high = gathered_low + 64;

    for (uint32_t code = 0U; code <= UINT8_MAX; ++code) {
        const int32_t centered =
            (int32_t)code - config->v_zero_point;
        const int32_t requantized =
            qbh_attention_u8_round_div_signed(
                centered * (int32_t)config->v_recenter_numerator,
                (int32_t)config->v_recenter_denominator);
        recenter_lut[code] = (int16_t)qbh_attention_u8_clip_s8(
            requantized, NULL);
    }

    for (uint32_t n_tile = 0U;
         n_tile < QBH_ATTENTION_HEAD_DIM_TILES; ++n_tile) {
        const uint8_t *source = v_head_tiles +
            (size_t)n_tile * QBH_HMX_ACTIVATION_BYTES;
        for (uint32_t k_tile = 0U;
             k_tile < QBH_ATTENTION_SCORE_TILES; ++k_tile) {
            int8_t *destination = weight_tiles +
                ((size_t)n_tile * QBH_ATTENTION_SCORE_TILES + k_tile) *
                    QBH_HMX_WEIGHT_BYTES;
            for (uint32_t input_group = 0U;
                 input_group < QBH_HMX_INPUT_CHANNELS / 4U;
                 ++input_group) {
                const uint32_t token =
                    k_tile * QBH_HMX_INPUT_CHANNELS +
                    input_group * 4U;
                const HVX_Vector values = *(const HVX_Vector *)(
                    source +
                    (size_t)token * QBH_HMX_OUTPUT_CHANNELS);
                const HVX_VectorPair value_h =
                    Q6_Wuh_vunpack_Vub(values);
                const HVX_Vector offsets_low = Q6_Vh_vasl_VhR(
                    Q6_V_lo_W(value_h), 1);
                const HVX_Vector offsets_high = Q6_Vh_vasl_VhR(
                    Q6_V_hi_W(value_h), 1);
                const HVX_VectorPred all_lanes = Q6_Q_vcmp_eq_VwVw(
                    offsets_low, offsets_low);
                HVX_Vector recentered;

                Q6_vgather_AQRMVh(
                    gathered_low, all_lanes,
                    (int32_t)(uintptr_t)recenter_lut,
                    QBH_ATTN_U8_VGATHER_LUT_BYTES - 1U,
                    offsets_low);
                Q6_vgather_AQRMVh(
                    gathered_high, all_lanes,
                    (int32_t)(uintptr_t)recenter_lut,
                    QBH_ATTN_U8_VGATHER_LUT_BYTES - 1U,
                    offsets_high);
                recentered = Q6_Vb_vpack_VhVh_sat(
                    *(volatile HVX_Vector *)gathered_high,
                    *(volatile HVX_Vector *)gathered_low);

                if (saturation_count != NULL) {
                    for (uint32_t index = 0U;
                         index < QBH_ATTN_U8_HVX_BYTES; ++index) {
                        const int32_t centered =
                            (int32_t)((const uint8_t *)&values)[index] -
                            config->v_zero_point;
                        const int32_t requantized =
                            qbh_attention_u8_round_div_signed(
                                centered * (int32_t)
                                    config->v_recenter_numerator,
                                (int32_t)
                                    config->v_recenter_denominator);
                        *saturation_count +=
                            requantized < INT8_MIN ||
                            requantized > INT8_MAX;
                    }
                }
                recentered = Q6_Vb_vdeal_Vb(recentered);
                recentered = Q6_Vb_vdeal_Vb(recentered);
                recentered = Q6_Vb_vdeal_Vb(recentered);
                recentered = Q6_Vb_vdeal_Vb(recentered);
                recentered = Q6_Vb_vdeal_Vb(recentered);
                *(HVX_Vector *)(destination +
                    (size_t)input_group * sizeof(HVX_Vector)) =
                    recentered;
            }
        }
        {
            uint32_t *bias = bias_words +
                (size_t)n_tile *
                    (QBH_HMX_BIAS_BYTES / sizeof(uint32_t));
            for (uint32_t output = 0U;
                 output < QBH_HMX_OUTPUT_CHANNELS; ++output) {
                bias[output] = conversion;
                bias[QBH_HMX_OUTPUT_CHANNELS + output] =
                    (uint32_t)(hmx_output_zero_point *
                                   (int32_t)divisor +
                               rounding);
            }
        }
    }
    asm volatile("barrier" ::: "memory");
}

static HVX_Vector qbh_attention_u8_requant_centered(
    HVX_Vector input, uint32_t multiplier,
    int32_t output_zero_point) {
    const HVX_Vector centered = Q6_V_vxor_VV(
        input, Q6_Vb_vsplat_R(0x80));
    const HVX_VectorPair product = Q6_Wh_vmpy_VbVb(
        centered, Q6_Vb_vsplat_R(multiplier));
    const HVX_VectorPair interleaved = Q6_W_vshuff_VVR(
        Q6_V_hi_W(product), Q6_V_lo_W(product), -2);
    const HVX_Vector output_zp = Q6_Vh_vsplat_R(output_zero_point);
    const HVX_Vector low = Q6_Vh_vadd_VhVh(
        Q6_V_lo_W(interleaved), output_zp);
    const HVX_Vector high = Q6_Vh_vadd_VhVh(
        Q6_V_hi_W(interleaved), output_zp);
    return Q6_Vub_vpack_VhVh_sat(high, low);
}

void qbh_attention_u8_requant_qk(
    uint8_t *score_tiles,
    const struct qbh_attention_config *config,
    uint32_t *saturation_count) {
    const uint32_t bytes =
        QBH_ATTENTION_Q_HEADS_PER_GROUP *
        QBH_ATTENTION_SCORE_TILES * QBH_HMX_OUTPUT_BYTES;
    if (saturation_count != NULL) {
        for (uint32_t offset = 0U; offset < bytes; ++offset) {
            const int32_t centered =
                (int32_t)score_tiles[offset] -
                (int32_t)QBH_ATTENTION_HMX_CENTER;
            const int32_t value =
                centered * (int32_t)config->score_multiplier +
                QBH_ATTN_U8_SCORE_ZP;
            *saturation_count += value < 0 || value > UINT8_MAX;
        }
    }
    for (uint32_t offset = 0U; offset < bytes;
         offset += sizeof(HVX_Vector)) {
        HVX_Vector *score = (HVX_Vector *)(score_tiles + offset);
        *score = qbh_attention_u8_requant_centered(
            *score, config->score_multiplier,
            QBH_ATTN_U8_SCORE_ZP);
    }
    asm volatile("barrier" ::: "memory");
}

static uint32_t qbh_attention_u8_floor_log2(uint32_t value) {
    uint32_t result = 0U;
    while (value > 1U) {
        value >>= 1U;
        ++result;
    }
    return result;
}

static uint8_t qbh_attention_u8_reduce_max(HVX_Vector value) {
    const HVX_Vector zero = Q6_V_vzero();
    for (uint32_t shift = 64U; shift != 0U; shift >>= 1U) {
        value = Q6_Vub_vmax_VubVub(
            value, Q6_V_vlalign_VVR(value, zero, shift));
    }
    return (uint8_t)(Q6_R_vextract_VR(value, 124) >> 24U);
}

static uint32_t qbh_attention_u8_reduce_sum_words(HVX_Vector value) {
    const HVX_Vector zero = Q6_V_vzero();
    for (uint32_t shift = 64U; shift >= 4U; shift >>= 1U) {
        value = Q6_Vw_vadd_VwVw(
            value, Q6_V_vlalign_VVR(value, zero, shift));
    }
    return (uint32_t)Q6_R_vextract_VR(value, 124);
}

static uint32_t qbh_attention_u8_sum_log2_weights(HVX_Vector codes) {
    const HVX_VectorPair code_h = Q6_Wuh_vunpack_Vub(codes);
    const HVX_Vector exponent_h = Q6_Vh_vsub_VhVh(
        Q6_Vh_vsplat_R(15), Q6_V_lo_W(code_h));
    const HVX_Vector weights_h = Q6_Vh_vasl_VhVh(
        Q6_Vh_vsplat_R(1), exponent_h);
    const HVX_VectorPair weights_w = Q6_Wuw_vunpack_Vuh(weights_h);
    return qbh_attention_u8_reduce_sum_words(
        Q6_Vw_vadd_VwVw(
            Q6_V_lo_W(weights_w), Q6_V_hi_W(weights_w)));
}

static uint32_t qbh_attention_u8_sum_probability(
    HVX_Vector probability) {
    const HVX_VectorPair probability_h =
        Q6_Wuh_vunpack_Vub(probability);
    const HVX_VectorPair probability_w =
        Q6_Wuw_vunpack_Vuh(Q6_V_lo_W(probability_h));
    return qbh_attention_u8_reduce_sum_words(
        Q6_Vw_vadd_VwVw(
            Q6_V_lo_W(probability_w),
            Q6_V_hi_W(probability_w)));
}

static HVX_Vector qbh_attention_u8_log2_codes(
    HVX_Vector score, uint32_t row, uint32_t fraction_bits) {
    const HVX_Vector lane =
        *(const HVX_Vector *)qbh_attention_u8_lane_index;
    const HVX_VectorPred valid = Q6_Q_not_Q(
        Q6_Q_vcmp_gt_VubVub(lane, Q6_Vb_vsplat_R(row)));
    const HVX_Vector masked_score = Q6_V_vmux_QVV(
        valid, score, Q6_V_vzero());
    const uint8_t maximum = qbh_attention_u8_reduce_max(masked_score);
    const HVX_VectorPair score_h = Q6_Wuh_vunpack_Vub(masked_score);
    const HVX_Vector maximum_h = Q6_Vh_vsplat_R(maximum);
    const HVX_Vector rounding_h = Q6_Vh_vsplat_R(
        UINT32_C(1) << (fraction_bits - 1U));
    const HVX_Vector limit_h = Q6_Vh_vsplat_R(15);
    HVX_Vector code_lo = Q6_Vuh_vsub_VuhVuh_sat(
        maximum_h, Q6_V_lo_W(score_h));
    HVX_Vector code_hi = Q6_Vuh_vsub_VuhVuh_sat(
        maximum_h, Q6_V_hi_W(score_h));
    code_lo = Q6_Vh_vadd_VhVh(code_lo, rounding_h);
    code_hi = Q6_Vh_vadd_VhVh(code_hi, rounding_h);
    code_lo = Q6_Vuh_vlsr_VuhR(code_lo, fraction_bits);
    code_hi = Q6_Vuh_vlsr_VuhR(code_hi, fraction_bits);
    code_lo = Q6_Vuh_vmin_VuhVuh(code_lo, limit_h);
    code_hi = Q6_Vuh_vmin_VuhVuh(code_hi, limit_h);
    return Q6_V_vmux_QVV(
        valid, Q6_Vub_vpack_VhVh_sat(code_hi, code_lo),
        Q6_Vb_vsplat_R(16));
}

static void qbh_attention_u8_build_probability_lut(
    uint8_t *lut, uint32_t sum, uint32_t mode,
    uint32_t valid_count) {
    memset(lut, 0, QBH_ATTN_U8_HVX_BYTES);
    if (valid_count == 1U) {
        for (uint32_t code = 0U; code <= 15U; ++code) {
            lut[2U * code] = UINT8_MAX;
        }
        return;
    }
    if (mode == QBH_ATTENTION_DIVISION_EXACT) {
        for (uint32_t code = 0U; code <= 15U; ++code) {
            const uint32_t weight = UINT32_C(1) <<
                (QBH_ATTN_U8_EXP_FRAC_BITS - code);
            lut[2U * code] = (uint8_t)(
                ((uint64_t)weight * UINT8_MAX + sum / 2U) / sum);
        }
        return;
    }
    {
        const uint32_t leading = qbh_attention_u8_floor_log2(sum);
        const uint32_t next = (sum >> (leading - 1U)) & 1U;
        const uint32_t coefficient =
            mode == QBH_ATTENTION_DIVISION_SOLE
                ? (next != 0U ? 145U : 209U)
                : (next != 0U ? 171U : 256U);
        const uint64_t numerator =
            (uint64_t)UINT8_MAX * coefficient;
        const uint32_t base_shift =
            8U + leading - QBH_ATTN_U8_EXP_FRAC_BITS;
        for (uint32_t code = 0U; code <= 15U; ++code) {
            const uint32_t shift = base_shift + code;
            lut[2U * code] = qbh_attention_u8_clip_u8(
                (int32_t)((numerator +
                           (UINT64_C(1) << (shift - 1U))) >> shift));
        }
    }
}

void qbh_attention_u8_softmax_group(
    const uint8_t *score_tiles, uint8_t *probability_tiles,
    uint8_t *scratch,
    const struct qbh_attention_config *config,
    struct qbh_attention_u8_telemetry *telemetry) {
    uint8_t *row_scratch = scratch;
    uint8_t *lut = scratch + QBH_ATTN_U8_HVX_BYTES;
    uint32_t row_sum_min = UINT_MAX;
    uint32_t row_sum_max = 0U;

    for (uint32_t head = 0U;
         head < QBH_ATTENTION_Q_HEADS_PER_GROUP; ++head) {
        const uint8_t *score_base = score_tiles +
            (size_t)head * QBH_ATTENTION_SCORE_TILES *
                QBH_HMX_OUTPUT_BYTES;
        uint8_t *probability_base = probability_tiles +
            (size_t)head * QBH_ATTENTION_SCORE_TILES *
                QBH_HMX_ACTIVATION_BYTES;
        for (uint32_t row = 0U; row < QBH_ATTENTION_M; ++row) {
            const uint32_t valid_count = row + 1U;
            HVX_Vector codes;
            HVX_Vector probabilities;
            uint32_t probability_sum;

            memcpy(row_scratch,
                   score_base +
                       (size_t)row * QBH_HMX_OUTPUT_CHANNELS,
                   QBH_HMX_OUTPUT_CHANNELS);
            memcpy(row_scratch + QBH_HMX_OUTPUT_CHANNELS,
                   score_base + QBH_HMX_OUTPUT_BYTES +
                       (size_t)row * QBH_HMX_OUTPUT_CHANNELS,
                   QBH_HMX_OUTPUT_CHANNELS);
            codes = qbh_attention_u8_log2_codes(
                *(const HVX_Vector *)row_scratch, row,
                config->fraction_bits);
            qbh_attention_u8_build_probability_lut(
                lut, qbh_attention_u8_sum_log2_weights(codes),
                config->division_mode, valid_count);
            probabilities = Q6_Vb_vlut32_VbVbR_nomatch(
                codes, *(const HVX_Vector *)lut, 0);
            *(HVX_Vector *)row_scratch = probabilities;
            probability_sum =
                qbh_attention_u8_sum_probability(probabilities);
            memcpy(probability_base +
                       (size_t)row * QBH_HMX_INPUT_CHANNELS,
                   row_scratch, QBH_HMX_INPUT_CHANNELS);
            memcpy(probability_base + QBH_HMX_ACTIVATION_BYTES +
                       (size_t)row * QBH_HMX_INPUT_CHANNELS,
                   row_scratch + QBH_HMX_INPUT_CHANNELS,
                   QBH_HMX_INPUT_CHANNELS);
            if (probability_sum < row_sum_min) {
                row_sum_min = probability_sum;
            }
            if (probability_sum > row_sum_max) {
                row_sum_max = probability_sum;
            }
            if (telemetry != NULL) {
                const uint8_t *bytes = (const uint8_t *)&probabilities;
                for (uint32_t column = valid_count;
                     column < QBH_ATTENTION_M; ++column) {
                    telemetry->probability_mask_violation_count +=
                        bytes[column] != 0U;
                }
            }
        }
    }
    if (telemetry != NULL) {
        telemetry->probability_row_sum_min = row_sum_min;
        telemetry->probability_row_sum_max = row_sum_max;
    }
    asm volatile("barrier" ::: "memory");
}

void qbh_attention_u8_requant_av(
    uint8_t *output_tiles,
    const struct qbh_attention_config *config) {
    if (config->av_multiplier != 1U) {
        const uint32_t bytes =
            QBH_ATTENTION_Q_HEADS_PER_GROUP *
            QBH_ATTENTION_HEAD_DIM_TILES * QBH_HMX_OUTPUT_BYTES;
        for (uint32_t offset = 0U; offset < bytes;
             offset += sizeof(HVX_Vector)) {
            HVX_Vector *output = (HVX_Vector *)(output_tiles + offset);
            *output = qbh_attention_u8_requant_centered(
                *output, config->av_multiplier,
                config->output_zero_point);
        }
        asm volatile("barrier" ::: "memory");
    }
}
