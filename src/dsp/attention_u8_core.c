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
#define QBH_ATTN_U8_LUT_LEADING_MIN UINT32_C(15)
#define QBH_ATTN_U8_LUT_LEADING_MAX UINT32_C(21)
#define QBH_ATTN_U8_LUT_TEMPLATE_BYTES UINT32_C(32)
#define QBH_ATTN_U8_LUT_TEMPLATE_COUNT UINT32_C(14)

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

static uint32_t qbh_attention_u8_sum_log2_weights_half(
    HVX_Vector code_h) {
    const HVX_Vector exponent_h = Q6_Vh_vsub_VhVh(
        Q6_Vh_vsplat_R(15), code_h);
    const HVX_Vector weights_h = Q6_Vh_vasl_VhVh(
        Q6_Vh_vsplat_R(1), exponent_h);
    const HVX_VectorPair weights_w = Q6_Wuw_vunpack_Vuh(weights_h);
    return qbh_attention_u8_reduce_sum_words(
        Q6_Vw_vadd_VwVw(
            Q6_V_lo_W(weights_w), Q6_V_hi_W(weights_w)));
}

static uint32_t qbh_attention_u8_sum_probability_half(
    HVX_Vector probability_h) {
    const HVX_VectorPair probability_w =
        Q6_Wuw_vunpack_Vuh(probability_h);
    return qbh_attention_u8_reduce_sum_words(
        Q6_Vw_vadd_VwVw(
            Q6_V_lo_W(probability_w),
            Q6_V_hi_W(probability_w)));
}

static HVX_Vector qbh_attention_u8_log2_code_half(
    HVX_Vector score_h, uint32_t maximum,
    uint32_t fraction_bits) {
    const HVX_Vector maximum_h = Q6_Vh_vsplat_R(maximum);
    const HVX_Vector rounding_h = Q6_Vh_vsplat_R(
        UINT32_C(1) << (fraction_bits - 1U));
    const HVX_Vector limit_h = Q6_Vh_vsplat_R(15);
    HVX_Vector code = Q6_Vuh_vsub_VuhVuh_sat(
        maximum_h, score_h);
    code = Q6_Vh_vadd_VhVh(code, rounding_h);
    code = Q6_Vuh_vlsr_VuhR(code, fraction_bits);
    return Q6_Vuh_vmin_VuhVuh(code, limit_h);
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

static HVX_Vector qbh_attention_u8_log2_codes_paired(
    HVX_Vector score, uint32_t row,
    uint32_t fraction_bits) {
    const HVX_Vector lane =
        *(const HVX_Vector *)qbh_attention_u8_lane_index;
    const HVX_Vector repeated_lane = Q6_V_vand_VV(
        lane, Q6_Vb_vsplat_R(63));
    const HVX_VectorPred valid = Q6_Q_not_Q(
        Q6_Q_vcmp_gt_VubVub(
            repeated_lane, Q6_Vb_vsplat_R(row)));
    const HVX_VectorPred lower_half = Q6_Q_vcmp_gt_VubVub(
        Q6_Vb_vsplat_R(64), lane);
    const HVX_Vector zero = Q6_V_vzero();
    const HVX_Vector masked_score = Q6_V_vmux_QVV(
        valid, score, zero);
    const uint8_t maximum0 = qbh_attention_u8_reduce_max(
        Q6_V_vmux_QVV(lower_half, masked_score, zero));
    const uint8_t maximum1 = qbh_attention_u8_reduce_max(
        Q6_V_vmux_QVV(lower_half, zero, masked_score));
    const HVX_VectorPair score_h = Q6_Wuh_vunpack_Vub(masked_score);
    const HVX_Vector code0 = qbh_attention_u8_log2_code_half(
        Q6_V_lo_W(score_h), maximum0, fraction_bits);
    const HVX_Vector code1 = qbh_attention_u8_log2_code_half(
        Q6_V_hi_W(score_h), maximum1, fraction_bits);
    return Q6_V_vmux_QVV(
        valid, Q6_Vub_vpack_VhVh_sat(code1, code0),
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

static void qbh_attention_u8_build_probability_lut_entries(
    uint8_t *lut, uint32_t entry_base, uint32_t sum,
    uint32_t mode, uint32_t valid_count) {
    if (valid_count == 1U) {
        for (uint32_t code = 0U; code <= 15U; ++code) {
            lut[2U * (entry_base + code)] = UINT8_MAX;
        }
        return;
    }
    if (mode == QBH_ATTENTION_DIVISION_EXACT) {
        for (uint32_t code = 0U; code <= 15U; ++code) {
            const uint32_t weight = UINT32_C(1) <<
                (QBH_ATTN_U8_EXP_FRAC_BITS - code);
            lut[2U * (entry_base + code)] = (uint8_t)(
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
            lut[2U * (entry_base + code)] =
                qbh_attention_u8_clip_u8(
                    (int32_t)((numerator +
                               (UINT64_C(1) << (shift - 1U))) >>
                              shift));
        }
    }
}

void qbh_attention_u8_build_sole_lut_template_bank(
    uint8_t *templates) {
    memset(templates, 0,
           QBH_ATTN_U8_LUT_TEMPLATE_COUNT *
               QBH_ATTN_U8_LUT_TEMPLATE_BYTES);
    for (uint32_t leading = QBH_ATTN_U8_LUT_LEADING_MIN;
         leading <= QBH_ATTN_U8_LUT_LEADING_MAX; ++leading) {
        for (uint32_t next = 0U; next < 2U; ++next) {
            const uint32_t slot =
                2U * (leading - QBH_ATTN_U8_LUT_LEADING_MIN) + next;
            const uint32_t representative_sum =
                (UINT32_C(1) << leading) |
                (next << (leading - 1U));
            qbh_attention_u8_build_probability_lut_entries(
                templates +
                    (size_t)slot * QBH_ATTN_U8_LUT_TEMPLATE_BYTES,
                0U, representative_sum,
                QBH_ATTENTION_DIVISION_SOLE, 2U);
        }
    }
}

static const uint8_t *qbh_attention_u8_select_sole_lut_template(
    const uint8_t *templates, uint32_t sum) {
    const uint32_t leading = qbh_attention_u8_floor_log2(sum);
    const uint32_t next = (sum >> (leading - 1U)) & 1U;
    const uint32_t slot =
        2U * (leading - QBH_ATTN_U8_LUT_LEADING_MIN) + next;
    return templates +
        (size_t)slot * QBH_ATTN_U8_LUT_TEMPLATE_BYTES;
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

static void qbh_attention_u8_requant_softmax_group_impl(
    uint8_t *score_tiles, uint8_t *probability_tiles,
    uint8_t *scratch,
    const struct qbh_attention_config *config,
    struct qbh_attention_u8_telemetry *telemetry,
    uint32_t use_lut_templates, uint32_t build_lut_templates,
    uint32_t first_row, uint32_t row_count) {
    uint8_t *row_scratch = scratch;
    uint8_t *lut = scratch + QBH_ATTN_U8_HVX_BYTES;
    uint8_t *lut_templates =
        scratch + 2U * QBH_ATTN_U8_HVX_BYTES;
    uint8_t *score0 = score_tiles;
    uint8_t *score1 = score_tiles +
        QBH_ATTENTION_SCORE_TILES * QBH_HMX_OUTPUT_BYTES;
    uint8_t *probability0 = probability_tiles;
    uint8_t *probability1 = probability_tiles +
        QBH_ATTENTION_SCORE_TILES * QBH_HMX_ACTIVATION_BYTES;
    const HVX_Vector lane =
        *(const HVX_Vector *)qbh_attention_u8_lane_index;
    const HVX_VectorPred lower_half = Q6_Q_vcmp_gt_VubVub(
        Q6_Vb_vsplat_R(64), lane);
    const uint32_t use_sole_templates =
        use_lut_templates != 0U &&
        config->division_mode == QBH_ATTENTION_DIVISION_SOLE;
    uint32_t row_sum_min = UINT_MAX;
    uint32_t row_sum_max = 0U;

    if (use_sole_templates != 0U && build_lut_templates != 0U) {
        qbh_attention_u8_build_sole_lut_template_bank(lut_templates);
    }

    for (uint32_t row = first_row;
         row < first_row + row_count; ++row) {
        const uint32_t valid_count = row + 1U;
        HVX_Vector score;
        HVX_Vector codes;
        HVX_Vector probabilities;
        HVX_Vector banked_codes;
        HVX_VectorPair code_h;
        HVX_VectorPair probability_h;
        uint32_t weight_sum0;
        uint32_t weight_sum1;
        uint32_t probability_sum0;
        uint32_t probability_sum1;

        memcpy(row_scratch,
               score0 + (size_t)row * QBH_HMX_OUTPUT_CHANNELS,
               QBH_HMX_OUTPUT_CHANNELS);
        memcpy(row_scratch + QBH_HMX_OUTPUT_CHANNELS,
               score0 + QBH_HMX_OUTPUT_BYTES +
                   (size_t)row * QBH_HMX_OUTPUT_CHANNELS,
               QBH_HMX_OUTPUT_CHANNELS);
        memcpy(row_scratch + 2U * QBH_HMX_OUTPUT_CHANNELS,
               score1 + (size_t)row * QBH_HMX_OUTPUT_CHANNELS,
               QBH_HMX_OUTPUT_CHANNELS);
        memcpy(row_scratch + 3U * QBH_HMX_OUTPUT_CHANNELS,
               score1 + QBH_HMX_OUTPUT_BYTES +
                   (size_t)row * QBH_HMX_OUTPUT_CHANNELS,
               QBH_HMX_OUTPUT_CHANNELS);

        score = *(const HVX_Vector *)row_scratch;
        if (telemetry != NULL) {
            const uint8_t *raw = (const uint8_t *)&score;
            for (uint32_t column = 0U;
                 column < QBH_ATTN_U8_HVX_BYTES; ++column) {
                const int32_t centered =
                    (int32_t)raw[column] -
                    (int32_t)QBH_ATTENTION_HMX_CENTER;
                const int32_t value =
                    centered * (int32_t)config->score_multiplier +
                    QBH_ATTN_U8_SCORE_ZP;
                telemetry->score_saturation_count +=
                    value < 0 || value > UINT8_MAX;
            }
        }
        score = qbh_attention_u8_requant_centered(
            score, config->score_multiplier,
            QBH_ATTN_U8_SCORE_ZP);
        if (telemetry != NULL) {
            const uint8_t *converted = (const uint8_t *)&score;
            memcpy(score0 +
                       (size_t)row * QBH_HMX_OUTPUT_CHANNELS,
                   converted, QBH_HMX_OUTPUT_CHANNELS);
            memcpy(score0 + QBH_HMX_OUTPUT_BYTES +
                       (size_t)row * QBH_HMX_OUTPUT_CHANNELS,
                   converted + QBH_HMX_OUTPUT_CHANNELS,
                   QBH_HMX_OUTPUT_CHANNELS);
            memcpy(score1 +
                       (size_t)row * QBH_HMX_OUTPUT_CHANNELS,
                   converted + 2U * QBH_HMX_OUTPUT_CHANNELS,
                   QBH_HMX_OUTPUT_CHANNELS);
            memcpy(score1 + QBH_HMX_OUTPUT_BYTES +
                       (size_t)row * QBH_HMX_OUTPUT_CHANNELS,
                   converted + 3U * QBH_HMX_OUTPUT_CHANNELS,
                   QBH_HMX_OUTPUT_CHANNELS);
        }

        codes = qbh_attention_u8_log2_codes_paired(
            score, row, config->fraction_bits);
        code_h = Q6_Wuh_vunpack_Vub(codes);
        weight_sum0 = qbh_attention_u8_sum_log2_weights_half(
            Q6_V_lo_W(code_h));
        weight_sum1 = qbh_attention_u8_sum_log2_weights_half(
            Q6_V_hi_W(code_h));
        if (use_sole_templates != 0U && valid_count != 1U) {
            memcpy(lut,
                   qbh_attention_u8_select_sole_lut_template(
                       lut_templates, weight_sum0),
                   QBH_ATTN_U8_LUT_TEMPLATE_BYTES);
            memcpy(lut + QBH_ATTN_U8_LUT_TEMPLATE_BYTES,
                   qbh_attention_u8_select_sole_lut_template(
                       lut_templates, weight_sum1),
                   QBH_ATTN_U8_LUT_TEMPLATE_BYTES);
        } else {
            memset(lut, 0, QBH_ATTN_U8_HVX_BYTES);
            qbh_attention_u8_build_probability_lut_entries(
                lut, 0U, weight_sum0,
                config->division_mode, valid_count);
            qbh_attention_u8_build_probability_lut_entries(
                lut, 16U, weight_sum1,
                config->division_mode, valid_count);
        }
        banked_codes = Q6_V_vmux_QVV(
            lower_half, codes,
            Q6_Vb_vadd_VbVb(codes, Q6_Vb_vsplat_R(16)));
        {
            const HVX_Vector repeated_lane = Q6_V_vand_VV(
                lane, Q6_Vb_vsplat_R(63));
            const HVX_VectorPred valid = Q6_Q_not_Q(
                Q6_Q_vcmp_gt_VubVub(
                    repeated_lane, Q6_Vb_vsplat_R(row)));
            probabilities = Q6_V_vmux_QVV(
                valid,
                Q6_Vb_vlut32_VbVbR_nomatch(
                    banked_codes, *(const HVX_Vector *)lut, 0),
                Q6_V_vzero());
        }
        *(HVX_Vector *)row_scratch = probabilities;
        probability_h = Q6_Wuh_vunpack_Vub(probabilities);
        probability_sum0 = qbh_attention_u8_sum_probability_half(
            Q6_V_lo_W(probability_h));
        probability_sum1 = qbh_attention_u8_sum_probability_half(
            Q6_V_hi_W(probability_h));

        memcpy(probability0 +
                   (size_t)row * QBH_HMX_INPUT_CHANNELS,
               row_scratch, QBH_HMX_INPUT_CHANNELS);
        memcpy(probability0 + QBH_HMX_ACTIVATION_BYTES +
                   (size_t)row * QBH_HMX_INPUT_CHANNELS,
               row_scratch + QBH_HMX_INPUT_CHANNELS,
               QBH_HMX_INPUT_CHANNELS);
        memcpy(probability1 +
                   (size_t)row * QBH_HMX_INPUT_CHANNELS,
               row_scratch + 2U * QBH_HMX_INPUT_CHANNELS,
               QBH_HMX_INPUT_CHANNELS);
        memcpy(probability1 + QBH_HMX_ACTIVATION_BYTES +
                   (size_t)row * QBH_HMX_INPUT_CHANNELS,
               row_scratch + 3U * QBH_HMX_INPUT_CHANNELS,
               QBH_HMX_INPUT_CHANNELS);

        if (probability_sum0 < row_sum_min) {
            row_sum_min = probability_sum0;
        }
        if (probability_sum1 < row_sum_min) {
            row_sum_min = probability_sum1;
        }
        if (probability_sum0 > row_sum_max) {
            row_sum_max = probability_sum0;
        }
        if (probability_sum1 > row_sum_max) {
            row_sum_max = probability_sum1;
        }
        if (telemetry != NULL) {
            const uint8_t *bytes =
                (const uint8_t *)&probabilities;
            for (uint32_t column = valid_count;
                 column < QBH_ATTENTION_M; ++column) {
                telemetry->probability_mask_violation_count +=
                    bytes[column] != 0U;
                telemetry->probability_mask_violation_count +=
                    bytes[QBH_ATTENTION_M + column] != 0U;
            }
        }
    }
    if (telemetry != NULL) {
        telemetry->probability_row_sum_min = row_sum_min;
        telemetry->probability_row_sum_max = row_sum_max;
    }
    asm volatile("barrier" ::: "memory");
}

static inline __attribute__((always_inline)) void
qbh_attention_u8_transpose_four_32byte_quarters(
    HVX_Vector source0, HVX_Vector source1,
    HVX_Vector source2, HVX_Vector source3,
    HVX_Vector *destination0, HVX_Vector *destination1,
    HVX_Vector *destination2, HVX_Vector *destination3) {
    const HVX_VectorPair source01 = Q6_W_vshuff_VVR(
        source1, source0, -32);
    const HVX_VectorPair source23 = Q6_W_vshuff_VVR(
        source3, source2, -32);
    const HVX_VectorPair destination01 = Q6_W_vshuff_VVR(
        Q6_V_lo_W(source23), Q6_V_lo_W(source01), -64);
    const HVX_VectorPair destination23 = Q6_W_vshuff_VVR(
        Q6_V_hi_W(source23), Q6_V_hi_W(source01), -64);

    *destination0 = Q6_V_lo_W(destination01);
    *destination1 = Q6_V_hi_W(destination01);
    *destination2 = Q6_V_lo_W(destination23);
    *destination3 = Q6_V_hi_W(destination23);
}

static inline __attribute__((always_inline)) void
qbh_attention_u8_update_probability_extrema(
    uint32_t sum0, uint32_t sum1,
    uint32_t *minimum, uint32_t *maximum) {
    if (sum0 < *minimum) {
        *minimum = sum0;
    }
    if (sum1 < *minimum) {
        *minimum = sum1;
    }
    if (sum0 > *maximum) {
        *maximum = sum0;
    }
    if (sum1 > *maximum) {
        *maximum = sum1;
    }
}

static __attribute__((noinline)) void
qbh_attention_u8_record_score_saturation(
    HVX_Vector score,
    const struct qbh_attention_config *config,
    struct qbh_attention_u8_telemetry *telemetry) {
    const uint8_t *raw = (const uint8_t *)&score;

    if (telemetry == NULL) {
        return;
    }
    for (uint32_t column = 0U;
         column < QBH_ATTN_U8_HVX_BYTES; ++column) {
        const int32_t centered =
            (int32_t)raw[column] -
            (int32_t)QBH_ATTENTION_HMX_CENTER;
        const int32_t value =
            centered * (int32_t)config->score_multiplier +
            QBH_ATTN_U8_SCORE_ZP;
        telemetry->score_saturation_count +=
            value < 0 || value > UINT8_MAX;
    }
}

static __attribute__((noinline)) HVX_Vector
qbh_attention_u8_softmax_requantized_pair(
    HVX_Vector score, uint32_t row,
    const struct qbh_attention_config *config,
    uint8_t *lut, const uint8_t *lut_templates,
    uint32_t use_sole_templates,
    struct qbh_attention_u8_telemetry *telemetry,
    HVX_Vector *converted_score,
    uint32_t *probability_sum0,
    uint32_t *probability_sum1) {
    const uint32_t valid_count = row + 1U;
    const HVX_Vector lane =
        *(const HVX_Vector *)qbh_attention_u8_lane_index;
    const HVX_VectorPred lower_half = Q6_Q_vcmp_gt_VubVub(
        Q6_Vb_vsplat_R(64), lane);
    if (telemetry != NULL) {
        qbh_attention_u8_record_score_saturation(
            score, config, telemetry);
    }
    score = qbh_attention_u8_requant_centered(
        score, config->score_multiplier,
        QBH_ATTN_U8_SCORE_ZP);
    if (converted_score != NULL) {
        *converted_score = score;
    }
    const HVX_Vector codes = qbh_attention_u8_log2_codes_paired(
        score, row, config->fraction_bits);
    const HVX_VectorPair code_h = Q6_Wuh_vunpack_Vub(codes);
    const uint32_t weight_sum0 =
        qbh_attention_u8_sum_log2_weights_half(
            Q6_V_lo_W(code_h));
    const uint32_t weight_sum1 =
        qbh_attention_u8_sum_log2_weights_half(
            Q6_V_hi_W(code_h));
    HVX_Vector probabilities;

    if (use_sole_templates != 0U && valid_count != 1U) {
        memcpy(lut,
               qbh_attention_u8_select_sole_lut_template(
                   lut_templates, weight_sum0),
               QBH_ATTN_U8_LUT_TEMPLATE_BYTES);
        memcpy(lut + QBH_ATTN_U8_LUT_TEMPLATE_BYTES,
               qbh_attention_u8_select_sole_lut_template(
                   lut_templates, weight_sum1),
               QBH_ATTN_U8_LUT_TEMPLATE_BYTES);
    } else {
        memset(lut, 0, QBH_ATTN_U8_HVX_BYTES);
        qbh_attention_u8_build_probability_lut_entries(
            lut, 0U, weight_sum0,
            config->division_mode, valid_count);
        qbh_attention_u8_build_probability_lut_entries(
            lut, 16U, weight_sum1,
            config->division_mode, valid_count);
    }
    {
        const HVX_Vector banked_codes = Q6_V_vmux_QVV(
            lower_half, codes,
            Q6_Vb_vadd_VbVb(codes, Q6_Vb_vsplat_R(16)));
        const HVX_Vector repeated_lane = Q6_V_vand_VV(
            lane, Q6_Vb_vsplat_R(63));
        const HVX_VectorPred valid = Q6_Q_not_Q(
            Q6_Q_vcmp_gt_VubVub(
                repeated_lane, Q6_Vb_vsplat_R(row)));
        probabilities = Q6_V_vmux_QVV(
            valid,
            Q6_Vb_vlut32_VbVbR_nomatch(
                banked_codes, *(const HVX_Vector *)lut, 0),
            Q6_V_vzero());
    }
    {
        const HVX_VectorPair probability_h =
            Q6_Wuh_vunpack_Vub(probabilities);
        *probability_sum0 =
            qbh_attention_u8_sum_probability_half(
                Q6_V_lo_W(probability_h));
        *probability_sum1 =
            qbh_attention_u8_sum_probability_half(
                Q6_V_hi_W(probability_h));
    }
    if (telemetry != NULL) {
        const uint8_t *bytes = (const uint8_t *)&probabilities;
        for (uint32_t column = valid_count;
             column < QBH_ATTENTION_M; ++column) {
            telemetry->probability_mask_violation_count +=
                bytes[column] != 0U;
            telemetry->probability_mask_violation_count +=
                bytes[QBH_ATTENTION_M + column] != 0U;
        }
    }
    return probabilities;
}

void qbh_attention_u8_requant_softmax_group_rows_prebuilt_templates_shuffle4(
    uint8_t *score_tiles, uint8_t *probability_tiles,
    uint8_t *scratch, uint8_t *carrier_scratch,
    const struct qbh_attention_config *config,
    struct qbh_attention_u8_telemetry *telemetry,
    uint32_t first_row, uint32_t row_count) {
    uint8_t *lut = scratch + QBH_ATTN_U8_HVX_BYTES;
    const uint8_t *lut_templates =
        scratch + 2U * QBH_ATTN_U8_HVX_BYTES;
    uint8_t *score0 = score_tiles;
    uint8_t *score1 = score_tiles +
        QBH_ATTENTION_SCORE_TILES * QBH_HMX_OUTPUT_BYTES;
    uint8_t *probability0 = probability_tiles;
    uint8_t *probability1 = probability_tiles +
        QBH_ATTENTION_SCORE_TILES * QBH_HMX_ACTIVATION_BYTES;
    const uint32_t use_sole_templates =
        config->division_mode == QBH_ATTENTION_DIVISION_SOLE;
    uint32_t row_sum_min = UINT_MAX;
    uint32_t row_sum_max = 0U;

    if ((first_row & 3U) != 0U || (row_count & 3U) != 0U ||
        carrier_scratch == NULL ||
        (((uintptr_t)carrier_scratch) &
         (QBH_ATTN_U8_HVX_BYTES - 1U)) != 0U) {
        qbh_attention_u8_requant_softmax_group_impl(
            score_tiles, probability_tiles, scratch,
            config, telemetry, 1U, 0U,
            first_row, row_count);
        return;
    }

    for (uint32_t row = first_row;
         row < first_row + row_count; row += 4U) {
        const size_t row_offset =
            (size_t)row * QBH_HMX_OUTPUT_CHANNELS;
        uint8_t *converted_scratch =
            carrier_scratch + 4U * QBH_ATTN_U8_HVX_BYTES;
        HVX_Vector tile0 = *(const HVX_Vector *)(
            score0 + row_offset);
        HVX_Vector tile1 = *(const HVX_Vector *)(
            score0 + QBH_HMX_OUTPUT_BYTES + row_offset);
        HVX_Vector tile2 = *(const HVX_Vector *)(
            score1 + row_offset);
        HVX_Vector tile3 = *(const HVX_Vector *)(
            score1 + QBH_HMX_OUTPUT_BYTES + row_offset);
        HVX_Vector row0;
        HVX_Vector row1;
        HVX_Vector row2;
        HVX_Vector row3;

        qbh_attention_u8_transpose_four_32byte_quarters(
            tile0, tile1, tile2, tile3,
            &row0, &row1, &row2, &row3);
        *(HVX_Vector *)(carrier_scratch +
                        0U * QBH_ATTN_U8_HVX_BYTES) = row0;
        *(HVX_Vector *)(carrier_scratch +
                        1U * QBH_ATTN_U8_HVX_BYTES) = row1;
        *(HVX_Vector *)(carrier_scratch +
                        2U * QBH_ATTN_U8_HVX_BYTES) = row2;
        *(HVX_Vector *)(carrier_scratch +
                        3U * QBH_ATTN_U8_HVX_BYTES) = row3;

        for (uint32_t local_row = 0U;
             local_row < 4U; ++local_row) {
            uint32_t probability_sum0;
            uint32_t probability_sum1;
            HVX_Vector value = *(const HVX_Vector *)(
                carrier_scratch +
                (size_t)local_row * QBH_ATTN_U8_HVX_BYTES);

            value = qbh_attention_u8_softmax_requantized_pair(
                value, row + local_row, config,
                lut, lut_templates,
                use_sole_templates, telemetry,
                telemetry != NULL
                    ? (HVX_Vector *)(converted_scratch +
                        (size_t)local_row * QBH_ATTN_U8_HVX_BYTES)
                    : NULL,
                &probability_sum0, &probability_sum1);
            *(HVX_Vector *)(carrier_scratch +
                (size_t)local_row * QBH_ATTN_U8_HVX_BYTES) = value;
            qbh_attention_u8_update_probability_extrema(
                probability_sum0, probability_sum1,
                &row_sum_min, &row_sum_max);
        }

        if (telemetry != NULL) {
            row0 = *(const HVX_Vector *)(converted_scratch +
                0U * QBH_ATTN_U8_HVX_BYTES);
            row1 = *(const HVX_Vector *)(converted_scratch +
                1U * QBH_ATTN_U8_HVX_BYTES);
            row2 = *(const HVX_Vector *)(converted_scratch +
                2U * QBH_ATTN_U8_HVX_BYTES);
            row3 = *(const HVX_Vector *)(converted_scratch +
                3U * QBH_ATTN_U8_HVX_BYTES);
            qbh_attention_u8_transpose_four_32byte_quarters(
                row0, row1, row2, row3,
                &tile0, &tile1, &tile2, &tile3);
            *(HVX_Vector *)(score0 + row_offset) = tile0;
            *(HVX_Vector *)(score0 + QBH_HMX_OUTPUT_BYTES +
                            row_offset) = tile1;
            *(HVX_Vector *)(score1 + row_offset) = tile2;
            *(HVX_Vector *)(score1 + QBH_HMX_OUTPUT_BYTES +
                            row_offset) = tile3;
        }

        row0 = *(const HVX_Vector *)(carrier_scratch +
            0U * QBH_ATTN_U8_HVX_BYTES);
        row1 = *(const HVX_Vector *)(carrier_scratch +
            1U * QBH_ATTN_U8_HVX_BYTES);
        row2 = *(const HVX_Vector *)(carrier_scratch +
            2U * QBH_ATTN_U8_HVX_BYTES);
        row3 = *(const HVX_Vector *)(carrier_scratch +
            3U * QBH_ATTN_U8_HVX_BYTES);
        qbh_attention_u8_transpose_four_32byte_quarters(
            row0, row1, row2, row3,
            &tile0, &tile1, &tile2, &tile3);
        *(HVX_Vector *)(probability0 + row_offset) = tile0;
        *(HVX_Vector *)(probability0 + QBH_HMX_ACTIVATION_BYTES +
                        row_offset) = tile1;
        *(HVX_Vector *)(probability1 + row_offset) = tile2;
        *(HVX_Vector *)(probability1 + QBH_HMX_ACTIVATION_BYTES +
                        row_offset) = tile3;
    }
    if (telemetry != NULL) {
        telemetry->probability_row_sum_min = row_sum_min;
        telemetry->probability_row_sum_max = row_sum_max;
    }
    asm volatile("barrier" ::: "memory");
}

void qbh_attention_u8_requant_softmax_group(
    uint8_t *score_tiles, uint8_t *probability_tiles,
    uint8_t *scratch,
    const struct qbh_attention_config *config,
    struct qbh_attention_u8_telemetry *telemetry) {
    qbh_attention_u8_requant_softmax_group_impl(
        score_tiles, probability_tiles, scratch,
        config, telemetry, 0U, 0U, 0U, QBH_ATTENTION_M);
}

void qbh_attention_u8_requant_softmax_group_lut_templates(
    uint8_t *score_tiles, uint8_t *probability_tiles,
    uint8_t *scratch,
    const struct qbh_attention_config *config,
    struct qbh_attention_u8_telemetry *telemetry) {
    qbh_attention_u8_requant_softmax_group_impl(
        score_tiles, probability_tiles, scratch,
        config, telemetry, 1U, 1U, 0U, QBH_ATTENTION_M);
}

void qbh_attention_u8_requant_softmax_group_rows_prebuilt_templates(
    uint8_t *score_tiles, uint8_t *probability_tiles,
    uint8_t *scratch,
    const struct qbh_attention_config *config,
    struct qbh_attention_u8_telemetry *telemetry,
    uint32_t first_row, uint32_t row_count) {
    qbh_attention_u8_requant_softmax_group_impl(
        score_tiles, probability_tiles, scratch,
        config, telemetry, 1U, 0U, first_row, row_count);
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

void qbh_attention_u8_native_head_to_row_major(
    const uint8_t *head_tiles, uint8_t *rows,
    uint32_t valid_rows) {
    if (head_tiles == NULL || rows == NULL ||
        valid_rows > QBH_ATTENTION_M) {
        return;
    }
    for (uint32_t row = 0U; row < valid_rows; ++row) {
        for (uint32_t tile = 0U;
             tile < QBH_ATTENTION_HEAD_DIM_TILES; ++tile) {
            memcpy(rows + (size_t)row * QBH_ATTENTION_HEAD_DIM +
                       tile * QBH_HMX_INPUT_CHANNELS,
                   head_tiles + (size_t)tile * QBH_HMX_ACTIVATION_BYTES +
                       (size_t)row * QBH_HMX_INPUT_CHANNELS,
                   QBH_HMX_INPUT_CHANNELS);
        }
    }
}

void qbh_attention_u8_pack_k_row_major(
    const uint8_t *rows, uint32_t valid_tokens,
    uint32_t padded_tokens,
    const struct qbh_attention_config *config,
    int8_t *weight_tiles, uint32_t *bias_words) {
    const uint32_t divisor = UINT32_C(1) << config->score_shift;
    const int32_t rounding = config->score_shift == 0U
                                 ? 0
                                 : (int32_t)(divisor / 2U);
    const uint16_t conversion = qbh_attention_u8_float_to_half_bits(
        512.0f / (float)divisor);
    const uint32_t n_tiles = padded_tokens / QBH_HMX_OUTPUT_CHANNELS;

    memset(weight_tiles, 0,
           (size_t)n_tiles * QBH_ATTENTION_HEAD_DIM_TILES *
               QBH_HMX_WEIGHT_BYTES);
    memset(bias_words, 0,
           (size_t)n_tiles * QBH_HMX_BIAS_BYTES);
    for (uint32_t n_tile = 0U; n_tile < n_tiles; ++n_tile) {
        int32_t sums[QBH_HMX_OUTPUT_CHANNELS] = {0};
        for (uint32_t k_tile = 0U;
             k_tile < QBH_ATTENTION_HEAD_DIM_TILES; ++k_tile) {
            int8_t *destination = weight_tiles +
                ((size_t)n_tile * QBH_ATTENTION_HEAD_DIM_TILES + k_tile) *
                    QBH_HMX_WEIGHT_BYTES;
            for (uint32_t input_group = 0U;
                 input_group < QBH_HMX_INPUT_CHANNELS / 4U;
                 ++input_group) {
                uint32_t *packed = (uint32_t *)(destination +
                    (size_t)input_group * 128U);
                for (uint32_t output = 0U;
                     output < QBH_HMX_OUTPUT_CHANNELS; ++output) {
                    const uint32_t token =
                        n_tile * QBH_HMX_OUTPUT_CHANNELS + output;
                    uint32_t word = 0U;
                    for (uint32_t lane = 0U; lane < 4U; ++lane) {
                        int32_t centered = 0;
                        if (token < valid_tokens) {
                            const uint32_t channel =
                                k_tile * QBH_HMX_INPUT_CHANNELS +
                                input_group * 4U + lane;
                            centered = (int32_t)rows[
                                (size_t)token * QBH_ATTENTION_HEAD_DIM +
                                channel] - config->k_zero_point;
                        }
                        if (centered < INT8_MIN) {
                            centered = INT8_MIN;
                        } else if (centered > INT8_MAX) {
                            centered = INT8_MAX;
                        }
                        sums[output] += centered;
                        word |= (uint32_t)(uint8_t)(int8_t)centered <<
                                (lane * 8U);
                    }
                    packed[output] = word;
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
                    (uint32_t)(-config->q_zero_point * sums[output] +
                               QBH_ATTN_U8_SCORE_ZP * (int32_t)divisor +
                               rounding);
            }
        }
    }
    asm volatile("barrier" ::: "memory");
}

void qbh_attention_u8_pack_v_row_major(
    const uint8_t *rows, uint32_t valid_tokens,
    uint32_t padded_tokens,
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
    const uint32_t k_tiles = padded_tokens / QBH_HMX_INPUT_CHANNELS;

    memset(weight_tiles, 0,
           (size_t)QBH_ATTENTION_HEAD_DIM_TILES * k_tiles *
               QBH_HMX_WEIGHT_BYTES);
    memset(bias_words, 0,
           (size_t)QBH_ATTENTION_HEAD_DIM_TILES * QBH_HMX_BIAS_BYTES);
    for (uint32_t n_tile = 0U;
         n_tile < QBH_ATTENTION_HEAD_DIM_TILES; ++n_tile) {
        for (uint32_t k_tile = 0U; k_tile < k_tiles; ++k_tile) {
            int8_t *destination = weight_tiles +
                ((size_t)n_tile * k_tiles + k_tile) *
                    QBH_HMX_WEIGHT_BYTES;
            for (uint32_t input_group = 0U;
                 input_group < QBH_HMX_INPUT_CHANNELS / 4U;
                 ++input_group) {
                uint32_t *packed = (uint32_t *)(destination +
                    (size_t)input_group * 128U);
                for (uint32_t output = 0U;
                     output < QBH_HMX_OUTPUT_CHANNELS; ++output) {
                    uint32_t word = 0U;
                    const uint32_t channel =
                        n_tile * QBH_HMX_OUTPUT_CHANNELS + output;
                    for (uint32_t lane = 0U; lane < 4U; ++lane) {
                        const uint32_t token =
                            k_tile * QBH_HMX_INPUT_CHANNELS +
                            input_group * 4U + lane;
                        int32_t requantized = 0;
                        if (token < valid_tokens) {
                            const int32_t centered =
                                (int32_t)rows[
                                    (size_t)token *
                                        QBH_ATTENTION_HEAD_DIM +
                                    channel] - config->v_zero_point;
                            requantized =
                                qbh_attention_u8_round_div_signed(
                                    centered * (int32_t)
                                        config->v_recenter_numerator,
                                    (int32_t)
                                        config->v_recenter_denominator);
                            if (saturation_count != NULL) {
                                *saturation_count +=
                                    requantized < INT8_MIN ||
                                    requantized > INT8_MAX;
                            }
                            requantized = qbh_attention_u8_clip_s8(
                                requantized, NULL);
                        }
                        word |=
                            (uint32_t)(uint8_t)(int8_t)requantized <<
                            (lane * 8U);
                    }
                    packed[output] = word;
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

void qbh_attention_u8_update_k_native_token(
    const uint8_t *k_head_tiles, uint32_t source_row,
    uint32_t output_lane,
    const struct qbh_attention_config *config,
    int8_t *n_tile_weight, uint32_t *n_tile_bias) {
    int32_t sum = 0;
    const uint32_t divisor = UINT32_C(1) << config->score_shift;
    const int32_t rounding = config->score_shift == 0U
                                 ? 0
                                 : (int32_t)(divisor / 2U);

    if (k_head_tiles == NULL || config == NULL ||
        n_tile_weight == NULL || n_tile_bias == NULL ||
        source_row >= QBH_ATTENTION_M ||
        output_lane >= QBH_HMX_OUTPUT_CHANNELS) {
        return;
    }
    for (uint32_t k_tile = 0U;
         k_tile < QBH_ATTENTION_HEAD_DIM_TILES; ++k_tile) {
        const uint8_t *source = k_head_tiles +
            (size_t)k_tile * QBH_HMX_ACTIVATION_BYTES +
            (size_t)source_row * QBH_HMX_INPUT_CHANNELS;
        int8_t *destination = n_tile_weight +
            (size_t)k_tile * QBH_HMX_WEIGHT_BYTES;
        for (uint32_t input_group = 0U;
             input_group < QBH_HMX_INPUT_CHANNELS / 4U;
             ++input_group) {
            uint32_t word = 0U;
            for (uint32_t lane = 0U; lane < 4U; ++lane) {
                int32_t centered =
                    (int32_t)source[input_group * 4U + lane] -
                    config->k_zero_point;
                centered = qbh_attention_u8_clip_s8(centered, NULL);
                sum += centered;
                word |= (uint32_t)(uint8_t)(int8_t)centered <<
                        (lane * 8U);
            }
            ((uint32_t *)(destination +
                (size_t)input_group * 128U))[output_lane] = word;
        }
    }
    n_tile_bias[output_lane] = qbh_attention_u8_float_to_half_bits(
        512.0f / (float)divisor);
    n_tile_bias[QBH_HMX_OUTPUT_CHANNELS + output_lane] =
        (uint32_t)(-config->q_zero_point * sum +
                   QBH_ATTN_U8_SCORE_ZP * (int32_t)divisor +
                   rounding);
    asm volatile("barrier" ::: "memory");
}

void qbh_attention_u8_update_v_native_token(
    const uint8_t *v_head_tiles, uint32_t source_row,
    uint32_t input_lane,
    const struct qbh_attention_config *config,
    int8_t *k_tile_weights, uint32_t *saturation_count) {
    const uint32_t input_group = input_lane / 4U;
    const uint32_t byte_lane = input_lane % 4U;

    if (v_head_tiles == NULL || config == NULL ||
        k_tile_weights == NULL || source_row >= QBH_ATTENTION_M ||
        input_lane >= QBH_HMX_INPUT_CHANNELS) {
        return;
    }
    for (uint32_t n_tile = 0U;
         n_tile < QBH_ATTENTION_HEAD_DIM_TILES; ++n_tile) {
        const uint8_t *source = v_head_tiles +
            (size_t)n_tile * QBH_HMX_ACTIVATION_BYTES +
            (size_t)source_row * QBH_HMX_INPUT_CHANNELS;
        int8_t *destination = k_tile_weights +
            (size_t)n_tile * QBH_HMX_WEIGHT_BYTES;
        for (uint32_t output = 0U;
             output < QBH_HMX_OUTPUT_CHANNELS; ++output) {
            const int32_t centered =
                (int32_t)source[output] - config->v_zero_point;
            int32_t requantized = qbh_attention_u8_round_div_signed(
                centered * (int32_t)config->v_recenter_numerator,
                (int32_t)config->v_recenter_denominator);
            if (saturation_count != NULL) {
                *saturation_count +=
                    requantized < INT8_MIN || requantized > INT8_MAX;
            }
            requantized = qbh_attention_u8_clip_s8(requantized, NULL);
            ((uint8_t *)(destination +
                (size_t)input_group * 128U +
                (size_t)output * sizeof(uint32_t)))[byte_lane] =
                    (uint8_t)(int8_t)requantized;
        }
    }
    asm volatile("barrier" ::: "memory");
}

void qbh_attention_u8_update_k_native_row(
    const uint8_t *row, uint32_t output_lane,
    const struct qbh_attention_config *config,
    int8_t *n_tile_weight, uint32_t *n_tile_bias) {
    int32_t sum = 0;
    const uint32_t divisor = UINT32_C(1) << config->score_shift;
    const int32_t rounding = config->score_shift == 0U
                                 ? 0
                                 : (int32_t)(divisor / 2U);

    if (row == NULL || config == NULL ||
        n_tile_weight == NULL || n_tile_bias == NULL ||
        output_lane >= QBH_HMX_OUTPUT_CHANNELS) {
        return;
    }
    for (uint32_t k_tile = 0U;
         k_tile < QBH_ATTENTION_HEAD_DIM_TILES; ++k_tile) {
        const uint8_t *source =
            row + (size_t)k_tile * QBH_HMX_INPUT_CHANNELS;
        int8_t *destination = n_tile_weight +
            (size_t)k_tile * QBH_HMX_WEIGHT_BYTES;
        for (uint32_t input_group = 0U;
             input_group < QBH_HMX_INPUT_CHANNELS / 4U;
             ++input_group) {
            uint32_t word = 0U;
            for (uint32_t lane = 0U; lane < 4U; ++lane) {
                int32_t centered =
                    (int32_t)source[input_group * 4U + lane] -
                    config->k_zero_point;
                centered = qbh_attention_u8_clip_s8(centered, NULL);
                sum += centered;
                word |= (uint32_t)(uint8_t)(int8_t)centered <<
                        (lane * 8U);
            }
            ((uint32_t *)(destination +
                (size_t)input_group * 128U))[output_lane] = word;
        }
    }
    n_tile_bias[output_lane] = qbh_attention_u8_float_to_half_bits(
        512.0f / (float)divisor);
    n_tile_bias[QBH_HMX_OUTPUT_CHANNELS + output_lane] =
        (uint32_t)(-config->q_zero_point * sum +
                   QBH_ATTN_U8_SCORE_ZP * (int32_t)divisor +
                   rounding);
}

void qbh_attention_u8_update_v_native_row(
    const uint8_t *row, uint32_t input_lane,
    const struct qbh_attention_config *config,
    int8_t *k_tile_weights, uint32_t k_tile_stride_bytes,
    uint32_t *saturation_count) {
    const uint32_t input_group = input_lane / 4U;
    const uint32_t byte_lane = input_lane % 4U;

    if (row == NULL || config == NULL ||
        k_tile_weights == NULL ||
        input_lane >= QBH_HMX_INPUT_CHANNELS ||
        k_tile_stride_bytes < QBH_HMX_WEIGHT_BYTES) {
        return;
    }
    for (uint32_t n_tile = 0U;
         n_tile < QBH_ATTENTION_HEAD_DIM_TILES; ++n_tile) {
        const uint8_t *source =
            row + (size_t)n_tile * QBH_HMX_OUTPUT_CHANNELS;
        int8_t *destination = k_tile_weights +
            (size_t)n_tile * k_tile_stride_bytes;
        for (uint32_t output = 0U;
             output < QBH_HMX_OUTPUT_CHANNELS; ++output) {
            const int32_t centered =
                (int32_t)source[output] - config->v_zero_point;
            int32_t requantized = qbh_attention_u8_round_div_signed(
                centered * (int32_t)config->v_recenter_numerator,
                (int32_t)config->v_recenter_denominator);
            if (saturation_count != NULL) {
                *saturation_count +=
                    requantized < INT8_MIN || requantized > INT8_MAX;
            }
            requantized = qbh_attention_u8_clip_s8(requantized, NULL);
            ((uint8_t *)(destination +
                (size_t)input_group * 128U +
                (size_t)output * sizeof(uint32_t)))[byte_lane] =
                    (uint8_t)(int8_t)requantized;
        }
    }
}

void qbh_attention_u8_patch_k_delta_rows_hvx(
    const uint8_t *rows, uint32_t row_count,
    const struct qbh_attention_config *config,
    int8_t *n_tile_weight, uint32_t *n_tile_bias) {
    const uint32_t divisor = UINT32_C(1) << config->score_shift;
    const int32_t rounding = config->score_shift == 0U
                                 ? 0
                                 : (int32_t)(divisor / 2U);
    const uint16_t conversion = qbh_attention_u8_float_to_half_bits(
        512.0f / (float)divisor);
    const HVX_Vector offsets_base =
        *(const HVX_Vector *)qbh_attention_u8_vscatter_offsets;

    if (rows == NULL || config == NULL ||
        n_tile_weight == NULL || n_tile_bias == NULL ||
        row_count > QBH_HMX_OUTPUT_CHANNELS) {
        return;
    }
    for (uint32_t output = 0U; output < row_count; ++output) {
        const HVX_Vector centered = qbh_attention_u8_center_u8_to_s8(
            *(const HVX_Vector *)(rows +
                (size_t)output * QBH_ATTENTION_HEAD_DIM),
            config->k_zero_point);
        const HVX_Vector offsets = Q6_Vw_vadd_VwVw(
            offsets_base, Q6_V_vsplat_R(output * sizeof(uint32_t)));
        const int32_t sum = qbh_attention_u8_sum_signed_bytes(centered);
        Q6_vscatter_RMVwV(
            (uint32_t)(uintptr_t)n_tile_weight,
            QBH_ATTENTION_HEAD_DIM_TILES * QBH_HMX_WEIGHT_BYTES - 1U,
            offsets, centered);
        n_tile_bias[output] = conversion;
        n_tile_bias[QBH_HMX_OUTPUT_CHANNELS + output] =
            (uint32_t)(-config->q_zero_point * sum +
                       QBH_ATTN_U8_SCORE_ZP * (int32_t)divisor +
                       rounding);
    }
    asm volatile("barrier" ::: "memory");
}

void qbh_attention_u8_prepare_v_delta_lut(
    const struct qbh_attention_config *config, uint8_t *scratch) {
    int16_t *recenter_lut = (int16_t *)(
        scratch + QBH_ATTN_U8_VGATHER_LUT_OFFSET);

    if (config == NULL || scratch == NULL) {
        return;
    }
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
    asm volatile("barrier" ::: "memory");
}

void qbh_attention_u8_patch_v_delta_rows_hvx(
    const uint8_t *rows, uint32_t row_count,
    const struct qbh_attention_config *config,
    int8_t *k_tile_weights, uint32_t k_tile_stride_bytes,
    uint8_t *scratch, uint32_t *saturation_count) {
    int16_t *recenter_lut = (int16_t *)(
        scratch + QBH_ATTN_U8_VGATHER_LUT_OFFSET);
    int16_t *gathered_low = (int16_t *)(
        scratch + QBH_ATTN_U8_VGATHER_SCRATCH_OFFSET);
    int16_t *gathered_high = gathered_low + 64;
    uint8_t row_group[QBH_ATTN_U8_HVX_BYTES]
        __attribute__((aligned(QBH_ATTN_U8_HVX_BYTES)));

    if (rows == NULL || config == NULL ||
        k_tile_weights == NULL || scratch == NULL ||
        row_count > QBH_HMX_INPUT_CHANNELS ||
        k_tile_stride_bytes < QBH_HMX_WEIGHT_BYTES) {
        return;
    }
    if (saturation_count != NULL) {
        for (uint32_t index = 0U;
             index < row_count * QBH_ATTENTION_HEAD_DIM; ++index) {
            const int32_t centered =
                (int32_t)rows[index] - config->v_zero_point;
            const int32_t requantized =
                qbh_attention_u8_round_div_signed(
                    centered * (int32_t)config->v_recenter_numerator,
                    (int32_t)config->v_recenter_denominator);
            *saturation_count +=
                requantized < INT8_MIN || requantized > INT8_MAX;
        }
    }

    for (uint32_t n_tile = 0U;
         n_tile < QBH_ATTENTION_HEAD_DIM_TILES; ++n_tile) {
        int8_t *destination = k_tile_weights +
            (size_t)n_tile * k_tile_stride_bytes;
        for (uint32_t input_group = 0U;
             input_group * 4U < row_count; ++input_group) {
            for (uint32_t lane = 0U; lane < 4U; ++lane) {
                const uint32_t row = input_group * 4U + lane;
                uint8_t *lane_destination = row_group +
                    lane * QBH_HMX_OUTPUT_CHANNELS;
                if (row < row_count) {
                    memcpy(
                        lane_destination,
                        rows + (size_t)row * QBH_ATTENTION_HEAD_DIM +
                            n_tile * QBH_HMX_OUTPUT_CHANNELS,
                        QBH_HMX_OUTPUT_CHANNELS);
                } else {
                    memset(
                        lane_destination,
                        (uint8_t)config->v_zero_point,
                        QBH_HMX_OUTPUT_CHANNELS);
                }
            }
            {
                const HVX_Vector values = *(const HVX_Vector *)row_group;
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
    }
    asm volatile("barrier" ::: "memory");
}

static uint8_t qbh_attention_u8_dynamic_score_at(
    const uint8_t *score_tiles, uint32_t head,
    uint32_t row, uint32_t column, uint32_t n_tiles) {
    const uint32_t tile = column / QBH_HMX_OUTPUT_CHANNELS;
    const uint32_t lane = column % QBH_HMX_OUTPUT_CHANNELS;
    return score_tiles[
        (size_t)head * n_tiles * QBH_HMX_OUTPUT_BYTES +
        (size_t)tile * QBH_HMX_OUTPUT_BYTES +
        (size_t)row * QBH_HMX_OUTPUT_CHANNELS + lane];
}

static void qbh_attention_u8_dynamic_probability_set(
    uint8_t *probability_tiles, uint32_t head,
    uint32_t row, uint32_t column, uint32_t k_tiles,
    uint8_t value) {
    const uint32_t tile = column / QBH_HMX_INPUT_CHANNELS;
    const uint32_t lane = column % QBH_HMX_INPUT_CHANNELS;
    probability_tiles[
        (size_t)head * k_tiles * QBH_HMX_ACTIVATION_BYTES +
        (size_t)tile * QBH_HMX_ACTIVATION_BYTES +
        (size_t)row * QBH_HMX_INPUT_CHANNELS + lane] = value;
}

static void qbh_attention_u8_requant_softmax_dynamic_scalar(
    uint8_t *score_tiles, uint8_t *probability_tiles,
    uint32_t query_rows, uint32_t past_tokens,
    uint32_t valid_tokens, uint32_t padded_tokens,
    const struct qbh_attention_config *config,
    struct qbh_attention_u8_telemetry *telemetry) {
    const uint32_t tiles = padded_tokens / QBH_HMX_INPUT_CHANNELS;
    uint32_t row_sum_min = UINT_MAX;
    uint32_t row_sum_max = 0U;
    uint8_t lut[QBH_ATTN_U8_HVX_BYTES]
        __attribute__((aligned(QBH_ATTN_U8_HVX_BYTES)));

    memset(probability_tiles, 0,
           (size_t)QBH_ATTENTION_Q_HEADS_PER_GROUP * tiles *
               QBH_HMX_ACTIVATION_BYTES);
    for (uint32_t head = 0U;
         head < QBH_ATTENTION_Q_HEADS_PER_GROUP; ++head) {
        for (uint32_t row = 0U; row < query_rows; ++row) {
            const uint32_t valid_count = past_tokens + row + 1U;
            uint32_t weight_sum = 0U;
            uint32_t probability_sum = 0U;
            uint8_t maximum = 0U;

            for (uint32_t column = 0U;
                 column < valid_count; ++column) {
                const uint8_t raw = qbh_attention_u8_dynamic_score_at(
                    score_tiles, head, row, column, tiles);
                const int32_t centered =
                    (int32_t)raw - (int32_t)QBH_ATTENTION_HMX_CENTER;
                const int32_t converted =
                    centered * (int32_t)config->score_multiplier +
                    QBH_ATTN_U8_SCORE_ZP;
                const uint8_t score =
                    qbh_attention_u8_clip_u8(converted);
                if (telemetry != NULL) {
                    telemetry->score_saturation_count +=
                        converted < 0 || converted > UINT8_MAX;
                }
                if (score > maximum) {
                    maximum = score;
                }
            }
            for (uint32_t column = 0U;
                 column < valid_count; ++column) {
                const uint8_t raw = qbh_attention_u8_dynamic_score_at(
                    score_tiles, head, row, column, tiles);
                const int32_t centered =
                    (int32_t)raw - (int32_t)QBH_ATTENTION_HMX_CENTER;
                const uint8_t score = qbh_attention_u8_clip_u8(
                    centered * (int32_t)config->score_multiplier +
                    QBH_ATTN_U8_SCORE_ZP);
                uint32_t code =
                    ((uint32_t)maximum - score +
                     (UINT32_C(1) << (config->fraction_bits - 1U))) >>
                    config->fraction_bits;
                if (code > 15U) {
                    code = 15U;
                }
                weight_sum += UINT32_C(1) <<
                    (QBH_ATTN_U8_EXP_FRAC_BITS - code);
            }
            qbh_attention_u8_build_probability_lut(
                lut, weight_sum, config->division_mode, valid_count);
            for (uint32_t column = 0U;
                 column < valid_count; ++column) {
                const uint8_t raw = qbh_attention_u8_dynamic_score_at(
                    score_tiles, head, row, column, tiles);
                const int32_t centered =
                    (int32_t)raw - (int32_t)QBH_ATTENTION_HMX_CENTER;
                const uint8_t score = qbh_attention_u8_clip_u8(
                    centered * (int32_t)config->score_multiplier +
                    QBH_ATTN_U8_SCORE_ZP);
                uint32_t code =
                    ((uint32_t)maximum - score +
                     (UINT32_C(1) << (config->fraction_bits - 1U))) >>
                    config->fraction_bits;
                uint8_t probability;
                if (code > 15U) {
                    code = 15U;
                }
                probability = lut[2U * code];
                probability_sum += probability;
                qbh_attention_u8_dynamic_probability_set(
                    probability_tiles, head, row, column, tiles,
                    probability);
            }
            if (valid_count > valid_tokens && telemetry != NULL) {
                telemetry->probability_mask_violation_count +=
                    valid_count - valid_tokens;
            }
            if (probability_sum < row_sum_min) {
                row_sum_min = probability_sum;
            }
            if (probability_sum > row_sum_max) {
                row_sum_max = probability_sum;
            }
        }
    }
    if (telemetry != NULL) {
        telemetry->probability_row_sum_min = row_sum_min;
        telemetry->probability_row_sum_max = row_sum_max;
    }
    asm volatile("barrier" ::: "memory");
}

static inline __attribute__((always_inline)) HVX_Vector
qbh_attention_u8_dynamic_load_head_pair(
    const uint8_t *score_tiles, uint32_t tiles,
    uint32_t first_head, uint32_t first_tile) {
    const size_t head_stride =
        (size_t)tiles * QBH_HMX_OUTPUT_BYTES;
    const size_t tile_offset =
        (size_t)first_tile * QBH_HMX_OUTPUT_BYTES;
    const HVX_Vector zero = Q6_V_vzero();
    const HVX_Vector tile0 = *(const HVX_Vector *)(
        score_tiles + (size_t)first_head * head_stride + tile_offset);
    const HVX_Vector tile1 = first_tile + 1U < tiles
        ? *(const HVX_Vector *)(
              score_tiles + (size_t)first_head * head_stride +
              tile_offset + QBH_HMX_OUTPUT_BYTES)
        : zero;
    const HVX_Vector tile2 = *(const HVX_Vector *)(
        score_tiles + (size_t)(first_head + 1U) * head_stride +
        tile_offset);
    const HVX_Vector tile3 = first_tile + 1U < tiles
        ? *(const HVX_Vector *)(
              score_tiles + (size_t)(first_head + 1U) * head_stride +
              tile_offset + QBH_HMX_OUTPUT_BYTES)
        : zero;
    HVX_Vector row0;
    HVX_Vector row1;
    HVX_Vector row2;
    HVX_Vector row3;

    qbh_attention_u8_transpose_four_32byte_quarters(
        tile0, tile1, tile2, tile3,
        &row0, &row1, &row2, &row3);
    return row0;
}

static inline __attribute__((always_inline)) void
qbh_attention_u8_dynamic_store_head_pair(
    uint8_t *probability_tiles, uint32_t tiles,
    uint32_t first_head, uint32_t first_tile,
    HVX_Vector probability) {
    const size_t head_stride =
        (size_t)tiles * QBH_HMX_ACTIVATION_BYTES;
    const size_t tile_offset =
        (size_t)first_tile * QBH_HMX_ACTIVATION_BYTES;
    const HVX_Vector zero = Q6_V_vzero();
    HVX_Vector tile0;
    HVX_Vector tile1;
    HVX_Vector tile2;
    HVX_Vector tile3;

    qbh_attention_u8_transpose_four_32byte_quarters(
        probability, zero, zero, zero,
        &tile0, &tile1, &tile2, &tile3);
    *(HVX_Vector *)(probability_tiles +
                    (size_t)first_head * head_stride + tile_offset) = tile0;
    *(HVX_Vector *)(probability_tiles +
                    (size_t)(first_head + 1U) * head_stride +
                    tile_offset) = tile2;
    if (first_tile + 1U < tiles) {
        *(HVX_Vector *)(probability_tiles +
                        (size_t)first_head * head_stride + tile_offset +
                        QBH_HMX_ACTIVATION_BYTES) = tile1;
        *(HVX_Vector *)(probability_tiles +
                        (size_t)(first_head + 1U) * head_stride +
                        tile_offset + QBH_HMX_ACTIVATION_BYTES) = tile3;
    }
}

static inline __attribute__((always_inline)) HVX_VectorPred
qbh_attention_u8_dynamic_pair_valid(uint32_t valid_in_block) {
    const HVX_Vector lane =
        *(const HVX_Vector *)qbh_attention_u8_lane_index;
    const HVX_Vector repeated_lane = Q6_V_vand_VV(
        lane, Q6_Vb_vsplat_R(63));
    return Q6_Q_vcmp_gt_VubVub(
        Q6_Vb_vsplat_R(valid_in_block), repeated_lane);
}

static inline __attribute__((always_inline)) uint32_t
qbh_attention_u8_sum_bytes(HVX_Vector value) {
    const HVX_VectorPair value_h = Q6_Wuh_vunpack_Vub(value);
    return qbh_attention_u8_sum_probability_half(Q6_V_lo_W(value_h)) +
           qbh_attention_u8_sum_probability_half(Q6_V_hi_W(value_h));
}

static inline __attribute__((always_inline)) uint32_t
qbh_attention_u8_dynamic_saturation_count(
    HVX_Vector raw, HVX_VectorPred valid, uint32_t multiplier) {
    uint32_t minimum_raw;
    uint32_t maximum_raw;
    HVX_VectorPred low;
    HVX_VectorPred high;
    HVX_Vector saturated;

    if (multiplier == 0U) {
        return 0U;
    }
    minimum_raw =
        (128U * multiplier - 128U + multiplier - 1U) / multiplier;
    maximum_raw = (128U * multiplier + 127U) / multiplier;
    if (minimum_raw > UINT8_MAX) {
        minimum_raw = UINT8_MAX;
    }
    if (maximum_raw > UINT8_MAX) {
        maximum_raw = UINT8_MAX;
    }
    low = Q6_Q_vcmp_gt_VubVub(
        Q6_Vb_vsplat_R(minimum_raw), raw);
    high = Q6_Q_vcmp_gt_VubVub(
        raw, Q6_Vb_vsplat_R(maximum_raw));
    saturated = Q6_V_vmux_QVV(
        Q6_Q_and_QQ(valid, Q6_Q_or_QQ(low, high)),
        Q6_Vb_vsplat_R(1), Q6_V_vzero());
    return qbh_attention_u8_sum_bytes(saturated);
}

static inline __attribute__((always_inline)) void
qbh_attention_u8_dynamic_pair_maximum(
    HVX_Vector score, HVX_VectorPred valid,
    uint8_t *maximum0, uint8_t *maximum1) {
    const HVX_Vector lane =
        *(const HVX_Vector *)qbh_attention_u8_lane_index;
    const HVX_VectorPred lower_half = Q6_Q_vcmp_gt_VubVub(
        Q6_Vb_vsplat_R(64), lane);
    const HVX_Vector zero = Q6_V_vzero();
    const HVX_Vector masked = Q6_V_vmux_QVV(valid, score, zero);
    const uint8_t local0 = qbh_attention_u8_reduce_max(
        Q6_V_vmux_QVV(lower_half, masked, zero));
    const uint8_t local1 = qbh_attention_u8_reduce_max(
        Q6_V_vmux_QVV(lower_half, zero, masked));

    if (local0 > *maximum0) {
        *maximum0 = local0;
    }
    if (local1 > *maximum1) {
        *maximum1 = local1;
    }
}

static inline __attribute__((always_inline)) HVX_Vector
qbh_attention_u8_dynamic_pair_codes(
    HVX_Vector score, HVX_VectorPred valid,
    uint32_t maximum0, uint32_t maximum1,
    uint32_t fraction_bits) {
    const HVX_VectorPair score_h = Q6_Wuh_vunpack_Vub(score);
    const HVX_Vector code0 = qbh_attention_u8_log2_code_half(
        Q6_V_lo_W(score_h), maximum0, fraction_bits);
    const HVX_Vector code1 = qbh_attention_u8_log2_code_half(
        Q6_V_hi_W(score_h), maximum1, fraction_bits);
    return Q6_V_vmux_QVV(
        valid, Q6_Vub_vpack_VhVh_sat(code1, code0),
        Q6_Vb_vsplat_R(16));
}

static void qbh_attention_u8_requant_softmax_dynamic_hvx_tile4(
    uint8_t *score_tiles, uint8_t *probability_tiles,
    uint32_t past_tokens, uint32_t valid_tokens,
    uint32_t padded_tokens,
    const struct qbh_attention_config *config,
    struct qbh_attention_u8_telemetry *telemetry) {
    const uint32_t tiles = padded_tokens / QBH_HMX_INPUT_CHANNELS;
    const uint32_t tile_blocks = (tiles + 1U) / 2U;
    const uint32_t valid_count = past_tokens + 1U;
    const HVX_Vector lane =
        *(const HVX_Vector *)qbh_attention_u8_lane_index;
    const HVX_VectorPred lower_half = Q6_Q_vcmp_gt_VubVub(
        Q6_Vb_vsplat_R(64), lane);
    uint32_t row_sum_min = UINT_MAX;
    uint32_t row_sum_max = 0U;
    uint8_t lut[QBH_ATTN_U8_HVX_BYTES]
        __attribute__((aligned(QBH_ATTN_U8_HVX_BYTES)));

    memset(probability_tiles, 0,
           (size_t)QBH_ATTENTION_Q_HEADS_PER_GROUP * tiles *
               QBH_HMX_ACTIVATION_BYTES);
    for (uint32_t pair = 0U;
         pair < QBH_ATTENTION_Q_HEADS_PER_GROUP / 2U; ++pair) {
        const uint32_t first_head = pair * 2U;
        uint8_t maximum0 = 0U;
        uint8_t maximum1 = 0U;
        uint32_t weight_sum0 = 0U;
        uint32_t weight_sum1 = 0U;
        uint32_t probability_sum0 = 0U;
        uint32_t probability_sum1 = 0U;

        for (uint32_t block = 0U; block < tile_blocks; ++block) {
            const uint32_t first_token = block * 64U;
            const uint32_t valid_in_block = valid_count - first_token < 64U
                ? valid_count - first_token : 64U;
            const HVX_VectorPred valid =
                qbh_attention_u8_dynamic_pair_valid(valid_in_block);
            HVX_Vector raw = qbh_attention_u8_dynamic_load_head_pair(
                score_tiles, tiles, first_head, block * 2U);
            HVX_Vector score;

            if (telemetry != NULL) {
                telemetry->score_saturation_count +=
                    qbh_attention_u8_dynamic_saturation_count(
                        raw, valid, config->score_multiplier);
            }
            score = qbh_attention_u8_requant_centered(
                raw, config->score_multiplier,
                QBH_ATTN_U8_SCORE_ZP);
            qbh_attention_u8_dynamic_pair_maximum(
                score, valid, &maximum0, &maximum1);
        }

        for (uint32_t block = 0U; block < tile_blocks; ++block) {
            const uint32_t first_token = block * 64U;
            const uint32_t valid_in_block = valid_count - first_token < 64U
                ? valid_count - first_token : 64U;
            const HVX_VectorPred valid =
                qbh_attention_u8_dynamic_pair_valid(valid_in_block);
            const HVX_Vector raw =
                qbh_attention_u8_dynamic_load_head_pair(
                    score_tiles, tiles, first_head, block * 2U);
            const HVX_Vector score = qbh_attention_u8_requant_centered(
                raw, config->score_multiplier,
                QBH_ATTN_U8_SCORE_ZP);
            const HVX_Vector codes =
                qbh_attention_u8_dynamic_pair_codes(
                    score, valid, maximum0, maximum1,
                    config->fraction_bits);
            const HVX_VectorPair code_h = Q6_Wuh_vunpack_Vub(codes);
            uint8_t *code_scratch = probability_tiles +
                (size_t)(pair * tile_blocks + block) *
                    QBH_HMX_ACTIVATION_BYTES +
                QBH_ATTN_U8_HVX_BYTES;

            weight_sum0 += qbh_attention_u8_sum_log2_weights_half(
                Q6_V_lo_W(code_h));
            weight_sum1 += qbh_attention_u8_sum_log2_weights_half(
                Q6_V_hi_W(code_h));
            *(HVX_Vector *)code_scratch = codes;
        }

        memset(lut, 0, sizeof(lut));
        qbh_attention_u8_build_probability_lut_entries(
            lut, 0U, weight_sum0,
            config->division_mode, valid_count);
        qbh_attention_u8_build_probability_lut_entries(
            lut, 16U, weight_sum1,
            config->division_mode, valid_count);

        for (uint32_t block = 0U; block < tile_blocks; ++block) {
            const uint32_t first_token = block * 64U;
            const uint32_t valid_in_block = valid_count - first_token < 64U
                ? valid_count - first_token : 64U;
            const HVX_VectorPred valid =
                qbh_attention_u8_dynamic_pair_valid(valid_in_block);
            uint8_t *code_scratch = probability_tiles +
                (size_t)(pair * tile_blocks + block) *
                    QBH_HMX_ACTIVATION_BYTES +
                QBH_ATTN_U8_HVX_BYTES;
            const HVX_Vector codes = *(const HVX_Vector *)code_scratch;
            const HVX_Vector banked_codes = Q6_V_vmux_QVV(
                lower_half, codes,
                Q6_Vb_vadd_VbVb(codes, Q6_Vb_vsplat_R(16)));
            const HVX_Vector probabilities = Q6_V_vmux_QVV(
                valid,
                Q6_Vb_vlut32_VbVbR_nomatch(
                    banked_codes, *(const HVX_Vector *)lut, 0),
                Q6_V_vzero());
            const HVX_VectorPair probability_h =
                Q6_Wuh_vunpack_Vub(probabilities);

            probability_sum0 += qbh_attention_u8_sum_probability_half(
                Q6_V_lo_W(probability_h));
            probability_sum1 += qbh_attention_u8_sum_probability_half(
                Q6_V_hi_W(probability_h));
            *(HVX_Vector *)code_scratch = Q6_V_vzero();
            qbh_attention_u8_dynamic_store_head_pair(
                probability_tiles, tiles, first_head,
                block * 2U, probabilities);
        }

        qbh_attention_u8_update_probability_extrema(
            probability_sum0, probability_sum1,
            &row_sum_min, &row_sum_max);
    }
    if (telemetry != NULL) {
        if (valid_count > valid_tokens) {
            telemetry->probability_mask_violation_count +=
                QBH_ATTENTION_Q_HEADS_PER_GROUP *
                (valid_count - valid_tokens);
        }
        telemetry->probability_row_sum_min = row_sum_min;
        telemetry->probability_row_sum_max = row_sum_max;
        ++telemetry->dynamic_hvx_tile4_call_count;
    }
    asm volatile("barrier" ::: "memory");
}

static void qbh_attention_u8_dynamic_stash_scalar_probability(
    uint8_t *score_tiles, const uint8_t *probability_tiles,
    uint32_t tiles) {
    for (uint32_t head = 0U;
         head < QBH_ATTENTION_Q_HEADS_PER_GROUP; ++head) {
        for (uint32_t tile = 0U; tile < tiles; ++tile) {
            uint8_t *score_carrier = score_tiles +
                ((size_t)head * tiles + tile) * QBH_HMX_OUTPUT_BYTES;
            const uint8_t *probability_carrier = probability_tiles +
                ((size_t)head * tiles + tile) *
                    QBH_HMX_ACTIVATION_BYTES;
            memcpy(score_carrier + QBH_HMX_OUTPUT_CHANNELS,
                   probability_carrier, QBH_HMX_INPUT_CHANNELS);
        }
    }
}

static uint32_t qbh_attention_u8_dynamic_compare_and_clear_scalar_probability(
    uint8_t *score_tiles, const uint8_t *probability_tiles,
    uint32_t tiles) {
    uint32_t mismatch_count = 0U;

    for (uint32_t head = 0U;
         head < QBH_ATTENTION_Q_HEADS_PER_GROUP; ++head) {
        for (uint32_t tile = 0U; tile < tiles; ++tile) {
            uint8_t *score_carrier = score_tiles +
                ((size_t)head * tiles + tile) * QBH_HMX_OUTPUT_BYTES;
            const uint8_t *probability_carrier = probability_tiles +
                ((size_t)head * tiles + tile) *
                    QBH_HMX_ACTIVATION_BYTES;
            const uint8_t *reference =
                score_carrier + QBH_HMX_OUTPUT_CHANNELS;

            for (uint32_t lane = 0U;
                 lane < QBH_HMX_INPUT_CHANNELS; ++lane) {
                mismatch_count +=
                    reference[lane] != probability_carrier[lane];
            }
            memset(score_carrier + QBH_HMX_OUTPUT_CHANNELS, 0,
                   QBH_HMX_OUTPUT_CHANNELS);
        }
    }
    return mismatch_count;
}

void qbh_attention_u8_requant_softmax_dynamic(
    uint8_t *score_tiles, uint8_t *probability_tiles,
    uint32_t query_rows, uint32_t past_tokens,
    uint32_t valid_tokens, uint32_t padded_tokens,
    const struct qbh_attention_config *config,
    struct qbh_attention_u8_telemetry *telemetry,
    uint32_t use_hvx_tile4, uint32_t verify_hvx_tile4) {
    if (use_hvx_tile4 != 0U && query_rows == 1U &&
        padded_tokens != 0U &&
        (padded_tokens % QBH_HMX_INPUT_CHANNELS) == 0U &&
        past_tokens + 1U <= valid_tokens &&
        valid_tokens <= padded_tokens &&
        config != NULL && config->fraction_bits != 0U &&
        config->score_multiplier <= INT8_MAX) {
        if (verify_hvx_tile4 != 0U) {
            qbh_attention_u8_requant_softmax_dynamic_scalar(
                score_tiles, probability_tiles, query_rows,
                past_tokens, valid_tokens, padded_tokens,
                config, NULL);
            qbh_attention_u8_dynamic_stash_scalar_probability(
                score_tiles, probability_tiles,
                padded_tokens / QBH_HMX_INPUT_CHANNELS);
        }
        qbh_attention_u8_requant_softmax_dynamic_hvx_tile4(
            score_tiles, probability_tiles, past_tokens,
            valid_tokens, padded_tokens, config, telemetry);
        if (verify_hvx_tile4 != 0U && telemetry != NULL) {
            telemetry->dynamic_hvx_tile4_mismatch_count +=
                qbh_attention_u8_dynamic_compare_and_clear_scalar_probability(
                    score_tiles, probability_tiles,
                    padded_tokens / QBH_HMX_INPUT_CHANNELS);
        }
        return;
    }
    qbh_attention_u8_requant_softmax_dynamic_scalar(
        score_tiles, probability_tiles, query_rows,
        past_tokens, valid_tokens, padded_tokens,
        config, telemetry);
}

void qbh_attention_u8_probability_map_from_raw_histogram(
    const uint32_t histogram[256], uint32_t valid_count,
    const struct qbh_attention_config *config,
    uint8_t probability_by_raw[256],
    uint32_t *probability_sum, uint32_t *score_saturation_count) {
    uint8_t code_lut[QBH_ATTN_U8_HVX_BYTES]
        __attribute__((aligned(QBH_ATTN_U8_HVX_BYTES)));
    uint8_t maximum = 0U;
    uint32_t weight_sum = 0U;
    uint32_t local_probability_sum = 0U;
    uint32_t local_saturation_count = 0U;

    if (histogram == NULL || config == NULL ||
        probability_by_raw == NULL || valid_count == 0U) {
        return;
    }
    for (uint32_t raw = 0U; raw <= UINT8_MAX; ++raw) {
        if (histogram[raw] != 0U) {
            const int32_t converted =
                ((int32_t)raw - (int32_t)QBH_ATTENTION_HMX_CENTER) *
                    (int32_t)config->score_multiplier +
                QBH_ATTN_U8_SCORE_ZP;
            const uint8_t score = qbh_attention_u8_clip_u8(converted);
            if (score > maximum) {
                maximum = score;
            }
            if (converted < 0 || converted > UINT8_MAX) {
                local_saturation_count += histogram[raw];
            }
        }
    }
    for (uint32_t raw = 0U; raw <= UINT8_MAX; ++raw) {
        if (histogram[raw] != 0U) {
            const int32_t converted =
                ((int32_t)raw - (int32_t)QBH_ATTENTION_HMX_CENTER) *
                    (int32_t)config->score_multiplier +
                QBH_ATTN_U8_SCORE_ZP;
            const uint8_t score = qbh_attention_u8_clip_u8(converted);
            uint32_t code =
                ((uint32_t)maximum - score +
                 (UINT32_C(1) << (config->fraction_bits - 1U))) >>
                config->fraction_bits;
            if (code > 15U) {
                code = 15U;
            }
            weight_sum += histogram[raw] *
                (UINT32_C(1) << (QBH_ATTN_U8_EXP_FRAC_BITS - code));
        }
    }
    qbh_attention_u8_build_probability_lut(
        code_lut, weight_sum, config->division_mode, valid_count);
    for (uint32_t raw = 0U; raw <= UINT8_MAX; ++raw) {
        const int32_t converted =
            ((int32_t)raw - (int32_t)QBH_ATTENTION_HMX_CENTER) *
                (int32_t)config->score_multiplier +
            QBH_ATTN_U8_SCORE_ZP;
        const uint8_t score = qbh_attention_u8_clip_u8(converted);
        uint32_t code =
            ((uint32_t)maximum - score +
             (UINT32_C(1) << (config->fraction_bits - 1U))) >>
            config->fraction_bits;
        if (code > 15U) {
            code = 15U;
        }
        probability_by_raw[raw] = code_lut[2U * code];
        local_probability_sum +=
            histogram[raw] * probability_by_raw[raw];
    }
    if (probability_sum != NULL) {
        *probability_sum = local_probability_sum;
    }
    if (score_saturation_count != NULL) {
        *score_saturation_count = local_saturation_count;
    }
    asm volatile("barrier" ::: "memory");
}

void qbh_attention_u8_probability_map_from_active_histogram(
    const uint32_t histogram[256], const uint8_t *active_scores,
    uint32_t active_count, uint32_t valid_count,
    const struct qbh_attention_config *config,
    uint8_t probability_by_raw[256],
    uint32_t *probability_sum, uint32_t *score_saturation_count) {
    uint8_t code_lut[QBH_ATTN_U8_HVX_BYTES]
        __attribute__((aligned(QBH_ATTN_U8_HVX_BYTES)));
    uint8_t maximum = 0U;
    uint32_t weight_sum = 0U;
    uint32_t local_probability_sum = 0U;
    uint32_t local_saturation_count = 0U;

    if (histogram == NULL || active_scores == NULL ||
        config == NULL || probability_by_raw == NULL ||
        active_count == 0U || active_count > 256U ||
        valid_count == 0U) {
        return;
    }
    for (uint32_t index = 0U; index < active_count; ++index) {
        const uint8_t raw = active_scores[index];
        const int32_t converted =
            ((int32_t)raw - (int32_t)QBH_ATTENTION_HMX_CENTER) *
                (int32_t)config->score_multiplier +
            QBH_ATTN_U8_SCORE_ZP;
        const uint8_t score = qbh_attention_u8_clip_u8(converted);
        if (score > maximum) {
            maximum = score;
        }
        if (converted < 0 || converted > UINT8_MAX) {
            local_saturation_count += histogram[raw];
        }
    }
    for (uint32_t index = 0U; index < active_count; ++index) {
        const uint8_t raw = active_scores[index];
        const int32_t converted =
            ((int32_t)raw - (int32_t)QBH_ATTENTION_HMX_CENTER) *
                (int32_t)config->score_multiplier +
            QBH_ATTN_U8_SCORE_ZP;
        const uint8_t score = qbh_attention_u8_clip_u8(converted);
        uint32_t code =
            ((uint32_t)maximum - score +
             (UINT32_C(1) << (config->fraction_bits - 1U))) >>
            config->fraction_bits;
        if (code > 15U) {
            code = 15U;
        }
        weight_sum += histogram[raw] *
            (UINT32_C(1) << (QBH_ATTN_U8_EXP_FRAC_BITS - code));
    }
    qbh_attention_u8_build_probability_lut(
        code_lut, weight_sum, config->division_mode, valid_count);
    for (uint32_t index = 0U; index < active_count; ++index) {
        const uint8_t raw = active_scores[index];
        const int32_t converted =
            ((int32_t)raw - (int32_t)QBH_ATTENTION_HMX_CENTER) *
                (int32_t)config->score_multiplier +
            QBH_ATTN_U8_SCORE_ZP;
        const uint8_t score = qbh_attention_u8_clip_u8(converted);
        uint32_t code =
            ((uint32_t)maximum - score +
             (UINT32_C(1) << (config->fraction_bits - 1U))) >>
            config->fraction_bits;
        if (code > 15U) {
            code = 15U;
        }
        probability_by_raw[raw] = code_lut[2U * code];
        local_probability_sum +=
            histogram[raw] * probability_by_raw[raw];
    }
    if (probability_sum != NULL) {
        *probability_sum = local_probability_sum;
    }
    if (score_saturation_count != NULL) {
        *score_saturation_count = local_saturation_count;
    }
    asm volatile("barrier" ::: "memory");
}
