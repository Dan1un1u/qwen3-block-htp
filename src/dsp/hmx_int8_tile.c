#include <hmx_hexagon_protos.h>
#include <stdint.h>

#include "hmx_int8_tile.h"

#define QBH_HMX_SPATIAL_MASK UINT32_C(0x38)
#define QBH_HMX_ACTIVATION_RT \
    (((QBH_HMX_SPATIAL_MASK >> 2U) << 7U) | \
     ((QBH_HMX_INPUT_CHANNELS - 1U) << 2U) | \
     (QBH_HMX_SPATIAL_MASK & 0x3U))
#define QBH_HMX_WEIGHT_RT (QBH_HMX_WEIGHT_BYTES - 1U)
#define QBH_HMX_WRITE_RT \
    (((QBH_HMX_SPATIAL_MASK >> 2U) << 7U) | \
     (QBH_HMX_SPATIAL_MASK & 0x3U))
#define QBH_HMX_IDENTITY_CONVERT_LOWER_WORD (UINT32_C(24) << 10U)

void qbh_pack_s8_weight(const int8_t *logical_weight, int8_t *packed_weight) {
    uint32_t input_channel;
    uint32_t output_channel;

    for (input_channel = 0; input_channel < QBH_HMX_INPUT_CHANNELS;
         ++input_channel) {
        for (output_channel = 0; output_channel < QBH_HMX_OUTPUT_CHANNELS;
             ++output_channel) {
            packed_weight[qbh_packed_weight_offset(input_channel,
                                                   output_channel)] =
                logical_weight[qbh_logical_weight_offset(input_channel,
                                                         output_channel)];
        }
    }
}

void qbh_fill_asymmetric_bias(const int8_t *packed_weight,
                              int32_t input_zero_point,
                              uint32_t *bias_words) {
    uint32_t output_channel;

    for (output_channel = 0; output_channel < QBH_HMX_OUTPUT_CHANNELS;
         ++output_channel) {
        int32_t weight_sum = 0;
        uint32_t input_channel;

        for (input_channel = 0; input_channel < QBH_HMX_INPUT_CHANNELS;
             ++input_channel) {
            weight_sum += packed_weight[qbh_packed_weight_offset(
                input_channel, output_channel)];
        }
        bias_words[output_channel] = QBH_HMX_IDENTITY_CONVERT_LOWER_WORD;
        bias_words[QBH_HMX_OUTPUT_CHANNELS + output_channel] =
            (uint32_t)(-input_zero_point * weight_sum);
    }
}

__attribute__((noinline)) void qbh_execute_u8s8_tile(
    const uint8_t *activation, const int8_t *packed_weight,
    const uint32_t *bias_words, uint8_t *output) {
    Q6_bias_mxmem2_A((void *)bias_words);
    Q6_mxclracc();
    asm volatile("{ activation.ub = mxmem(%0, %1):cm\n"
                 "  weight.b = mxmem(%2, %3) }\n"
                 :
                 : "r"(activation), "r"(QBH_HMX_ACTIVATION_RT),
                   "r"(packed_weight), "r"(QBH_HMX_WEIGHT_RT)
                 : "memory");
    Q6_mxmem_AR_after_cm_sat_ub(output, QBH_HMX_WRITE_RT);
    asm volatile("barrier" ::: "memory");
}
