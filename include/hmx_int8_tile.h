#ifndef QWEN3_BLOCK_HTP_HMX_INT8_TILE_H
#define QWEN3_BLOCK_HTP_HMX_INT8_TILE_H

#include <stddef.h>
#include <stdint.h>

#include "probe_protocol.h"

static inline size_t qbh_activation_offset(uint32_t spatial,
                                           uint32_t input_channel) {
    return (size_t)spatial * QBH_HMX_INPUT_CHANNELS + input_channel;
}

static inline size_t qbh_logical_weight_offset(uint32_t input_channel,
                                               uint32_t output_channel) {
    return (size_t)input_channel * QBH_HMX_OUTPUT_CHANNELS + output_channel;
}

static inline size_t qbh_packed_weight_offset(uint32_t input_channel,
                                              uint32_t output_channel) {
    return ((size_t)(input_channel / 4U) * QBH_HMX_OUTPUT_CHANNELS +
            output_channel) * 4U + input_channel % 4U;
}

static inline size_t qbh_output_offset(uint32_t spatial,
                                       uint32_t output_channel) {
    return (size_t)spatial * QBH_HMX_OUTPUT_CHANNELS + output_channel;
}

void qbh_pack_s8_weight(const int8_t *logical_weight, int8_t *packed_weight);
void qbh_fill_asymmetric_bias(const int8_t *packed_weight,
                              int32_t input_zero_point,
                              uint32_t *bias_words);
void qbh_execute_u8s8_tile(const uint8_t *activation,
                           const int8_t *packed_weight,
                           const uint32_t *bias_words, uint8_t *output);

#endif
