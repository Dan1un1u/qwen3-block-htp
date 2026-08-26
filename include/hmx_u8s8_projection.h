#ifndef QWEN3_BLOCK_HTP_HMX_U8S8_PROJECTION_H
#define QWEN3_BLOCK_HTP_HMX_U8S8_PROJECTION_H

#include <stddef.h>
#include <stdint.h>

#include "hmx_int8_tile.h"
#include "probe_protocol.h"

#define QBH_HMX_IDENTITY_CONVERT_LOWER_WORD (UINT32_C(24) << 10U)

static inline size_t qbh_projection_activation_offset(uint32_t row,
                                                       uint32_t channel) {
    return (size_t)row * QBH_PROJ_K + channel;
}

static inline size_t qbh_projection_logical_weight_offset(
    uint32_t input_channel, uint32_t output_channel) {
    return (size_t)input_channel * QBH_PROJ_N + output_channel;
}

static inline size_t qbh_projection_bundle_offset(uint32_t output_tile) {
    return (size_t)output_tile * QBH_PROJ_WEIGHT_BUNDLE_BYTES;
}

static inline size_t qbh_projection_packed_weight_offset(
    uint32_t output_tile, uint32_t input_tile, uint32_t input_channel,
    uint32_t output_channel) {
    return qbh_projection_bundle_offset(output_tile) +
           (size_t)input_tile * QBH_PROJ_WEIGHT_TILE_BYTES +
           qbh_packed_weight_offset(input_channel, output_channel);
}

static inline size_t qbh_projection_bias_offset(uint32_t output_tile) {
    return qbh_projection_bundle_offset(output_tile) +
           QBH_PROJ_WEIGHT_CHUNK_BYTES;
}

static inline size_t qbh_projection_output_offset(uint32_t row,
                                                   uint32_t channel) {
    return (size_t)row * QBH_PROJ_N + channel;
}

void qbh_hmx_begin_u8s8_output(const uint32_t *bias_words);
void qbh_hmx_accumulate_u8s8_projection(const uint8_t *activation_tiles,
                                        const int8_t *packed_weight_tiles);
void qbh_hmx_store_u8_output(uint8_t *output);

#endif
