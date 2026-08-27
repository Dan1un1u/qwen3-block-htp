#ifndef QWEN3_BLOCK_HTP_HMX_U8S8_PROJECTION_H
#define QWEN3_BLOCK_HTP_HMX_U8S8_PROJECTION_H

#include <stddef.h>
#include <stdint.h>

#include "hmx_int8_tile.h"
#include "probe_protocol.h"

#define QBH_HMX_IDENTITY_CONVERT_LOWER_WORD (UINT32_C(24) << 10U)

struct qbh_projection_layout {
    uint32_t variant;
    uint32_t m;
    uint32_t k;
    uint32_t n;
    uint32_t k_tiles;
    uint32_t n_tiles;
    uint32_t activation_bytes;
    uint32_t logical_weight_bytes;
    uint32_t weight_chunk_bytes;
    uint32_t weight_bundle_bytes;
    uint32_t packed_weight_bytes;
    uint32_t output_bytes;
    uint32_t output_tiles_bytes;
    uint32_t hmx_pairs_per_repeat;
    uint32_t k_streams_per_output;
    uint32_t hmx_streams_per_repeat;
    uint32_t vtcm_activation_offset;
    uint32_t vtcm_weight_slot0_offset;
    uint32_t vtcm_weight_slot1_offset;
    uint32_t vtcm_output_offset;
    uint32_t vtcm_plan_bytes;
};

static inline uint32_t qbh_align_up_u32(uint32_t value,
                                        uint32_t alignment) {
    return ((value + alignment - 1U) / alignment) * alignment;
}

static inline int qbh_projection_layout_init(
    uint32_t variant, struct qbh_projection_layout *layout) {
    uint32_t k;
    uint32_t n;

    if (layout == NULL) {
        return -1;
    }
    if (variant == QBH_PROJECTION_GATE_UP) {
        k = QBH_GATE_UP_K;
        n = QBH_GATE_UP_N;
    } else if (variant == QBH_PROJECTION_DOWN) {
        k = QBH_DOWN_K;
        n = QBH_DOWN_N;
    } else {
        return -1;
    }

    layout->variant = variant;
    layout->m = QBH_PROJ_M;
    layout->k = k;
    layout->n = n;
    layout->k_tiles = k / QBH_HMX_INPUT_CHANNELS;
    layout->n_tiles = n / QBH_HMX_OUTPUT_CHANNELS;
    layout->activation_bytes = layout->m * layout->k;
    layout->logical_weight_bytes = layout->k * layout->n;
    layout->weight_chunk_bytes =
        layout->k_tiles * QBH_HMX_WEIGHT_BYTES;
    layout->weight_bundle_bytes =
        layout->weight_chunk_bytes + QBH_HMX_BIAS_BYTES;
    layout->packed_weight_bytes =
        layout->n_tiles * layout->weight_bundle_bytes;
    layout->output_bytes = layout->m * layout->n;
    layout->output_tiles_bytes =
        layout->n_tiles * QBH_HMX_OUTPUT_BYTES;
    layout->hmx_pairs_per_repeat =
        layout->k_tiles * layout->n_tiles;
    layout->k_streams_per_output =
        (layout->k_tiles + QBH_HMX_MAX_STREAM_TILES - 1U) /
        QBH_HMX_MAX_STREAM_TILES;
    layout->hmx_streams_per_repeat =
        layout->n_tiles * layout->k_streams_per_output;

    layout->vtcm_activation_offset = 0;
    layout->vtcm_weight_slot0_offset = qbh_align_up_u32(
        layout->vtcm_activation_offset + layout->activation_bytes,
        QBH_HMX_ACTIVATION_BYTES);
    layout->vtcm_weight_slot1_offset = qbh_align_up_u32(
        layout->vtcm_weight_slot0_offset + layout->weight_bundle_bytes,
        UINT32_C(256));
    layout->vtcm_output_offset = qbh_align_up_u32(
        layout->vtcm_weight_slot1_offset + layout->weight_bundle_bytes,
        QBH_HMX_OUTPUT_BYTES);
    layout->vtcm_plan_bytes =
        layout->vtcm_output_offset + layout->output_tiles_bytes;

    return layout->hmx_pairs_per_repeat ==
                       QBH_QWEN3_HMX_PAIRS_PER_REPEAT &&
                   layout->vtcm_plan_bytes <= QBH_QWEN3_VTCM_BYTES
               ? 0
               : -1;
}

static inline size_t qbh_projection_activation_offset(
    const struct qbh_projection_layout *layout, uint32_t row,
    uint32_t channel) {
    return (size_t)row * layout->k + channel;
}

static inline size_t qbh_projection_logical_weight_offset(
    const struct qbh_projection_layout *layout, uint32_t input_channel,
    uint32_t output_channel) {
    return (size_t)input_channel * layout->n + output_channel;
}

static inline size_t qbh_projection_bundle_offset(
    const struct qbh_projection_layout *layout, uint32_t output_tile) {
    return (size_t)output_tile * layout->weight_bundle_bytes;
}

static inline size_t qbh_projection_packed_weight_offset(
    const struct qbh_projection_layout *layout, uint32_t output_tile,
    uint32_t input_tile, uint32_t input_channel,
    uint32_t output_channel) {
    return qbh_projection_bundle_offset(layout, output_tile) +
           (size_t)input_tile * QBH_HMX_WEIGHT_BYTES +
           qbh_packed_weight_offset(input_channel, output_channel);
}

static inline size_t qbh_projection_bias_offset(
    const struct qbh_projection_layout *layout, uint32_t output_tile) {
    return qbh_projection_bundle_offset(layout, output_tile) +
           layout->weight_chunk_bytes;
}

static inline size_t qbh_projection_output_offset(
    const struct qbh_projection_layout *layout, uint32_t row,
    uint32_t channel) {
    return (size_t)row * layout->n + channel;
}

void qbh_hmx_begin_u8s8_output(const uint32_t *bias_words);
uint32_t qbh_hmx_accumulate_u8s8_projection(
    const uint8_t *activation_tiles, const int8_t *packed_weight_tiles,
    uint32_t k_tiles);
void qbh_hmx_store_u8_output(uint8_t *output);

#endif
