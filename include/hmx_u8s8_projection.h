#ifndef QWEN3_BLOCK_HTP_HMX_U8S8_PROJECTION_H
#define QWEN3_BLOCK_HTP_HMX_U8S8_PROJECTION_H

#include <stddef.h>
#include <stdint.h>

#include "hmx_int8_tile.h"
#include "probe_protocol.h"

#define QBH_HMX_IDENTITY_CONVERT_LOWER_WORD (UINT32_C(24) << 10U)

struct qbh_projection_layout {
    uint32_t variant;
    uint32_t weight_storage_variant;
    uint32_t physical_plan;
    uint32_t compressed_slot_count;
    uint32_t expanded_slot_count;
    uint32_t m;
    uint32_t k;
    uint32_t n;
    uint32_t k_tiles;
    uint32_t n_tiles;
    uint32_t activation_bytes;
    uint32_t logical_weight_bytes;
    uint32_t expanded_weight_chunk_bytes;
    uint32_t expanded_weight_bundle_bytes;
    uint32_t expanded_weight_bytes;
    uint32_t w4_packed_chunk_bytes;
    uint32_t w4_scale_offset;
    uint32_t w4_bias_offset;
    uint32_t w4_bundle_bytes;
    uint32_t w4_weight_bytes;
    uint32_t stored_weight_bundle_bytes;
    uint32_t stored_weight_bytes;
    uint32_t output_bytes;
    uint32_t output_tiles_bytes;
    uint32_t hmx_pairs_per_repeat;
    uint32_t k_streams_per_output;
    uint32_t hmx_streams_per_repeat;
    uint32_t chunk_tiles;
    uint32_t chunks_per_output;
    uint32_t expanded_chunk_weight_bytes;
    uint32_t expanded_chunk_slot_bytes;
    uint32_t vtcm_activation_offset;
    uint32_t vtcm_compressed_slot0_offset;
    uint32_t vtcm_compressed_slot1_offset;
    uint32_t vtcm_expanded_slot0_offset;
    uint32_t vtcm_expanded_slot1_offset;
    uint32_t vtcm_output_offset;
    uint32_t vtcm_chunked_expanded_slots_offset;
    uint32_t vtcm_chunked_output_offset;
    uint32_t vtcm_full_bundle_plan_bytes;
    uint32_t vtcm_chunked_plan_bytes;
    uint32_t vtcm_plan_bytes;
};

static inline uint32_t qbh_align_up_u32(uint32_t value,
                                        uint32_t alignment) {
    return ((value + alignment - 1U) / alignment) * alignment;
}

static inline int qbh_projection_layout_init(
    uint32_t variant, uint32_t weight_storage_variant,
    uint32_t physical_plan, uint32_t compressed_slot_count,
    uint32_t chunk_tiles,
    struct qbh_projection_layout *layout) {
    uint32_t k;
    uint32_t n;

    if (layout == NULL) {
        return -1;
    }
    if (variant == QBH_PROJECTION_GATE_UP) {
        k = QBH_GATE_UP_K;
        n = QBH_GATE_UP_N;
    } else if (variant == QBH_PROJECTION_GATE_UP_PAIR) {
        k = QBH_GATE_UP_K;
        n = QBH_GATE_UP_PAIR_N;
    } else if (variant == QBH_PROJECTION_DOWN) {
        k = QBH_DOWN_K;
        n = QBH_DOWN_N;
    } else {
        return -1;
    }
    if (weight_storage_variant != QBH_WEIGHT_EXPANDED_S8 &&
        !qbh_weight_storage_is_packed_w4(weight_storage_variant)) {
        return -1;
    }
    if (!qbh_physical_plan_is_full_bundle(physical_plan) &&
        !qbh_physical_plan_is_chunked(physical_plan)) {
        return -1;
    }
    if (qbh_physical_plan_is_chunked(physical_plan) &&
        !qbh_weight_storage_is_packed_w4(weight_storage_variant)) {
        return -1;
    }
    if ((physical_plan == QBH_PHYSICAL_PLAN_FULL_BUNDLE_DMA_BATCH2 ||
         physical_plan == QBH_PHYSICAL_PLAN_FULL_BUNDLE_DMA_CHAIN2) &&
        weight_storage_variant != QBH_WEIGHT_EXPANDED_S8) {
        return -1;
    }
    if ((physical_plan == QBH_PHYSICAL_PLAN_CHUNKED_E7_DMA_BATCH2 &&
         compressed_slot_count != 4U) ||
        ((physical_plan == QBH_PHYSICAL_PLAN_CHUNKED_E7_DMA_BATCH4 ||
          physical_plan == QBH_PHYSICAL_PLAN_CHUNKED_E7_DMA_CHAIN4 ||
          physical_plan ==
              QBH_PHYSICAL_PLAN_STREAMING_E7_DMA_CHAIN4 ||
          physical_plan ==
              QBH_PHYSICAL_PLAN_STREAMING_CAP2_E7_DMA_CHAIN4 ||
          physical_plan ==
              QBH_PHYSICAL_PLAN_STREAMING_BITWISE_E7_DMA_CHAIN4 ||
          physical_plan ==
              QBH_PHYSICAL_PLAN_STREAMING_DIRECT_E7_DMA_CHAIN4 ||
          physical_plan ==
              QBH_PHYSICAL_PLAN_STREAMING_LOCKFREE_E7_DMA_CHAIN4) &&
         compressed_slot_count != 8U)) {
        return -1;
    }
    if (((physical_plan == QBH_PHYSICAL_PLAN_STREAMING_DMA_BATCH2 ||
          physical_plan ==
              QBH_PHYSICAL_PLAN_STREAMING_CAP2_DMA_BATCH2) &&
         (variant != QBH_PROJECTION_DOWN || compressed_slot_count != 4U ||
          chunk_tiles != QBH_W4_WIDE_CHUNK_TILES)) ||
        ((physical_plan ==
              QBH_PHYSICAL_PLAN_STREAMING_E7_DMA_CHAIN4 ||
          physical_plan ==
              QBH_PHYSICAL_PLAN_STREAMING_CAP2_E7_DMA_CHAIN4 ||
          physical_plan ==
              QBH_PHYSICAL_PLAN_STREAMING_BITWISE_E7_DMA_CHAIN4 ||
          physical_plan ==
              QBH_PHYSICAL_PLAN_STREAMING_DIRECT_E7_DMA_CHAIN4 ||
          physical_plan ==
              QBH_PHYSICAL_PLAN_STREAMING_LOCKFREE_E7_DMA_CHAIN4) &&
         (variant != QBH_PROJECTION_GATE_UP_PAIR ||
          compressed_slot_count != 8U ||
          chunk_tiles != QBH_W4_COARSE_CHUNK_TILES))) {
        return -1;
    }
    if (weight_storage_variant == QBH_WEIGHT_PACKED_W4_HMX_SCALE &&
        !qbh_physical_plan_is_chunked(physical_plan)) {
        return -1;
    }
    if (compressed_slot_count <
            QBH_W4_DEFAULT_COMPRESSED_SLOT_COUNT ||
        compressed_slot_count > QBH_W4_MAX_COMPRESSED_SLOT_COUNT ||
        (qbh_physical_plan_is_full_bundle(physical_plan) &&
         compressed_slot_count !=
             QBH_W4_DEFAULT_COMPRESSED_SLOT_COUNT) ||
        (chunk_tiles != QBH_W4_DEFAULT_CHUNK_TILES &&
         chunk_tiles != QBH_W4_FINE_CHUNK_TILES &&
         chunk_tiles != QBH_W4_COARSE_CHUNK_TILES &&
         chunk_tiles != QBH_W4_WIDE_CHUNK_TILES)) {
        return -1;
    }

    layout->variant = variant;
    layout->weight_storage_variant = weight_storage_variant;
    layout->physical_plan = physical_plan;
    layout->compressed_slot_count = compressed_slot_count;
    layout->expanded_slot_count =
        qbh_physical_plan_expanded_slot_count(physical_plan);
    layout->m = QBH_PROJ_M;
    layout->k = k;
    layout->n = n;
    layout->k_tiles = k / QBH_HMX_INPUT_CHANNELS;
    layout->n_tiles = n / QBH_HMX_OUTPUT_CHANNELS;
    layout->activation_bytes = layout->m * layout->k;
    layout->logical_weight_bytes = layout->k * layout->n;
    layout->expanded_weight_chunk_bytes =
        layout->k_tiles * QBH_HMX_WEIGHT_BYTES;
    layout->expanded_weight_bundle_bytes =
        layout->expanded_weight_chunk_bytes + QBH_HMX_BIAS_BYTES;
    layout->expanded_weight_bytes =
        layout->n_tiles * layout->expanded_weight_bundle_bytes;
    layout->w4_packed_chunk_bytes =
        layout->k_tiles * QBH_W4_PACKED_TILE_BYTES;
    layout->w4_scale_offset = layout->w4_packed_chunk_bytes;
    layout->w4_bias_offset = qbh_align_up_u32(
        layout->w4_scale_offset + QBH_W4_CHANNEL_SCALE_BYTES,
        QBH_W4_METADATA_ALIGNMENT);
    layout->w4_bundle_bytes =
        layout->w4_bias_offset + QBH_HMX_BIAS_BYTES;
    layout->w4_weight_bytes =
        layout->n_tiles * layout->w4_bundle_bytes;
    layout->stored_weight_bundle_bytes =
        qbh_weight_storage_is_packed_w4(weight_storage_variant)
            ? layout->w4_bundle_bytes
            : layout->expanded_weight_bundle_bytes;
    layout->stored_weight_bytes =
        layout->n_tiles * layout->stored_weight_bundle_bytes;
    layout->output_bytes = layout->m * layout->n;
    layout->output_tiles_bytes =
        layout->n_tiles * QBH_HMX_OUTPUT_BYTES;
    layout->hmx_pairs_per_repeat =
        layout->k_tiles * layout->n_tiles;
    layout->k_streams_per_output =
        (layout->k_tiles + QBH_HMX_MAX_STREAM_TILES - 1U) /
        QBH_HMX_MAX_STREAM_TILES;
    layout->chunk_tiles = chunk_tiles;
    layout->chunks_per_output =
        (layout->k_tiles + chunk_tiles - 1U) / chunk_tiles;
    layout->hmx_streams_per_repeat =
        layout->n_tiles * layout->k_streams_per_output;
    layout->expanded_chunk_weight_bytes =
        layout->chunk_tiles * QBH_HMX_WEIGHT_BYTES;
    layout->expanded_chunk_slot_bytes = qbh_align_up_u32(
        layout->expanded_chunk_weight_bytes + QBH_HMX_BIAS_BYTES,
        QBH_W4_METADATA_ALIGNMENT);

    layout->vtcm_activation_offset = 0;
    layout->vtcm_compressed_slot0_offset = qbh_align_up_u32(
        layout->vtcm_activation_offset + layout->activation_bytes,
        QBH_W4_METADATA_ALIGNMENT);
    layout->vtcm_compressed_slot1_offset = qbh_align_up_u32(
        layout->vtcm_compressed_slot0_offset + layout->w4_bundle_bytes,
        QBH_W4_METADATA_ALIGNMENT);
    layout->vtcm_expanded_slot0_offset = qbh_align_up_u32(
        layout->vtcm_compressed_slot1_offset + layout->w4_bundle_bytes,
        QBH_HMX_ACTIVATION_BYTES);
    layout->vtcm_expanded_slot1_offset = qbh_align_up_u32(
        layout->vtcm_expanded_slot0_offset +
            layout->expanded_weight_bundle_bytes,
        QBH_W4_METADATA_ALIGNMENT);
    layout->vtcm_output_offset = qbh_align_up_u32(
        layout->vtcm_expanded_slot1_offset +
            layout->expanded_weight_bundle_bytes,
        QBH_HMX_OUTPUT_BYTES);
    layout->vtcm_full_bundle_plan_bytes =
        layout->vtcm_output_offset + layout->output_tiles_bytes;

    layout->vtcm_chunked_expanded_slots_offset = qbh_align_up_u32(
        layout->vtcm_compressed_slot0_offset +
            compressed_slot_count * layout->w4_bundle_bytes,
        QBH_W4_METADATA_ALIGNMENT);
    layout->vtcm_chunked_output_offset = qbh_align_up_u32(
        layout->vtcm_chunked_expanded_slots_offset +
            layout->expanded_slot_count *
                layout->expanded_chunk_slot_bytes,
        QBH_HMX_OUTPUT_BYTES);
    layout->vtcm_chunked_plan_bytes =
        layout->vtcm_chunked_output_offset + layout->output_tiles_bytes;
    if (qbh_physical_plan_is_chunked(physical_plan)) {
        layout->vtcm_output_offset = layout->vtcm_chunked_output_offset;
        layout->vtcm_plan_bytes = layout->vtcm_chunked_plan_bytes;
    } else {
        layout->vtcm_plan_bytes = layout->vtcm_full_bundle_plan_bytes;
    }

    return layout->hmx_pairs_per_repeat ==
                       (variant == QBH_PROJECTION_GATE_UP_PAIR
                            ? QBH_QWEN3_PAIRED_HMX_PAIRS_PER_REPEAT
                            : QBH_QWEN3_HMX_PAIRS_PER_REPEAT) &&
                   layout->vtcm_full_bundle_plan_bytes <=
                       QBH_W4U8_VTCM_BYTES &&
                   layout->vtcm_chunked_plan_bytes <=
                       QBH_W4U8_VTCM_BYTES &&
                   layout->vtcm_plan_bytes <= QBH_W4U8_VTCM_BYTES
               ? 0
               : -1;
}

static inline uint32_t qbh_projection_chunk_tiles(
    const struct qbh_projection_layout *layout, uint32_t chunk_index) {
    uint32_t first_tile = chunk_index * layout->chunk_tiles;
    uint32_t remaining = layout->k_tiles - first_tile;
    return remaining < layout->chunk_tiles ? remaining
                                           : layout->chunk_tiles;
}

static inline uint32_t qbh_projection_expanded_chunk_offset(
    const struct qbh_projection_layout *layout, uint32_t slot) {
    return layout->vtcm_chunked_expanded_slots_offset +
           slot * layout->expanded_chunk_slot_bytes;
}

static inline uint32_t qbh_projection_compressed_slot_offset(
    const struct qbh_projection_layout *layout, uint32_t slot) {
    return layout->vtcm_compressed_slot0_offset +
           slot * layout->w4_bundle_bytes;
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

int32_t qbh_hmx_accumulate_u8s8_streaming(
    const uint8_t *activation_tiles, const int8_t *expanded_weight_tiles,
    const uint32_t *bias_words, uint32_t begin_output,
    const volatile uint32_t *ready_generations,
    uint32_t expected_generation, uint32_t stream_count,
    volatile int32_t *abort_status, uint64_t timeout_ticks,
    uint64_t *ready_wait_ticks,
    volatile uint32_t *hmx_consumption_started);

static inline size_t qbh_projection_expanded_bundle_offset(
    const struct qbh_projection_layout *layout, uint32_t output_tile) {
    return (size_t)output_tile * layout->expanded_weight_bundle_bytes;
}

static inline size_t qbh_projection_expanded_weight_offset(
    const struct qbh_projection_layout *layout, uint32_t output_tile,
    uint32_t input_tile, uint32_t input_channel,
    uint32_t output_channel) {
    return qbh_projection_expanded_bundle_offset(layout, output_tile) +
           (size_t)input_tile * QBH_HMX_WEIGHT_BYTES +
           qbh_packed_weight_offset(input_channel, output_channel);
}

static inline size_t qbh_projection_expanded_bias_offset(
    const struct qbh_projection_layout *layout, uint32_t output_tile) {
    return qbh_projection_expanded_bundle_offset(layout, output_tile) +
           layout->expanded_weight_chunk_bytes;
}

static inline size_t qbh_projection_w4_bundle_offset(
    const struct qbh_projection_layout *layout, uint32_t output_tile) {
    return (size_t)output_tile * layout->w4_bundle_bytes;
}

static inline size_t qbh_projection_w4_scale_offset(
    const struct qbh_projection_layout *layout, uint32_t output_tile) {
    return qbh_projection_w4_bundle_offset(layout, output_tile) +
           layout->w4_scale_offset;
}

static inline size_t qbh_projection_w4_bias_offset(
    const struct qbh_projection_layout *layout, uint32_t output_tile) {
    return qbh_projection_w4_bundle_offset(layout, output_tile) +
           layout->w4_bias_offset;
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
