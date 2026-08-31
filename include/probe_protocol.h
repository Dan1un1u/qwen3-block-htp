#ifndef QWEN3_BLOCK_HTP_PROBE_PROTOCOL_H
#define QWEN3_BLOCK_HTP_PROBE_PROTOCOL_H

#include <stdint.h>

#define QBH_PROBE_MAGIC UINT32_C(0x51424850)
#define QBH_PROBE_ABI_VERSION UINT32_C(21)
#define QBH_PROBE_ALIGNMENT UINT32_C(4096)

#define QBH_HMX_SPATIAL UINT32_C(64)
#define QBH_HMX_INPUT_CHANNELS UINT32_C(32)
#define QBH_HMX_OUTPUT_CHANNELS UINT32_C(32)
#define QBH_HMX_ACTIVATION_BYTES UINT32_C(2048)
#define QBH_HMX_WEIGHT_BYTES UINT32_C(1024)
#define QBH_HMX_OUTPUT_BYTES UINT32_C(2048)
#define QBH_HMX_BIAS_BYTES UINT32_C(256)
#define QBH_HMX_DEFAULT_ZERO_POINT UINT32_C(128)
#define QBH_HMX_DEFAULT_REPEATS UINT32_C(1)
#define QBH_HMX_MAX_REPEATS UINT32_C(100000)
#define QBH_HMX_MAX_STREAM_TILES UINT32_C(32)

#define QBH_MAX_HVX_WORKERS UINT32_C(6)
#define QBH_W4_DEFAULT_COMPRESSED_SLOT_COUNT UINT32_C(2)
#define QBH_W4_MAX_COMPRESSED_SLOT_COUNT UINT32_C(16)
#define QBH_W4_EXPANDED_CHUNK_SLOT_COUNT UINT32_C(8)
#define QBH_W4_MAX_EXPANDED_CHUNK_SLOT_COUNT UINT32_C(16)
#define QBH_W4_REBALANCED_EXPANDED_SLOT_COUNT UINT32_C(7)
#define QBH_W4_TASK_QUEUE_DEPTH UINT32_C(32)
#define QBH_W4_DEFAULT_CHUNK_TILES UINT32_C(32)
#define QBH_W4_FINE_CHUNK_TILES UINT32_C(16)
#define QBH_W4_COARSE_CHUNK_TILES UINT32_C(64)
#define QBH_W4_WIDE_CHUNK_TILES UINT32_C(96)
#define QBH_W4_STREAM_REGION_TILES UINT32_C(32)
#define QBH_W4_MAX_STREAM_REGIONS UINT32_C(3)

#define QBH_W4_PACKED_TILE_BYTES UINT32_C(512)
#define QBH_W4_CHANNEL_SCALE_BYTES UINT32_C(32)
#define QBH_W4_METADATA_ALIGNMENT UINT32_C(256)
#define QBH_W4_MIN_VALUE INT32_C(-7)
#define QBH_W4_MAX_VALUE INT32_C(7)
#define QBH_W4_MAX_INTEGER_SCALE UINT32_C(18)

#define QBH_PROJ_M UINT32_C(64)
#define QBH_GATE_UP_K UINT32_C(2048)
#define QBH_GATE_UP_N UINT32_C(6144)
#define QBH_GATE_UP_PAIR_N UINT32_C(12288)
#define QBH_MAX_PROJECTION_N QBH_GATE_UP_PAIR_N
#define QBH_DOWN_K UINT32_C(6144)
#define QBH_DOWN_N UINT32_C(2048)
#define QBH_EXPECTED_FULL_VTCM_BYTES UINT32_C(8388608)
#define QBH_FULL_VTCM_MIN_PAGE_BYTES UINT32_C(4194304)
#define QBH_W4U8_VTCM_BYTES QBH_EXPECTED_FULL_VTCM_BYTES
#define QBH_QWEN3_HMX_PAIRS_PER_REPEAT UINT32_C(12288)
#define QBH_QWEN3_PAIRED_HMX_PAIRS_PER_REPEAT UINT32_C(24576)

enum qbh_projection_variant {
    QBH_PROJECTION_GATE_UP = 1,
    QBH_PROJECTION_DOWN = 2,
    QBH_PROJECTION_GATE_UP_PAIR = 3,
};

enum qbh_weight_storage_variant {
    QBH_WEIGHT_EXPANDED_S8 = 1,
    QBH_WEIGHT_PACKED_W4 = 2,
    QBH_WEIGHT_PACKED_W4_HMX_SCALE = 3,
};

static inline int qbh_weight_storage_is_packed_w4(uint32_t storage) {
    return storage == QBH_WEIGHT_PACKED_W4 ||
           storage == QBH_WEIGHT_PACKED_W4_HMX_SCALE;
}

enum qbh_physical_plan {
    QBH_PHYSICAL_PLAN_FULL_BUNDLE = 1,
    QBH_PHYSICAL_PLAN_CHUNKED = 2,
    QBH_PHYSICAL_PLAN_FULL_BUNDLE_DMA_BATCH2 = 3,
    QBH_PHYSICAL_PLAN_CHUNKED_DMA_BATCH2 = 4,
    QBH_PHYSICAL_PLAN_CHUNKED_DMA_BATCH4 = 5,
    QBH_PHYSICAL_PLAN_FULL_BUNDLE_DMA_CHAIN2 = 6,
    QBH_PHYSICAL_PLAN_CHUNKED_DMA_CHAIN2 = 7,
    QBH_PHYSICAL_PLAN_CHUNKED_DMA_CHAIN4 = 8,
    QBH_PHYSICAL_PLAN_CHUNKED_E7_DMA_BATCH2 = 9,
    QBH_PHYSICAL_PLAN_CHUNKED_E7_DMA_BATCH4 = 10,
    QBH_PHYSICAL_PLAN_CHUNKED_E7_DMA_CHAIN4 = 11,
    QBH_PHYSICAL_PLAN_STREAMING_DMA_BATCH2 = 12,
    QBH_PHYSICAL_PLAN_STREAMING_E7_DMA_CHAIN4 = 13,
    QBH_PHYSICAL_PLAN_STREAMING_CAP2_DMA_BATCH2 = 14,
    QBH_PHYSICAL_PLAN_STREAMING_CAP2_E7_DMA_CHAIN4 = 15,
    QBH_PHYSICAL_PLAN_STREAMING_E7_DMA_CHAIN8 = 16,
};

static inline int qbh_physical_plan_is_full_bundle(uint32_t plan) {
    return plan == QBH_PHYSICAL_PLAN_FULL_BUNDLE ||
           plan == QBH_PHYSICAL_PLAN_FULL_BUNDLE_DMA_BATCH2 ||
           plan == QBH_PHYSICAL_PLAN_FULL_BUNDLE_DMA_CHAIN2;
}

static inline int qbh_physical_plan_is_chunked(uint32_t plan) {
    return plan == QBH_PHYSICAL_PLAN_CHUNKED ||
           plan == QBH_PHYSICAL_PLAN_CHUNKED_DMA_BATCH2 ||
           plan == QBH_PHYSICAL_PLAN_CHUNKED_DMA_BATCH4 ||
           plan == QBH_PHYSICAL_PLAN_CHUNKED_DMA_CHAIN2 ||
           plan == QBH_PHYSICAL_PLAN_CHUNKED_DMA_CHAIN4 ||
           plan == QBH_PHYSICAL_PLAN_CHUNKED_E7_DMA_BATCH2 ||
           plan == QBH_PHYSICAL_PLAN_CHUNKED_E7_DMA_BATCH4 ||
           plan == QBH_PHYSICAL_PLAN_CHUNKED_E7_DMA_CHAIN4 ||
           plan == QBH_PHYSICAL_PLAN_STREAMING_DMA_BATCH2 ||
           plan == QBH_PHYSICAL_PLAN_STREAMING_E7_DMA_CHAIN4 ||
           plan == QBH_PHYSICAL_PLAN_STREAMING_E7_DMA_CHAIN8 ||
           plan == QBH_PHYSICAL_PLAN_STREAMING_CAP2_DMA_BATCH2 ||
           plan == QBH_PHYSICAL_PLAN_STREAMING_CAP2_E7_DMA_CHAIN4;
}

static inline uint32_t qbh_physical_plan_dma_bundle_batch(uint32_t plan) {
    if (plan == QBH_PHYSICAL_PLAN_FULL_BUNDLE_DMA_BATCH2 ||
        plan == QBH_PHYSICAL_PLAN_CHUNKED_DMA_BATCH2 ||
        plan == QBH_PHYSICAL_PLAN_FULL_BUNDLE_DMA_CHAIN2 ||
        plan == QBH_PHYSICAL_PLAN_CHUNKED_DMA_CHAIN2 ||
        plan == QBH_PHYSICAL_PLAN_CHUNKED_E7_DMA_BATCH2 ||
        plan == QBH_PHYSICAL_PLAN_STREAMING_DMA_BATCH2 ||
        plan == QBH_PHYSICAL_PLAN_STREAMING_CAP2_DMA_BATCH2) {
        return 2U;
    }
    if (plan == QBH_PHYSICAL_PLAN_CHUNKED_DMA_BATCH4 ||
        plan == QBH_PHYSICAL_PLAN_CHUNKED_DMA_CHAIN4 ||
        plan == QBH_PHYSICAL_PLAN_CHUNKED_E7_DMA_BATCH4 ||
        plan == QBH_PHYSICAL_PLAN_CHUNKED_E7_DMA_CHAIN4 ||
        plan == QBH_PHYSICAL_PLAN_STREAMING_E7_DMA_CHAIN4 ||
        plan == QBH_PHYSICAL_PLAN_STREAMING_CAP2_E7_DMA_CHAIN4) {
        return 4U;
    }
    if (plan == QBH_PHYSICAL_PLAN_STREAMING_E7_DMA_CHAIN8) {
        return 8U;
    }
    return 1U;
}

static inline int qbh_physical_plan_uses_linked_dma(uint32_t plan) {
    return plan == QBH_PHYSICAL_PLAN_FULL_BUNDLE_DMA_CHAIN2 ||
           plan == QBH_PHYSICAL_PLAN_CHUNKED_DMA_CHAIN2 ||
           plan == QBH_PHYSICAL_PLAN_CHUNKED_DMA_CHAIN4 ||
           plan == QBH_PHYSICAL_PLAN_CHUNKED_E7_DMA_CHAIN4 ||
           plan == QBH_PHYSICAL_PLAN_STREAMING_E7_DMA_CHAIN4 ||
           plan == QBH_PHYSICAL_PLAN_STREAMING_E7_DMA_CHAIN8 ||
           plan == QBH_PHYSICAL_PLAN_STREAMING_CAP2_E7_DMA_CHAIN4;
}

static inline uint32_t qbh_physical_plan_expanded_slot_count(
    uint32_t plan) {
    return plan == QBH_PHYSICAL_PLAN_CHUNKED_E7_DMA_BATCH2 ||
                   plan == QBH_PHYSICAL_PLAN_CHUNKED_E7_DMA_BATCH4 ||
                   plan == QBH_PHYSICAL_PLAN_CHUNKED_E7_DMA_CHAIN4 ||
                   plan == QBH_PHYSICAL_PLAN_STREAMING_E7_DMA_CHAIN4 ||
                   plan == QBH_PHYSICAL_PLAN_STREAMING_E7_DMA_CHAIN8 ||
                   plan == QBH_PHYSICAL_PLAN_STREAMING_CAP2_E7_DMA_CHAIN4
               ? QBH_W4_REBALANCED_EXPANDED_SLOT_COUNT
               : QBH_W4_EXPANDED_CHUNK_SLOT_COUNT;
}

static inline int qbh_physical_plan_is_streaming(uint32_t plan) {
    return plan == QBH_PHYSICAL_PLAN_STREAMING_DMA_BATCH2 ||
           plan == QBH_PHYSICAL_PLAN_STREAMING_E7_DMA_CHAIN4 ||
           plan == QBH_PHYSICAL_PLAN_STREAMING_E7_DMA_CHAIN8 ||
           plan == QBH_PHYSICAL_PLAN_STREAMING_CAP2_DMA_BATCH2 ||
           plan == QBH_PHYSICAL_PLAN_STREAMING_CAP2_E7_DMA_CHAIN4;
}

static inline int qbh_physical_plan_is_capped_streaming(uint32_t plan) {
    return plan == QBH_PHYSICAL_PLAN_STREAMING_CAP2_DMA_BATCH2 ||
           plan == QBH_PHYSICAL_PLAN_STREAMING_CAP2_E7_DMA_CHAIN4;
}

enum qbh_probe_pattern {
    QBH_PATTERN_IDENTITY = 1,
    QBH_PATTERN_SIGNED = 2,
    QBH_PATTERN_STRUCTURED = 3,
    QBH_PATTERN_BOUNDARY = 4,
};

enum qbh_output_assembly_mode {
    QBH_OUTPUT_ASSEMBLY_SCALAR = 1,
    QBH_OUTPUT_ASSEMBLY_LINKED_2D_DMA = 2,
};

enum qbh_resource_lifetime_mode {
    QBH_RESOURCE_LIFETIME_TRANSIENT = 1,
    QBH_RESOURCE_LIFETIME_PREPARED_SESSION = 2,
};

enum qbh_probe_status {
    QBH_PROBE_STATUS_HOST_INITIALIZED = 1,
    QBH_PROBE_STATUS_DSP_RUNNING = 2,
    QBH_PROBE_STATUS_OK = 0,
    QBH_PROBE_STATUS_BAD_HEADER = -1,
    QBH_PROBE_STATUS_CACHE_INVALIDATE_FAILED = -2,
    QBH_PROBE_STATUS_VTCM_CONFIG_FAILED = -3,
    QBH_PROBE_STATUS_VTCM_ACQUIRE_FAILED = -4,
    QBH_PROBE_STATUS_VTCM_POINTER_FAILED = -5,
    QBH_PROBE_STATUS_CACHE_FLUSH_FAILED = -6,
    QBH_PROBE_STATUS_HMX_CONFIG_FAILED = -7,
    QBH_PROBE_STATUS_HMX_LOCK_FAILED = -8,
    QBH_PROBE_STATUS_VTCM_ALIGNMENT_FAILED = -9,
    QBH_PROBE_STATUS_HVX_LOCK_FAILED = -10,
    QBH_PROBE_STATUS_HMX_RELEASE_FAILED = -11,
    QBH_PROBE_STATUS_HMX_THREAD_FAILED = -12,
    QBH_PROBE_STATUS_HMX_POWER_FAILED = -13,
    QBH_PROBE_STATUS_DMA_FAILED = -14,
    QBH_PROBE_STATUS_SYNC_FAILED = -15,
    QBH_PROBE_STATUS_LAYOUT_FAILED = -16,
    QBH_PROBE_STATUS_W4_EXPAND_FAILED = -17,
    QBH_PROBE_STATUS_DCVS_POWER_FAILED = -18,
    QBH_PROBE_STATUS_OUTPUT_DMA_FAILED = -19,
};

struct qbh_probe_header {
    uint32_t magic;
    uint32_t abi_version;
    uint32_t header_bytes;
    uint32_t total_bytes;

    uint32_t pattern;
    uint32_t projection_variant;
    uint32_t weight_storage_variant;
    uint32_t physical_plan;
    uint32_t requested_hvx_workers;
    uint32_t compressed_slot_count;
    uint32_t expanded_chunk_slot_count;
    uint32_t chunk_tiles;
    uint32_t output_assembly_mode;
    uint32_t resource_lifetime_mode;
    uint32_t activation_offset;
    uint32_t weight_offset;
    uint32_t output_offset;
    uint32_t input_zero_point;
    uint32_t repeat_count;

    int32_t dsp_status;
    uint32_t vtcm_requested_bytes;
    uint32_t vtcm_acquired_bytes;
    int32_t cache_status;
    int32_t hmx_resource_status;
    int32_t hmx_lock_status;
    int32_t hmx_unlock_status;
    int32_t hmx_release_status;
    int32_t hmx_thread_create_status;
    int32_t hmx_thread_join_status;
    int32_t hmx_power_up_status;
    int32_t hmx_power_down_status;
    int32_t dcvs_power_setup_status;
    int32_t dcvs_power_reset_status;
    uint32_t hmx_execution_count;
    uint32_t hmx_stream_count;
    int32_t hvx_lock_status;
    int32_t hvx_unlock_status;

    uint32_t projection_m;
    uint32_t projection_k;
    uint32_t projection_n;
    uint32_t k_tile_count;
    uint32_t n_tile_count;
    uint32_t stored_weight_bundle_bytes;
    uint32_t expanded_weight_bundle_bytes;
    uint32_t vtcm_plan_bytes;
    uint32_t k_streams_per_output;
    uint32_t stored_weight_bytes_per_repeat;
    uint32_t expanded_weight_bytes_per_repeat;
    uint32_t weight_expand_count;
    uint32_t activation_stage_count;
    uint32_t weight_bundle_stage_count;
    uint32_t output_tile_count;
    uint32_t dma_submit_count;
    uint32_t dma_wait_count;
    uint32_t dma_descriptor_count;
    uint32_t dma_chain_count;
    uint32_t dma_descriptor_completion_count;
    uint32_t dma_descriptor_timeout_count;
    uint32_t output_dma_submit_count;
    uint32_t output_dma_wait_count;
    uint32_t output_dma_descriptor_count;
    uint32_t output_dma_chain_count;
    uint32_t output_dma_descriptor_completion_count;
    uint32_t output_dma_descriptor_timeout_count;
    int32_t output_dma_status;
    uint32_t resource_setup_in_run;
    uint32_t resource_release_in_run;
    uint32_t prepared_session_run_index;
    uint32_t resource_vtcm_address;
    uint32_t resource_hmx_context_id;
    uint32_t weight_slot_reuse_count;
    uint32_t expanded_chunk_slot_reuse_count;
    uint32_t chunks_per_output;
    uint32_t chunk_expand_count;
    int32_t dma_status;
    int32_t sync_status;
    uint32_t streaming_region_publish_count;
    uint32_t streaming_ready_timeout_count;
    uint32_t hmx_batch_output_count;
    uint32_t hmx_in_command_slot_release_count;
    uint32_t hmx_producer_progress_command_count;
    uint32_t hmx_batch_reserved;

    uint32_t hvx_units_128b;
    uint32_t hvx_workers_created;
    uint32_t hvx_workers_locked;
    uint32_t hvx_max_active_workers;
    uint32_t hvx_hmx_overlap_observed;
    uint32_t hvx_parallel_overlap_observed;
    int32_t hvx_thread_create_status;
    int32_t hvx_thread_join_status;
    int32_t hvx_worker_lock_status[QBH_MAX_HVX_WORKERS];
    int32_t hvx_worker_unlock_status[QBH_MAX_HVX_WORKERS];

    uint64_t qtimer_start;
    uint64_t qtimer_end;
    uint64_t qtimer_elapsed;
    uint64_t pcycles_start;
    uint64_t pcycles_end;

    uint64_t activation_stage_ticks;
    uint64_t weight_stage_ticks;
    uint64_t weight_expand_ticks;
    uint64_t hmx_compute_ticks;
    uint64_t hmx_ready_wait_ticks;
    uint64_t producer_slot_wait_ticks;
    uint64_t expanded_slot_wait_ticks;
    uint64_t pipeline_ticks;
    uint64_t output_assembly_ticks;
    uint64_t dsp_total_ticks;
    uint64_t input_cache_ticks;
    uint64_t output_cache_ticks;
    uint64_t expand_window_start;
    uint64_t expand_window_end;
    uint64_t hmx_window_start;
    uint64_t hmx_window_end;
    uint64_t hvx_worker_ticks[QBH_MAX_HVX_WORKERS];
};

_Static_assert(sizeof(struct qbh_probe_header) == 624,
               "probe header ABI changed");

#endif
