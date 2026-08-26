#ifndef QWEN3_BLOCK_HTP_PROBE_PROTOCOL_H
#define QWEN3_BLOCK_HTP_PROBE_PROTOCOL_H

#include <stdint.h>

#define QBH_PROBE_MAGIC UINT32_C(0x51424850)
#define QBH_PROBE_ABI_VERSION UINT32_C(3)
#define QBH_PROBE_ALIGNMENT UINT32_C(4096)

#define QBH_HMX_SPATIAL UINT32_C(64)
#define QBH_HMX_INPUT_CHANNELS UINT32_C(32)
#define QBH_HMX_OUTPUT_CHANNELS UINT32_C(32)
#define QBH_HMX_ACTIVATION_BYTES UINT32_C(2048)
#define QBH_HMX_WEIGHT_BYTES UINT32_C(1024)
#define QBH_HMX_OUTPUT_BYTES UINT32_C(2048)
#define QBH_HMX_BIAS_BYTES UINT32_C(256)
#define QBH_HMX_VTCM_BYTES UINT32_C(8192)
#define QBH_HMX_DEFAULT_ZERO_POINT UINT32_C(128)
#define QBH_HMX_DEFAULT_REPEATS UINT32_C(1)
#define QBH_HMX_MAX_REPEATS UINT32_C(100000)

#define QBH_PROJ_M UINT32_C(64)
#define QBH_PROJ_K UINT32_C(128)
#define QBH_PROJ_N UINT32_C(96)
#define QBH_PROJ_K_TILES (QBH_PROJ_K / QBH_HMX_INPUT_CHANNELS)
#define QBH_PROJ_N_TILES (QBH_PROJ_N / QBH_HMX_OUTPUT_CHANNELS)
#define QBH_PROJ_ACTIVATION_BYTES (QBH_PROJ_M * QBH_PROJ_K)
#define QBH_PROJ_LOGICAL_WEIGHT_BYTES (QBH_PROJ_K * QBH_PROJ_N)
#define QBH_PROJ_WEIGHT_TILE_BYTES QBH_HMX_WEIGHT_BYTES
#define QBH_PROJ_WEIGHT_CHUNK_BYTES \
    (QBH_PROJ_K_TILES * QBH_PROJ_WEIGHT_TILE_BYTES)
#define QBH_PROJ_WEIGHT_BUNDLE_BYTES \
    (QBH_PROJ_WEIGHT_CHUNK_BYTES + QBH_HMX_BIAS_BYTES)
#define QBH_PROJ_PACKED_WEIGHT_BYTES \
    (QBH_PROJ_N_TILES * QBH_PROJ_WEIGHT_BUNDLE_BYTES)
#define QBH_PROJ_OUTPUT_BYTES (QBH_PROJ_M * QBH_PROJ_N)
#define QBH_PROJ_OUTPUT_TILES_BYTES \
    (QBH_PROJ_N_TILES * QBH_HMX_OUTPUT_BYTES)
#define QBH_PROJ_HMX_PAIRS_PER_REPEAT \
    (QBH_PROJ_K_TILES * QBH_PROJ_N_TILES)
#define QBH_PROJ_VTCM_BYTES UINT32_C(32768)

enum qbh_probe_pattern {
    QBH_PATTERN_IDENTITY = 1,
    QBH_PATTERN_SIGNED = 2,
    QBH_PATTERN_STRUCTURED = 3,
    QBH_PATTERN_BOUNDARY = 4,
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
};

struct qbh_probe_header {
    uint32_t magic;
    uint32_t abi_version;
    uint32_t header_bytes;
    uint32_t total_bytes;

    uint32_t pattern;
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
    uint32_t hmx_execution_count;
    int32_t hvx_lock_status;
    int32_t hvx_unlock_status;

    uint32_t projection_m;
    uint32_t projection_k;
    uint32_t projection_n;
    uint32_t k_tile_count;
    uint32_t n_tile_count;
    uint32_t activation_stage_count;
    uint32_t weight_bundle_stage_count;
    uint32_t output_tile_count;
    uint32_t dma_submit_count;
    uint32_t dma_wait_count;
    uint32_t weight_slot_reuse_count;
    int32_t dma_status;
    int32_t sync_status;

    uint64_t qtimer_start;
    uint64_t qtimer_end;
    uint64_t qtimer_elapsed;
    uint64_t pcycles_start;
    uint64_t pcycles_end;

    uint64_t activation_stage_ticks;
    uint64_t weight_stage_ticks;
    uint64_t hmx_compute_ticks;
    uint64_t hmx_ready_wait_ticks;
    uint64_t producer_slot_wait_ticks;
    uint64_t pipeline_ticks;
    uint64_t output_assembly_ticks;
    uint64_t dsp_total_ticks;
};

_Static_assert(sizeof(struct qbh_probe_header) == 256,
               "probe header ABI changed");

#endif
