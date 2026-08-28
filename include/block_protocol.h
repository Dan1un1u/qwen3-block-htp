#ifndef QWEN3_BLOCK_HTP_BLOCK_PROTOCOL_H
#define QWEN3_BLOCK_HTP_BLOCK_PROTOCOL_H

#include <stdint.h>

#include "probe_protocol.h"

#define QBH_BLOCK_MAGIC UINT32_C(0x5142424c)
#define QBH_BLOCK_ABI_VERSION UINT32_C(6)
#define QBH_BLOCK_EXPERIMENT UINT32_C(27)

#define QBH_BLOCK_M UINT32_C(64)
#define QBH_BLOCK_HIDDEN UINT32_C(2048)
#define QBH_BLOCK_INTERMEDIATE UINT32_C(6144)
#define QBH_BLOCK_HEADS UINT32_C(16)
#define QBH_BLOCK_KV_HEADS UINT32_C(8)
#define QBH_BLOCK_HEAD_DIM UINT32_C(128)
#define QBH_BLOCK_KV_HIDDEN \
    (QBH_BLOCK_KV_HEADS * QBH_BLOCK_HEAD_DIM)
#define QBH_BLOCK_PROJECTION_COUNT UINT32_C(7)
#define QBH_BLOCK_QPARAM_COUNT UINT32_C(17)
#define QBH_BLOCK_QPARAM_RECORD_BYTES UINT32_C(48)

enum qbh_block_variant {
    QBH_BLOCK_F16F16 = 1,
    QBH_BLOCK_W4F16 = 2,
    QBH_BLOCK_W4U8 = 3,
};

enum qbh_block_common_ops_mask {
    QBH_BLOCK_COMMON_OPS_SCALAR = 0,
    QBH_BLOCK_COMMON_OP_RMS_NORM = 1U << 0,
    QBH_BLOCK_COMMON_OP_ROPE = 1U << 1,
    QBH_BLOCK_COMMON_OP_SOFTMAX = 1U << 2,
    QBH_BLOCK_COMMON_OP_SILU = 1U << 3,
    QBH_BLOCK_COMMON_OPS_HVX_FP16 =
        QBH_BLOCK_COMMON_OP_RMS_NORM |
        QBH_BLOCK_COMMON_OP_ROPE |
        QBH_BLOCK_COMMON_OP_SOFTMAX |
        QBH_BLOCK_COMMON_OP_SILU,
};

enum qbh_block_residual_mode {
    QBH_BLOCK_RESIDUAL_SCALAR = 0,
    QBH_BLOCK_RESIDUAL_HVX = 1,
    QBH_BLOCK_RESIDUAL_HVX_FUSED_POST_NORM = 2,
};

enum qbh_block_f16f16_projection_mode {
    QBH_BLOCK_F16F16_PROJECTION_SERIAL = 0,
    QBH_BLOCK_F16F16_PROJECTION_ASYNC_SINGLE = 1,
    QBH_BLOCK_F16F16_PROJECTION_BATCH2 = 2,
};

enum qbh_block_projection_index {
    QBH_BLOCK_PROJ_Q = 0,
    QBH_BLOCK_PROJ_K = 1,
    QBH_BLOCK_PROJ_V = 2,
    QBH_BLOCK_PROJ_O = 3,
    QBH_BLOCK_PROJ_GATE = 4,
    QBH_BLOCK_PROJ_UP = 5,
    QBH_BLOCK_PROJ_DOWN = 6,
};

enum qbh_block_qparam_index {
    QBH_BLOCK_QP_BLOCK_INPUT = 0,
    QBH_BLOCK_QP_INPUT_NORM = 1,
    QBH_BLOCK_QP_Q_PROJECTION = 2,
    QBH_BLOCK_QP_K_PROJECTION = 3,
    QBH_BLOCK_QP_V = 4,
    QBH_BLOCK_QP_Q_ROPE = 5,
    QBH_BLOCK_QP_K_ROPE = 6,
    QBH_BLOCK_QP_ATTENTION_PROBABILITY = 7,
    QBH_BLOCK_QP_ATTENTION_CONCAT = 8,
    QBH_BLOCK_QP_ATTENTION_PROJECTION = 9,
    QBH_BLOCK_QP_POST_ATTENTION_RESIDUAL = 10,
    QBH_BLOCK_QP_POST_ATTENTION_NORM = 11,
    QBH_BLOCK_QP_GATE = 12,
    QBH_BLOCK_QP_UP = 13,
    QBH_BLOCK_QP_MIDDLE = 14,
    QBH_BLOCK_QP_DOWN = 15,
    QBH_BLOCK_QP_BLOCK_OUTPUT = 16,
};

enum qbh_block_status {
    QBH_BLOCK_STATUS_HOST_READY = 1,
    QBH_BLOCK_STATUS_DSP_RUNNING = 2,
    QBH_BLOCK_STATUS_OK = 3,
    QBH_BLOCK_STATUS_BAD_HEADER = -1,
    QBH_BLOCK_STATUS_CACHE_FAILED = -2,
    QBH_BLOCK_STATUS_ARENA_FAILED = -3,
    QBH_BLOCK_STATUS_INPUT_DMA_FAILED = -4,
    QBH_BLOCK_STATUS_METADATA_DMA_FAILED = -5,
    QBH_BLOCK_STATUS_HMX_WORKER_FAILED = -6,
    QBH_BLOCK_STATUS_INPUT_NORM_FAILED = -7,
    QBH_BLOCK_STATUS_QKV_FAILED = -8,
    QBH_BLOCK_STATUS_QK_NORM_ROPE_FAILED = -9,
    QBH_BLOCK_STATUS_ATTENTION_FAILED = -10,
    QBH_BLOCK_STATUS_O_PROJECTION_FAILED = -11,
    QBH_BLOCK_STATUS_POST_NORM_FAILED = -12,
    QBH_BLOCK_STATUS_GATE_UP_FAILED = -13,
    QBH_BLOCK_STATUS_ACTIVATION_FAILED = -14,
    QBH_BLOCK_STATUS_DOWN_FAILED = -15,
    QBH_BLOCK_STATUS_OUTPUT_DMA_FAILED = -16,
    QBH_BLOCK_STATUS_W4F16_PIPELINE_FAILED = -17,
};

enum qbh_block_numerical_status {
    QBH_BLOCK_NUMERICAL_UNCHECKED = 0,
    QBH_BLOCK_NUMERICAL_OK = 1,
    QBH_BLOCK_NUMERICAL_INPUT_NORM = -1,
    QBH_BLOCK_NUMERICAL_Q = -2,
    QBH_BLOCK_NUMERICAL_K = -3,
    QBH_BLOCK_NUMERICAL_V = -4,
    QBH_BLOCK_NUMERICAL_Q_ROPE = -5,
    QBH_BLOCK_NUMERICAL_K_ROPE = -6,
    QBH_BLOCK_NUMERICAL_ATTENTION_QK = -7,
    QBH_BLOCK_NUMERICAL_O = -8,
    QBH_BLOCK_NUMERICAL_POST_RESIDUAL = -9,
    QBH_BLOCK_NUMERICAL_POST_NORM = -10,
    QBH_BLOCK_NUMERICAL_GATE = -11,
    QBH_BLOCK_NUMERICAL_UP = -12,
    QBH_BLOCK_NUMERICAL_MIDDLE = -13,
    QBH_BLOCK_NUMERICAL_DOWN = -14,
    QBH_BLOCK_NUMERICAL_OUTPUT = -15,
    QBH_BLOCK_NUMERICAL_ATTENTION_SOFTMAX = -16,
    QBH_BLOCK_NUMERICAL_ATTENTION_AV = -17,
};

struct qbh_block_qparam {
    float scale;
    int32_t zero_point;
    float minimum;
    float maximum;
};

struct qbh_block_projection_desc {
    uint32_t k;
    uint32_t n;
    uint32_t weight_offset;
    uint32_t weight_bytes;
    uint32_t scale_offset;
    uint32_t scale_bytes;
    uint32_t bias_offset;
    uint32_t bias_bytes;
};

struct qbh_block_header {
    uint32_t magic;
    uint32_t abi_version;
    uint32_t experiment;
    uint32_t header_bytes;
    uint32_t shared_bytes;
    uint32_t variant;
    uint32_t repeat_count;
    uint32_t w4f16_requested_hvx_workers;
    uint32_t w4f16_region_tiles;
    uint32_t common_ops_mask;
    uint32_t attribution_enabled;
    uint32_t numerical_audit_enabled;
    uint32_t residual_mode;
    uint32_t f16f16_projection_mode;

    uint32_t input_offset;
    uint32_t input_bytes;
    uint32_t output_offset;
    uint32_t output_bytes;
    uint32_t reference_offset;
    uint32_t reference_bytes;
    uint32_t qparam_offset;
    uint32_t qparam_bytes;

    uint32_t input_norm_weight_offset;
    uint32_t input_norm_weight_bytes;
    uint32_t post_norm_weight_offset;
    uint32_t post_norm_weight_bytes;
    uint32_t q_norm_weight_offset;
    uint32_t q_norm_weight_bytes;
    uint32_t k_norm_weight_offset;
    uint32_t k_norm_weight_bytes;
    uint32_t rope_cos_offset;
    uint32_t rope_cos_bytes;
    uint32_t rope_sin_offset;
    uint32_t rope_sin_bytes;

    struct qbh_block_projection_desc
        projections[QBH_BLOCK_PROJECTION_COUNT];
    struct qbh_block_qparam qparams[QBH_BLOCK_QPARAM_COUNT];

    int32_t dsp_status;
    int32_t cache_status;
    int32_t hmx_worker_status;
    int32_t hmx_lock_status;
    int32_t hmx_unlock_status;
    int32_t input_dma_status;
    int32_t output_dma_status;
    int32_t numerical_status;
    float attention_qk_max_abs;
    float attention_probability_max_abs;
    float attention_av_max_abs;
    float common_op_rms_max_abs;
    float common_op_rms_cosine;
    float common_op_rope_max_abs;
    float common_op_rope_cosine;
    float common_op_softmax_max_abs;
    float common_op_softmax_cosine;
    float common_op_silu_max_abs;
    float common_op_silu_cosine;
    uint32_t common_op_nonfinite_count;
    uint32_t common_op_softmax_mask_violation_count;
    int32_t projection_failure_result;
    uint32_t projection_failure_index;
    uint32_t projection_failure_n_tile;
    uint32_t projection_failure_step;
    uint32_t w4f16_expand_mismatch_count;
    uint32_t w4f16_expand_first_logical_index;
    uint32_t w4f16_expand_expected_half_bits;
    uint32_t w4f16_expand_actual_half_bits;
    uint32_t w4f16_hvx_workers_created;
    uint32_t w4f16_hvx_workers_locked;
    int32_t w4f16_pool_status;
    uint32_t f16f16_weight_batch_n_tiles;

    uint32_t prepared_session_run_index;
    uint32_t resource_vtcm_address;
    uint32_t resource_hmx_context_id;
    uint32_t vtcm_requested_bytes;
    uint32_t vtcm_acquired_bytes;
    uint32_t vtcm_peak_plan_bytes;
    uint32_t block_invocation_count;
    uint32_t hmx_command_count;
    uint32_t hmx_fp16_tile_pair_count;
    uint32_t hmx_u8s8_tile_pair_count;
    uint32_t weight_dma_descriptor_count;
    uint32_t boundary_dma_descriptor_count;
    uint32_t intermediate_ddr_read_bytes;
    uint32_t intermediate_ddr_write_bytes;
    uint32_t intermediate_dma_descriptor_count;
    uint32_t intermediate_spill_fill_count;
    uint64_t weight_ddr_read_bytes;
    uint64_t boundary_ddr_read_bytes;
    uint64_t boundary_ddr_write_bytes;

    uint64_t qtimer_start;
    uint64_t qtimer_end;
    uint64_t total_ticks;
    uint64_t input_stage_ticks;
    uint64_t metadata_stage_ticks;
    uint64_t input_norm_ticks;
    uint64_t qkv_projection_ticks;
    uint64_t qk_norm_rope_ticks;
    uint64_t attention_ticks;
    uint64_t o_projection_ticks;
    uint64_t post_attention_residual_ticks;
    uint64_t post_attention_norm_ticks;
    uint64_t gate_up_ticks;
    uint64_t activation_ticks;
    uint64_t down_ticks;
    uint64_t final_residual_ticks;
    uint64_t output_stage_ticks;
    uint64_t weight_dma_ticks;
    uint64_t hmx_compute_ticks;
    uint64_t projection_pack_ticks;
    uint64_t w4f16_expand_ticks;
    uint64_t projection_hmx_wait_ticks;
    uint64_t projection_unpack_ticks;
    uint64_t hmx_ready_wait_ticks;
    uint64_t w4f16_streamed_command_count;
    uint64_t w4f16_expand_work_ticks;
    uint64_t w4f16_expand_region_count;
    uint64_t w4f16_prefetch_count;
    uint64_t w4f16_prefetch_wait_ticks;
    uint64_t f16f16_prefetch_count;
    uint64_t f16f16_prefetch_wait_ticks;
    uint64_t scalar_math_ticks;

    uint64_t invocation_ticks;
    uint64_t runtime_setup_ticks;
    uint64_t runtime_teardown_ticks;
    uint64_t ledger_named_ticks;
    uint64_t ledger_unattributed_ticks;

    uint64_t input_norm_audit_ticks;
    uint64_t qkv_audit_ticks;
    uint64_t qk_norm_rope_audit_ticks;
    uint64_t o_projection_audit_ticks;
    uint64_t post_attention_residual_audit_ticks;
    uint64_t post_attention_norm_audit_ticks;
    uint64_t gate_up_audit_ticks;
    uint64_t activation_audit_ticks;
    uint64_t down_audit_ticks;
    uint64_t final_residual_audit_ticks;

    uint64_t attention_setup_ticks;
    uint64_t attention_qk_pack_ticks;
    uint64_t attention_qk_hmx_ticks;
    uint64_t attention_qk_unpack_ticks;
    uint64_t attention_qk_audit_ticks;
    uint64_t attention_softmax_ticks;
    uint64_t attention_softmax_audit_ticks;
    uint64_t attention_av_pack_ticks;
    uint64_t attention_av_hmx_ticks;
    uint64_t attention_av_unpack_ticks;
    uint64_t attention_av_audit_ticks;
    uint64_t attention_unattributed_ticks;
};

#endif
