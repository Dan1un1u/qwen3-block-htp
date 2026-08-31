#ifndef QWEN3_BLOCK_HTP_BLOCK_PROTOCOL_H
#define QWEN3_BLOCK_HTP_BLOCK_PROTOCOL_H

#include <stdint.h>

#include "attention_protocol.h"
#include "probe_protocol.h"

#define QBH_BLOCK_MAGIC UINT32_C(0x5142424c)
#define QBH_BLOCK_ABI_VERSION UINT32_C(47)
#define QBH_BLOCK_EXPERIMENT UINT32_C(112)

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
#define QBH_BLOCK_ATTENTION_CONFIG_COUNT QBH_BLOCK_KV_HEADS
#define QBH_BLOCK_ATTENTION_CONFIG_BYTES \
    (QBH_BLOCK_ATTENTION_CONFIG_COUNT * \
     sizeof(struct qbh_attention_config))
#define QBH_BLOCK_U8_ATTENTION_Q_BYTES \
    (QBH_BLOCK_M * QBH_BLOCK_HIDDEN)
#define QBH_BLOCK_U8_ATTENTION_KV_BYTES \
    (QBH_BLOCK_M * QBH_BLOCK_KV_HIDDEN)
#define QBH_BLOCK_U8_ATTENTION_SCORE_BYTES \
    (QBH_BLOCK_HEADS * QBH_BLOCK_M * QBH_BLOCK_M)
#define QBH_BLOCK_U8_ATTENTION_AV_BYTES \
    QBH_BLOCK_U8_ATTENTION_Q_BYTES
#define QBH_BLOCK_U8_ATTENTION_AUDIT_BYTES \
    (QBH_BLOCK_U8_ATTENTION_Q_BYTES + \
     2U * QBH_BLOCK_U8_ATTENTION_KV_BYTES + \
     2U * QBH_BLOCK_U8_ATTENTION_SCORE_BYTES + \
     QBH_BLOCK_U8_ATTENTION_AV_BYTES)

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
    QBH_BLOCK_RESIDUAL_HVX_FUSED_POST_NORM_POOL4 = 3,
    QBH_BLOCK_RESIDUAL_HVX_FUSED_POST_NORM_POOL6 = 4,
    QBH_BLOCK_RESIDUAL_HVX_FUSED_POST_NORM_POOL6_SHUFFLE4 = 5,
};

enum qbh_block_f16f16_projection_mode {
    QBH_BLOCK_F16F16_PROJECTION_SERIAL = 0,
    QBH_BLOCK_F16F16_PROJECTION_ASYNC_SINGLE = 1,
    QBH_BLOCK_F16F16_PROJECTION_BATCH2 = 2,
    QBH_BLOCK_F16F16_PROJECTION_GATE4 = 3,
    QBH_BLOCK_F16F16_PROJECTION_GATE8 = 4,
    QBH_BLOCK_F16F16_PROJECTION_GATE8_INTERLEAVED = 5,
};

enum qbh_block_w4f16_pipeline_mode {
    QBH_BLOCK_W4F16_PIPELINE_CONTROL = 0,
    QBH_BLOCK_W4F16_PIPELINE_EARLY_REGION = 1,
    QBH_BLOCK_W4F16_PIPELINE_HYBRID_WORKERS = 2,
    QBH_BLOCK_W4F16_PIPELINE_MAIN_HALF = 3,
    QBH_BLOCK_W4F16_PIPELINE_MAIN_TWO_THIRDS = 4,
    QBH_BLOCK_W4F16_PIPELINE_CROSS_PREFETCH = 5,
    QBH_BLOCK_W4F16_PIPELINE_HYBRID_CROSS_PREFETCH = 6,
    QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN64_CROSS_PREFETCH = 7,
    QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN48_CROSS_PREFETCH = 8,
    QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_CROSS_PREFETCH = 9,
    QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_GATE16_CROSS_PREFETCH = 10,
    QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_GATE8_CROSS_PREFETCH = 11,
    QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_GATE4_CROSS_PREFETCH = 12,
    QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_GATE4_DMA8_CROSS_PREFETCH = 13,
};

enum qbh_block_w4u8_qkvo_pipeline_mode {
    QBH_BLOCK_W4U8_QKVO_SERIAL = 0,
    QBH_BLOCK_W4U8_QKV_BATCH2 = 1,
    QBH_BLOCK_W4U8_QKV_BATCH4 = 2,
    QBH_BLOCK_W4U8_QKVO_BATCH4 = 3,
    QBH_BLOCK_W4U8_QKVO_BATCH4_QK_HEAD_TASKS = 4,
    QBH_BLOCK_W4U8_QKVO_BATCH4_QK_HEAD_PAIRS = 5,
};

enum qbh_block_u8_norm_reduction_mode {
    QBH_BLOCK_U8_NORM_REDUCTION_SCALAR = 0,
    QBH_BLOCK_U8_NORM_REDUCTION_HVX_TREE = 1,
    QBH_BLOCK_U8_NORM_REDUCTION_HVX_TREE_QK_BATCHED_RSQRT = 2,
    QBH_BLOCK_U8_NORM_REDUCTION_HVX_TREE_QK_BATCHED_RSQRT_SHARED_ROPE = 3,
    QBH_BLOCK_U8_NORM_REDUCTION_HVX_TREE_QK_BATCHED_RSQRT_SHARED_ROPE_PARALLEL_INPUT = 4,
};

enum qbh_block_w4u8_qk_pair_kernel_mode {
    QBH_BLOCK_W4U8_QK_PAIR_SERIAL_INNER = 0,
    QBH_BLOCK_W4U8_QK_PAIR_QUARTER_TILED = 1,
};

enum qbh_block_fp16_common_schedule_mode {
    QBH_BLOCK_FP16_COMMON_SCHEDULE_CONTROL = 0,
    QBH_BLOCK_FP16_COMMON_SCHEDULE_QK_HEAD_PAIRS = 1U << 0,
    QBH_BLOCK_FP16_COMMON_SCHEDULE_INPUT_NORM_POOL = 1U << 1,
    QBH_BLOCK_FP16_COMMON_SCHEDULE_POST_RESIDUAL_NORM_POOL = 1U << 2,
    QBH_BLOCK_FP16_COMMON_SCHEDULE_ALL =
        QBH_BLOCK_FP16_COMMON_SCHEDULE_QK_HEAD_PAIRS |
        QBH_BLOCK_FP16_COMMON_SCHEDULE_INPUT_NORM_POOL |
        QBH_BLOCK_FP16_COMMON_SCHEDULE_POST_RESIDUAL_NORM_POOL,
};

enum qbh_block_qkv_schedule_mode {
    QBH_BLOCK_QKV_SCHEDULE_CONTROL = 0,
    QBH_BLOCK_QKV_SCHEDULE_Q_PREFIX4_K_ALL = 1,
};

enum qbh_block_w4f16_group_fence_mode {
    QBH_BLOCK_W4F16_GROUP_FENCE_CONTROL = 0,
    QBH_BLOCK_W4F16_GROUP_FENCE_JOIN_ONLY = 1,
};

enum qbh_block_w4u8_stream_fence_mode {
    QBH_BLOCK_W4U8_STREAM_FENCE_CONTROL = 0,
    QBH_BLOCK_W4U8_STREAM_FENCE_SINGLE = 1,
    QBH_BLOCK_W4U8_STREAM_FENCE_RELEASE_ONLY = 2,
};

enum qbh_block_attention_pack_mode {
    QBH_BLOCK_ATTENTION_PACK_CONTROL = 0,
    QBH_BLOCK_ATTENTION_PACK_QK_HVX = 1U << 0,
    QBH_BLOCK_ATTENTION_PACK_AV_HVX = 1U << 1,
    QBH_BLOCK_ATTENTION_PACK_HVX =
        QBH_BLOCK_ATTENTION_PACK_QK_HVX |
        QBH_BLOCK_ATTENTION_PACK_AV_HVX,
};

enum qbh_block_attention_pipeline_mode {
    QBH_BLOCK_ATTENTION_PIPELINE_CONTROL = 0,
    QBH_BLOCK_ATTENTION_PIPELINE_PARALLEL_QK_NORM_ROPE = 1,
    QBH_BLOCK_ATTENTION_PIPELINE_PARALLEL_SOFTMAX = 2,
    QBH_BLOCK_ATTENTION_PIPELINE_PARALLEL_HVX = 3,
    QBH_BLOCK_ATTENTION_PIPELINE_GQA = 4,
    QBH_BLOCK_ATTENTION_PIPELINE_GQA_QKV_OVERLAP = 5,
    QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA = 6,
    QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_FUSED_K = 7,
    QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP = 8,
    QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER = 9,
    QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER_VDEAL = 10,
    QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER_VDEAL_FUSED_QK_REQUANT = 11,
    QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER_VDEAL_FUSED_QK_REQUANT_HMX_BATCH = 12,
    QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER_VDEAL_FUSED_QK_REQUANT_HMX_BATCH_LUT_TEMPLATES = 13,
    QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER_VDEAL_FUSED_QK_REQUANT_HMX_BATCH_LUT_TEMPLATES_GQA_BATCH = 14,
    QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER_VDEAL_FUSED_QK_REQUANT_HMX_BATCH_LUT_TEMPLATES_GQA_BATCH_DEPENDENCY_STREAM = 15,
    QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER_VDEAL_FUSED_QK_REQUANT_HMX_BATCH_LUT_TEMPLATES_GQA_BATCH_DEPENDENCY_STREAM_SOFTMAX_SHUFFLE4 = 16,
};

enum qbh_block_mlp_mode {
    QBH_BLOCK_MLP_CONTROL = 0,
    QBH_BLOCK_MLP_MULTI_WORKER_SILU = 1,
    QBH_BLOCK_MLP_STREAMING = 2,
    QBH_BLOCK_MLP_CROUTON_NATIVE = 3,
    QBH_BLOCK_MLP_CROUTON_NATIVE_BATCH8 = 4,
    QBH_BLOCK_MLP_W4U8_STREAMING = 5,
    QBH_BLOCK_MLP_W4U8_STREAMING_PERSISTENT_GATE_UP_HVX = 6,
    QBH_BLOCK_MLP_W4U8_STREAMING_PERSISTENT_MLP_HVX = 7,
};

static inline uint32_t qbh_block_mlp_is_w4u8_streaming(uint32_t mode) {
    return mode == QBH_BLOCK_MLP_W4U8_STREAMING ||
           mode ==
               QBH_BLOCK_MLP_W4U8_STREAMING_PERSISTENT_GATE_UP_HVX ||
           mode == QBH_BLOCK_MLP_W4U8_STREAMING_PERSISTENT_MLP_HVX;
}

static inline uint32_t qbh_block_mlp_uses_persistent_gate_up_hvx(
    uint32_t mode) {
    return mode ==
               QBH_BLOCK_MLP_W4U8_STREAMING_PERSISTENT_GATE_UP_HVX ||
           mode == QBH_BLOCK_MLP_W4U8_STREAMING_PERSISTENT_MLP_HVX;
}

enum qbh_block_crouton_boundary_mode {
    QBH_BLOCK_CROUTON_BOUNDARY_CONTROL = 0,
    QBH_BLOCK_CROUTON_BOUNDARY_QKV = 1U << 0,
    QBH_BLOCK_CROUTON_BOUNDARY_AV_TO_O = 1U << 1,
    QBH_BLOCK_CROUTON_BOUNDARY_INPUT_NORM = 1U << 2,
    QBH_BLOCK_CROUTON_BOUNDARY_POST_NORM = 1U << 3,
    QBH_BLOCK_CROUTON_BOUNDARY_W4U8_MLP_INPUT = 1U << 4,
    QBH_BLOCK_CROUTON_BOUNDARY_W4U8_MLP_OUTPUT = 1U << 5,
    QBH_BLOCK_CROUTON_BOUNDARY_W4U8_QKV_INPUT = 1U << 6,
    QBH_BLOCK_CROUTON_BOUNDARY_W4U8_O_OUTPUT = 1U << 7,
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
    QBH_BLOCK_STATUS_MLP_POOL_FAILED = -18,
    QBH_BLOCK_STATUS_MLP_STREAM_FAILED = -19,
    QBH_BLOCK_STATUS_ATTENTION_POOL_FAILED = -20,
    QBH_BLOCK_STATUS_ATTENTION_PIPELINE_FAILED = -21,
    QBH_BLOCK_STATUS_RESIDUAL_POOL_FAILED = -22,
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
    uint32_t w4f16_pipeline_mode;
    uint32_t attention_pack_mode;
    uint32_t attention_pipeline_mode;
    uint32_t attention_hvx_contexts;
    uint32_t mlp_mode;
    uint32_t mlp_hvx_contexts;
    uint32_t mlp_chunk_vectors;
    uint32_t crouton_boundary_mode;
    uint32_t w4u8_qkvo_pipeline_mode;
    uint32_t u8_norm_reduction_mode;
    uint32_t w4u8_qk_pair_kernel_mode;
    uint32_t fp16_common_schedule_mode;
    uint32_t fp16_norm_rows_per_task;
    uint32_t fp16_norm_contexts;
    uint32_t w4u8_down_hmx_batch_outputs;
    uint32_t qkv_schedule_mode;
    uint32_t w4f16_group_fence_mode;
    uint32_t w4u8_stream_fence_mode;

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
    uint32_t w4u8_gate_up_bundle_offset;
    uint32_t w4u8_gate_up_bundle_bytes;
    uint32_t w4u8_down_bundle_offset;
    uint32_t w4u8_down_bundle_bytes;
    uint32_t w4u8_silu_lut_offset;
    uint32_t w4u8_silu_lut_bytes;
    uint32_t attention_config_offset;
    uint32_t attention_config_bytes;

    struct qbh_block_projection_desc
        projections[QBH_BLOCK_PROJECTION_COUNT];
    struct qbh_block_qparam qparams[QBH_BLOCK_QPARAM_COUNT];

    /* Host-computed FNV-1a hashes over independent, physical tile-order
     * references.  They are part of the immutable request and therefore sit
     * before dsp_status, the boundary at which the DSP clears run telemetry. */
    uint64_t u8_attention_expected_score_hash;
    uint64_t u8_attention_expected_probability_hash;
    uint64_t u8_attention_expected_av_hash;
    uint32_t u8_attention_audit_output_offset;
    uint32_t u8_attention_audit_output_bytes;

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
    uint32_t w4f16_active_worker_min;
    uint32_t w4f16_active_worker_max;
    uint32_t w4f16_effective_region_min;
    uint32_t w4f16_effective_region_max;
    uint32_t mlp_hvx_workers_created;
    uint32_t mlp_hvx_workers_locked;
    int32_t mlp_pool_status;
    uint32_t mlp_silu_chunk_count;
    uint32_t mlp_stream_group_count;
    uint32_t mlp_down_pack_skipped;
    uint64_t mlp_down_input_hash;
    uint32_t attention_hvx_workers_created;
    uint32_t attention_hvx_workers_locked;
    int32_t attention_pool_status;
    uint32_t attention_qk_norm_task_count;
    uint32_t fp16_qk_norm_pair_task_count;
    uint32_t attention_softmax_task_count;
    uint32_t u8_attention_softmax_shuffle4_row_group_count;
    uint32_t attention_gqa_group_count;
    uint32_t crouton_qkv_projection_count;
    uint32_t crouton_qkv_unpack_skipped;
    uint32_t crouton_qk_operand_count;
    uint32_t crouton_av_weight_count;
    uint32_t crouton_av_o_head_count;
    uint32_t crouton_av_unpack_skipped;
    uint32_t crouton_norm_projection_count;
    uint32_t crouton_q_operand_mismatch_count;
    uint32_t crouton_k_operand_mismatch_count;
    uint32_t crouton_v_operand_mismatch_count;
    uint32_t qkv_operand_audit_tensor_count;
    uint32_t qkv_schedule_command_count;
    uint64_t qkv_schedule_trace_hash;
    uint32_t u8_attention_group_count;
    uint32_t u8_attention_qk_execution_count;
    uint32_t u8_attention_av_execution_count;
    uint32_t u8_attention_score_saturation_count;
    uint32_t u8_attention_v_recenter_saturation_count;
    uint32_t u8_attention_probability_mask_violation_count;
    uint32_t u8_attention_probability_row_sum_min;
    uint32_t u8_attention_probability_row_sum_max;
    uint32_t u8_attention_direct_o_tile_count;
    uint32_t u8_attention_qkv_unpack_skipped;
    uint32_t u8_attention_fused_k_operand_mismatch_count;
    uint32_t w4u8_qkv_batch_n_tiles;
    uint32_t w4u8_qkv_batch_count;
    uint32_t w4u8_qkvo_prefetch_count;
    uint32_t w4u8_qkvo_overlap_schedule_count;
    uint32_t w4u8_qk_pair_kernel_mode_observed;
    uint32_t w4u8_qk_quarter_pair_count;
    uint64_t u8_attention_actual_score_hash;
    uint64_t u8_attention_actual_probability_hash;
    uint64_t u8_attention_actual_av_hash;
    uint64_t u8_input_norm_actual_hash;
    uint32_t u8_attention_audit_ddr_write_bytes;

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
    uint64_t w4f16_first_expand_ticks;
    uint64_t w4f16_steady_expand_ticks;
    uint64_t w4f16_expand_pool_wait_ticks;
    uint64_t w4f16_hmx_tail_wait_ticks;
    uint64_t w4f16_early_region_command_count;
    uint64_t w4f16_cross_prefetch_count;
    uint64_t w4f16_cross_prefetch_wait_ticks;
    uint64_t w4f16_cross_prefetch_lifetime_ticks;
    uint64_t mlp_silu_main_work_ticks;
    uint64_t mlp_silu_worker_work_ticks;
    uint64_t mlp_silu_pool_wait_ticks;
    uint64_t mlp_stream_worker_work_ticks;
    uint64_t mlp_stream_main_work_ticks;
    uint64_t mlp_stream_ready_wait_ticks;
    uint64_t mlp_stream_join_wait_ticks;
    uint64_t attention_qk_norm_main_work_ticks;
    uint64_t attention_qk_norm_worker_work_ticks;
    uint64_t attention_qk_norm_pool_wait_ticks;
    uint64_t attention_softmax_main_work_ticks;
    uint64_t attention_softmax_worker_work_ticks;
    uint64_t attention_softmax_pool_wait_ticks;
    uint64_t attention_gqa_worker_work_ticks;
    uint64_t attention_gqa_hmx_wait_ticks;
    uint64_t attention_gqa_queue_wait_ticks;
    uint64_t crouton_qkv_transform_ticks;
    uint64_t crouton_av_o_copy_ticks;
    uint64_t crouton_norm_store_ticks;
    uint64_t scalar_math_ticks;
    uint64_t u8_attention_qk_norm_rope_ticks;
    uint64_t u8_attention_k_pack_ticks;
    uint64_t u8_attention_v_pack_ticks;
    uint64_t u8_attention_qk_hmx_ticks;
    uint64_t u8_attention_qk_requant_ticks;
    uint64_t u8_attention_softmax_ticks;
    uint64_t u8_attention_av_hmx_ticks;
    uint64_t u8_attention_av_requant_ticks;
    uint64_t u8_attention_pipeline_wait_ticks;
    uint64_t w4u8_qkvo_weight_expand_ticks;
    uint64_t w4u8_qkvo_prefetch_wait_ticks;
    uint64_t w4u8_qkvo_hmx_lifetime_ticks;
    uint32_t w4u8_input_norm_task_count;
    uint64_t w4u8_input_norm_main_work_ticks;
    uint64_t w4u8_input_norm_worker_work_ticks;
    uint64_t w4u8_input_norm_pool_wait_ticks;
    uint32_t w4u8_residual_active_contexts;
    uint32_t w4u8_post_residual_task_count;
    uint32_t w4u8_final_residual_task_count;
    uint64_t w4u8_post_residual_main_work_ticks;
    uint64_t w4u8_post_residual_worker_work_ticks;
    uint64_t w4u8_post_residual_pool_wait_ticks;
    uint64_t w4u8_final_residual_main_work_ticks;
    uint64_t w4u8_final_residual_worker_work_ticks;
    uint64_t w4u8_final_residual_pool_wait_ticks;
    uint32_t fp16_input_norm_task_count;
    uint32_t fp16_input_norm_active_contexts;
    uint32_t fp16_post_residual_norm_task_count;
    uint32_t fp16_post_residual_norm_active_contexts;
    uint64_t fp16_input_norm_main_work_ticks;
    uint64_t fp16_input_norm_worker_work_ticks;
    uint64_t fp16_input_norm_pool_wait_ticks;
    uint64_t fp16_post_residual_norm_main_work_ticks;
    uint64_t fp16_post_residual_norm_worker_work_ticks;
    uint64_t fp16_post_residual_norm_pool_wait_ticks;

    uint64_t invocation_ticks;
    uint64_t runtime_setup_ticks;
    uint64_t runtime_teardown_ticks;
    uint64_t stage_boundary_ticks;
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
    uint64_t attention_gqa_pipeline_ticks;
    uint64_t attention_unattributed_ticks;

    uint32_t w4f16_gate_up_effective_region_tiles;
    uint32_t w4f16_gate_up_scale_cache_bytes;
    uint64_t w4f16_gate_up_weight_dma_ticks;
    uint64_t w4f16_gate_up_expand_ticks;
    uint64_t w4f16_gate_up_expand_work_ticks;
    uint64_t w4f16_gate_up_expand_pool_wait_ticks;
    uint64_t w4f16_gate_up_prefetch_wait_ticks;
    uint64_t w4f16_gate_up_hmx_wait_ticks;
    uint64_t w4f16_gate_up_hmx_tail_wait_ticks;
    uint64_t w4f16_gate_up_unpack_ticks;
    uint64_t w4f16_gate_up_stream_work_ticks;
    uint64_t w4f16_gate_up_stream_ready_wait_ticks;
    uint64_t w4f16_gate_up_stream_join_wait_ticks;
    uint64_t w4f16_gate_up_hmx_command_count;
    uint64_t w4f16_gate_up_scale_init_ticks;

    uint32_t w4u8_mlp_vtcm_base_offset;
    uint32_t w4u8_mlp_vtcm_plan_bytes;
    uint32_t w4u8_mlp_lut_vtcm_bytes;
    uint32_t w4u8_mlp_gather_scratch_vtcm_bytes;
    uint32_t w4u8_mlp_gate_up_hvx_workers;
    uint32_t w4u8_mlp_down_hvx_workers;
    uint32_t w4u8_mlp_gate_up_hmx_batch_n_tiles;
    uint32_t w4u8_mlp_down_hmx_batch_n_tiles;
    uint32_t w4u8_mlp_down_in_command_slot_release_count;
    uint32_t w4u8_mlp_down_producer_progress_command_count;
    uint32_t w4u8_mlp_gate_up_expanded_slot_count;
    uint32_t w4u8_mlp_pair_publish_count;
    uint32_t w4u8_mlp_pair_consume_count;
    uint32_t w4u8_mlp_gate_up_hvx_hmx_overlap;
    uint32_t w4u8_mlp_down_hvx_hmx_overlap;
    uint32_t w4u8_mlp_gate_up_hvx_parallel_overlap;
    uint32_t w4u8_mlp_down_hvx_parallel_overlap;
    uint32_t w4u8_mlp_input_pack_skipped;
    uint32_t w4u8_mlp_output_unpack_skipped;
    uint64_t w4u8_mlp_input_pack_ticks;
    uint64_t w4u8_mlp_output_unpack_ticks;
    uint64_t w4u8_mlp_gate_up_pipeline_ticks;
    uint64_t w4u8_mlp_gate_up_hmx_command_count;
    uint64_t w4u8_mlp_down_pipeline_ticks;
    uint64_t w4u8_mlp_down_hmx_command_count;
    uint64_t w4u8_mlp_activation_work_ticks;
    uint64_t w4u8_mlp_weight_stage_ticks;
    uint64_t w4u8_mlp_weight_expand_ticks;
    uint64_t w4u8_mlp_hmx_compute_ticks;
    uint64_t w4u8_mlp_hmx_ready_wait_ticks;
    uint64_t w4u8_mlp_producer_slot_wait_ticks;
    uint64_t w4u8_mlp_expanded_slot_wait_ticks;
    uint32_t w4u8_gate_up_persistent_hvx_dispatch_count;
    uint32_t w4u8_gate_up_persistent_hvx_worker_count;
    uint32_t w4u8_gate_up_transient_hvx_thread_count;
    uint32_t w4u8_down_persistent_hvx_dispatch_count;
    uint32_t w4u8_down_persistent_hvx_worker_count;
    uint32_t w4u8_down_transient_hvx_thread_count;
};

#endif
