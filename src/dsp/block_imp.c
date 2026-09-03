#include <AEEStdErr.h>
#include <HAP_compute_res.h>
#include <HAP_mem.h>
#include <HAP_perf.h>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include <math.h>
#include <qurt.h>
#include <qurt_hvx.h>
#include <remote.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "block_imp.h"
#include "block_protocol.h"
#include "attention_u8_core.h"
#include "hmx_fp16.h"
#include "hmx_u8s8_projection.h"
#include "hvx_fp16_ops.h"
#include "hvx_u8_ops.h"
#include "mlp_u8.h"
#include "qbh_user_dma.h"
#include "w4_parallel_pipeline.h"
#include "w4_u8_expand.h"

#define QBH_BLOCK_ALIGNMENT UINT32_C(128)
#define QBH_BLOCK_HMX_STACK_BYTES UINT32_C(16384)
#define QBH_BLOCK_MAX_K QBH_BLOCK_INTERMEDIATE
#define QBH_BLOCK_SCORE_ELEMENTS \
    (QBH_BLOCK_HEADS * QBH_BLOCK_M * QBH_BLOCK_M)
#define QBH_BLOCK_HMX_OUTPUT_MAX_BYTES \
    (2U * 4U * QBH_HMX_FP16_TILE_BYTES)
#define QBH_BLOCK_PROJECTION_SCALE_CHANNELS \
    (3U * QBH_BLOCK_HIDDEN + 2U * QBH_BLOCK_KV_HIDDEN + \
     2U * QBH_BLOCK_INTERMEDIATE)
#define QBH_BLOCK_PROJECTION_SCALE_BYTES \
    (QBH_BLOCK_PROJECTION_SCALE_CHANNELS * sizeof(float))

enum qbh_block_hmx_command_kind {
    QBH_BLOCK_HMX_NONE = 0,
    QBH_BLOCK_HMX_FP16 = 1,
    QBH_BLOCK_HMX_U8S8 = 2,
    QBH_BLOCK_HMX_FP16_STREAMING = 3,
    QBH_BLOCK_HMX_FP16_TILE_SCALES = 4,
    QBH_BLOCK_HMX_FP16_TILE_SCALES_STREAMING = 5,
    QBH_BLOCK_HMX_U8S8_PIPELINE = 6,
    QBH_BLOCK_HMX_U8S8_QKV_RING = 7,
};

#define QBH_BLOCK_W4F16_MIN_REGION_TILES UINT32_C(8)
#define QBH_BLOCK_W4F16_HMX_BATCH_N_TILES UINT32_C(2)
#define QBH_BLOCK_W4F16_MAX_REGIONS \
    (QBH_BLOCK_MAX_K / 32U * QBH_BLOCK_W4F16_HMX_BATCH_N_TILES / \
     QBH_BLOCK_W4F16_MIN_REGION_TILES)
#define QBH_BLOCK_W4F16_HVX_WORKERS UINT32_C(4)
#define QBH_BLOCK_W4F16_MAX_REQUESTED_HVX_WORKERS UINT32_C(4)
#define QBH_BLOCK_MAX_ATTENTION_HVX_CONTEXTS UINT32_C(6)
#define QBH_BLOCK_MAX_POOL_HVX_WORKERS \
    (QBH_BLOCK_MAX_ATTENTION_HVX_CONTEXTS - 1U)
#define QBH_BLOCK_W4F16_HVX_STACK_BYTES UINT32_C(8192)
#define QBH_BLOCK_W4F16_DMA_BATCH_N_TILES UINT32_C(4)
#define QBH_BLOCK_F16F16_BATCH_N_TILES UINT32_C(2)

_Static_assert(
    QBH_BLOCK_W4F16_HVX_WORKERS *
            (QBH_ATTN_U8_K_WEIGHT_BYTES + QBH_ATTN_U8_QK_BIAS_BYTES) <=
        QBH_QK_PAIR_RSQRT_SCRATCH_OFFSET,
    "Q/K rsqrt scratch must follow numerical-audit scratch");
_Static_assert(
    QBH_QK_PAIR_RSQRT_SCRATCH_OFFSET +
            QBH_BLOCK_W4F16_HVX_WORKERS *
                QBH_QK_PAIR_RSQRT_SCRATCH_BYTES <=
        QBH_QK_ROPE_SF32_CACHE_OFFSET,
    "Q/K rsqrt scratch must not overlap shared RoPE cache");
_Static_assert(
    QBH_QK_ROPE_SF32_CACHE_OFFSET + QBH_QK_ROPE_SF32_CACHE_BYTES <=
        QBH_BLOCK_M * QBH_BLOCK_HIDDEN,
    "shared RoPE cache must fit the W4U8 Attention projection overlay");
#define QBH_BLOCK_DMA_DESCRIPTOR_TIMEOUT_TICKS UINT64_C(1920000)
#define QBH_BLOCK_HVX_F16_LANES UINT32_C(64)
#define QBH_BLOCK_MLP_STREAM_CHANNELS UINT32_C(64)
#define QBH_BLOCK_MLP_STREAM_GROUPS \
    (QBH_BLOCK_INTERMEDIATE / QBH_BLOCK_MLP_STREAM_CHANNELS)
#define QBH_BLOCK_MLP_CROUTON_GROUP_TILES UINT32_C(4)
#define QBH_BLOCK_MLP_CROUTON_GROUPS \
    (QBH_BLOCK_INTERMEDIATE / QBH_HMX_FP16_COLS / \
     QBH_BLOCK_MLP_CROUTON_GROUP_TILES)
#define QBH_BLOCK_MLP_CROUTON_RING_SLOTS \
    QBH_BLOCK_MLP_CROUTON_GROUPS
#define QBH_BLOCK_W4U8_GATE_UP_HVX_WORKERS UINT32_C(3)
#define QBH_BLOCK_W4U8_DOWN_HVX_WORKERS UINT32_C(6)
#define QBH_BLOCK_W4U8_DOWN_PERSISTENT_HVX_WORKERS UINT32_C(5)
#define QBH_BLOCK_W4U8_GATE_UP_PAIR_SLOTS UINT32_C(8)
#define QBH_BLOCK_W4U8_GATE_UP_HMX_BATCH_N_TILES UINT32_C(8)
#define QBH_BLOCK_W4U8_QKVO_MAX_BATCH_N_TILES UINT32_C(4)
#define QBH_BLOCK_W4U8_QKV_RING_SLOTS UINT32_C(4)
#define QBH_BLOCK_W4U8_QKV_RING_BATCHES UINT32_C(32)
#define QBH_BLOCK_W4U8_QKV_RING_TILES_PER_BATCH UINT32_C(4)
#define QBH_BLOCK_U8_SEGMENT_RING_SLOTS UINT32_C(26)
#define QBH_BLOCK_U8_SEGMENT_ASYNC_BANK_SLOTS \
    (QBH_BLOCK_U8_SEGMENT_RING_SLOTS / 2U)
#define QBH_BLOCK_U8_SEGMENT_K_WEIGHT_OFFSET UINT32_C(0)
#define QBH_BLOCK_U8_SEGMENT_K_BIAS_OFFSET \
    (QBH_BLOCK_U8_SEGMENT_K_WEIGHT_OFFSET + \
     QBH_KV_CACHE_HMX_U8_SEGMENT_WEIGHT_BYTES)
#define QBH_BLOCK_U8_SEGMENT_V_WEIGHT_OFFSET \
    (QBH_BLOCK_U8_SEGMENT_K_BIAS_OFFSET + QBH_HMX_BIAS_BYTES)
#define QBH_BLOCK_U8_SEGMENT_SCORE_OFFSET \
    (QBH_BLOCK_U8_SEGMENT_V_WEIGHT_OFFSET + \
     QBH_KV_CACHE_HMX_U8_SEGMENT_WEIGHT_BYTES)
#define QBH_BLOCK_U8_SEGMENT_SCORE_BYTES \
    (QBH_ATTENTION_Q_HEADS_PER_GROUP * QBH_HMX_OUTPUT_BYTES)
#define QBH_BLOCK_U8_SEGMENT_SLOT_BYTES \
    (QBH_BLOCK_U8_SEGMENT_SCORE_OFFSET + \
     QBH_BLOCK_U8_SEGMENT_SCORE_BYTES)
#define QBH_BLOCK_U8_SHORT_K_WEIGHT_OFFSET UINT32_C(0)
#define QBH_BLOCK_U8_SHORT_K_BIAS_OFFSET \
    (QBH_BLOCK_U8_SHORT_K_WEIGHT_OFFSET + \
     QBH_BLOCK_U8_SEGMENT_RING_SLOTS * \
         QBH_KV_CACHE_HMX_U8_SEGMENT_WEIGHT_BYTES)
#define QBH_BLOCK_U8_SHORT_V_WEIGHT_OFFSET \
    (QBH_BLOCK_U8_SHORT_K_BIAS_OFFSET + \
     QBH_BLOCK_U8_SEGMENT_RING_SLOTS * QBH_HMX_BIAS_BYTES)
#define QBH_BLOCK_U8_SHORT_SCORE_OFFSET \
    (QBH_BLOCK_U8_SHORT_V_WEIGHT_OFFSET + \
     QBH_BLOCK_U8_SEGMENT_RING_SLOTS * \
         QBH_KV_CACHE_HMX_U8_SEGMENT_WEIGHT_BYTES)
#define QBH_BLOCK_U8_SHORT_SCORE_HEAD_BYTES \
    (QBH_BLOCK_U8_SEGMENT_RING_SLOTS * QBH_HMX_OUTPUT_BYTES)
#define QBH_BLOCK_U8_SHORT_LAYOUT_BYTES \
    (QBH_BLOCK_U8_SHORT_SCORE_OFFSET + \
     QBH_ATTENTION_Q_HEADS_PER_GROUP * \
         QBH_BLOCK_U8_SHORT_SCORE_HEAD_BYTES)
#define QBH_BLOCK_U8_SEGMENT_HISTOGRAM_BYTES \
    (QBH_ATTENTION_Q_HEADS_PER_GROUP * 256U * sizeof(uint32_t))
#define QBH_BLOCK_U8_SEGMENT_ACTIVE_SCORE_BYTES \
    (QBH_ATTENTION_Q_HEADS_PER_GROUP * 256U)
#define QBH_BLOCK_U8_SEGMENT_ACCUMULATOR_BYTES \
    (QBH_ATTENTION_Q_HEADS_PER_GROUP * QBH_BLOCK_HEAD_DIM * \
     sizeof(int32_t))
#define QBH_BLOCK_W4U8_INPUT_NORM_ROWS_PER_TASK UINT32_C(4)
#define QBH_BLOCK_W4U8_RESIDUAL_ROWS_PER_TASK UINT32_C(4)
#define QBH_BLOCK_W4U8_SOFTMAX_ROW_SLICES UINT32_C(2)
#define QBH_BLOCK_W4U8_SOFTMAX_ROWS_PER_SLICE \
    (QBH_ATTENTION_M / QBH_BLOCK_W4U8_SOFTMAX_ROW_SLICES)
#define QBH_BLOCK_W4U8_GATHER_SCRATCH_BYTES \
    (QBH_BLOCK_W4U8_GATE_UP_HVX_WORKERS * \
     QBH_MLP_GATHER_SCRATCH_BYTES)

enum qbh_block_hvx_pool_job_kind {
    QBH_BLOCK_HVX_POOL_NONE = 0,
    QBH_BLOCK_HVX_POOL_W4_EXPAND = 1,
    QBH_BLOCK_HVX_POOL_SILU = 2,
    QBH_BLOCK_HVX_POOL_MLP_STREAM = 3,
    QBH_BLOCK_HVX_POOL_QK_NORM_ROPE = 4,
    QBH_BLOCK_HVX_POOL_SOFTMAX = 5,
    QBH_BLOCK_HVX_POOL_GQA_ATTENTION = 6,
    QBH_BLOCK_HVX_POOL_U8_GQA_ATTENTION = 7,
    QBH_BLOCK_HVX_POOL_U8_QK_PREP = 8,
    QBH_BLOCK_HVX_POOL_U8_RESIDUAL = 9,
    QBH_BLOCK_HVX_POOL_U8_INPUT_NORM = 10,
    QBH_BLOCK_HVX_POOL_W4U8_PIPELINE = 11,
    QBH_BLOCK_HVX_POOL_FP16_INPUT_NORM = 12,
    QBH_BLOCK_HVX_POOL_FP16_POST_RESIDUAL_NORM = 13,
    QBH_BLOCK_HVX_POOL_W4U8_QKV_RING = 14,
    QBH_BLOCK_HVX_POOL_U8_SEGMENTED_ATTENTION = 15,
    QBH_BLOCK_HVX_POOL_W4U8_GENERATION_EXPAND = 16,
};

enum qbh_block_u8_residual_kind {
    QBH_BLOCK_U8_RESIDUAL_POST_NORM = 1,
    QBH_BLOCK_U8_RESIDUAL_FINAL = 2,
};

struct qbh_block_arena {
    uint8_t *base;
    uint32_t capacity;
    uint32_t cursor;
    uint32_t peak;
};

struct qbh_block_buffers {
    uint8_t *input_norm_weight;
    uint8_t *post_norm_weight;
    uint8_t *q_norm_weight;
    uint8_t *k_norm_weight;
    uint8_t *rope_cos;
    uint8_t *rope_sin;
    uint8_t *residual;
    uint8_t *normalized;
    uint8_t *q;
    uint8_t *k;
    uint8_t *v;
    uint8_t *scores;
    uint8_t *probability;
    uint8_t *attention_concat;
    uint8_t *attention_projection;
    uint8_t *gate;
    uint8_t *up;
    uint8_t *middle;
    uint8_t *down;
    uint8_t *hmx_activation;
    uint8_t *compressed_weight;
    uint8_t *compressed_weight_alt;
    uint8_t *expanded_weight;
    uint8_t *expanded_weight_alt;
    uint8_t *hmx_output;
    uint8_t *scale_or_bias;
    uint8_t *channel_scale;
    uint8_t *channel_scale_alt;
    uint8_t *projection_scales;
    uint8_t *gate_up_scale_cache;
    uint8_t *w4u8_silu_lut;
    uint8_t *w4u8_gather_scratch;
    struct qbh_attention_config *attention_configs;
};

struct qbh_block_hmx_worker {
    uint32_t hmx_context_id;
    qurt_sem_t command_ready;
    qurt_sem_t command_done;
    qurt_sem_t worker_started;
    volatile uint32_t stop;
    volatile uint32_t kind;
    const void *activation;
    const void *weight;
    const void *scale_or_bias;
    void *output;
    uint32_t m_tiles;
    uint32_t k_tiles;
    uint32_t n_tiles;
    uint32_t region_tiles;
    const volatile uint32_t *ready_generations;
    uint32_t expected_generation;
    const struct qbh_w4_hmx_request *w4_pipeline_request;
    const void *qkv_ring_request;
    int32_t lock_status;
    int32_t unlock_status;
    int32_t command_status;
    uint64_t compute_ticks;
    uint64_t ready_wait_ticks;
};

struct qbh_block_w4f16_pool;

struct qbh_block_w4f16_job {
    struct qbh_block_w4f16_pool *pool;
    uint32_t worker_index;
    int32_t lock_status;
    int32_t unlock_status;
    uint32_t expand_count;
    uint64_t expand_ticks;
    uint32_t silu_chunk_count;
    uint64_t silu_ticks;
    volatile uint32_t command_kind;
    uint32_t stream_group_count;
    uint64_t stream_ticks;
    uint64_t stream_ready_wait_ticks;
    uint32_t attention_qk_norm_task_count;
    uint32_t fp16_qk_norm_pair_task_count;
    uint64_t attention_qk_norm_ticks;
    uint32_t attention_softmax_task_count;
    uint32_t u8_attention_softmax_shuffle4_row_group_count;
    uint64_t attention_softmax_ticks;
    uint32_t attention_gqa_group_count;
    uint64_t attention_gqa_ticks;
    uint64_t attention_gqa_hmx_wait_ticks;
    uint64_t attention_gqa_queue_wait_ticks;
    uint64_t attention_av_o_copy_ticks;
    uint32_t u8_attention_group_count;
    uint32_t u8_attention_score_saturation_count;
    uint32_t u8_attention_v_recenter_saturation_count;
    uint32_t u8_attention_probability_mask_violation_count;
    uint32_t u8_attention_probability_row_sum_min;
    uint32_t u8_attention_probability_row_sum_max;
    uint32_t u8_attention_fused_k_operand_mismatch_count;
    uint32_t u8_attention_prepared_group_count;
    uint32_t u8_qk_quarter_pair_count;
    uint64_t u8_attention_qk_norm_rope_ticks;
    uint64_t u8_attention_k_pack_ticks;
    uint64_t u8_attention_v_pack_ticks;
    uint64_t u8_attention_qk_hmx_ticks;
    uint64_t u8_attention_qk_requant_ticks;
    uint64_t u8_attention_softmax_ticks;
    uint64_t u8_attention_av_hmx_ticks;
    uint64_t u8_attention_av_requant_ticks;
    uint64_t u8_attention_hmx_queue_wait_ticks;
    uint32_t u8_post_residual_task_count;
    uint32_t u8_final_residual_task_count;
    uint64_t u8_post_residual_ticks;
    uint64_t u8_final_residual_ticks;
    uint32_t u8_input_norm_task_count;
    uint64_t u8_input_norm_ticks;
    uint32_t fp16_input_norm_task_count;
    uint64_t fp16_input_norm_ticks;
    uint32_t fp16_post_residual_norm_task_count;
    uint64_t fp16_post_residual_norm_ticks;
    uint32_t generation_expand_count;
    uint64_t generation_expand_ticks;
    void *w4u8_pipeline_worker_context;
    int32_t w4u8_pipeline_worker_status;
};

struct qbh_block_w4f16_pool {
    qurt_sem_t command_ready[QBH_BLOCK_MAX_POOL_HVX_WORKERS];
    qurt_sem_t command_done[QBH_BLOCK_MAX_POOL_HVX_WORKERS];
    qurt_sem_t worker_started[QBH_BLOCK_MAX_POOL_HVX_WORKERS];
    qurt_thread_t threads[QBH_BLOCK_MAX_POOL_HVX_WORKERS];
    struct qbh_block_w4f16_job jobs[QBH_BLOCK_MAX_POOL_HVX_WORKERS];
    volatile uint32_t stop;
    volatile uint32_t next_region;
    const uint8_t *compressed_weight;
    const float *channel_scale;
    uint8_t *expanded_weight;
    volatile uint32_t *ready_generations;
    uint32_t expected_generation;
    uint32_t region_count;
    uint32_t region_tiles;
    uint32_t publish_ready;
    uint32_t claim_regions;
    uint32_t extra_expand_worker_index;
    uint32_t worker_count;
    uint32_t active_worker_count;
    uint32_t created_workers;
    const __fp16 *silu_gate;
    const __fp16 *silu_up;
    __fp16 *silu_middle;
    volatile uint32_t next_silu_chunk;
    uint32_t silu_chunk_count;
    uint32_t silu_chunk_vectors;
    uint32_t silu_vector_count;
    const __fp16 *mlp_gate;
    const __fp16 *mlp_up;
    __fp16 *mlp_middle;
    __fp16 *mlp_hmx_activation;
    volatile uint32_t next_mlp_group;
    volatile uint32_t mlp_stream_abort;
    volatile uint32_t mlp_up_group_ready[QBH_BLOCK_MLP_STREAM_GROUPS];
    volatile uint32_t mlp_crouton_slot_consumed[
        QBH_BLOCK_MLP_CROUTON_RING_SLOTS];
    volatile uint32_t mlp_crouton_subgroup_done[
        QBH_BLOCK_MLP_CROUTON_GROUPS];
    uint32_t mlp_stream_generation;
    uint32_t mlp_stream_first_worker;
    uint32_t mlp_stream_worker_count;
    uint32_t mlp_crouton_native;
    uint32_t mlp_stream_group_limit;
    uint32_t mlp_crouton_group_tiles;
    uint32_t mlp_crouton_stream_group_tiles;
    uint32_t mlp_crouton_slot_elements;
    __fp16 *attention_q;
    __fp16 *attention_k;
    __fp16 *attention_q_destination;
    __fp16 *attention_k_weight;
    uint32_t attention_crouton_qkv;
    const __fp16 *attention_q_gamma;
    const __fp16 *attention_k_gamma;
    const __fp16 *attention_rope_cos;
    const __fp16 *attention_rope_sin;
    __fp16 *attention_scores;
    __fp16 *attention_probability;
    struct qbh_block_header *attention_header;
    struct qbh_block_buffers *attention_buffers;
    struct qbh_block_hmx_worker *attention_hmx_worker;
    qurt_mutex_t attention_hmx_mutex;
    volatile uint32_t attention_gqa_abort;
    float attention_gqa_qk_max_abs[QBH_BLOCK_HEADS];
    volatile uint32_t next_attention_task;
    uint32_t attention_task_base;
    uint32_t attention_task_count;
    volatile uint32_t next_attention_softmax_task;
    volatile uint32_t next_attention_av_task;
    volatile uint32_t u8_attention_qk_ready[QBH_BLOCK_KV_HEADS];
    volatile uint32_t u8_attention_softmax_ready[
        QBH_BLOCK_KV_HEADS * QBH_BLOCK_W4U8_SOFTMAX_ROW_SLICES];
    uint32_t u8_attention_dependency_generation;
    volatile uint32_t attention_qk_ready[
        QBH_BLOCK_HEADS + QBH_BLOCK_KV_HEADS];
    uint32_t attention_qk_generation;
    volatile uint32_t attention_qk_stream_abort;
    uint32_t attention_qk_streaming;
    uint32_t fp16_common_schedule_mode;
    uint8_t *u8_residual;
    const struct qbh_block_qparam *u8_residual_qparam;
    const uint8_t *u8_residual_addition_tiles;
    const struct qbh_block_qparam *u8_residual_addition_qparam;
    const struct qbh_block_qparam *u8_residual_sum_qparam;
    const __fp16 *u8_residual_gamma;
    uint8_t *u8_residual_normalized_tiles;
    const struct qbh_block_qparam *u8_residual_normalized_qparam;
    volatile uint32_t next_u8_residual_task;
    uint32_t u8_residual_task_count;
    uint32_t u8_residual_kind;
    uint32_t u8_residual_shuffle4;
    const uint8_t *u8_input_norm_input;
    const struct qbh_block_qparam *u8_input_norm_input_qparam;
    const __fp16 *u8_input_norm_gamma;
    uint8_t *u8_input_norm_output_tiles;
    const struct qbh_block_qparam *u8_input_norm_output_qparam;
    volatile uint32_t next_u8_input_norm_task;
    uint32_t u8_input_norm_task_count;
    const __fp16 *fp16_input_norm_input;
    const __fp16 *fp16_input_norm_gamma;
    __fp16 *fp16_input_norm_output;
    volatile uint32_t next_fp16_input_norm_task;
    uint32_t fp16_input_norm_task_count;
    uint32_t fp16_input_norm_crouton;
    __fp16 *fp16_post_residual;
    const __fp16 *fp16_post_addition;
    const __fp16 *fp16_post_gamma;
    __fp16 *fp16_post_output;
    volatile uint32_t next_fp16_post_residual_norm_task;
    uint32_t fp16_post_residual_norm_task_count;
    uint32_t fp16_post_residual_norm_crouton;
    uint32_t fp16_norm_rows_per_task;
    void *qkv_ring_state;
    uint8_t *segmented_attention_scratch;
    uint32_t segmented_attention_group_stride;
    uint32_t segmented_attention_ring_offset;
    uint32_t segmented_attention_batch_count;
    uint32_t segmented_attention_phase;
    uint32_t segmented_attention_valid_tokens;
    uint32_t segmented_attention_rows[
        QBH_BLOCK_U8_SEGMENT_RING_SLOTS];
    uint32_t segmented_attention_active_counts
        [QBH_BLOCK_KV_HEADS][QBH_ATTENTION_Q_HEADS_PER_GROUP];
    uint32_t segmented_attention_probability_sums
        [QBH_BLOCK_KV_HEADS][QBH_ATTENTION_Q_HEADS_PER_GROUP];
    uint32_t segmented_attention_saturation_counts
        [QBH_BLOCK_KV_HEADS][QBH_ATTENTION_Q_HEADS_PER_GROUP];
    volatile uint32_t segmented_short_qk_ready[QBH_BLOCK_KV_HEADS];
    volatile uint32_t segmented_short_probability_head_count[
        QBH_BLOCK_KV_HEADS];
    volatile uint32_t segmented_short_probability_ready[
        QBH_BLOCK_KV_HEADS];
    struct qbh_attention_u8_telemetry
        *segmented_attention_telemetry;
};

struct qbh_w4u8_qkv_ring_batch {
    const struct qbh_block_projection_desc *desc;
    uint8_t *output;
    uint32_t first_n_tile;
    uint32_t n_tiles;
};

struct qbh_w4u8_qkv_ring_state {
    struct qbh_block_header *header;
    const uint8_t *shared;
    struct qbh_block_w4f16_pool *pool;
    const uint8_t *activation;
    uint8_t *compressed_slots[QBH_BLOCK_W4U8_QKV_RING_SLOTS];
    uint8_t *expanded_slots[QBH_BLOCK_W4U8_QKV_RING_SLOTS];
    uint8_t *bias_slots[QBH_BLOCK_W4U8_QKV_RING_SLOTS];
    struct qbh_w4u8_qkv_ring_batch
        batches[QBH_BLOCK_W4U8_QKV_RING_BATCHES];
    qurt_sem_t compressed_free[QBH_BLOCK_W4U8_QKV_RING_SLOTS];
    qurt_sem_t expanded_free[QBH_BLOCK_W4U8_QKV_RING_SLOTS];
    qurt_sem_t expanded_ready[QBH_BLOCK_W4U8_QKV_RING_SLOTS];
    volatile uint32_t dma_ready[QBH_BLOCK_W4U8_QKV_RING_BATCHES];
    volatile uint32_t expanded_count[QBH_BLOCK_W4U8_QKV_RING_BATCHES];
    volatile uint32_t next_expand_task;
    volatile uint32_t abort_status;
    uint32_t generation;
    uint32_t batch_count;
    uint32_t k_tiles;
    uint32_t expand_worker_count;
    uint64_t hmx_ready_wait_ticks;
    uint64_t hmx_compute_ticks;
    uint32_t hmx_batch_count;
    uint32_t head_publish_count;
};

struct qbh_block_w4f16_cross_prefetch {
    struct qbh_dma_aligned_desc_1d descriptor;
    const struct qbh_block_projection_desc *target;
    uint64_t start_ticks;
    uint32_t active;
};

struct qbh_block_w4u8_hybrid_hvx_runner {
    struct qbh_block_w4f16_pool *pool;
    qurt_thread_t transient_thread;
    uint32_t transient_thread_created;
};

static uint8_t qbh_block_hmx_stack[QBH_BLOCK_HMX_STACK_BYTES]
    __attribute__((aligned(128)));
static uint8_t qbh_block_w4f16_hvx_stacks
    [QBH_BLOCK_MAX_POOL_HVX_WORKERS][QBH_BLOCK_W4F16_HVX_STACK_BYTES]
    __attribute__((aligned(128)));
static uint8_t qbh_block_w4u8_down_transient_hvx_stack
    [QBH_BLOCK_W4F16_HVX_STACK_BYTES]
    __attribute__((aligned(128)));

static int qbh_attention_parallel_qk_norm_enabled(uint32_t mode);
static int qbh_attention_parallel_softmax_enabled(uint32_t mode);
static int qbh_attention_gqa_enabled(uint32_t mode);
static int qbh_attention_u8_enabled(uint32_t mode);
static int qbh_attention_u8_fused_k_enabled(uint32_t mode);
static int qbh_attention_u8_qkv_overlap_enabled(uint32_t mode);
static int qbh_attention_u8_vgather_enabled(uint32_t mode);
static int qbh_attention_u8_vdeal_enabled(uint32_t mode);
static int qbh_attention_u8_fused_qk_requant_enabled(uint32_t mode);
static int qbh_attention_u8_hmx_batch_enabled(uint32_t mode);
static int qbh_attention_u8_gqa_hmx_batch_enabled(uint32_t mode);
static int qbh_attention_u8_dependency_stream_enabled(uint32_t mode);
static int qbh_attention_u8_softmax_shuffle4_enabled(uint32_t mode);
static int qbh_hmx_native_f16_cache_formats(uint32_t k_format,
                                             uint32_t v_format);
static void qbh_hvx_copy_aligned_bytes(
    void *destination, const void *source, uint32_t bytes);
static void qbh_hvx_zero_aligned_bytes(void *destination,
                                        uint32_t bytes);
static void qbh_attention_gqa_pool_run_tasks(
    struct qbh_block_w4f16_pool *pool,
    struct qbh_block_w4f16_job *job);
static void qbh_attention_u8_pool_run_tasks(
    struct qbh_block_w4f16_pool *pool,
    struct qbh_block_w4f16_job *job);
static void qbh_attention_u8_qk_prep_pool_run_tasks(
    struct qbh_block_w4f16_pool *pool,
    struct qbh_block_w4f16_job *job);
static void qbh_attention_u8_segmented_pool_run_tasks(
    struct qbh_block_w4f16_pool *pool,
    struct qbh_block_w4f16_job *job);
static void qbh_u8_residual_pool_run_tasks(
    struct qbh_block_w4f16_pool *pool,
    struct qbh_block_w4f16_job *job);
static void qbh_u8_input_norm_pool_run_tasks(
    struct qbh_block_w4f16_pool *pool,
    struct qbh_block_w4f16_job *job);
static int qbh_hvx_pool_u8_qk_prep_start_async(
    struct qbh_block_header *header,
    struct qbh_block_w4f16_pool *pool,
    struct qbh_block_buffers *buffers);
static void qbh_hvx_pool_u8_qk_prep_abort_async(
    struct qbh_block_w4f16_pool *pool);
static void qbh_hvx_pool_u8_qk_prep_publish(
    const struct qbh_block_header *header,
    const struct qbh_block_projection_desc *desc,
    struct qbh_block_w4f16_pool *pool,
    uint32_t first_n_tile, uint32_t n_tiles);
static int qbh_hvx_pool_u8_qk_prep_wait_async(
    struct qbh_block_header *header,
    struct qbh_block_w4f16_pool *pool);
static float qbh_f16_max_abs(
    const struct qbh_block_header *header,
    const __fp16 *data, uint32_t elements);

static uint32_t qbh_atomic_fetch_increment(volatile uint32_t *target) {
    uint32_t original;
    uint32_t updated;
    __asm__ __volatile__(
        "1:     %0 = memw_locked(%3)\n"
        "       %1 = add(%0, #1)\n"
        "       memw_locked(%3, p0) = %1\n"
        "       if !p0 jump 1b\n"
        : "=&r"(original), "=&r"(updated), "+m"(*target)
        : "r"(target)
        : "p0");
    return original;
}

static uint32_t qbh_atomic_fetch_add_two(volatile uint32_t *target) {
    uint32_t original;
    uint32_t updated;
    __asm__ __volatile__(
        "1:     %0 = memw_locked(%3)\n"
        "       %1 = add(%0, #2)\n"
        "       memw_locked(%3, p0) = %1\n"
        "       if !p0 jump 1b\n"
        : "=&r"(original), "=&r"(updated), "+m"(*target)
        : "r"(target)
        : "p0");
    return original;
}

static uint32_t qbh_align_up(uint32_t value, uint32_t alignment) {
    return (value + alignment - 1U) / alignment * alignment;
}

static void *qbh_arena_alloc_aligned(struct qbh_block_arena *arena,
                                     uint32_t bytes,
                                     uint32_t alignment) {
    uint32_t offset;
    if (arena == NULL || bytes == 0U || alignment == 0U) {
        return NULL;
    }
    offset = qbh_align_up(arena->cursor, alignment);
    if (offset > arena->capacity || bytes > arena->capacity - offset) {
        return NULL;
    }
    arena->cursor = offset + bytes;
    if (arena->cursor > arena->peak) {
        arena->peak = arena->cursor;
    }
    return arena->base + offset;
}

static void *qbh_arena_alloc(struct qbh_block_arena *arena,
                             uint32_t bytes) {
    return qbh_arena_alloc_aligned(
        arena, bytes, QBH_BLOCK_ALIGNMENT);
}

static uint64_t qbh_fnv1a64_bytes(const uint8_t *data, uint32_t bytes) {
    uint64_t hash = UINT64_C(1469598103934665603);
    for (uint32_t index = 0U; index < bytes; ++index) {
        hash ^= data[index];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static uint64_t qbh_map_gate_mix_range(
    uint64_t hash, const uint8_t *shared, uint32_t offset, uint32_t bytes) {
    if (bytes == 0U) {
        return hash;
    }
    hash ^= shared[offset];
    hash *= UINT64_C(1099511628211);
    hash ^= shared[offset + bytes - 1U];
    hash *= UINT64_C(1099511628211);
    return hash;
}

static uint64_t qbh_map_gate_layer_hash(
    const uint8_t *shared, const struct qbh_block_layer_desc *layer) {
    uint64_t hash = qbh_fnv1a64_bytes(
        (const uint8_t *)layer, (uint32_t)sizeof(*layer));
#define QBH_MAP_GATE_RANGE(name) \
    hash = qbh_map_gate_mix_range( \
        hash, shared, layer->name##_offset, layer->name##_bytes)
    QBH_MAP_GATE_RANGE(qparam);
    QBH_MAP_GATE_RANGE(input_norm_weight);
    QBH_MAP_GATE_RANGE(post_norm_weight);
    QBH_MAP_GATE_RANGE(q_norm_weight);
    QBH_MAP_GATE_RANGE(k_norm_weight);
    QBH_MAP_GATE_RANGE(w4u8_gate_up_bundle);
    QBH_MAP_GATE_RANGE(w4u8_down_bundle);
    QBH_MAP_GATE_RANGE(w4u8_silu_lut);
    QBH_MAP_GATE_RANGE(attention_config);
    QBH_MAP_GATE_RANGE(kv_cache_k);
    QBH_MAP_GATE_RANGE(kv_cache_v);
    QBH_MAP_GATE_RANGE(w4f16_gate_up_scale_cache);
#undef QBH_MAP_GATE_RANGE
    for (uint32_t projection = 0U;
         projection < QBH_BLOCK_PROJECTION_COUNT; ++projection) {
        const struct qbh_block_projection_desc *desc =
            &layer->projections[projection];
        hash = qbh_map_gate_mix_range(
            hash, shared, desc->weight_offset, desc->weight_bytes);
        hash = qbh_map_gate_mix_range(
            hash, shared, desc->scale_offset, desc->scale_bytes);
        hash = qbh_map_gate_mix_range(
            hash, shared, desc->bias_offset, desc->bias_bytes);
    }
    return hash;
}

static int qbh_plan_buffers(uint8_t *vtcm, uint32_t vtcm_bytes,
                            uint32_t variant,
                            uint32_t f16f16_projection_mode,
                            uint32_t w4f16_pipeline_mode,
                            uint32_t attention_pipeline_mode,
                            uint32_t mlp_mode,
                            uint32_t generation_mode,
                            struct qbh_block_buffers *buffers,
                            uint32_t *peak_bytes) {
    struct qbh_block_arena arena = {vtcm, vtcm_bytes, 0U, 0U};
    uint32_t element_bytes = variant == QBH_BLOCK_W4U8 ? 1U : 2U;
    uint32_t hidden_bytes = QBH_BLOCK_M * QBH_BLOCK_HIDDEN * element_bytes;
    uint32_t intermediate_bytes =
        QBH_BLOCK_M * QBH_BLOCK_INTERMEDIATE * element_bytes;
    uint32_t compressed_batch_factor =
        variant == QBH_BLOCK_W4F16
            ? QBH_BLOCK_W4F16_DMA_BATCH_N_TILES
            : (variant == QBH_BLOCK_W4U8
                   ? QBH_BLOCK_W4U8_QKVO_MAX_BATCH_N_TILES : 1U);
    uint32_t expanded_batch_factor =
        variant == QBH_BLOCK_W4F16
            ? QBH_BLOCK_W4F16_HMX_BATCH_N_TILES
            : (variant == QBH_BLOCK_F16F16 &&
               (f16f16_projection_mode ==
                    QBH_BLOCK_F16F16_PROJECTION_BATCH2 ||
                f16f16_projection_mode ==
                    QBH_BLOCK_F16F16_PROJECTION_GATE4 ||
                f16f16_projection_mode ==
                    QBH_BLOCK_F16F16_PROJECTION_GATE8 ||
                f16f16_projection_mode ==
                    QBH_BLOCK_F16F16_PROJECTION_GATE8_INTERLEAVED)
                   ? QBH_BLOCK_F16F16_BATCH_N_TILES
                   : 1U);
    uint32_t expanded_buffer_bytes =
        QBH_BLOCK_MAX_K * QBH_HMX_OUTPUT_CHANNELS *
        sizeof(uint16_t) * expanded_batch_factor;
    uint32_t scale_batch_factor =
        variant == QBH_BLOCK_W4F16
            ? 4U
            : (variant == QBH_BLOCK_W4U8
                   ? 2U * QBH_BLOCK_W4U8_QKVO_MAX_BATCH_N_TILES : 1U);

    memset(buffers, 0, sizeof(*buffers));
    buffers->input_norm_weight = qbh_arena_alloc(
        &arena, QBH_BLOCK_HIDDEN * sizeof(uint16_t));
    buffers->post_norm_weight = qbh_arena_alloc(
        &arena, QBH_BLOCK_HIDDEN * sizeof(uint16_t));
    buffers->q_norm_weight = qbh_arena_alloc(
        &arena, QBH_BLOCK_HEAD_DIM * sizeof(uint16_t));
    buffers->k_norm_weight = qbh_arena_alloc(
        &arena, QBH_BLOCK_HEAD_DIM * sizeof(uint16_t));
    buffers->rope_cos = qbh_arena_alloc(
        &arena, QBH_BLOCK_M * QBH_BLOCK_HEAD_DIM * sizeof(uint16_t));
    buffers->rope_sin = qbh_arena_alloc(
        &arena, QBH_BLOCK_M * QBH_BLOCK_HEAD_DIM * sizeof(uint16_t));
    buffers->residual = qbh_arena_alloc(&arena, hidden_bytes);
    buffers->normalized = qbh_arena_alloc(&arena, hidden_bytes);
    buffers->q = qbh_arena_alloc_aligned(
        &arena, QBH_BLOCK_M * QBH_BLOCK_HIDDEN * sizeof(uint16_t),
        QBH_HMX_FP16_TILE_BYTES);
    if (qbh_attention_u8_enabled(attention_pipeline_mode)) {
        buffers->k = qbh_arena_alloc_aligned(
            &arena,
            QBH_BLOCK_M * QBH_BLOCK_KV_HIDDEN * sizeof(uint16_t),
            QBH_HMX_FP16_TILE_BYTES);
        buffers->v = qbh_arena_alloc_aligned(
            &arena,
            QBH_BLOCK_M * QBH_BLOCK_KV_HIDDEN * sizeof(uint16_t),
            QBH_HMX_FP16_TILE_BYTES);
        buffers->scores = qbh_arena_alloc_aligned(
            &arena, QBH_BLOCK_SCORE_ELEMENTS * sizeof(uint16_t),
            QBH_HMX_FP16_TILE_BYTES);
        buffers->probability = qbh_arena_alloc_aligned(
            &arena, QBH_BLOCK_SCORE_ELEMENTS * sizeof(uint16_t),
            QBH_HMX_FP16_TILE_BYTES);
    } else {
        buffers->k = qbh_arena_alloc(
            &arena,
            QBH_BLOCK_M * QBH_BLOCK_KV_HIDDEN * sizeof(uint16_t));
        buffers->v = qbh_arena_alloc(
            &arena,
            QBH_BLOCK_M * QBH_BLOCK_KV_HIDDEN * sizeof(uint16_t));
        buffers->scores = qbh_arena_alloc(
            &arena, QBH_BLOCK_SCORE_ELEMENTS * sizeof(uint16_t));
        buffers->probability = qbh_arena_alloc(
            &arena, QBH_BLOCK_SCORE_ELEMENTS * sizeof(uint16_t));
    }
    buffers->attention_concat = qbh_arena_alloc_aligned(
        &arena, QBH_BLOCK_M * QBH_BLOCK_HIDDEN * sizeof(uint16_t),
        512U);
    buffers->attention_projection = qbh_arena_alloc(&arena, hidden_bytes);
    if (mlp_mode == QBH_BLOCK_MLP_CROUTON_NATIVE ||
        mlp_mode == QBH_BLOCK_MLP_CROUTON_NATIVE_BATCH8) {
        buffers->gate = qbh_arena_alloc_aligned(
            &arena, intermediate_bytes, QBH_HMX_FP16_TILE_BYTES);
        buffers->up = qbh_arena_alloc_aligned(
            &arena, intermediate_bytes, QBH_HMX_FP16_TILE_BYTES);
    } else {
        buffers->gate = qbh_arena_alloc(&arena, intermediate_bytes);
        buffers->up = qbh_arena_alloc(&arena, intermediate_bytes);
    }
    if (mlp_mode == QBH_BLOCK_MLP_CROUTON_NATIVE_BATCH8) {
        buffers->middle = qbh_arena_alloc_aligned(
            &arena,
            QBH_BLOCK_W4F16_HVX_WORKERS *
                QBH_BLOCK_HMX_OUTPUT_MAX_BYTES,
            QBH_HMX_FP16_TILE_BYTES);
    } else {
        buffers->middle = qbh_arena_alloc(&arena, intermediate_bytes);
    }
    buffers->down = qbh_arena_alloc(&arena, hidden_bytes);
    buffers->hmx_activation = qbh_arena_alloc_aligned(
        &arena, QBH_BLOCK_M * QBH_BLOCK_MAX_K * sizeof(uint16_t),
        QBH_HMX_FP16_TILE_BYTES);
    buffers->compressed_weight = qbh_arena_alloc(
        &arena, QBH_BLOCK_MAX_K * QBH_HMX_OUTPUT_CHANNELS / 2U *
                    compressed_batch_factor);
    buffers->compressed_weight_alt = qbh_arena_alloc(
        &arena, QBH_BLOCK_MAX_K * QBH_HMX_OUTPUT_CHANNELS / 2U *
                    compressed_batch_factor);
    if (mlp_mode == QBH_BLOCK_MLP_CROUTON_NATIVE_BATCH8 ||
        generation_mode ==
            QBH_BLOCK_GENERATION_GREEDY_W4U8_BATCH8_RESIDENT_BIAS) {
        expanded_buffer_bytes = variant == QBH_BLOCK_W4F16
            ? QBH_BLOCK_HIDDEN * 8U * QBH_HMX_FP16_COLS *
                  sizeof(uint16_t) +
              8U * (QBH_BLOCK_HIDDEN / QBH_HMX_FP16_COLS) *
                  QBH_W4_PACKED_TILE_BYTES
            : QBH_BLOCK_HIDDEN * 8U * QBH_HMX_FP16_COLS *
                  sizeof(uint16_t);
    }
    buffers->expanded_weight = qbh_arena_alloc_aligned(
        &arena, expanded_buffer_bytes,
        QBH_HMX_FP16_TILE_BYTES);
    buffers->expanded_weight_alt = qbh_arena_alloc_aligned(
        &arena, expanded_buffer_bytes,
        QBH_HMX_FP16_TILE_BYTES);
    buffers->hmx_output = qbh_arena_alloc_aligned(
        &arena, QBH_BLOCK_HMX_OUTPUT_MAX_BYTES,
        QBH_HMX_FP16_TILE_BYTES);
    buffers->scale_or_bias = qbh_arena_alloc_aligned(
        &arena, QBH_HMX_FP16_SCALE_BYTES * scale_batch_factor,
        QBH_HMX_FP16_SCALE_BYTES);
    buffers->channel_scale = qbh_arena_alloc_aligned(
        &arena, QBH_HMX_FP16_SCALE_BYTES,
        QBH_HMX_FP16_SCALE_BYTES);
    buffers->channel_scale_alt = qbh_arena_alloc_aligned(
        &arena, QBH_HMX_FP16_SCALE_BYTES,
        QBH_HMX_FP16_SCALE_BYTES);
    if (qbh_block_mlp_is_w4u8_streaming(mlp_mode)) {
        buffers->w4u8_silu_lut = qbh_arena_alloc_aligned(
            &arena, QBH_MLP_LUT_BYTES,
            QBH_MLP_GATHER_HALF_BYTES);
        buffers->w4u8_gather_scratch = qbh_arena_alloc_aligned(
            &arena, QBH_BLOCK_W4U8_GATHER_SCRATCH_BYTES,
            QBH_BLOCK_ALIGNMENT);
    }
    if (qbh_attention_u8_enabled(attention_pipeline_mode)) {
        buffers->attention_configs = qbh_arena_alloc_aligned(
            &arena, QBH_BLOCK_ATTENTION_CONFIG_BYTES,
            QBH_BLOCK_ALIGNMENT);
    }
    if (variant == QBH_BLOCK_W4F16) {
        buffers->projection_scales = qbh_arena_alloc_aligned(
            &arena, QBH_BLOCK_PROJECTION_SCALE_BYTES,
            QBH_HMX_FP16_SCALE_BYTES);
        if (w4f16_pipeline_mode ==
            QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_GATE4_DMA8_CROSS_PREFETCH) {
            buffers->gate_up_scale_cache = qbh_arena_alloc_aligned(
                &arena,
                2U * (QBH_BLOCK_INTERMEDIATE / QBH_HMX_FP16_COLS) *
                    QBH_HMX_FP16_SCALE_BYTES,
                QBH_HMX_FP16_SCALE_BYTES);
        }
    }

    if (buffers->input_norm_weight == NULL ||
        buffers->post_norm_weight == NULL ||
        buffers->q_norm_weight == NULL ||
        buffers->k_norm_weight == NULL || buffers->rope_cos == NULL ||
        buffers->rope_sin == NULL || buffers->residual == NULL ||
        buffers->normalized == NULL || buffers->q == NULL ||
        buffers->k == NULL || buffers->v == NULL ||
        buffers->scores == NULL || buffers->probability == NULL ||
        buffers->attention_concat == NULL ||
        buffers->attention_projection == NULL || buffers->gate == NULL ||
        buffers->up == NULL || buffers->middle == NULL ||
        buffers->down == NULL || buffers->hmx_activation == NULL ||
        buffers->compressed_weight == NULL ||
        buffers->compressed_weight_alt == NULL ||
        buffers->expanded_weight == NULL ||
        buffers->expanded_weight_alt == NULL ||
        buffers->hmx_output == NULL ||
        buffers->scale_or_bias == NULL || buffers->channel_scale == NULL ||
        buffers->channel_scale_alt == NULL ||
        (qbh_block_mlp_is_w4u8_streaming(mlp_mode) &&
         (buffers->w4u8_silu_lut == NULL ||
          buffers->w4u8_gather_scratch == NULL)) ||
        (qbh_attention_u8_enabled(attention_pipeline_mode) &&
         buffers->attention_configs == NULL) ||
        (variant == QBH_BLOCK_W4F16 &&
         (buffers->projection_scales == NULL ||
          (w4f16_pipeline_mode ==
               QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_GATE4_DMA8_CROSS_PREFETCH &&
           buffers->gate_up_scale_cache == NULL)))) {
        return -1;
    }
    if ((uintptr_t)buffers->attention_projection -
            (uintptr_t)buffers->q <
            QBH_BLOCK_M * QBH_BLOCK_INTERMEDIATE * sizeof(uint16_t) ||
        ((uintptr_t)buffers->q &
         (QBH_HMX_FP16_TILE_BYTES - 1U)) != 0U ||
        ((uintptr_t)buffers->hmx_activation &
         (QBH_HMX_FP16_TILE_BYTES - 1U)) != 0U ||
        (qbh_attention_u8_enabled(attention_pipeline_mode) &&
         ((((uintptr_t)buffers->k |
             (uintptr_t)buffers->v |
             (uintptr_t)buffers->scores |
             (uintptr_t)buffers->probability) &
            (QBH_HMX_FP16_TILE_BYTES - 1U)) != 0U ||
          (((uintptr_t)buffers->attention_concat) &
           (QBH_BLOCK_ALIGNMENT - 1U)) != 0U)) ||
        ((uintptr_t)buffers->expanded_weight &
         (QBH_HMX_FP16_TILE_BYTES - 1U)) != 0U ||
        ((uintptr_t)buffers->expanded_weight_alt &
         (QBH_HMX_FP16_TILE_BYTES - 1U)) != 0U ||
        ((uintptr_t)buffers->hmx_output &
         (QBH_HMX_FP16_TILE_BYTES - 1U)) != 0U ||
        ((uintptr_t)buffers->scale_or_bias &
         (QBH_HMX_FP16_SCALE_BYTES - 1U)) != 0U ||
        ((uintptr_t)buffers->channel_scale &
         (QBH_HMX_FP16_SCALE_BYTES - 1U)) != 0U ||
        ((uintptr_t)buffers->channel_scale_alt &
         (QBH_HMX_FP16_SCALE_BYTES - 1U)) != 0U ||
        (variant == QBH_BLOCK_W4F16 &&
         (((uintptr_t)buffers->projection_scales &
           (QBH_HMX_FP16_SCALE_BYTES - 1U)) != 0U ||
          (buffers->gate_up_scale_cache != NULL &&
           ((uintptr_t)buffers->gate_up_scale_cache &
            (QBH_HMX_FP16_SCALE_BYTES - 1U)) != 0U)))) {
        return -1;
    }
    *peak_bytes = arena.peak;
    return 0;
}

static int qbh_projection_shape_valid(
    uint32_t index, const struct qbh_block_projection_desc *desc);

static int qbh_range_valid(uint32_t offset, uint32_t bytes,
                           uint32_t shared_bytes) {
    return offset >= sizeof(struct qbh_block_header) &&
           offset <= shared_bytes && bytes <= shared_bytes - offset;
}

static int qbh_hmx_native_u8_cache_formats(uint32_t k_format,
                                            uint32_t v_format) {
    return (k_format == QBH_KV_CACHE_FORMAT_HMX_U8_K_WEIGHT_V1 &&
            v_format == QBH_KV_CACHE_FORMAT_HMX_U8_V_WEIGHT_V1) ||
           (k_format ==
                QBH_KV_CACHE_FORMAT_HMX_U8_K_WEIGHT_DELTA_V2 &&
            v_format ==
                QBH_KV_CACHE_FORMAT_HMX_U8_V_WEIGHT_DELTA_V2) ||
           (k_format ==
                QBH_KV_CACHE_FORMAT_HMX_U8_K_SEGMENTED_V4 &&
            v_format ==
                QBH_KV_CACHE_FORMAT_HMX_U8_V_SEGMENTED_V4);
}

static int qbh_hmx_native_u8_delta_cache_formats(
    uint32_t k_format, uint32_t v_format) {
    return k_format ==
               QBH_KV_CACHE_FORMAT_HMX_U8_K_WEIGHT_DELTA_V2 &&
           v_format ==
               QBH_KV_CACHE_FORMAT_HMX_U8_V_WEIGHT_DELTA_V2;
}

static int qbh_hmx_native_u8_segmented_cache_formats(
    uint32_t k_format, uint32_t v_format) {
    return k_format ==
               QBH_KV_CACHE_FORMAT_HMX_U8_K_SEGMENTED_V4 &&
           v_format ==
               QBH_KV_CACHE_FORMAT_HMX_U8_V_SEGMENTED_V4;
}

static int qbh_hmx_native_f16_cache_formats(uint32_t k_format,
                                             uint32_t v_format) {
    return k_format == QBH_KV_CACHE_FORMAT_HMX_F16_K_WEIGHT_V1 &&
           v_format == QBH_KV_CACHE_FORMAT_HMX_F16_V_WEIGHT_V1;
}

static int qbh_hmx_native_cache_formats(uint32_t k_format,
                                         uint32_t v_format) {
    return qbh_hmx_native_u8_cache_formats(k_format, v_format) ||
           qbh_hmx_native_f16_cache_formats(k_format, v_format);
}

static uint32_t qbh_expected_k_cache_bytes(uint32_t variant,
                                           uint32_t capacity,
                                           uint32_t element_bytes,
                                           uint32_t k_format) {
    if (variant == QBH_BLOCK_W4U8 &&
        k_format == QBH_KV_CACHE_FORMAT_HMX_U8_K_WEIGHT_V1) {
        return QBH_KV_CACHE_HMX_K_BYTES(capacity);
    }
    if (variant == QBH_BLOCK_W4U8 &&
        k_format ==
            QBH_KV_CACHE_FORMAT_HMX_U8_K_WEIGHT_DELTA_V2) {
        return QBH_KV_CACHE_HMX_U8_K_DELTA_BYTES(capacity);
    }
    if (variant == QBH_BLOCK_W4U8 &&
        k_format == QBH_KV_CACHE_FORMAT_HMX_U8_K_SEGMENTED_V4) {
        return QBH_KV_CACHE_HMX_U8_K_SEGMENTED_BYTES(capacity);
    }
    if (variant != QBH_BLOCK_W4U8 &&
        k_format == QBH_KV_CACHE_FORMAT_HMX_F16_K_WEIGHT_V1) {
        return QBH_KV_CACHE_HMX_F16_K_BYTES(capacity);
    }
    return capacity * QBH_BLOCK_KV_HIDDEN * element_bytes;
}

static uint32_t qbh_expected_v_cache_bytes(uint32_t variant,
                                           uint32_t capacity,
                                           uint32_t element_bytes,
                                           uint32_t v_format) {
    if (variant == QBH_BLOCK_W4U8 &&
        v_format == QBH_KV_CACHE_FORMAT_HMX_U8_V_WEIGHT_V1) {
        return QBH_KV_CACHE_HMX_V_BYTES(capacity);
    }
    if (variant == QBH_BLOCK_W4U8 &&
        v_format ==
            QBH_KV_CACHE_FORMAT_HMX_U8_V_WEIGHT_DELTA_V2) {
        return QBH_KV_CACHE_HMX_U8_V_DELTA_BYTES(capacity);
    }
    if (variant == QBH_BLOCK_W4U8 &&
        v_format == QBH_KV_CACHE_FORMAT_HMX_U8_V_SEGMENTED_V4) {
        return QBH_KV_CACHE_HMX_U8_V_SEGMENTED_BYTES(capacity);
    }
    if (variant != QBH_BLOCK_W4U8 &&
        v_format == QBH_KV_CACHE_FORMAT_HMX_F16_V_WEIGHT_V1) {
        return QBH_KV_CACHE_HMX_F16_V_BYTES(capacity);
    }
    return capacity * QBH_BLOCK_KV_HIDDEN * element_bytes;
}

static int qbh_slice_enabled(const struct qbh_block_header *header) {
    return header->slice_mode == QBH_BLOCK_SLICE_ACTIVE_RANGE &&
           header->slice_first_layer == QBH_VERTICAL_SLICE_FIRST_LAYER &&
           header->slice_layer_count == QBH_VERTICAL_SLICE_LAYER_COUNT;
}

static int qbh_full_stack_hidden_capture_valid(
    const struct qbh_block_header *header, uint32_t shared_bytes,
    uint32_t element_bytes) {
    const uint32_t layer_bytes =
        QBH_BLOCK_M * QBH_BLOCK_HIDDEN * element_bytes;
    const uint64_t capture_bytes =
        (uint64_t)QBH_VERTICAL_SLICE_LAYER_COUNT * layer_bytes;

    if (header->full_stack_stage_mode !=
        QBH_BLOCK_FULL_STACK_HIDDEN_CAPTURE) {
        return header->full_stack_hidden_capture_offset == 0U &&
               header->full_stack_hidden_capture_bytes == 0U &&
               header->full_stack_hidden_capture_layer_bytes == 0U;
    }
    return qbh_slice_enabled(header) &&
           header->scan_mode == QBH_BLOCK_SCAN_PREFILL &&
           header->logical_m == QBH_BLOCK_M &&
           header->initial_kv_length == 0U &&
           header->repeat_count == 1U &&
           capture_bytes <= UINT32_MAX &&
           header->full_stack_hidden_capture_layer_bytes == layer_bytes &&
           header->full_stack_hidden_capture_bytes ==
               (uint32_t)capture_bytes &&
           qbh_range_valid(
               header->full_stack_hidden_capture_offset,
               header->full_stack_hidden_capture_bytes, shared_bytes);
}

static int qbh_w4u8_boundary_audit_valid(
    const struct qbh_block_header *header, uint32_t shared_bytes) {
    const uint32_t expected_bytes = QBH_BLOCK_M * QBH_BLOCK_HIDDEN;

    if (header->w4u8_boundary_audit_enabled == 0U) {
        return header->w4u8_boundary_audit_output_offset == 0U &&
               header->w4u8_boundary_audit_output_bytes == 0U;
    }
    return header->w4u8_boundary_audit_enabled == 1U &&
           header->variant == QBH_BLOCK_W4U8 &&
           header->slice_mode == QBH_BLOCK_SLICE_DISABLED &&
           header->numerical_audit_enabled != 0U &&
           (header->crouton_boundary_mode &
            QBH_BLOCK_CROUTON_BOUNDARY_W4U8_QKV_INPUT) != 0U &&
           header->w4u8_boundary_audit_output_bytes == expected_bytes &&
           qbh_range_valid(
               header->w4u8_boundary_audit_output_offset,
               header->w4u8_boundary_audit_output_bytes, shared_bytes);
}

static void qbh_bind_slice_layer(struct qbh_block_header *header,
                                 uint32_t slice_index) {
    const struct qbh_block_layer_desc *layer =
        &header->slice_layers[slice_index];
    header->qparam_offset = layer->qparam_offset;
    header->qparam_bytes = layer->qparam_bytes;
    header->input_norm_weight_offset = layer->input_norm_weight_offset;
    header->input_norm_weight_bytes = layer->input_norm_weight_bytes;
    header->post_norm_weight_offset = layer->post_norm_weight_offset;
    header->post_norm_weight_bytes = layer->post_norm_weight_bytes;
    header->q_norm_weight_offset = layer->q_norm_weight_offset;
    header->q_norm_weight_bytes = layer->q_norm_weight_bytes;
    header->k_norm_weight_offset = layer->k_norm_weight_offset;
    header->k_norm_weight_bytes = layer->k_norm_weight_bytes;
    header->w4u8_gate_up_bundle_offset =
        layer->w4u8_gate_up_bundle_offset;
    header->w4u8_gate_up_bundle_bytes =
        layer->w4u8_gate_up_bundle_bytes;
    header->w4u8_down_bundle_offset = layer->w4u8_down_bundle_offset;
    header->w4u8_down_bundle_bytes = layer->w4u8_down_bundle_bytes;
    header->w4u8_silu_lut_offset = layer->w4u8_silu_lut_offset;
    header->w4u8_silu_lut_bytes = layer->w4u8_silu_lut_bytes;
    header->attention_config_offset = layer->attention_config_offset;
    header->attention_config_bytes = layer->attention_config_bytes;
    header->kv_cache_k_format = layer->kv_cache_k_format;
    header->kv_cache_v_format = layer->kv_cache_v_format;
    header->kv_cache_padded_capacity =
        layer->kv_cache_padded_capacity;
    header->kv_cache_k_offset = layer->kv_cache_k_offset;
    header->kv_cache_k_bytes = layer->kv_cache_k_bytes;
    header->kv_cache_v_offset = layer->kv_cache_v_offset;
    header->kv_cache_v_bytes = layer->kv_cache_v_bytes;
    header->w4f16_gate_up_scale_cache_offset =
        layer->w4f16_gate_up_scale_cache_offset;
    header->w4f16_gate_up_scale_cache_bytes =
        layer->w4f16_gate_up_scale_cache_bytes;
    memcpy(header->projections, layer->projections,
           sizeof(header->projections));
    memcpy(header->qparams, layer->qparams, sizeof(header->qparams));
}

static int qbh_slice_layer_desc_valid(
    const struct qbh_block_header *header,
    const struct qbh_block_layer_desc *layer,
    uint32_t expected_layer, uint32_t shared_bytes,
    uint32_t element_bytes) {
    const uint32_t expected_k_bytes = qbh_expected_k_cache_bytes(
        header->variant, header->kv_cache_capacity, element_bytes,
        layer->kv_cache_k_format);
    const uint32_t expected_v_bytes = qbh_expected_v_cache_bytes(
        header->variant, header->kv_cache_capacity, element_bytes,
        layer->kv_cache_v_format);
    const int row_major =
        layer->kv_cache_k_format ==
            QBH_KV_CACHE_FORMAT_HEAD_MAJOR_ROW_V1 &&
        layer->kv_cache_v_format ==
            QBH_KV_CACHE_FORMAT_HEAD_MAJOR_ROW_V1;
    const int hmx_native = qbh_hmx_native_cache_formats(
        layer->kv_cache_k_format, layer->kv_cache_v_format);
    const int hmx_native_matches_variant =
        header->variant == QBH_BLOCK_W4U8
            ? qbh_hmx_native_u8_cache_formats(
                  layer->kv_cache_k_format,
                  layer->kv_cache_v_format)
            : qbh_hmx_native_f16_cache_formats(
                  layer->kv_cache_k_format,
                  layer->kv_cache_v_format);

    if (layer->layer_index != expected_layer ||
        layer->qparam_bytes !=
            QBH_BLOCK_QPARAM_COUNT * QBH_BLOCK_QPARAM_RECORD_BYTES ||
        !qbh_range_valid(layer->qparam_offset, layer->qparam_bytes,
                         shared_bytes) ||
        layer->input_norm_weight_bytes !=
            QBH_BLOCK_HIDDEN * sizeof(uint16_t) ||
        layer->post_norm_weight_bytes !=
            QBH_BLOCK_HIDDEN * sizeof(uint16_t) ||
        layer->q_norm_weight_bytes !=
            QBH_BLOCK_HEAD_DIM * sizeof(uint16_t) ||
        layer->k_norm_weight_bytes !=
            QBH_BLOCK_HEAD_DIM * sizeof(uint16_t) ||
        !qbh_range_valid(layer->input_norm_weight_offset,
                         layer->input_norm_weight_bytes, shared_bytes) ||
        !qbh_range_valid(layer->post_norm_weight_offset,
                         layer->post_norm_weight_bytes, shared_bytes) ||
        !qbh_range_valid(layer->q_norm_weight_offset,
                         layer->q_norm_weight_bytes, shared_bytes) ||
        !qbh_range_valid(layer->k_norm_weight_offset,
                         layer->k_norm_weight_bytes, shared_bytes) ||
        (!row_major && !hmx_native_matches_variant) ||
        layer->kv_cache_padded_capacity !=
            (hmx_native
                 ? QBH_KV_CACHE_HMX_PADDED_CAPACITY(
                       header->kv_cache_capacity)
                 : header->kv_cache_capacity) ||
        layer->kv_cache_k_bytes != expected_k_bytes ||
        layer->kv_cache_v_bytes != expected_v_bytes ||
        !qbh_range_valid(layer->kv_cache_k_offset,
                         layer->kv_cache_k_bytes, shared_bytes) ||
        !qbh_range_valid(layer->kv_cache_v_offset,
                         layer->kv_cache_v_bytes, shared_bytes)) {
        return 0;
    }
    if (qbh_attention_u8_enabled(header->attention_pipeline_mode)) {
        if (layer->attention_config_bytes !=
                QBH_BLOCK_ATTENTION_CONFIG_BYTES ||
            !qbh_range_valid(layer->attention_config_offset,
                             layer->attention_config_bytes,
                             shared_bytes)) {
            return 0;
        }
    } else if (layer->attention_config_offset != 0U ||
               layer->attention_config_bytes != 0U) {
        return 0;
    }
    if (qbh_block_mlp_is_w4u8_streaming(header->mlp_mode)) {
        if (layer->w4u8_silu_lut_bytes != QBH_MLP_LUT_BYTES ||
            layer->w4u8_gate_up_bundle_bytes == 0U ||
            layer->w4u8_down_bundle_bytes == 0U ||
            !qbh_range_valid(layer->w4u8_silu_lut_offset,
                             layer->w4u8_silu_lut_bytes, shared_bytes) ||
            !qbh_range_valid(layer->w4u8_gate_up_bundle_offset,
                             layer->w4u8_gate_up_bundle_bytes,
                             shared_bytes) ||
            !qbh_range_valid(layer->w4u8_down_bundle_offset,
                             layer->w4u8_down_bundle_bytes,
                             shared_bytes)) {
            return 0;
        }
    }
    if (header->w4f16_pipeline_mode ==
            QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_GATE4_DMA8_CROSS_PREFETCH &&
        (layer->w4f16_gate_up_scale_cache_bytes == 0U ||
         !qbh_range_valid(layer->w4f16_gate_up_scale_cache_offset,
                          layer->w4f16_gate_up_scale_cache_bytes,
                          shared_bytes))) {
        return 0;
    }
    for (uint32_t index = 0U;
         index < QBH_BLOCK_PROJECTION_COUNT; ++index) {
        const struct qbh_block_projection_desc *desc =
            &layer->projections[index];
        uint32_t expected_weight =
            header->variant == QBH_BLOCK_F16F16
                ? desc->k * desc->n * sizeof(uint16_t)
                : desc->k * desc->n / 2U;
        if (!qbh_projection_shape_valid(index, desc) ||
            desc->weight_bytes != expected_weight ||
            !qbh_range_valid(desc->weight_offset, desc->weight_bytes,
                             shared_bytes)) {
            return 0;
        }
        if (header->variant != QBH_BLOCK_F16F16 &&
            (desc->scale_bytes != desc->n * sizeof(float) ||
             !qbh_range_valid(desc->scale_offset, desc->scale_bytes,
                              shared_bytes))) {
            return 0;
        }
        if (header->variant == QBH_BLOCK_W4U8 &&
            (desc->bias_bytes !=
                 desc->n / QBH_HMX_OUTPUT_CHANNELS * QBH_HMX_BIAS_BYTES ||
             !qbh_range_valid(desc->bias_offset, desc->bias_bytes,
                              shared_bytes))) {
            return 0;
        }
    }
    for (uint32_t index = 0U; index < QBH_BLOCK_QPARAM_COUNT; ++index) {
        if (!(layer->qparams[index].scale > 0.0f) ||
            !isfinite(layer->qparams[index].scale)) {
            return 0;
        }
    }
    return 1;
}

static int qbh_replay_session_valid(
    const struct qbh_block_header *header, uint32_t shared_bytes,
    uint32_t element_bytes) {
    const struct qbh_decode_session_state *state;
    const struct qbh_decode_layer_state *layer;
    uint32_t expected_k_bytes;
    uint32_t expected_v_bytes;
    uint32_t expected_element;

    if (header->replay_mode == QBH_BLOCK_REPLAY_DISABLED) {
        return header->replay_session_offset == 0U &&
               header->replay_session_bytes == 0U &&
               header->replay_expected_step == 0U &&
               header->replay_first_position == 0U &&
               header->slice_mode == QBH_BLOCK_SLICE_DISABLED;
    }
    if (header->replay_mode != QBH_BLOCK_REPLAY_CONTINUOUS ||
        header->repeat_count != 1U ||
        header->scan_mode == QBH_BLOCK_SCAN_DISABLED ||
        header->replay_session_bytes !=
            sizeof(struct qbh_decode_session_state) ||
        !qbh_range_valid(header->replay_session_offset,
                         header->replay_session_bytes, shared_bytes)) {
        return 0;
    }
    state = (const struct qbh_decode_session_state *)(
        (const uint8_t *)header + header->replay_session_offset);
    if (state->magic != QBH_DECODE_SESSION_MAGIC ||
        state->abi_version != QBH_DECODE_SESSION_ABI_VERSION ||
        state->state_bytes != sizeof(*state) ||
        state->declared_layer_count != QBH_QWEN3_TRANSFORMER_LAYERS ||
        state->active_layer != QBH_VERTICAL_SLICE_FIRST_LAYER ||
        state->active_layer_count != QBH_VERTICAL_SLICE_LAYER_COUNT ||
        state->completed_step_count != header->replay_expected_step ||
        state->next_position != header->replay_first_position ||
        state->next_position != header->initial_kv_length ||
        !qbh_slice_enabled(header)) {
        return 0;
    }
    expected_element = header->variant == QBH_BLOCK_W4U8
        ? QBH_KV_CACHE_ELEMENT_U8 : QBH_KV_CACHE_ELEMENT_F16;
    for (uint32_t slice_index = 0U;
         slice_index < QBH_VERTICAL_SLICE_LAYER_COUNT; ++slice_index) {
        const uint32_t layer_index =
            QBH_VERTICAL_SLICE_FIRST_LAYER + slice_index;
        const struct qbh_block_layer_desc *desc =
            &header->slice_layers[slice_index];
        layer = &state->layers[layer_index];
        expected_k_bytes = qbh_expected_k_cache_bytes(
            header->variant, layer->capacity, element_bytes,
            layer->k_format);
        expected_v_bytes = qbh_expected_v_cache_bytes(
            header->variant, layer->capacity, element_bytes,
            layer->v_format);
        const int row_major =
            layer->k_format == QBH_KV_CACHE_FORMAT_HEAD_MAJOR_ROW_V1 &&
            layer->v_format == QBH_KV_CACHE_FORMAT_HEAD_MAJOR_ROW_V1;
        const int hmx_native = qbh_hmx_native_cache_formats(
            layer->k_format, layer->v_format);
        const int hmx_native_matches_variant =
            header->variant == QBH_BLOCK_W4U8
                ? qbh_hmx_native_u8_cache_formats(
                      layer->k_format, layer->v_format)
                : qbh_hmx_native_f16_cache_formats(
                      layer->k_format, layer->v_format);
        const int hmx_native_u8_segmented =
            header->variant == QBH_BLOCK_W4U8 &&
            qbh_hmx_native_u8_segmented_cache_formats(
                layer->k_format, layer->v_format);
        if (layer->layer_index != layer_index ||
            layer->element_type != expected_element ||
            (!row_major && !hmx_native_matches_variant) ||
            layer->capacity != header->kv_cache_capacity ||
            layer->valid_length != header->initial_kv_length ||
            layer->valid_length + header->logical_m > layer->capacity ||
            layer->k_offset != desc->kv_cache_k_offset ||
            layer->k_bytes != desc->kv_cache_k_bytes ||
            layer->v_offset != desc->kv_cache_v_offset ||
            layer->v_bytes != desc->kv_cache_v_bytes ||
            layer->k_bytes != expected_k_bytes ||
            layer->v_bytes != expected_v_bytes ||
            layer->head_count != QBH_BLOCK_KV_HEADS ||
            layer->head_dim != QBH_BLOCK_HEAD_DIM ||
            layer->padded_capacity !=
                (hmx_native
                     ? QBH_KV_CACHE_HMX_PADDED_CAPACITY(layer->capacity)
                     : layer->capacity) ||
            layer->k_head_stride_bytes !=
                (hmx_native
                     ? (header->variant == QBH_BLOCK_W4U8
                            ? (hmx_native_u8_segmented
                                   ? QBH_KV_CACHE_HMX_U8_K_SEGMENTED_HEAD_BYTES(
                                         layer->capacity)
                                   : (qbh_hmx_native_u8_delta_cache_formats(
                                   layer->k_format, layer->v_format)
                                   ? QBH_KV_CACHE_HMX_U8_K_DELTA_HEAD_BYTES(
                                         layer->capacity)
                                   : QBH_KV_CACHE_HMX_K_HEAD_BYTES(
                                         layer->capacity)))
                            : QBH_KV_CACHE_HMX_F16_K_HEAD_BYTES(
                                  layer->capacity))
                     : layer->capacity * QBH_BLOCK_HEAD_DIM *
                           element_bytes) ||
            layer->v_head_stride_bytes !=
                (hmx_native
                     ? (header->variant == QBH_BLOCK_W4U8
                            ? (hmx_native_u8_segmented
                                   ? QBH_KV_CACHE_HMX_U8_V_SEGMENTED_HEAD_BYTES(
                                         layer->capacity)
                                   : (qbh_hmx_native_u8_delta_cache_formats(
                                   layer->k_format, layer->v_format)
                                   ? QBH_KV_CACHE_HMX_U8_V_DELTA_HEAD_BYTES(
                                         layer->capacity)
                                   : QBH_KV_CACHE_HMX_V_HEAD_BYTES(
                                         layer->capacity)))
                            : QBH_KV_CACHE_HMX_F16_V_HEAD_BYTES(
                                  layer->capacity))
                     : layer->capacity * QBH_BLOCK_HEAD_DIM *
                           element_bytes) ||
            layer->head_stride_bytes != layer->k_head_stride_bytes ||
            layer->token_stride_bytes !=
                (hmx_native ? 0U : QBH_BLOCK_HEAD_DIM * element_bytes) ||
            layer->k_weight_bytes_per_head !=
                (hmx_native
                     ? (header->variant == QBH_BLOCK_W4U8
                            ? (hmx_native_u8_segmented
                                   ? QBH_KV_CACHE_HMX_U8_SEGMENT_COUNT(
                                         layer->capacity) *
                                         QBH_KV_CACHE_HMX_U8_SEGMENT_K_BYTES
                                   : (qbh_hmx_native_u8_delta_cache_formats(
                                   layer->k_format, layer->v_format)
                                   ? QBH_KV_CACHE_HMX_U8_BASE_WEIGHT_BYTES_PER_HEAD
                                   : QBH_KV_CACHE_HMX_WEIGHT_BYTES_PER_HEAD(
                                         layer->capacity)))
                            : QBH_KV_CACHE_HMX_F16_WEIGHT_BYTES_PER_HEAD(
                                  layer->capacity))
                     : 0U) ||
            layer->v_weight_bytes_per_head !=
                (hmx_native
                     ? (header->variant == QBH_BLOCK_W4U8
                            ? (hmx_native_u8_segmented
                                   ? QBH_KV_CACHE_HMX_U8_SEGMENT_COUNT(
                                         layer->capacity) *
                                         QBH_KV_CACHE_HMX_U8_SEGMENT_K_BYTES
                                   : (qbh_hmx_native_u8_delta_cache_formats(
                                   layer->k_format, layer->v_format)
                                   ? QBH_KV_CACHE_HMX_U8_BASE_WEIGHT_BYTES_PER_HEAD
                                   : QBH_KV_CACHE_HMX_WEIGHT_BYTES_PER_HEAD(
                                         layer->capacity)))
                            : QBH_KV_CACHE_HMX_F16_WEIGHT_BYTES_PER_HEAD(
                                  layer->capacity))
                     : 0U) ||
            layer->k_bias_bytes_per_head !=
                (hmx_native && header->variant == QBH_BLOCK_W4U8
                     ? (hmx_native_u8_segmented
                            ? 0U
                            : (qbh_hmx_native_u8_delta_cache_formats(
                            layer->k_format, layer->v_format)
                            ? QBH_KV_CACHE_HMX_U8_K_BASE_BIAS_BYTES_PER_HEAD
                            : QBH_KV_CACHE_HMX_K_BIAS_BYTES_PER_HEAD(
                                  layer->capacity)))
                     : 0U) ||
            layer->v_bias_bytes_per_head !=
                (hmx_native && header->variant == QBH_BLOCK_W4U8
                     ? QBH_KV_CACHE_HMX_V_BIAS_BYTES_PER_HEAD
                     : 0U)) {
            return 0;
        }
    }
    return 1;
}

static int qbh_projection_shape_valid(uint32_t index,
                                      const struct qbh_block_projection_desc *desc) {
    static const uint32_t expected_k[QBH_BLOCK_PROJECTION_COUNT] = {
        QBH_BLOCK_HIDDEN, QBH_BLOCK_HIDDEN, QBH_BLOCK_HIDDEN,
        QBH_BLOCK_HIDDEN, QBH_BLOCK_HIDDEN, QBH_BLOCK_HIDDEN,
        QBH_BLOCK_INTERMEDIATE,
    };
    static const uint32_t expected_n[QBH_BLOCK_PROJECTION_COUNT] = {
        QBH_BLOCK_HIDDEN, QBH_BLOCK_KV_HIDDEN, QBH_BLOCK_KV_HIDDEN,
        QBH_BLOCK_HIDDEN, QBH_BLOCK_INTERMEDIATE,
        QBH_BLOCK_INTERMEDIATE, QBH_BLOCK_HIDDEN,
    };
    return index < QBH_BLOCK_PROJECTION_COUNT && desc != NULL &&
           desc->k == expected_k[index] && desc->n == expected_n[index];
}

static int qbh_generation_w4f16_enabled(uint32_t mode) {
    return mode == QBH_BLOCK_GENERATION_GREEDY_W4F16 ||
           mode == QBH_BLOCK_GENERATION_GREEDY_W4F16_HVX_ARGMAX ||
           mode == QBH_BLOCK_GENERATION_GREEDY_W4F16_HVX_ARGMAX_BATCH4 ||
           mode == QBH_BLOCK_GENERATION_GREEDY_W4F16_HVX_ARGMAX_BATCH8 ||
           mode == QBH_BLOCK_GENERATION_GREEDY_W4F16_LM_HEAD_OVERLAP ||
           mode == QBH_BLOCK_GENERATION_GREEDY_W4F16_DMA_HVX_OVERLAP ||
           mode == QBH_BLOCK_GENERATION_GREEDY_W4F16_COARSE_PIPELINE;
}

static int qbh_generation_w4u8_enabled(uint32_t mode) {
    return mode == QBH_BLOCK_GENERATION_GREEDY_W4U8_COARSE_PIPELINE ||
           mode ==
               QBH_BLOCK_GENERATION_GREEDY_W4U8_BATCH8_RESIDENT_BIAS;
}

static int qbh_generation_enabled(uint32_t mode) {
    return qbh_generation_w4f16_enabled(mode) ||
           qbh_generation_w4u8_enabled(mode);
}

static int qbh_generation_u8_qparam_valid(
    const struct qbh_block_qparam *qparam) {
    return qparam != NULL && qparam->scale > 0.0f &&
           isfinite(qparam->scale) && qparam->zero_point >= 0 &&
           qparam->zero_point <= UINT8_MAX;
}

static int qbh_generation_request_valid(
    const struct qbh_block_header *header, uint32_t shared_bytes) {
    const struct qbh_block_projection_desc *head;

    if (header->generation_mode == QBH_BLOCK_GENERATION_DISABLED) {
        return header->generation_token_ids_offset == 0U &&
               header->generation_token_ids_bytes == 0U &&
               header->generation_token_count == 0U &&
               header->generation_embedding_offset == 0U &&
               header->generation_embedding_bytes == 0U &&
               header->generation_final_norm_offset == 0U &&
               header->generation_final_norm_bytes == 0U &&
               header->generation_lm_head.k == 0U &&
               header->generation_lm_head.n == 0U &&
               header->generation_lm_head.weight_offset == 0U &&
               header->generation_lm_head.weight_bytes == 0U &&
               header->generation_lm_head.scale_offset == 0U &&
               header->generation_lm_head.scale_bytes == 0U &&
               header->generation_lm_head.bias_offset == 0U &&
               header->generation_lm_head.bias_bytes == 0U &&
               header->generation_boundary_audit_enabled == 0U &&
               header->generation_expected_token_ids_offset == 0U &&
               header->generation_expected_token_ids_bytes == 0U &&
               header->generation_expected_token_count == 0U;
    }
    if (!qbh_generation_enabled(header->generation_mode) ||
        (qbh_generation_w4f16_enabled(header->generation_mode)
             ? header->variant != QBH_BLOCK_W4F16
             : header->variant != QBH_BLOCK_W4U8) ||
        header->replay_mode != QBH_BLOCK_REPLAY_CONTINUOUS ||
        !qbh_slice_enabled(header) ||
        header->full_stack_stage_mode != QBH_BLOCK_FULL_STACK_RUN ||
        header->repeat_count != 1U ||
        header->generation_token_count != header->logical_m ||
        (header->generation_token_count != 1U &&
         header->generation_token_count != QBH_BLOCK_M) ||
        header->generation_token_ids_bytes !=
            QBH_BLOCK_M * sizeof(uint32_t) ||
        header->generation_embedding_bytes !=
            QBH_QWEN3_VOCAB_SIZE * QBH_BLOCK_HIDDEN *
                (qbh_generation_w4u8_enabled(header->generation_mode)
                     ? sizeof(uint8_t) : sizeof(uint16_t)) ||
        header->generation_final_norm_bytes !=
            QBH_BLOCK_HIDDEN * sizeof(uint16_t) ||
        header->generation_expected_token_ids_bytes !=
            QBH_GENERATION_DEFAULT_TOKENS * sizeof(uint32_t) ||
        header->generation_expected_token_count !=
            QBH_GENERATION_DEFAULT_TOKENS ||
        header->generation_boundary_audit_enabled > 1U ||
        !qbh_range_valid(header->generation_token_ids_offset,
                         header->generation_token_ids_bytes,
                         shared_bytes) ||
        !qbh_range_valid(header->generation_embedding_offset,
                         header->generation_embedding_bytes,
                         shared_bytes) ||
        !qbh_range_valid(header->generation_final_norm_offset,
                         header->generation_final_norm_bytes,
                         shared_bytes) ||
        !qbh_range_valid(header->generation_expected_token_ids_offset,
                         header->generation_expected_token_ids_bytes,
                         shared_bytes)) {
        return 0;
    }
    head = &header->generation_lm_head;
    if (head->k != QBH_BLOCK_HIDDEN ||
        head->n != QBH_QWEN3_VOCAB_SIZE ||
        head->weight_bytes != head->k * head->n / 2U ||
        head->scale_bytes != head->n * sizeof(float) ||
        !qbh_range_valid(head->weight_offset, head->weight_bytes,
                         shared_bytes) ||
        !qbh_range_valid(head->scale_offset, head->scale_bytes,
                         shared_bytes)) {
        return 0;
    }
    if (qbh_generation_w4u8_enabled(header->generation_mode)) {
        return head->bias_bytes ==
                   (head->n / QBH_HMX_OUTPUT_CHANNELS) *
                       QBH_HMX_BIAS_BYTES &&
               qbh_range_valid(head->bias_offset, head->bias_bytes,
                               shared_bytes) &&
               qbh_generation_u8_qparam_valid(
                   &header->generation_final_norm_output_qparam) &&
               qbh_generation_u8_qparam_valid(
                   &header->generation_lm_head_output_qparam);
    }
    return head->k == QBH_BLOCK_HIDDEN &&
           head->n == QBH_QWEN3_VOCAB_SIZE &&
           head->weight_bytes == head->k * head->n / 2U &&
           head->scale_bytes == head->n * sizeof(float) &&
           head->bias_offset == 0U && head->bias_bytes == 0U &&
           qbh_range_valid(head->weight_offset, head->weight_bytes,
                           shared_bytes) &&
           qbh_range_valid(head->scale_offset, head->scale_bytes,
                           shared_bytes);
}

static uint32_t qbh_scan_physical_chunks(
    const struct qbh_block_header *header) {
    if (header->scan_mode == QBH_BLOCK_SCAN_DISABLED) {
        return 1U;
    }
    return (header->logical_m + QBH_BLOCK_M - 1U) / QBH_BLOCK_M;
}

static int qbh_scan_request_valid(const struct qbh_block_header *header,
                                  uint32_t shared_bytes,
                                  uint32_t element_bytes) {
    uint32_t chunks;
    uint32_t tensor_bytes;
    uint32_t expected_k_bytes;
    uint32_t expected_v_bytes;
    int row_major;
    int hmx_native;

    if (header->scan_mode == QBH_BLOCK_SCAN_DISABLED) {
        return header->logical_m == QBH_BLOCK_M &&
               header->initial_kv_length == 0U &&
               header->kv_cache_capacity == 0U &&
               header->kv_cache_k_format == QBH_KV_CACHE_FORMAT_NONE &&
               header->kv_cache_v_format == QBH_KV_CACHE_FORMAT_NONE &&
               header->kv_cache_padded_capacity == 0U &&
               header->kv_cache_k_offset == 0U &&
               header->kv_cache_k_bytes == 0U &&
               header->kv_cache_v_offset == 0U &&
               header->kv_cache_v_bytes == 0U;
    }
    if (header->scan_mode == QBH_BLOCK_SCAN_PREFILL) {
        if ((header->logical_m != 16U && header->logical_m != 32U &&
             header->logical_m != 64U && header->logical_m != 128U) ||
            header->initial_kv_length != 0U) {
            return 0;
        }
    } else if (header->scan_mode == QBH_BLOCK_SCAN_DECODE) {
        const int replay_length_valid =
            header->replay_mode == QBH_BLOCK_REPLAY_CONTINUOUS &&
            header->initial_kv_length >= 64U &&
            header->initial_kv_length < header->kv_cache_capacity;
        const int scan_length_valid =
            header->initial_kv_length == 64U ||
            header->initial_kv_length == 256U ||
            header->initial_kv_length == 1024U ||
            header->initial_kv_length == 4096U;
        if (header->logical_m != 1U ||
            (!replay_length_valid && !scan_length_valid)) {
            return 0;
        }
    } else {
        return 0;
    }
    if (header->initial_kv_length + header->logical_m >
            QBH_BLOCK_SCAN_MAX_KV ||
        header->kv_cache_capacity <
            header->initial_kv_length + header->logical_m ||
        header->kv_cache_capacity > QBH_BLOCK_SCAN_MAX_KV) {
        return 0;
    }
    chunks = qbh_scan_physical_chunks(header);
    tensor_bytes = chunks * QBH_BLOCK_M * QBH_BLOCK_HIDDEN *
                   element_bytes;
    row_major =
        header->kv_cache_k_format ==
            QBH_KV_CACHE_FORMAT_HEAD_MAJOR_ROW_V1 &&
        header->kv_cache_v_format ==
            QBH_KV_CACHE_FORMAT_HEAD_MAJOR_ROW_V1;
    hmx_native = qbh_hmx_native_cache_formats(
        header->kv_cache_k_format, header->kv_cache_v_format);
    if (!row_major &&
        !(header->variant == QBH_BLOCK_W4U8
              ? qbh_hmx_native_u8_cache_formats(
                    header->kv_cache_k_format,
                    header->kv_cache_v_format)
              : qbh_hmx_native_f16_cache_formats(
                    header->kv_cache_k_format,
                    header->kv_cache_v_format))) {
        return 0;
    }
    expected_k_bytes = qbh_expected_k_cache_bytes(
        header->variant, header->kv_cache_capacity, element_bytes,
        header->kv_cache_k_format);
    expected_v_bytes = qbh_expected_v_cache_bytes(
        header->variant, header->kv_cache_capacity, element_bytes,
        header->kv_cache_v_format);
    return header->input_bytes == tensor_bytes &&
           header->output_bytes == tensor_bytes &&
           header->reference_bytes == tensor_bytes &&
           header->kv_cache_padded_capacity ==
               (hmx_native
                    ? QBH_KV_CACHE_HMX_PADDED_CAPACITY(
                          header->kv_cache_capacity)
                    : header->kv_cache_capacity) &&
           header->kv_cache_k_bytes == expected_k_bytes &&
           header->kv_cache_v_bytes == expected_v_bytes &&
           qbh_range_valid(header->kv_cache_k_offset,
                           header->kv_cache_k_bytes, shared_bytes) &&
           qbh_range_valid(header->kv_cache_v_offset,
                           header->kv_cache_v_bytes, shared_bytes);
}

static int qbh_header_valid(const struct qbh_block_header *header,
                            uint32_t shared_bytes) {
    uint32_t element_bytes;
    if (header == NULL || header->magic != QBH_BLOCK_MAGIC ||
        header->abi_version != QBH_BLOCK_ABI_VERSION ||
        header->experiment != QBH_BLOCK_EXPERIMENT ||
        header->header_bytes != sizeof(*header) ||
        header->shared_bytes != shared_bytes ||
        (header->variant != QBH_BLOCK_F16F16 &&
         header->variant != QBH_BLOCK_W4F16 &&
        header->variant != QBH_BLOCK_W4U8) ||
        header->repeat_count == 0U || header->repeat_count > 100U ||
        header->w4u8_prefill_cache_mode >
            QBH_BLOCK_W4U8_PREFILL_CACHE_REUSE_ATTENTION_CARRIERS ||
        header->w4u8_delta_reconstruction_mode >
            QBH_BLOCK_W4U8_DELTA_RECONSTRUCTION_PIPELINE ||
        (header->w4u8_prefill_cache_mode ==
             QBH_BLOCK_W4U8_PREFILL_CACHE_REUSE_ATTENTION_CARRIERS &&
         (header->variant != QBH_BLOCK_W4U8 ||
          !qbh_hmx_native_u8_cache_formats(
              header->kv_cache_k_format,
              header->kv_cache_v_format)))) {
        return 0;
    }
    element_bytes = header->variant == QBH_BLOCK_W4U8 ? 1U : 2U;
    if (!qbh_replay_session_valid(
            header, shared_bytes, element_bytes)) {
        return 0;
    }
    if (!qbh_full_stack_hidden_capture_valid(
            header, shared_bytes, element_bytes)) {
        return 0;
    }
    if (!qbh_w4u8_boundary_audit_valid(header, shared_bytes)) {
        return 0;
    }
    if (!qbh_generation_request_valid(header, shared_bytes)) {
        return 0;
    }
    if (qbh_slice_enabled(header)) {
        for (uint32_t slice_index = 0U;
             slice_index < QBH_VERTICAL_SLICE_LAYER_COUNT;
             ++slice_index) {
            if (!qbh_slice_layer_desc_valid(
                    header, &header->slice_layers[slice_index],
                    QBH_VERTICAL_SLICE_FIRST_LAYER + slice_index,
                    shared_bytes, element_bytes)) {
                return 0;
            }
        }
    } else if (header->slice_mode != QBH_BLOCK_SLICE_DISABLED) {
        return 0;
    }
    if ((header->common_ops_mask &
         ~((uint32_t)QBH_BLOCK_COMMON_OPS_HVX_FP16)) != 0U ||
        header->attribution_enabled > 1U ||
        header->numerical_audit_enabled > 1U ||
        header->full_stack_stage_mode >
            QBH_BLOCK_FULL_STACK_HIDDEN_CAPTURE ||
        header->residual_mode >
            QBH_BLOCK_RESIDUAL_HVX_FUSED_POST_NORM_POOL6_SHUFFLE4 ||
        header->f16f16_projection_mode >
            QBH_BLOCK_F16F16_PROJECTION_GATE8_INTERLEAVED ||
        header->w4f16_pipeline_mode >
            QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_GATE4_DMA8_CROSS_PREFETCH ||
        header->u8_norm_reduction_mode >
            QBH_BLOCK_U8_NORM_REDUCTION_HVX_TREE_QK_BATCHED_RSQRT_SHARED_ROPE_PARALLEL_INPUT ||
        header->w4u8_qk_pair_kernel_mode >
            QBH_BLOCK_W4U8_QK_PAIR_QUARTER_TILED_SIMD_IO ||
        header->fp16_common_schedule_mode >
            QBH_BLOCK_FP16_COMMON_SCHEDULE_ALL ||
        header->qkv_schedule_mode >
            QBH_BLOCK_QKV_SCHEDULE_KV_BATCH4 ||
        header->w4f16_group_fence_mode >
            QBH_BLOCK_W4F16_GROUP_FENCE_JOIN_ONLY_DOWN ||
        (header->w4f16_group_fence_mode !=
             QBH_BLOCK_W4F16_GROUP_FENCE_CONTROL &&
         header->variant != QBH_BLOCK_W4F16) ||
        (header->w4f16_expand_claim_regions != 1U &&
         header->w4f16_expand_claim_regions != 2U &&
         header->w4f16_expand_claim_regions != 3U) ||
        (header->w4f16_expand_claim_regions != 1U &&
         (header->variant != QBH_BLOCK_W4F16 ||
          (header->w4f16_group_fence_mode !=
               QBH_BLOCK_W4F16_GROUP_FENCE_JOIN_ONLY &&
           header->w4f16_group_fence_mode !=
               QBH_BLOCK_W4F16_GROUP_FENCE_JOIN_ONLY_DOWN))) ||
        header->w4f16_gate_up_extra_expand_worker > 1U ||
        (header->w4f16_gate_up_extra_expand_worker != 0U &&
         (header->variant != QBH_BLOCK_W4F16 ||
          (header->w4f16_group_fence_mode !=
               QBH_BLOCK_W4F16_GROUP_FENCE_JOIN_ONLY &&
           header->w4f16_group_fence_mode !=
               QBH_BLOCK_W4F16_GROUP_FENCE_JOIN_ONLY_DOWN) ||
          header->w4f16_requested_hvx_workers != 4U)) ||
        header->w4f16_gate_up_extra_stream_worker > 1U ||
        (header->w4f16_gate_up_extra_stream_worker != 0U &&
         (header->w4f16_gate_up_extra_expand_worker == 0U ||
          header->variant != QBH_BLOCK_W4F16 ||
          header->w4f16_requested_hvx_workers != 4U)) ||
        (header->w4f16_gate_up_stream_group_tiles != 4U &&
         header->w4f16_gate_up_stream_group_tiles != 8U) ||
        (header->w4f16_gate_up_stream_group_tiles != 8U &&
         (header->w4f16_gate_up_extra_expand_worker == 0U ||
          header->w4f16_gate_up_extra_stream_worker == 0U ||
          header->variant != QBH_BLOCK_W4F16 ||
          header->w4f16_requested_hvx_workers != 4U ||
          header->mlp_mode != QBH_BLOCK_MLP_CROUTON_NATIVE_BATCH8)) ||
        header->w4u8_stream_fence_mode >
            QBH_BLOCK_W4U8_STREAM_FENCE_RELEASE_ONLY ||
        (header->w4u8_stream_fence_mode !=
             QBH_BLOCK_W4U8_STREAM_FENCE_CONTROL &&
         header->variant != QBH_BLOCK_W4U8) ||
        (header->w4u8_gate_up_ring_slots != 8U &&
         header->w4u8_gate_up_ring_slots != 16U) ||
        (header->variant != QBH_BLOCK_W4U8 &&
         header->w4u8_gate_up_ring_slots != 8U) ||
        header->w4u8_qkv_ring_expand_workers > 3U ||
        (header->variant != QBH_BLOCK_W4U8 &&
         header->w4u8_qkv_ring_expand_workers != 0U) ||
        (header->w4u8_qkv_ring_expand_workers != 0U &&
         (header->w4u8_qkvo_pipeline_mode !=
              QBH_BLOCK_W4U8_QKVO_BATCH4_QK_HEAD_PAIRS ||
          !qbh_attention_u8_qkv_overlap_enabled(
              header->attention_pipeline_mode) ||
          header->attention_hvx_contexts !=
              QBH_BLOCK_MAX_ATTENTION_HVX_CONTEXTS ||
          (header->crouton_boundary_mode &
           QBH_BLOCK_CROUTON_BOUNDARY_W4U8_QKV_INPUT) == 0U)) ||
        (header->qkv_schedule_mode !=
             QBH_BLOCK_QKV_SCHEDULE_CONTROL &&
         (header->variant != QBH_BLOCK_W4F16 ||
          header->attention_pipeline_mode !=
              QBH_BLOCK_ATTENTION_PIPELINE_GQA_QKV_OVERLAP ||
          (header->crouton_boundary_mode &
           QBH_BLOCK_CROUTON_BOUNDARY_INPUT_NORM) == 0U)) ||
        (header->fp16_norm_rows_per_task != 2U &&
         header->fp16_norm_rows_per_task != 4U &&
         header->fp16_norm_rows_per_task != 8U) ||
        header->fp16_norm_contexts < 2U ||
        header->fp16_norm_contexts > 4U ||
        (header->w4u8_down_hmx_batch_outputs != 1U &&
         header->w4u8_down_hmx_batch_outputs != 4U) ||
        (header->variant != QBH_BLOCK_W4U8 &&
         header->w4u8_down_hmx_batch_outputs != 1U) ||
        (header->variant != QBH_BLOCK_W4U8 &&
         header->u8_norm_reduction_mode !=
             QBH_BLOCK_U8_NORM_REDUCTION_SCALAR) ||
        (header->variant != QBH_BLOCK_W4U8 &&
         header->w4u8_qk_pair_kernel_mode !=
             QBH_BLOCK_W4U8_QK_PAIR_SERIAL_INNER) ||
        (header->w4u8_qk_pair_kernel_mode >=
             QBH_BLOCK_W4U8_QK_PAIR_QUARTER_TILED &&
         header->u8_norm_reduction_mode <
             QBH_BLOCK_U8_NORM_REDUCTION_HVX_TREE_QK_BATCHED_RSQRT) ||
        (header->variant == QBH_BLOCK_W4U8 &&
         header->fp16_common_schedule_mode !=
             QBH_BLOCK_FP16_COMMON_SCHEDULE_CONTROL) ||
        (header->attention_pack_mode &
         ~((uint32_t)QBH_BLOCK_ATTENTION_PACK_HVX)) != 0U ||
        header->attention_pipeline_mode >
            QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER_VDEAL_FUSED_QK_REQUANT_HMX_BATCH_LUT_TEMPLATES_GQA_BATCH_DEPENDENCY_STREAM_SOFTMAX_SHUFFLE4 ||
        header->attention_hvx_contexts == 0U ||
        header->attention_hvx_contexts >
            QBH_BLOCK_MAX_ATTENTION_HVX_CONTEXTS ||
        (header->attention_pipeline_mode ==
             QBH_BLOCK_ATTENTION_PIPELINE_CONTROL &&
         header->attention_hvx_contexts != 1U) ||
        (header->attention_pipeline_mode !=
             QBH_BLOCK_ATTENTION_PIPELINE_CONTROL &&
         (qbh_attention_u8_enabled(header->attention_pipeline_mode)
              ? (header->variant != QBH_BLOCK_W4U8 ||
                 header->attention_hvx_contexts < 4U ||
                 header->attention_hvx_contexts >
                     QBH_BLOCK_MAX_ATTENTION_HVX_CONTEXTS)
              : (header->variant == QBH_BLOCK_W4U8 ||
                 header->attention_hvx_contexts != 4U))) ||
        (qbh_attention_parallel_qk_norm_enabled(
             header->attention_pipeline_mode) &&
         (header->common_ops_mask & QBH_BLOCK_COMMON_OP_ROPE) == 0U) ||
        (qbh_attention_parallel_softmax_enabled(
             header->attention_pipeline_mode) &&
         (header->common_ops_mask & QBH_BLOCK_COMMON_OP_SOFTMAX) == 0U) ||
        (qbh_attention_gqa_enabled(
             header->attention_pipeline_mode) &&
         (((header->common_ops_mask &
            (QBH_BLOCK_COMMON_OP_ROPE |
             QBH_BLOCK_COMMON_OP_SOFTMAX)) !=
           (QBH_BLOCK_COMMON_OP_ROPE |
            QBH_BLOCK_COMMON_OP_SOFTMAX)) ||
          header->attention_pack_mode !=
              QBH_BLOCK_ATTENTION_PACK_HVX)) ||
        (qbh_attention_u8_enabled(
             header->attention_pipeline_mode) &&
         (((header->common_ops_mask &
              (QBH_BLOCK_COMMON_OP_RMS_NORM |
               QBH_BLOCK_COMMON_OP_ROPE |
               QBH_BLOCK_COMMON_OP_SOFTMAX)) !=
             (QBH_BLOCK_COMMON_OP_RMS_NORM |
              QBH_BLOCK_COMMON_OP_ROPE |
              QBH_BLOCK_COMMON_OP_SOFTMAX)) ||
          header->attention_pack_mode !=
              QBH_BLOCK_ATTENTION_PACK_HVX)) ||
        (header->attention_pipeline_mode ==
             QBH_BLOCK_ATTENTION_PIPELINE_GQA_QKV_OVERLAP &&
         header->variant == QBH_BLOCK_W4F16 &&
         header->w4f16_requested_hvx_workers != 3U &&
         header->w4f16_requested_hvx_workers != 4U) ||
        (header->crouton_boundary_mode &
         ~((uint32_t)(QBH_BLOCK_CROUTON_BOUNDARY_QKV |
                      QBH_BLOCK_CROUTON_BOUNDARY_AV_TO_O |
                      QBH_BLOCK_CROUTON_BOUNDARY_INPUT_NORM |
                      QBH_BLOCK_CROUTON_BOUNDARY_POST_NORM |
                      QBH_BLOCK_CROUTON_BOUNDARY_W4U8_MLP_INPUT |
                      QBH_BLOCK_CROUTON_BOUNDARY_W4U8_MLP_OUTPUT |
                      QBH_BLOCK_CROUTON_BOUNDARY_W4U8_QKV_INPUT |
                      QBH_BLOCK_CROUTON_BOUNDARY_W4U8_O_OUTPUT))) != 0U ||
        (header->variant == QBH_BLOCK_W4U8 &&
         ((header->crouton_boundary_mode &
           ~((uint32_t)(QBH_BLOCK_CROUTON_BOUNDARY_W4U8_MLP_INPUT |
                        QBH_BLOCK_CROUTON_BOUNDARY_W4U8_MLP_OUTPUT |
                        QBH_BLOCK_CROUTON_BOUNDARY_W4U8_QKV_INPUT |
                        QBH_BLOCK_CROUTON_BOUNDARY_W4U8_O_OUTPUT))) != 0U ||
          ((header->crouton_boundary_mode &
            QBH_BLOCK_CROUTON_BOUNDARY_W4U8_MLP_OUTPUT) != 0U &&
           (header->crouton_boundary_mode &
            QBH_BLOCK_CROUTON_BOUNDARY_W4U8_MLP_INPUT) == 0U) ||
          ((header->crouton_boundary_mode &
            QBH_BLOCK_CROUTON_BOUNDARY_W4U8_O_OUTPUT) != 0U &&
           (header->crouton_boundary_mode &
            (QBH_BLOCK_CROUTON_BOUNDARY_W4U8_MLP_INPUT |
             QBH_BLOCK_CROUTON_BOUNDARY_W4U8_MLP_OUTPUT |
             QBH_BLOCK_CROUTON_BOUNDARY_W4U8_QKV_INPUT)) !=
            (QBH_BLOCK_CROUTON_BOUNDARY_W4U8_MLP_INPUT |
             QBH_BLOCK_CROUTON_BOUNDARY_W4U8_MLP_OUTPUT |
             QBH_BLOCK_CROUTON_BOUNDARY_W4U8_QKV_INPUT)) ||
          (header->crouton_boundary_mode !=
               QBH_BLOCK_CROUTON_BOUNDARY_CONTROL &&
           !qbh_block_mlp_is_w4u8_streaming(
               header->mlp_mode)))) ||
        (header->variant != QBH_BLOCK_W4U8 &&
         (header->crouton_boundary_mode &
          (QBH_BLOCK_CROUTON_BOUNDARY_W4U8_MLP_INPUT |
           QBH_BLOCK_CROUTON_BOUNDARY_W4U8_MLP_OUTPUT |
           QBH_BLOCK_CROUTON_BOUNDARY_W4U8_QKV_INPUT |
           QBH_BLOCK_CROUTON_BOUNDARY_W4U8_O_OUTPUT)) != 0U) ||
        header->w4u8_qkvo_pipeline_mode >
            QBH_BLOCK_W4U8_QKVO_BATCH4_QK_HEAD_PAIRS ||
        (header->variant != QBH_BLOCK_W4U8 &&
         header->w4u8_qkvo_pipeline_mode !=
             QBH_BLOCK_W4U8_QKVO_SERIAL) ||
        ((header->crouton_boundary_mode &
          QBH_BLOCK_CROUTON_BOUNDARY_QKV) != 0U &&
         header->attention_pipeline_mode !=
             QBH_BLOCK_ATTENTION_PIPELINE_GQA_QKV_OVERLAP) ||
        ((header->crouton_boundary_mode &
          QBH_BLOCK_CROUTON_BOUNDARY_AV_TO_O) != 0U &&
         header->attention_pipeline_mode !=
             QBH_BLOCK_ATTENTION_PIPELINE_GQA_QKV_OVERLAP) ||
        ((header->crouton_boundary_mode &
          QBH_BLOCK_CROUTON_BOUNDARY_W4U8_QKV_INPUT) != 0U &&
         (!qbh_attention_u8_qkv_overlap_enabled(
              header->attention_pipeline_mode) ||
          header->w4u8_qkvo_pipeline_mode <
              QBH_BLOCK_W4U8_QKVO_BATCH4)) ||
        ((header->crouton_boundary_mode &
          QBH_BLOCK_CROUTON_BOUNDARY_W4U8_O_OUTPUT) != 0U &&
         (!qbh_attention_u8_qkv_overlap_enabled(
              header->attention_pipeline_mode) ||
          header->w4u8_qkvo_pipeline_mode <
              QBH_BLOCK_W4U8_QKVO_BATCH4 ||
          (header->residual_mode !=
               QBH_BLOCK_RESIDUAL_HVX_FUSED_POST_NORM &&
           header->residual_mode !=
               QBH_BLOCK_RESIDUAL_HVX_FUSED_POST_NORM_POOL4 &&
           header->residual_mode !=
               QBH_BLOCK_RESIDUAL_HVX_FUSED_POST_NORM_POOL6 &&
           header->residual_mode !=
               QBH_BLOCK_RESIDUAL_HVX_FUSED_POST_NORM_POOL6_SHUFFLE4))) ||
        ((header->residual_mode ==
              QBH_BLOCK_RESIDUAL_HVX_FUSED_POST_NORM_POOL4 ||
          header->residual_mode ==
              QBH_BLOCK_RESIDUAL_HVX_FUSED_POST_NORM_POOL6 ||
          header->residual_mode ==
              QBH_BLOCK_RESIDUAL_HVX_FUSED_POST_NORM_POOL6_SHUFFLE4) &&
         (header->variant != QBH_BLOCK_W4U8 ||
          header->attention_hvx_contexts < 4U ||
          header->attention_hvx_contexts >
              QBH_BLOCK_MAX_ATTENTION_HVX_CONTEXTS ||
          (header->crouton_boundary_mode &
           (QBH_BLOCK_CROUTON_BOUNDARY_W4U8_MLP_INPUT |
            QBH_BLOCK_CROUTON_BOUNDARY_W4U8_MLP_OUTPUT |
            QBH_BLOCK_CROUTON_BOUNDARY_W4U8_QKV_INPUT |
            QBH_BLOCK_CROUTON_BOUNDARY_W4U8_O_OUTPUT)) !=
           (QBH_BLOCK_CROUTON_BOUNDARY_W4U8_MLP_INPUT |
            QBH_BLOCK_CROUTON_BOUNDARY_W4U8_MLP_OUTPUT |
            QBH_BLOCK_CROUTON_BOUNDARY_W4U8_QKV_INPUT |
            QBH_BLOCK_CROUTON_BOUNDARY_W4U8_O_OUTPUT))) ||
        ((header->residual_mode ==
              QBH_BLOCK_RESIDUAL_HVX_FUSED_POST_NORM_POOL6 ||
          header->residual_mode ==
              QBH_BLOCK_RESIDUAL_HVX_FUSED_POST_NORM_POOL6_SHUFFLE4) &&
         header->attention_hvx_contexts !=
             QBH_BLOCK_MAX_ATTENTION_HVX_CONTEXTS) ||
        header->mlp_mode >
            QBH_BLOCK_MLP_W4U8_STREAMING_PERSISTENT_MLP_HVX ||
        header->mlp_hvx_contexts == 0U ||
        header->mlp_hvx_contexts > QBH_BLOCK_W4F16_HVX_WORKERS ||
        (header->mlp_chunk_vectors != 16U &&
         header->mlp_chunk_vectors != 32U &&
         header->mlp_chunk_vectors != 64U &&
         header->mlp_chunk_vectors != 128U &&
         header->mlp_chunk_vectors != 256U) ||
        (header->mlp_mode == QBH_BLOCK_MLP_CONTROL &&
         header->mlp_hvx_contexts != 1U) ||
        (header->mlp_mode != QBH_BLOCK_MLP_CONTROL &&
         !qbh_block_mlp_is_w4u8_streaming(header->mlp_mode) &&
         (header->variant == QBH_BLOCK_W4U8 ||
          (header->common_ops_mask & QBH_BLOCK_COMMON_OP_SILU) == 0U)) ||
        (qbh_block_mlp_is_w4u8_streaming(header->mlp_mode) &&
         (header->variant != QBH_BLOCK_W4U8 ||
          header->mlp_hvx_contexts !=
              QBH_BLOCK_W4U8_GATE_UP_HVX_WORKERS)) ||
        (header->mlp_mode == QBH_BLOCK_MLP_STREAMING &&
         (header->mlp_hvx_contexts != 4U ||
          (header->variant == QBH_BLOCK_F16F16 &&
           header->f16f16_projection_mode !=
               QBH_BLOCK_F16F16_PROJECTION_BATCH2 &&
           header->f16f16_projection_mode !=
               QBH_BLOCK_F16F16_PROJECTION_GATE4) ||
          (header->variant == QBH_BLOCK_W4F16 &&
           header->w4f16_pipeline_mode !=
               QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_CROSS_PREFETCH &&
           header->w4f16_pipeline_mode !=
               QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_GATE16_CROSS_PREFETCH &&
           header->w4f16_pipeline_mode !=
               QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_GATE8_CROSS_PREFETCH &&
           header->w4f16_pipeline_mode !=
               QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_GATE4_CROSS_PREFETCH &&
           header->w4f16_pipeline_mode !=
               QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_GATE4_DMA8_CROSS_PREFETCH))) ||
        (header->mlp_mode == QBH_BLOCK_MLP_CROUTON_NATIVE &&
         (header->variant != QBH_BLOCK_W4F16 ||
          header->mlp_hvx_contexts != 4U ||
          header->w4f16_pipeline_mode !=
              QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_GATE4_DMA8_CROSS_PREFETCH)) ||
        (header->mlp_mode == QBH_BLOCK_MLP_CROUTON_NATIVE_BATCH8 &&
         (header->mlp_hvx_contexts != 4U ||
          (header->variant == QBH_BLOCK_W4F16 &&
           header->w4f16_pipeline_mode !=
               QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_GATE4_DMA8_CROSS_PREFETCH) ||
          (header->variant == QBH_BLOCK_F16F16 &&
           header->f16f16_projection_mode !=
               QBH_BLOCK_F16F16_PROJECTION_GATE8 &&
           header->f16f16_projection_mode !=
               QBH_BLOCK_F16F16_PROJECTION_GATE8_INTERLEAVED))) ||
        (header->variant != QBH_BLOCK_F16F16 &&
         header->f16f16_projection_mode !=
             QBH_BLOCK_F16F16_PROJECTION_SERIAL) ||
        (header->variant != QBH_BLOCK_W4F16 &&
         header->w4f16_pipeline_mode !=
             QBH_BLOCK_W4F16_PIPELINE_CONTROL) ||
        ((header->w4f16_pipeline_mode ==
              QBH_BLOCK_W4F16_PIPELINE_HYBRID_WORKERS ||
          header->w4f16_pipeline_mode ==
              QBH_BLOCK_W4F16_PIPELINE_HYBRID_CROSS_PREFETCH ||
          header->w4f16_pipeline_mode ==
              QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN64_CROSS_PREFETCH ||
          header->w4f16_pipeline_mode ==
              QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN48_CROSS_PREFETCH ||
          header->w4f16_pipeline_mode ==
              QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_CROSS_PREFETCH ||
          header->w4f16_pipeline_mode ==
              QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_GATE16_CROSS_PREFETCH ||
          header->w4f16_pipeline_mode ==
              QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_GATE8_CROSS_PREFETCH ||
          header->w4f16_pipeline_mode ==
              QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_GATE4_CROSS_PREFETCH ||
          header->w4f16_pipeline_mode ==
              QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_GATE4_DMA8_CROSS_PREFETCH) &&
         header->w4f16_requested_hvx_workers != 3U &&
         header->w4f16_requested_hvx_workers != 4U) ||
        ((header->w4f16_pipeline_mode ==
              QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN64_CROSS_PREFETCH ||
          header->w4f16_pipeline_mode ==
              QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN48_CROSS_PREFETCH ||
          header->w4f16_pipeline_mode ==
              QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_CROSS_PREFETCH ||
          header->w4f16_pipeline_mode ==
              QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_GATE16_CROSS_PREFETCH ||
          header->w4f16_pipeline_mode ==
              QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_GATE8_CROSS_PREFETCH ||
          header->w4f16_pipeline_mode ==
              QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_GATE4_CROSS_PREFETCH ||
          header->w4f16_pipeline_mode ==
              QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_GATE4_DMA8_CROSS_PREFETCH) &&
         header->w4f16_region_tiles != 32U) ||
        (header->variant == QBH_BLOCK_W4U8 &&
         ((!qbh_attention_u8_enabled(
                header->attention_pipeline_mode) &&
           (header->attention_pack_mode !=
                QBH_BLOCK_ATTENTION_PACK_CONTROL ||
            header->attention_pipeline_mode !=
                QBH_BLOCK_ATTENTION_PIPELINE_CONTROL)) ||
          (header->mlp_mode != QBH_BLOCK_MLP_CONTROL &&
           !qbh_block_mlp_is_w4u8_streaming(
               header->mlp_mode))))) {
        return 0;
    }
    if (header->w4f16_requested_hvx_workers == 0U ||
        header->w4f16_requested_hvx_workers >
            QBH_BLOCK_W4F16_MAX_REQUESTED_HVX_WORKERS ||
        (header->w4f16_region_tiles != 8U &&
         header->w4f16_region_tiles != 16U &&
         header->w4f16_region_tiles != 32U &&
         header->w4f16_region_tiles != 64U) ||
        (header->w4f16_pipeline_mode ==
             QBH_BLOCK_W4F16_PIPELINE_EARLY_REGION &&
         header->w4f16_region_tiles > 32U)) {
        return 0;
    }
    element_bytes = header->variant == QBH_BLOCK_W4U8 ? 1U : 2U;
    if (!qbh_scan_request_valid(header, shared_bytes, element_bytes) ||
        header->qparam_bytes !=
            QBH_BLOCK_QPARAM_COUNT * QBH_BLOCK_QPARAM_RECORD_BYTES ||
        !qbh_range_valid(header->input_offset, header->input_bytes,
                         shared_bytes) ||
        !qbh_range_valid(header->output_offset, header->output_bytes,
                         shared_bytes) ||
        !qbh_range_valid(header->reference_offset,
                         header->reference_bytes, shared_bytes) ||
        !qbh_range_valid(header->qparam_offset, header->qparam_bytes,
                         shared_bytes) ||
        header->input_norm_weight_bytes !=
            QBH_BLOCK_HIDDEN * sizeof(uint16_t) ||
        header->post_norm_weight_bytes !=
            QBH_BLOCK_HIDDEN * sizeof(uint16_t) ||
        header->q_norm_weight_bytes !=
            QBH_BLOCK_HEAD_DIM * sizeof(uint16_t) ||
        header->k_norm_weight_bytes !=
            QBH_BLOCK_HEAD_DIM * sizeof(uint16_t) ||
        header->rope_cos_bytes !=
            qbh_scan_physical_chunks(header) * QBH_BLOCK_M *
                QBH_BLOCK_HEAD_DIM * sizeof(uint16_t) ||
        header->rope_sin_bytes != header->rope_cos_bytes) {
        return 0;
    }
    if (qbh_attention_u8_enabled(
            header->attention_pipeline_mode) &&
        (header->attention_config_bytes !=
             QBH_BLOCK_ATTENTION_CONFIG_BYTES ||
         !qbh_range_valid(header->attention_config_offset,
                          header->attention_config_bytes,
                          shared_bytes))) {
        return 0;
    }
    if (qbh_attention_u8_enabled(
            header->attention_pipeline_mode) &&
        header->numerical_audit_enabled != 0U &&
        (header->u8_attention_audit_output_bytes !=
             QBH_BLOCK_U8_ATTENTION_AUDIT_BYTES ||
         !qbh_range_valid(
             header->u8_attention_audit_output_offset,
             header->u8_attention_audit_output_bytes,
             shared_bytes))) {
        return 0;
    }
    if ((!qbh_attention_u8_enabled(
             header->attention_pipeline_mode) ||
         header->numerical_audit_enabled == 0U) &&
        (header->u8_attention_audit_output_offset != 0U ||
         header->u8_attention_audit_output_bytes != 0U)) {
        return 0;
    }
    if (!qbh_attention_u8_enabled(
            header->attention_pipeline_mode) &&
        (header->attention_config_offset != 0U ||
         header->attention_config_bytes != 0U)) {
        return 0;
    }
    if (qbh_block_mlp_is_w4u8_streaming(header->mlp_mode) &&
        (header->w4u8_silu_lut_bytes != QBH_MLP_LUT_BYTES ||
         header->w4u8_gate_up_bundle_bytes == 0U ||
         header->w4u8_down_bundle_bytes == 0U ||
         !qbh_range_valid(header->w4u8_silu_lut_offset,
                          header->w4u8_silu_lut_bytes, shared_bytes) ||
         !qbh_range_valid(header->w4u8_gate_up_bundle_offset,
                          header->w4u8_gate_up_bundle_bytes,
                          shared_bytes) ||
         !qbh_range_valid(header->w4u8_down_bundle_offset,
                          header->w4u8_down_bundle_bytes,
                          shared_bytes))) {
        return 0;
    }
    if (!qbh_range_valid(header->input_norm_weight_offset,
                         header->input_norm_weight_bytes, shared_bytes) ||
        !qbh_range_valid(header->post_norm_weight_offset,
                         header->post_norm_weight_bytes, shared_bytes) ||
        !qbh_range_valid(header->q_norm_weight_offset,
                         header->q_norm_weight_bytes, shared_bytes) ||
        !qbh_range_valid(header->k_norm_weight_offset,
                         header->k_norm_weight_bytes, shared_bytes) ||
        !qbh_range_valid(header->rope_cos_offset, header->rope_cos_bytes,
                         shared_bytes) ||
        !qbh_range_valid(header->rope_sin_offset, header->rope_sin_bytes,
                         shared_bytes)) {
        return 0;
    }
    for (uint32_t index = 0; index < QBH_BLOCK_PROJECTION_COUNT;
         ++index) {
        const struct qbh_block_projection_desc *desc =
            &header->projections[index];
        uint32_t expected_weight =
            header->variant == QBH_BLOCK_F16F16
                ? desc->k * desc->n * sizeof(uint16_t)
                : desc->k * desc->n / 2U;
        if (!qbh_projection_shape_valid(index, desc) ||
            desc->weight_bytes != expected_weight ||
            !qbh_range_valid(desc->weight_offset, desc->weight_bytes,
                             shared_bytes)) {
            return 0;
        }
        if (header->variant != QBH_BLOCK_F16F16 &&
            (desc->scale_bytes != desc->n * sizeof(float) ||
             !qbh_range_valid(desc->scale_offset, desc->scale_bytes,
                              shared_bytes))) {
            return 0;
        }
        if (header->variant == QBH_BLOCK_W4U8 &&
            (desc->bias_bytes !=
                 desc->n / QBH_HMX_OUTPUT_CHANNELS * QBH_HMX_BIAS_BYTES ||
             !qbh_range_valid(desc->bias_offset, desc->bias_bytes,
                              shared_bytes))) {
            return 0;
        }
    }
    for (uint32_t index = 0; index < QBH_BLOCK_QPARAM_COUNT; ++index) {
        if (!(header->qparams[index].scale > 0.0f) ||
            !isfinite(header->qparams[index].scale)) {
            return 0;
        }
    }
    return 1;
}

static int qbh_dma_copy(struct qbh_block_header *header, void *destination,
                        const void *source, uint32_t bytes,
                        uint32_t ddr_to_vtcm) {
    struct qbh_dma_aligned_desc_1d aligned;
    struct qbh_dma_desc_1d *descriptor = &aligned.descriptor;
    uint64_t start = HAP_perf_get_qtimer_count();
    if (destination == NULL || source == NULL || bytes == 0U ||
        bytes >= UINT32_C(0x01000000)) {
        return -1;
    }
    if (qbh_dma_wait_idle() != 0) {
        return -2;
    }
    memset(&aligned, 0, sizeof(aligned));
    descriptor->length = bytes;
    descriptor->type = QBH_DMA_TYPE_1D;
    descriptor->src_bypass = ddr_to_vtcm != 0U;
    descriptor->dst_bypass = ddr_to_vtcm == 0U;
    descriptor->ordered = 1;
    descriptor->dstate = QBH_DMA_DESC_PENDING;
    descriptor->src = (uint32_t)(uintptr_t)source;
    descriptor->dst = (uint32_t)(uintptr_t)destination;
    if (qbh_dma_start(descriptor) != 0) {
        return -3;
    }
    if (qbh_dma_wait_idle() != 0) {
        return -4;
    }
    if (descriptor->dstate != QBH_DMA_DESC_COMPLETE) {
        return -5;
    }
    if (ddr_to_vtcm != 0U) {
        header->weight_dma_ticks += HAP_perf_get_qtimer_count() - start;
    }
    return 0;
}

static int qbh_dma_start_weight_prefetch(
    struct qbh_dma_aligned_desc_1d *aligned,
    void *weight_destination, const void *weight_source,
    uint32_t weight_bytes) {
    struct qbh_dma_desc_1d *weight = &aligned->descriptor;

    if (qbh_dma_wait_idle() != 0) {
        return -1;
    }
    memset(aligned, 0, sizeof(*aligned));
    weight->length = weight_bytes;
    weight->type = QBH_DMA_TYPE_1D;
    weight->src_bypass = 1;
    weight->ordered = 1;
    weight->dstate = QBH_DMA_DESC_PENDING;
    weight->src = (uint32_t)(uintptr_t)weight_source;
    weight->dst = (uint32_t)(uintptr_t)weight_destination;
    return qbh_dma_start(weight) == 0 ? 0 : -2;
}

static int qbh_dma_wait_weight_prefetch(
    struct qbh_dma_aligned_desc_1d *aligned) {
    struct qbh_dma_desc_1d *descriptor = &aligned->descriptor;
    uint64_t start = HAP_perf_get_qtimer_count();
    uint32_t spins = 0U;

    for (;;) {
        uint32_t status = Q6_R_dmpoll() & QBH_DMA_STATUS_MASK;
        if (((volatile struct qbh_dma_desc_1d *)descriptor)->dstate ==
            QBH_DMA_DESC_COMPLETE) {
            asm volatile("barrier" ::: "memory");
            return 0;
        }
        if (status == QBH_DMA_STATUS_ERROR) {
            return -1;
        }
        ++spins;
        if ((spins & UINT32_C(255)) == 0U &&
            HAP_perf_get_qtimer_count() - start >
                QBH_BLOCK_DMA_DESCRIPTOR_TIMEOUT_TICKS) {
            return -2;
        }
    }
}

static void qbh_fp16_input_norm_pool_run_tasks(
    struct qbh_block_w4f16_pool *pool,
    struct qbh_block_w4f16_job *job) {
    for (;;) {
        const uint32_t task = qbh_atomic_fetch_increment(
            &pool->next_fp16_input_norm_task);
        const uint32_t first_row =
            task * pool->fp16_norm_rows_per_task;
        uint32_t row_count;
        uint64_t start;

        if (task >= pool->fp16_input_norm_task_count ||
            first_row >= QBH_BLOCK_M) {
            break;
        }
        row_count = QBH_BLOCK_M - first_row;
        if (row_count > pool->fp16_norm_rows_per_task) {
            row_count = pool->fp16_norm_rows_per_task;
        }
        start = HAP_perf_get_qtimer_count();
        if (pool->fp16_input_norm_crouton != 0U) {
            qbh_hvx_rms_norm_f16_crouton_rows(
                pool->fp16_input_norm_input,
                pool->fp16_input_norm_gamma,
                pool->fp16_input_norm_output,
                first_row, row_count, QBH_BLOCK_HIDDEN);
        } else {
            qbh_hvx_rms_norm_f16_rows(
                pool->fp16_input_norm_input,
                pool->fp16_input_norm_gamma,
                pool->fp16_input_norm_output,
                first_row, row_count, QBH_BLOCK_HIDDEN);
        }
        job->fp16_input_norm_ticks +=
            HAP_perf_get_qtimer_count() - start;
        ++job->fp16_input_norm_task_count;
    }
}

static void qbh_fp16_post_residual_norm_pool_run_tasks(
    struct qbh_block_w4f16_pool *pool,
    struct qbh_block_w4f16_job *job) {
    for (;;) {
        const uint32_t task = qbh_atomic_fetch_increment(
            &pool->next_fp16_post_residual_norm_task);
        const uint32_t first_row =
            task * pool->fp16_norm_rows_per_task;
        uint32_t row_count;
        uint64_t start;

        if (task >= pool->fp16_post_residual_norm_task_count ||
            first_row >= QBH_BLOCK_M) {
            break;
        }
        row_count = QBH_BLOCK_M - first_row;
        if (row_count > pool->fp16_norm_rows_per_task) {
            row_count = pool->fp16_norm_rows_per_task;
        }
        start = HAP_perf_get_qtimer_count();
        if (pool->fp16_post_residual_norm_crouton != 0U) {
            qbh_hvx_residual_rms_norm_f16_crouton_rows(
                pool->fp16_post_residual,
                pool->fp16_post_addition,
                pool->fp16_post_gamma,
                pool->fp16_post_output,
                first_row, row_count, QBH_BLOCK_HIDDEN);
        } else {
            qbh_hvx_residual_rms_norm_f16_rows(
                pool->fp16_post_residual,
                pool->fp16_post_addition,
                pool->fp16_post_gamma,
                pool->fp16_post_output,
                first_row, row_count, QBH_BLOCK_HIDDEN);
        }
        job->fp16_post_residual_norm_ticks +=
            HAP_perf_get_qtimer_count() - start;
        ++job->fp16_post_residual_norm_task_count;
    }
}

static int qbh_dma_start_w4u8_batch_prefetch(
    struct qbh_dma_aligned_desc_1d descriptors[2],
    void *weight_destination, const void *weight_source,
    uint32_t weight_bytes, void *bias_destination,
    const void *bias_source, uint32_t bias_bytes) {
    struct qbh_dma_desc_1d *weight = &descriptors[0].descriptor;
    struct qbh_dma_desc_1d *bias = &descriptors[1].descriptor;

    if (weight_destination == NULL || weight_source == NULL ||
        bias_destination == NULL || bias_source == NULL ||
        weight_bytes == 0U || bias_bytes == 0U ||
        qbh_dma_wait_idle() != 0) {
        return -1;
    }
    memset(descriptors, 0, 2U * sizeof(*descriptors));
    weight->next = (uint32_t)(uintptr_t)bias;
    weight->length = weight_bytes;
    weight->type = QBH_DMA_TYPE_1D;
    weight->src_bypass = 1;
    weight->ordered = 1;
    weight->dstate = QBH_DMA_DESC_PENDING;
    weight->src = (uint32_t)(uintptr_t)weight_source;
    weight->dst = (uint32_t)(uintptr_t)weight_destination;

    bias->length = bias_bytes;
    bias->type = QBH_DMA_TYPE_1D;
    bias->src_bypass = 1;
    bias->ordered = 1;
    bias->dstate = QBH_DMA_DESC_PENDING;
    bias->src = (uint32_t)(uintptr_t)bias_source;
    bias->dst = (uint32_t)(uintptr_t)bias_destination;
    asm volatile("release(%0):at" : : "r"(bias) : "memory");
    asm volatile("release(%0):at" : : "r"(weight) : "memory");
    return qbh_dma_start(weight) == 0 ? 0 : -2;
}

static int qbh_dma_wait_w4u8_batch_prefetch(
    struct qbh_dma_aligned_desc_1d descriptors[2]) {
    int result = qbh_dma_wait_weight_prefetch(&descriptors[1]);
    if (result != 0) {
        return result;
    }
    return qbh_dma_wait_idle() == 0 ? 0 : -3;
}

static int qbh_w4f16_cross_prefetch_enabled(
    const struct qbh_block_header *header) {
    return header->w4f16_pipeline_mode ==
               QBH_BLOCK_W4F16_PIPELINE_CROSS_PREFETCH ||
           header->w4f16_pipeline_mode ==
               QBH_BLOCK_W4F16_PIPELINE_HYBRID_CROSS_PREFETCH ||
           header->w4f16_pipeline_mode ==
               QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN64_CROSS_PREFETCH ||
           header->w4f16_pipeline_mode ==
               QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN48_CROSS_PREFETCH ||
           header->w4f16_pipeline_mode ==
               QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_CROSS_PREFETCH ||
           header->w4f16_pipeline_mode ==
               QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_GATE16_CROSS_PREFETCH ||
           header->w4f16_pipeline_mode ==
               QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_GATE8_CROSS_PREFETCH ||
           header->w4f16_pipeline_mode ==
               QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_GATE4_CROSS_PREFETCH ||
           header->w4f16_pipeline_mode ==
               QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_GATE4_DMA8_CROSS_PREFETCH;
}

static uint32_t qbh_w4f16_dma_batch_tiles(
    const struct qbh_block_header *header,
    const struct qbh_block_projection_desc *desc) {
    if (header->w4f16_pipeline_mode ==
            QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_GATE4_DMA8_CROSS_PREFETCH &&
        desc != NULL &&
        (desc == &header->projections[QBH_BLOCK_PROJ_GATE] ||
         desc == &header->projections[QBH_BLOCK_PROJ_UP])) {
        return 8U;
    }
    return QBH_BLOCK_W4F16_DMA_BATCH_N_TILES;
}

static int qbh_w4f16_start_cross_prefetch(
    struct qbh_block_header *header, const uint8_t *shared,
    const struct qbh_block_projection_desc *next_desc,
    struct qbh_block_buffers *buffers,
    struct qbh_block_w4f16_cross_prefetch *state) {
    uint32_t k_tiles;
    uint32_t n_tiles;
    uint32_t first_batch_tiles;
    uint32_t first_batch_bytes;
    uint32_t dma_batch_tiles;
    int result;

    if (!qbh_w4f16_cross_prefetch_enabled(header) ||
        next_desc == NULL) {
        return 0;
    }
    if (state == NULL || state->active != 0U) {
        return -1;
    }
    k_tiles = next_desc->k / QBH_HMX_FP16_COLS;
    n_tiles = next_desc->n / QBH_HMX_FP16_COLS;
    dma_batch_tiles = qbh_w4f16_dma_batch_tiles(header, next_desc);
    first_batch_tiles =
        n_tiles < dma_batch_tiles ? n_tiles : dma_batch_tiles;
    first_batch_bytes = first_batch_tiles * k_tiles *
                        QBH_W4_PACKED_TILE_BYTES;
    state->start_ticks = HAP_perf_get_qtimer_count();
    result = qbh_dma_start_weight_prefetch(
        &state->descriptor, buffers->compressed_weight,
        shared + next_desc->weight_offset, first_batch_bytes);
    if (result != 0) {
        state->start_ticks = 0U;
        return result;
    }
    state->target = next_desc;
    state->active = 1U;
    ++header->w4f16_cross_prefetch_count;
    header->weight_ddr_read_bytes += first_batch_bytes;
    ++header->weight_dma_descriptor_count;
    return 0;
}

static int qbh_w4f16_consume_cross_prefetch(
    struct qbh_block_header *header,
    const struct qbh_block_projection_desc *desc,
    struct qbh_block_w4f16_cross_prefetch *state) {
    uint64_t wait_start;
    uint64_t end;
    int result;

    if (state == NULL || state->active == 0U) {
        return 0;
    }
    if (state->target != desc) {
        return -1;
    }
    wait_start = HAP_perf_get_qtimer_count();
    result = qbh_dma_wait_weight_prefetch(&state->descriptor);
    end = HAP_perf_get_qtimer_count();
    header->w4f16_cross_prefetch_wait_ticks += end - wait_start;
    header->w4f16_cross_prefetch_lifetime_ticks +=
        end - state->start_ticks;
    state->active = 0U;
    state->target = NULL;
    state->start_ticks = 0U;
    return result == 0 ? 1 : -2;
}

static void qbh_w4f16_drain_cross_prefetch(
    struct qbh_block_header *header,
    struct qbh_block_w4f16_cross_prefetch *state) {
    uint64_t wait_start;
    uint64_t end;

    if (state == NULL || state->active == 0U) {
        return;
    }
    wait_start = HAP_perf_get_qtimer_count();
    (void)qbh_dma_wait_weight_prefetch(&state->descriptor);
    end = HAP_perf_get_qtimer_count();
    header->w4f16_cross_prefetch_wait_ticks += end - wait_start;
    header->w4f16_cross_prefetch_lifetime_ticks +=
        end - state->start_ticks;
    state->active = 0U;
    state->target = NULL;
    state->start_ticks = 0U;
}

static int qbh_hmx_run_w4u8_qkv_ring(
    struct qbh_w4u8_qkv_ring_state *state) {
    if (state == NULL || state->header == NULL || state->pool == NULL ||
        state->activation == NULL ||
        state->batch_count != QBH_BLOCK_W4U8_QKV_RING_BATCHES ||
        state->k_tiles != QBH_BLOCK_HIDDEN / QBH_HMX_INPUT_CHANNELS) {
        return -1;
    }
    for (uint32_t batch_index = 0U;
         batch_index < state->batch_count; ++batch_index) {
        const uint32_t slot =
            batch_index % QBH_BLOCK_W4U8_QKV_RING_SLOTS;
        const struct qbh_w4u8_qkv_ring_batch *batch =
            &state->batches[batch_index];
        uint64_t wait_start = HAP_perf_get_qtimer_count();
        uint64_t core_start;

        qurt_sem_down(&state->expanded_ready[slot]);
        state->hmx_ready_wait_ticks +=
            HAP_perf_get_qtimer_count() - wait_start;
        asm volatile("barrier" ::: "memory");
        if (state->abort_status != 0U || batch->desc == NULL ||
            batch->output == NULL ||
            batch->n_tiles !=
                QBH_BLOCK_W4U8_QKV_RING_TILES_PER_BATCH) {
            qurt_sem_up(&state->expanded_free[slot]);
            return -1;
        }

        core_start = HAP_perf_get_qtimer_count();
        for (uint32_t tile = 0U; tile < batch->n_tiles; ++tile) {
            qbh_hmx_begin_u8s8_output(
                (const uint32_t *)(state->bias_slots[slot] +
                    (size_t)tile * QBH_HMX_BIAS_BYTES));
            (void)qbh_hmx_accumulate_u8s8_projection(
                state->activation,
                (const int8_t *)(state->expanded_slots[slot] +
                    (size_t)tile * state->k_tiles *
                        QBH_HMX_WEIGHT_BYTES),
                state->k_tiles);
            qbh_hmx_store_u8_output(
                batch->output +
                    (size_t)(batch->first_n_tile + tile) *
                        QBH_HMX_OUTPUT_BYTES);
        }
        state->hmx_compute_ticks +=
            HAP_perf_get_qtimer_count() - core_start;
        ++state->hmx_batch_count;
        state->header->hmx_u8s8_tile_pair_count +=
            state->k_tiles * batch->n_tiles;
        ++state->header->hmx_command_count;
        ++state->header->w4u8_qkv_batch_count;
        state->header->u8_attention_qkv_unpack_skipped +=
            batch->n_tiles;

        if (batch->desc ==
                &state->header->projections[QBH_BLOCK_PROJ_Q] ||
            batch->desc ==
                &state->header->projections[QBH_BLOCK_PROJ_K]) {
            qbh_hvx_pool_u8_qk_prep_publish(
                state->header, batch->desc, state->pool,
                batch->first_n_tile, batch->n_tiles);
            ++state->head_publish_count;
        }
        qurt_sem_up(&state->expanded_free[slot]);
    }
    return 0;
}

static void qbh_hmx_worker_main(void *opaque) {
    struct qbh_block_hmx_worker *worker =
        (struct qbh_block_hmx_worker *)opaque;
    worker->lock_status = HAP_compute_res_hmx_lock2(
        worker->hmx_context_id, HAP_COMPUTE_RES_HMX_SHARED);
    (void)qurt_sem_up(&worker->worker_started);
    if (worker->lock_status != AEE_SUCCESS) {
        qurt_thread_exit(worker->lock_status);
    }
    for (;;) {
        uint64_t start;
        qurt_sem_down(&worker->command_ready);
        if (worker->stop != 0U) {
            break;
        }
        start = HAP_perf_get_qtimer_count();
        worker->command_status = AEE_SUCCESS;
        if (worker->kind == QBH_BLOCK_HMX_FP16) {
            qbh_hmx_fp16_matmul_tiles(
                (const __fp16 *)worker->activation,
                (const __fp16 *)worker->weight,
                worker->scale_or_bias, (__fp16 *)worker->output,
                worker->m_tiles, worker->k_tiles, worker->n_tiles);
        } else if (worker->kind == QBH_BLOCK_HMX_FP16_STREAMING &&
                   worker->n_tiles == 1U) {
            worker->command_status = qbh_hmx_fp16_matmul_streaming(
                (const __fp16 *)worker->activation,
                (const __fp16 *)worker->weight,
                worker->scale_or_bias, (__fp16 *)worker->output,
                worker->m_tiles, worker->k_tiles,
                worker->region_tiles, worker->ready_generations,
                worker->expected_generation,
                &worker->ready_wait_ticks) == 0
                ? AEE_SUCCESS : AEE_EFAILED;
        } else if (worker->kind == QBH_BLOCK_HMX_FP16_TILE_SCALES) {
            qbh_hmx_fp16_matmul_tile_scales(
                (const __fp16 *)worker->activation,
                (const __fp16 *)worker->weight,
                worker->scale_or_bias, (__fp16 *)worker->output,
                worker->m_tiles, worker->k_tiles, worker->n_tiles);
        } else if (worker->kind ==
                       QBH_BLOCK_HMX_FP16_TILE_SCALES_STREAMING &&
                   worker->m_tiles != 0U && worker->n_tiles != 0U &&
                   worker->m_tiles <= 8U / worker->n_tiles) {
            worker->command_status =
                qbh_hmx_fp16_matmul_tile_scales_streaming(
                    (const __fp16 *)worker->activation,
                    (const __fp16 *)worker->weight,
                    worker->scale_or_bias, (__fp16 *)worker->output,
                    worker->m_tiles, worker->k_tiles,
                    worker->n_tiles, worker->region_tiles,
                    worker->ready_generations,
                    worker->expected_generation,
                    &worker->ready_wait_ticks) == 0
                    ? AEE_SUCCESS : AEE_EFAILED;
        } else if (worker->kind == QBH_BLOCK_HMX_U8S8 &&
                   worker->m_tiles >= 1U &&
                   worker->m_tiles <= QBH_ATTENTION_Q_HEADS_PER_GROUP &&
                   worker->n_tiles >= 1U &&
                   worker->n_tiles <=
                       QBH_BLOCK_SCAN_MAX_KV_TILES) {
            for (uint32_t m_tile = 0U;
                 m_tile < worker->m_tiles; ++m_tile) {
                const uint8_t *activation =
                    (const uint8_t *)worker->activation +
                    (size_t)m_tile * worker->k_tiles *
                        QBH_HMX_ACTIVATION_BYTES;
                uint8_t *output = (uint8_t *)worker->output +
                    (size_t)m_tile * worker->n_tiles *
                        QBH_HMX_OUTPUT_BYTES;
                for (uint32_t output_tile = 0U;
                     output_tile < worker->n_tiles; ++output_tile) {
                    qbh_hmx_begin_u8s8_output(
                        (const uint32_t *)worker->scale_or_bias +
                        (size_t)output_tile *
                            (QBH_HMX_BIAS_BYTES / sizeof(uint32_t)));
                    (void)qbh_hmx_accumulate_u8s8_projection(
                        activation,
                        (const int8_t *)worker->weight +
                            (size_t)output_tile * worker->k_tiles *
                                QBH_HMX_WEIGHT_BYTES,
                        worker->k_tiles);
                    qbh_hmx_store_u8_output(
                        output + (size_t)output_tile *
                            QBH_HMX_OUTPUT_BYTES);
                }
            }
        } else if (worker->kind == QBH_BLOCK_HMX_U8S8_QKV_RING &&
                   worker->qkv_ring_request != NULL) {
            worker->command_status = qbh_hmx_run_w4u8_qkv_ring(
                (struct qbh_w4u8_qkv_ring_state *)
                    worker->qkv_ring_request) == 0
                ? AEE_SUCCESS : AEE_EFAILED;
        } else if (worker->kind == QBH_BLOCK_HMX_U8S8_PIPELINE &&
                   worker->w4_pipeline_request != NULL) {
            const struct qbh_w4_hmx_request *request =
                worker->w4_pipeline_request;
            uint32_t streams = 0U;
            if (request->streaming != 0U) {
                uint32_t batch_output_count =
                    request->batch_output_count != 0U
                        ? request->batch_output_count : 1U;
                uint64_t ready_wait_before =
                    request->ready_wait_ticks != NULL
                        ? *request->ready_wait_ticks : 0U;
                if (request->activation_tiles == NULL ||
                    request->abort_status == NULL ||
                    request->ready_wait_ticks == NULL ||
                    request->hmx_consumption_started == NULL ||
                    batch_output_count >
                        QBH_W4_HMX_MAX_BATCH_OUTPUTS) {
                    worker->command_status = AEE_EBADPARM;
                } else {
                    for (uint32_t batch_index = 0U;
                         batch_index < batch_output_count; ++batch_index) {
                        const int8_t *expanded_weight_tiles =
                            request->batch_output_count != 0U
                                ? request->batch_outputs[batch_index]
                                      .expanded_weight_tiles
                                : request->expanded_weight_tiles;
                        const uint32_t *bias_words =
                            request->batch_output_count != 0U
                                ? request->batch_outputs[batch_index]
                                      .bias_words
                                : request->bias_words;
                        uint8_t *output_tiles =
                            request->batch_output_count != 0U
                                ? request->batch_outputs[batch_index]
                                      .output_tiles
                                : request->output_tiles;
                        const volatile uint32_t *ready_generations =
                            request->batch_output_count != 0U
                                ? request->batch_outputs[batch_index]
                                      .ready_generations
                                : request->ready_generations;
                        uint32_t expected_generation =
                            request->batch_output_count != 0U
                                ? request->batch_outputs[batch_index]
                                      .expected_generation
                                : request->expected_generation;
                        int32_t stream_result;

                        if (expanded_weight_tiles == NULL ||
                            ready_generations == NULL ||
                            (request->begin_output != 0U &&
                             bias_words == NULL) ||
                            (request->store_output != 0U &&
                             output_tiles == NULL)) {
                            worker->command_status = AEE_EBADPARM;
                            break;
                        }
                        stream_result = qbh_hmx_accumulate_u8s8_streaming(
                            request->activation_tiles,
                            expanded_weight_tiles, bias_words,
                            request->begin_output, ready_generations,
                            expected_generation, request->stream_count,
                            request->abort_status, request->timeout_ticks,
                            request->ready_wait_ticks,
                            request->hmx_consumption_started);
                        if (stream_result < 0) {
                            worker->command_status = AEE_EFAILED;
                            break;
                        }
                        streams += (uint32_t)stream_result;
                        if (request->store_output != 0U) {
                            qbh_hmx_store_u8_output(output_tiles);
                        }
                    }
                    worker->ready_wait_ticks +=
                        *request->ready_wait_ticks - ready_wait_before;
                }
            } else {
                uint64_t continuation_ready_wait_before =
                    request->ready_wait_ticks != NULL
                        ? *request->ready_wait_ticks : 0U;
                if (request->batch_output_count != 0U) {
                    const uint8_t *continuation_activation =
                        request->continuation_chunks[0]
                            .activation_tiles;
                    uint32_t continuation_chunk_tiles =
                        request->continuation_chunks[0].chunk_tiles;

                    if (request->batch_output_count >
                            QBH_W4_HMX_MAX_BATCH_OUTPUTS ||
                        request->activation_tiles == NULL ||
                        request->chunk_tiles == 0U ||
                        request->continuation_chunk_count != 1U ||
                        continuation_activation == NULL ||
                        continuation_chunk_tiles == 0U ||
                        request->ready_wait_ticks == NULL ||
                        request->abort_status == NULL ||
                        request->in_command_slot_release_count == NULL) {
                        worker->command_status = AEE_EBADPARM;
                    }
                    for (uint32_t batch_index = 0U;
                         worker->command_status == AEE_SUCCESS &&
                         batch_index < request->batch_output_count;
                         ++batch_index) {
                        const int8_t *first_weight =
                            request->batch_outputs[batch_index]
                                .expanded_weight_tiles;
                        const uint32_t *bias =
                            request->batch_outputs[batch_index]
                                .bias_words;
                        uint8_t *output =
                            request->batch_outputs[batch_index]
                                .output_tiles;
                        qurt_sem_t *first_ready = (qurt_sem_t *)
                            request->batch_outputs[batch_index]
                                .ready_semaphore;
                        qurt_sem_t *first_free = (qurt_sem_t *)
                            request->batch_outputs[batch_index]
                                .free_semaphore;
                        const int8_t *continuation_weight =
                            request->batch_outputs[batch_index]
                                .continuation_expanded_weight_tiles;
                        qurt_sem_t *continuation_ready = (qurt_sem_t *)
                            request->batch_outputs[batch_index]
                                .continuation_ready_semaphore;
                        qurt_sem_t *continuation_free = (qurt_sem_t *)
                            request->batch_outputs[batch_index]
                                .continuation_free_semaphore;

                        if (first_weight == NULL || bias == NULL ||
                            output == NULL || first_free == NULL ||
                            continuation_weight == NULL ||
                            continuation_ready == NULL ||
                            continuation_free == NULL) {
                            worker->command_status = AEE_EBADPARM;
                            break;
                        }
                        if (first_ready != NULL) {
                            uint64_t wait_start =
                                HAP_perf_get_qtimer_count();
                            qurt_sem_down(first_ready);
                            *request->ready_wait_ticks +=
                                HAP_perf_get_qtimer_count() - wait_start;
                        }
                        if (*request->abort_status != 0) {
                            worker->command_status = AEE_EFAILED;
                            break;
                        }
                        qbh_hmx_begin_u8s8_output(bias);
                        streams += qbh_hmx_accumulate_u8s8_projection(
                            request->activation_tiles, first_weight,
                            request->chunk_tiles);
                        {
                            uint64_t wait_start =
                                HAP_perf_get_qtimer_count();
                            qurt_sem_down(continuation_ready);
                            *request->ready_wait_ticks +=
                                HAP_perf_get_qtimer_count() - wait_start;
                        }
                        if (*request->abort_status != 0) {
                            worker->command_status = AEE_EFAILED;
                            break;
                        }
                        streams += qbh_hmx_accumulate_u8s8_projection(
                            continuation_activation,
                            continuation_weight,
                            continuation_chunk_tiles);
                        qbh_hmx_store_u8_output(output);
                        qurt_sem_up(first_free);
                        qurt_sem_up(continuation_free);
                        *request->in_command_slot_release_count += 2U;
                    }
                    worker->ready_wait_ticks +=
                        *request->ready_wait_ticks -
                        continuation_ready_wait_before;
                } else {
                if (request->begin_output != 0U) {
                    if (request->bias_words == NULL) {
                        worker->command_status = AEE_EBADPARM;
                    } else {
                        qbh_hmx_begin_u8s8_output(request->bias_words);
                    }
                }
                if (worker->command_status == AEE_SUCCESS &&
                    request->chunk_tiles != 0U) {
                    if (request->activation_tiles == NULL ||
                        request->expanded_weight_tiles == NULL) {
                        worker->command_status = AEE_EBADPARM;
                    } else {
                        streams = qbh_hmx_accumulate_u8s8_projection(
                            request->activation_tiles,
                            request->expanded_weight_tiles,
                            request->chunk_tiles);
                    }
                }
                if (worker->command_status == AEE_SUCCESS &&
                    request->continuation_chunk_count >
                        QBH_W4_HMX_MAX_CONTINUATION_CHUNKS) {
                    worker->command_status = AEE_EBADPARM;
                }
                for (uint32_t chunk_index = 0U;
                     worker->command_status == AEE_SUCCESS &&
                     chunk_index < request->continuation_chunk_count;
                     ++chunk_index) {
                    const uint8_t *activation_tiles =
                        request->continuation_chunks[chunk_index]
                            .activation_tiles;
                    const int8_t *expanded_weight_tiles =
                        request->continuation_chunks[chunk_index]
                            .expanded_weight_tiles;
                    uint32_t chunk_tiles =
                        request->continuation_chunks[chunk_index]
                            .chunk_tiles;
                    qurt_sem_t *ready_semaphore = (qurt_sem_t *)
                        request->continuation_chunks[chunk_index]
                            .ready_semaphore;

                    if (activation_tiles == NULL ||
                        expanded_weight_tiles == NULL ||
                        chunk_tiles == 0U || ready_semaphore == NULL ||
                        request->ready_wait_ticks == NULL ||
                        request->abort_status == NULL) {
                        worker->command_status = AEE_EBADPARM;
                        break;
                    }
                    {
                        uint64_t wait_start =
                            HAP_perf_get_qtimer_count();
                        qurt_sem_down(ready_semaphore);
                        *request->ready_wait_ticks +=
                            HAP_perf_get_qtimer_count() - wait_start;
                    }
                    if (*request->abort_status != 0) {
                        worker->command_status = AEE_EFAILED;
                        break;
                    }
                    streams += qbh_hmx_accumulate_u8s8_projection(
                        activation_tiles, expanded_weight_tiles,
                        chunk_tiles);
                }
                if (request->ready_wait_ticks != NULL) {
                    worker->ready_wait_ticks +=
                        *request->ready_wait_ticks -
                        continuation_ready_wait_before;
                }
                }
            }
            if (worker->command_status == AEE_SUCCESS &&
                request->streaming == 0U &&
                request->batch_output_count == 0U &&
                request->store_output != 0U) {
                if (request->output_tiles == NULL) {
                    worker->command_status = AEE_EBADPARM;
                } else {
                    qbh_hmx_store_u8_output(request->output_tiles);
                }
            }
            if (request->executed_stream_count != NULL) {
                *request->executed_stream_count = streams;
            }
        } else {
            worker->command_status = AEE_EBADPARM;
        }
        worker->compute_ticks += HAP_perf_get_qtimer_count() - start;
        (void)qurt_sem_up(&worker->command_done);
    }
    worker->unlock_status = HAP_compute_res_hmx_unlock2(
        worker->hmx_context_id, HAP_COMPUTE_RES_HMX_SHARED);
    qurt_thread_exit(worker->unlock_status);
}

static void qbh_hmx_start(struct qbh_block_hmx_worker *worker,
                          uint32_t kind, const void *activation,
                          const void *weight, const void *scale_or_bias,
                          void *output, uint32_t m_tiles,
                          uint32_t k_tiles, uint32_t n_tiles) {
    worker->kind = kind;
    worker->activation = activation;
    worker->weight = weight;
    worker->scale_or_bias = scale_or_bias;
    worker->output = output;
    worker->m_tiles = m_tiles;
    worker->k_tiles = k_tiles;
    worker->n_tiles = n_tiles;
    worker->command_status = AEE_EFAILED;
    asm volatile("barrier" ::: "memory");
    (void)qurt_sem_up(&worker->command_ready);
}

static int qbh_hmx_submit(struct qbh_block_hmx_worker *worker,
                          uint32_t kind, const void *activation,
                          const void *weight, const void *scale_or_bias,
                          void *output, uint32_t m_tiles,
                          uint32_t k_tiles, uint32_t n_tiles) {
    qbh_hmx_start(worker, kind, activation, weight, scale_or_bias,
                  output, m_tiles, k_tiles, n_tiles);
    qurt_sem_down(&worker->command_done);
    asm volatile("barrier" ::: "memory");
    return worker->command_status == AEE_SUCCESS ? 0 : -1;
}

static int qbh_block_w4_hmx_submit(
    void *context, const struct qbh_w4_hmx_request *request) {
    struct qbh_block_hmx_worker *worker =
        (struct qbh_block_hmx_worker *)context;
    if (worker == NULL || request == NULL) {
        return -1;
    }
    worker->kind = QBH_BLOCK_HMX_U8S8_PIPELINE;
    worker->w4_pipeline_request = request;
    worker->command_status = AEE_EFAILED;
    asm volatile("barrier" ::: "memory");
    (void)qurt_sem_up(&worker->command_ready);
    qurt_sem_down(&worker->command_done);
    asm volatile("barrier" ::: "memory");
    worker->w4_pipeline_request = NULL;
    return worker->command_status == AEE_SUCCESS ? 0 : -1;
}

static void qbh_hmx_start_fp16_tile_scales(
    struct qbh_block_hmx_worker *worker, const void *activation,
    const void *weight, const void *scale_blocks, void *output,
    uint32_t m_tiles, uint32_t k_tiles, uint32_t n_tiles) {
    worker->kind = QBH_BLOCK_HMX_FP16_TILE_SCALES;
    worker->activation = activation;
    worker->weight = weight;
    worker->scale_or_bias = scale_blocks;
    worker->output = output;
    worker->m_tiles = m_tiles;
    worker->k_tiles = k_tiles;
    worker->n_tiles = n_tiles;
    worker->command_status = AEE_EFAILED;
    asm volatile("barrier" ::: "memory");
    (void)qurt_sem_up(&worker->command_ready);
}

static void qbh_hmx_start_fp16_streaming(
    struct qbh_block_hmx_worker *worker, const void *activation,
    const void *weight, const void *scale, void *output,
    uint32_t m_tiles, uint32_t k_tiles, uint32_t region_tiles,
    const volatile uint32_t *ready_generations,
    uint32_t expected_generation) {
    worker->kind = QBH_BLOCK_HMX_FP16_STREAMING;
    worker->activation = activation;
    worker->weight = weight;
    worker->scale_or_bias = scale;
    worker->output = output;
    worker->m_tiles = m_tiles;
    worker->k_tiles = k_tiles;
    worker->n_tiles = 1U;
    worker->region_tiles = region_tiles;
    worker->ready_generations = ready_generations;
    worker->expected_generation = expected_generation;
    worker->command_status = AEE_EFAILED;
    asm volatile("barrier" ::: "memory");
    (void)qurt_sem_up(&worker->command_ready);
}

static void qbh_hmx_start_fp16_tile_scales_streaming(
    struct qbh_block_hmx_worker *worker, const void *activation,
    const void *weight, const void *scale_blocks, void *output,
    uint32_t m_tiles, uint32_t k_tiles, uint32_t n_tiles,
    uint32_t region_tiles,
    const volatile uint32_t *ready_generations,
    uint32_t expected_generation) {
    worker->kind = QBH_BLOCK_HMX_FP16_TILE_SCALES_STREAMING;
    worker->activation = activation;
    worker->weight = weight;
    worker->scale_or_bias = scale_blocks;
    worker->output = output;
    worker->m_tiles = m_tiles;
    worker->k_tiles = k_tiles;
    worker->n_tiles = n_tiles;
    worker->region_tiles = region_tiles;
    worker->ready_generations = ready_generations;
    worker->expected_generation = expected_generation;
    worker->command_status = AEE_EFAILED;
    asm volatile("barrier" ::: "memory");
    (void)qurt_sem_up(&worker->command_ready);
}

static int qbh_hmx_wait(struct qbh_block_hmx_worker *worker) {
    qurt_sem_down(&worker->command_done);
    asm volatile("barrier" ::: "memory");
    return worker->command_status == AEE_SUCCESS ? 0 : -1;
}

static void qbh_silu_pool_run_chunks(
    struct qbh_block_w4f16_pool *pool, uint64_t *work_ticks,
    uint32_t *completed_chunks) {
    for (;;) {
        uint32_t chunk =
            qbh_atomic_fetch_increment(&pool->next_silu_chunk);
        uint32_t first_vector;
        uint32_t vector_count;
        uint64_t start;
        if (chunk >= pool->silu_chunk_count) {
            break;
        }
        first_vector = chunk * pool->silu_chunk_vectors;
        vector_count = pool->silu_vector_count - first_vector;
        if (vector_count > pool->silu_chunk_vectors) {
            vector_count = pool->silu_chunk_vectors;
        }
        start = HAP_perf_get_qtimer_count();
        qbh_hvx_silu_multiply_f16_vectors(
            pool->silu_gate, pool->silu_up, pool->silu_middle,
            first_vector, vector_count);
        *work_ticks += HAP_perf_get_qtimer_count() - start;
        ++*completed_chunks;
    }
}

static void qbh_pack_fp16_activation_channel64(
    const __fp16 *source, uint32_t source_stride,
    uint32_t first_channel, __fp16 *destination) {
    const uint32_t k_tiles =
        QBH_BLOCK_INTERMEDIATE / QBH_HMX_FP16_COLS;

    for (uint32_t row = 0U; row < QBH_BLOCK_M; row += 2U) {
        uint32_t row_tile = row / QBH_HMX_FP16_ROWS;
        uint32_t row_pair = (row % QBH_HMX_FP16_ROWS) / 2U;
        const HVX_Vector source0 = *(const HVX_Vector *)(
            source + (size_t)row * source_stride + first_channel);
        const HVX_Vector source1 = *(const HVX_Vector *)(
            source + (size_t)(row + 1U) * source_stride +
            first_channel);
        HVX_VectorPair packed = Q6_W_vshuff_VVR(
            source1, source0, -2);
        size_t tile = qbh_hmx_fp16_matrix_tile_offset(
            row_tile, first_channel / QBH_HMX_FP16_COLS,
            k_tiles);
        HVX_Vector *output0 =
            (HVX_Vector *)(destination + tile) + row_pair;
        HVX_Vector *output1 = output0 +
            QBH_HMX_FP16_TILE_BYTES / sizeof(HVX_Vector);
        *output0 = Q6_V_lo_W(packed);
        *output1 = Q6_V_hi_W(packed);
    }
    asm volatile("barrier" ::: "memory");
}

static void qbh_mlp_stream_worker_run(
    struct qbh_block_w4f16_pool *pool,
    struct qbh_block_w4f16_job *job) {
    for (;;) {
        uint32_t group =
            qbh_atomic_fetch_increment(&pool->next_mlp_group);
        uint64_t wait_start;
        uint64_t work_start;

        if (group >= pool->mlp_stream_group_limit) {
            break;
        }
        wait_start = HAP_perf_get_qtimer_count();
        while (pool->mlp_up_group_ready[group] !=
               pool->mlp_stream_generation) {
            if (pool->mlp_stream_abort != 0U) {
                job->stream_ready_wait_ticks +=
                    HAP_perf_get_qtimer_count() - wait_start;
                return;
            }
            asm volatile("pause(#8)" : : : "memory");
        }
        asm volatile("barrier" ::: "memory");
        job->stream_ready_wait_ticks +=
            HAP_perf_get_qtimer_count() - wait_start;

        work_start = HAP_perf_get_qtimer_count();
        if (pool->mlp_crouton_native != 0U) {
            const uint32_t subgroups_per_hmx =
                pool->mlp_crouton_group_tiles /
                pool->mlp_crouton_stream_group_tiles;
            const uint32_t hmx_group = group / subgroups_per_hmx;
            const uint32_t first_source_tile =
                (group % subgroups_per_hmx) *
                pool->mlp_crouton_stream_group_tiles;
            const uint32_t slot =
                hmx_group % QBH_BLOCK_MLP_CROUTON_RING_SLOTS;
            const __fp16 *gate_tiles = pool->mlp_gate +
                (size_t)slot * pool->mlp_crouton_slot_elements;
            const __fp16 *up_tiles = pool->mlp_up +
                (size_t)slot * pool->mlp_crouton_slot_elements;

            qbh_hvx_silu_multiply_f16_crouton_tile_range(
                gate_tiles, up_tiles, pool->mlp_hmx_activation,
                QBH_BLOCK_M / QBH_HMX_FP16_ROWS,
                pool->mlp_crouton_group_tiles,
                first_source_tile,
                pool->mlp_crouton_stream_group_tiles,
                QBH_BLOCK_INTERMEDIATE / QBH_HMX_FP16_COLS,
                group * pool->mlp_crouton_stream_group_tiles);
            if (qbh_atomic_fetch_increment(
                    &pool->mlp_crouton_subgroup_done[hmx_group]) + 1U ==
                subgroups_per_hmx) {
                pool->mlp_crouton_slot_consumed[slot] = hmx_group + 1U;
                asm volatile("release(%0):at"
                             :
                             : "r"(&pool->mlp_crouton_slot_consumed[slot])
                             : "memory");
            }
            job->stream_group_count +=
                pool->mlp_crouton_stream_group_tiles /
                (QBH_BLOCK_MLP_STREAM_CHANNELS /
                 QBH_HMX_FP16_COLS);
        } else {
            const uint32_t first_channel =
                group * QBH_BLOCK_MLP_STREAM_CHANNELS;
            qbh_hvx_silu_multiply_f16_channel64(
                pool->mlp_gate, pool->mlp_up, pool->mlp_middle,
                QBH_BLOCK_M, QBH_BLOCK_INTERMEDIATE, first_channel);
            qbh_pack_fp16_activation_channel64(
                pool->mlp_middle, QBH_BLOCK_INTERMEDIATE,
                first_channel, pool->mlp_hmx_activation);
            ++job->stream_group_count;
        }
        job->stream_ticks +=
            HAP_perf_get_qtimer_count() - work_start;
    }
}

static int qbh_attention_parallel_qk_norm_enabled(uint32_t mode) {
    return mode == QBH_BLOCK_ATTENTION_PIPELINE_PARALLEL_QK_NORM_ROPE ||
           mode == QBH_BLOCK_ATTENTION_PIPELINE_PARALLEL_HVX;
}

static int qbh_attention_parallel_softmax_enabled(uint32_t mode) {
    return mode == QBH_BLOCK_ATTENTION_PIPELINE_PARALLEL_SOFTMAX ||
           mode == QBH_BLOCK_ATTENTION_PIPELINE_PARALLEL_HVX;
}

static int qbh_attention_gqa_enabled(uint32_t mode) {
    return mode == QBH_BLOCK_ATTENTION_PIPELINE_GQA ||
           mode == QBH_BLOCK_ATTENTION_PIPELINE_GQA_QKV_OVERLAP;
}

static uint32_t qbh_attention_u8_base_mode(uint32_t mode) {
    return mode ==
                   QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER_VDEAL_FUSED_QK_REQUANT_HMX_BATCH_LUT_TEMPLATES_GQA_BATCH_DEPENDENCY_STREAM_SOFTMAX_SHUFFLE4
               ? QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER_VDEAL_FUSED_QK_REQUANT_HMX_BATCH_LUT_TEMPLATES_GQA_BATCH_DEPENDENCY_STREAM
               : mode;
}

static int qbh_attention_u8_enabled(uint32_t mode) {
    mode = qbh_attention_u8_base_mode(mode);
    return mode == QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA ||
           mode ==
               QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_FUSED_K ||
           mode ==
               QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP ||
           mode ==
               QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER ||
           mode ==
               QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER_VDEAL ||
           mode ==
               QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER_VDEAL_FUSED_QK_REQUANT ||
           mode ==
               QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER_VDEAL_FUSED_QK_REQUANT_HMX_BATCH ||
           mode ==
               QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER_VDEAL_FUSED_QK_REQUANT_HMX_BATCH_LUT_TEMPLATES ||
           mode ==
               QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER_VDEAL_FUSED_QK_REQUANT_HMX_BATCH_LUT_TEMPLATES_GQA_BATCH ||
           mode ==
               QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER_VDEAL_FUSED_QK_REQUANT_HMX_BATCH_LUT_TEMPLATES_GQA_BATCH_DEPENDENCY_STREAM;
}

static int qbh_attention_u8_fused_k_enabled(uint32_t mode) {
    mode = qbh_attention_u8_base_mode(mode);
    return mode ==
               QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_FUSED_K ||
           mode ==
               QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP ||
           mode ==
               QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER ||
           mode ==
               QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER_VDEAL ||
           mode ==
               QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER_VDEAL_FUSED_QK_REQUANT ||
           mode ==
               QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER_VDEAL_FUSED_QK_REQUANT_HMX_BATCH ||
           mode ==
               QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER_VDEAL_FUSED_QK_REQUANT_HMX_BATCH_LUT_TEMPLATES ||
           mode ==
               QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER_VDEAL_FUSED_QK_REQUANT_HMX_BATCH_LUT_TEMPLATES_GQA_BATCH ||
           mode ==
               QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER_VDEAL_FUSED_QK_REQUANT_HMX_BATCH_LUT_TEMPLATES_GQA_BATCH_DEPENDENCY_STREAM;
}

static int qbh_attention_u8_qkv_overlap_enabled(uint32_t mode) {
    mode = qbh_attention_u8_base_mode(mode);
    return mode ==
               QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP ||
           mode ==
               QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER ||
           mode ==
               QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER_VDEAL ||
           mode ==
               QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER_VDEAL_FUSED_QK_REQUANT ||
           mode ==
               QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER_VDEAL_FUSED_QK_REQUANT_HMX_BATCH ||
           mode ==
               QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER_VDEAL_FUSED_QK_REQUANT_HMX_BATCH_LUT_TEMPLATES ||
           mode ==
               QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER_VDEAL_FUSED_QK_REQUANT_HMX_BATCH_LUT_TEMPLATES_GQA_BATCH ||
           mode ==
               QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER_VDEAL_FUSED_QK_REQUANT_HMX_BATCH_LUT_TEMPLATES_GQA_BATCH_DEPENDENCY_STREAM;
}

static int qbh_attention_u8_vgather_enabled(uint32_t mode) {
    mode = qbh_attention_u8_base_mode(mode);
    return mode ==
               QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER ||
           mode ==
               QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER_VDEAL ||
           mode ==
               QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER_VDEAL_FUSED_QK_REQUANT ||
           mode ==
               QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER_VDEAL_FUSED_QK_REQUANT_HMX_BATCH ||
           mode ==
               QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER_VDEAL_FUSED_QK_REQUANT_HMX_BATCH_LUT_TEMPLATES ||
           mode ==
               QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER_VDEAL_FUSED_QK_REQUANT_HMX_BATCH_LUT_TEMPLATES_GQA_BATCH ||
           mode ==
               QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER_VDEAL_FUSED_QK_REQUANT_HMX_BATCH_LUT_TEMPLATES_GQA_BATCH_DEPENDENCY_STREAM;
}

static int qbh_attention_u8_vdeal_enabled(uint32_t mode) {
    mode = qbh_attention_u8_base_mode(mode);
    return mode ==
               QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER_VDEAL ||
           mode ==
               QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER_VDEAL_FUSED_QK_REQUANT ||
           mode ==
               QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER_VDEAL_FUSED_QK_REQUANT_HMX_BATCH ||
           mode ==
               QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER_VDEAL_FUSED_QK_REQUANT_HMX_BATCH_LUT_TEMPLATES ||
           mode ==
               QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER_VDEAL_FUSED_QK_REQUANT_HMX_BATCH_LUT_TEMPLATES_GQA_BATCH ||
           mode ==
               QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER_VDEAL_FUSED_QK_REQUANT_HMX_BATCH_LUT_TEMPLATES_GQA_BATCH_DEPENDENCY_STREAM;
}

static int qbh_attention_u8_fused_qk_requant_enabled(uint32_t mode) {
    mode = qbh_attention_u8_base_mode(mode);
    return mode ==
               QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER_VDEAL_FUSED_QK_REQUANT ||
           mode ==
               QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER_VDEAL_FUSED_QK_REQUANT_HMX_BATCH ||
           mode ==
               QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER_VDEAL_FUSED_QK_REQUANT_HMX_BATCH_LUT_TEMPLATES ||
           mode ==
               QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER_VDEAL_FUSED_QK_REQUANT_HMX_BATCH_LUT_TEMPLATES_GQA_BATCH ||
           mode ==
               QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER_VDEAL_FUSED_QK_REQUANT_HMX_BATCH_LUT_TEMPLATES_GQA_BATCH_DEPENDENCY_STREAM;
}

static int qbh_attention_u8_hmx_batch_enabled(uint32_t mode) {
    mode = qbh_attention_u8_base_mode(mode);
    return mode ==
               QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER_VDEAL_FUSED_QK_REQUANT_HMX_BATCH ||
           mode ==
               QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER_VDEAL_FUSED_QK_REQUANT_HMX_BATCH_LUT_TEMPLATES ||
           mode ==
               QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER_VDEAL_FUSED_QK_REQUANT_HMX_BATCH_LUT_TEMPLATES_GQA_BATCH ||
           mode ==
               QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER_VDEAL_FUSED_QK_REQUANT_HMX_BATCH_LUT_TEMPLATES_GQA_BATCH_DEPENDENCY_STREAM;
}

static int qbh_attention_u8_lut_templates_enabled(uint32_t mode) {
    mode = qbh_attention_u8_base_mode(mode);
    return mode ==
               QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER_VDEAL_FUSED_QK_REQUANT_HMX_BATCH_LUT_TEMPLATES ||
           mode ==
               QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER_VDEAL_FUSED_QK_REQUANT_HMX_BATCH_LUT_TEMPLATES_GQA_BATCH ||
           mode ==
               QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER_VDEAL_FUSED_QK_REQUANT_HMX_BATCH_LUT_TEMPLATES_GQA_BATCH_DEPENDENCY_STREAM;
}

static int qbh_attention_u8_gqa_hmx_batch_enabled(uint32_t mode) {
    mode = qbh_attention_u8_base_mode(mode);
    return mode ==
               QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER_VDEAL_FUSED_QK_REQUANT_HMX_BATCH_LUT_TEMPLATES_GQA_BATCH ||
           mode ==
               QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER_VDEAL_FUSED_QK_REQUANT_HMX_BATCH_LUT_TEMPLATES_GQA_BATCH_DEPENDENCY_STREAM;
}

static int qbh_attention_u8_dependency_stream_enabled(uint32_t mode) {
    mode = qbh_attention_u8_base_mode(mode);
    return mode ==
           QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER_VDEAL_FUSED_QK_REQUANT_HMX_BATCH_LUT_TEMPLATES_GQA_BATCH_DEPENDENCY_STREAM;
}

static int qbh_attention_u8_softmax_shuffle4_enabled(uint32_t mode) {
    return mode ==
           QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER_VDEAL_FUSED_QK_REQUANT_HMX_BATCH_LUT_TEMPLATES_GQA_BATCH_DEPENDENCY_STREAM_SOFTMAX_SHUFFLE4;
}

static uint32_t qbh_w4f16_projection_group_tiles(
    const struct qbh_block_header *header,
    const struct qbh_block_projection_desc *desc);

static int qbh_attention_qk_norm_wait_head(
    struct qbh_block_w4f16_pool *pool, uint32_t task) {
    if (pool->attention_qk_streaming == 0U) {
        return 0;
    }
    while (pool->attention_qk_ready[task] !=
           pool->attention_qk_generation) {
        if (pool->attention_qk_stream_abort != 0U) {
            return -1;
        }
        asm volatile("pause(#8)" : : : "memory");
    }
    asm volatile("barrier" ::: "memory");
    return 0;
}

static void qbh_attention_qk_norm_run_head(
    struct qbh_block_w4f16_pool *pool, uint32_t task) {
    const struct qbh_block_header *header = pool->attention_header;
    if (pool->attention_crouton_qkv != 0U &&
        task < QBH_BLOCK_HEADS) {
        qbh_hvx_qk_norm_rope_f16_crouton_head(
            pool->attention_q, pool->attention_q_destination,
            task, qbh_w4f16_projection_group_tiles(
                      header, &header->projections[QBH_BLOCK_PROJ_Q]),
            0U,
            pool->attention_q_gamma,
            pool->attention_rope_cos, pool->attention_rope_sin);
    } else if (pool->attention_crouton_qkv != 0U) {
        const uint32_t head = task - QBH_BLOCK_HEADS;
        qbh_hvx_qk_norm_rope_f16_crouton_head(
            pool->attention_k, pool->attention_k_weight,
            head, qbh_w4f16_projection_group_tiles(
                      header, &header->projections[QBH_BLOCK_PROJ_K]),
            1U,
            pool->attention_k_gamma,
            pool->attention_rope_cos, pool->attention_rope_sin);
    } else if (task < QBH_BLOCK_HEADS) {
        qbh_hvx_qk_norm_rope_f16_head(
            pool->attention_q, QBH_BLOCK_M, QBH_BLOCK_HIDDEN,
            QBH_BLOCK_HEAD_DIM, task,
            pool->attention_q_gamma, pool->attention_rope_cos,
            pool->attention_rope_sin);
    } else {
        const uint32_t head = task - QBH_BLOCK_HEADS;
        qbh_hvx_qk_norm_rope_f16_head(
            pool->attention_k, QBH_BLOCK_M,
            QBH_BLOCK_KV_HIDDEN, QBH_BLOCK_HEAD_DIM, head,
            pool->attention_k_gamma, pool->attention_rope_cos,
            pool->attention_rope_sin);
    }
}

static void qbh_attention_qk_norm_pool_run_tasks(
    struct qbh_block_w4f16_pool *pool,
    struct qbh_block_w4f16_job *job) {
    const uint32_t paired =
        (pool->fp16_common_schedule_mode &
         QBH_BLOCK_FP16_COMMON_SCHEDULE_QK_HEAD_PAIRS) != 0U;
    const uint32_t q_prefix4 =
        pool->attention_header != NULL &&
        pool->attention_header->qkv_schedule_mode ==
            QBH_BLOCK_QKV_SCHEDULE_Q_PREFIX4_K_ALL;
    for (;;) {
        const uint32_t task_offset =
            qbh_atomic_fetch_increment(&pool->next_attention_task);
        uint32_t task0;
        uint32_t task1 = UINT32_MAX;
        uint64_t start;
        if (task_offset >= pool->attention_task_count) {
            break;
        }
        if (q_prefix4 != 0U) {
            const uint32_t prefix_q_heads = 8U;
            if (task_offset < prefix_q_heads) {
                task0 = task_offset;
            } else if (task_offset <
                       prefix_q_heads + QBH_BLOCK_KV_HEADS) {
                task0 = QBH_BLOCK_HEADS +
                    task_offset - prefix_q_heads;
            } else {
                task0 = task_offset - QBH_BLOCK_KV_HEADS;
            }
        } else if (paired != 0U) {
            const uint32_t q_pair_count = QBH_BLOCK_HEADS / 2U;
            task0 = task_offset < q_pair_count
                ? task_offset * 2U
                : QBH_BLOCK_HEADS +
                      (task_offset - q_pair_count) * 2U;
            task1 = task0 + 1U;
        } else {
            task0 = pool->attention_task_base + task_offset;
        }
        if (qbh_attention_qk_norm_wait_head(pool, task0) != 0 ||
            (q_prefix4 == 0U && paired != 0U &&
             qbh_attention_qk_norm_wait_head(pool, task1) != 0)) {
            return;
        }
        start = HAP_perf_get_qtimer_count();
        qbh_attention_qk_norm_run_head(pool, task0);
        if (q_prefix4 == 0U && paired != 0U) {
            qbh_attention_qk_norm_run_head(pool, task1);
            job->attention_qk_norm_task_count += 2U;
            ++job->fp16_qk_norm_pair_task_count;
        } else {
            ++job->attention_qk_norm_task_count;
        }
        job->attention_qk_norm_ticks +=
            HAP_perf_get_qtimer_count() - start;
    }
}

static void qbh_attention_softmax_pool_run_tasks(
    struct qbh_block_w4f16_pool *pool,
    struct qbh_block_w4f16_job *job) {
    const size_t head_elements =
        (size_t)QBH_BLOCK_M * QBH_BLOCK_M;
    for (;;) {
        uint32_t task =
            qbh_atomic_fetch_increment(&pool->next_attention_task);
        uint64_t start;
        if (task >= pool->attention_task_count) {
            break;
        }
        start = HAP_perf_get_qtimer_count();
        qbh_hvx_stable_causal_softmax_f16(
            pool->attention_scores + (size_t)task * head_elements,
            pool->attention_probability + (size_t)task * head_elements,
            1U, QBH_BLOCK_M, QBH_BLOCK_M,
            0.08838834764831845f, NULL);
        job->attention_softmax_ticks +=
            HAP_perf_get_qtimer_count() - start;
        ++job->attention_softmax_task_count;
    }
}

static void qbh_u8_residual_pool_run_tasks(
    struct qbh_block_w4f16_pool *pool,
    struct qbh_block_w4f16_job *job) {
    for (;;) {
        const uint32_t task = qbh_atomic_fetch_increment(
            &pool->next_u8_residual_task);
        const uint32_t first_row =
            task * QBH_BLOCK_W4U8_RESIDUAL_ROWS_PER_TASK;
        uint32_t row_count;
        uint64_t start;

        if (task >= pool->u8_residual_task_count ||
            first_row >= QBH_BLOCK_M) {
            break;
        }
        row_count = QBH_BLOCK_M - first_row;
        if (row_count > QBH_BLOCK_W4U8_RESIDUAL_ROWS_PER_TASK) {
            row_count = QBH_BLOCK_W4U8_RESIDUAL_ROWS_PER_TASK;
        }
        start = HAP_perf_get_qtimer_count();
        if (pool->u8_residual_kind ==
            QBH_BLOCK_U8_RESIDUAL_POST_NORM) {
            if (pool->u8_residual_shuffle4 != 0U) {
                qbh_hvx_residual_rms_norm_u8_native_io_rows_shuffle4(
                    pool->u8_residual,
                    pool->u8_residual_qparam,
                    pool->u8_residual_addition_tiles,
                    pool->u8_residual_addition_qparam,
                    pool->u8_residual_sum_qparam,
                    pool->u8_residual_gamma,
                    pool->u8_residual_normalized_tiles,
                    pool->u8_residual_normalized_qparam,
                    first_row, row_count, QBH_BLOCK_HIDDEN);
            } else {
                qbh_hvx_residual_rms_norm_u8_native_io_rows(
                    pool->u8_residual,
                    pool->u8_residual_qparam,
                    pool->u8_residual_addition_tiles,
                    pool->u8_residual_addition_qparam,
                    pool->u8_residual_sum_qparam,
                    pool->u8_residual_gamma,
                    pool->u8_residual_normalized_tiles,
                    pool->u8_residual_normalized_qparam,
                    first_row, row_count, QBH_BLOCK_HIDDEN);
            }
            job->u8_post_residual_ticks +=
                HAP_perf_get_qtimer_count() - start;
            ++job->u8_post_residual_task_count;
        } else {
            if (pool->u8_residual_shuffle4 != 0U) {
                qbh_hvx_residual_add_u8_native_output_rows_shuffle4(
                    pool->u8_residual,
                    pool->u8_residual_qparam,
                    pool->u8_residual_addition_tiles,
                    pool->u8_residual_addition_qparam,
                    pool->u8_residual_sum_qparam,
                    first_row, row_count, QBH_BLOCK_HIDDEN);
            } else {
                qbh_hvx_residual_add_u8_native_output_rows(
                    pool->u8_residual,
                    pool->u8_residual_qparam,
                    pool->u8_residual_addition_tiles,
                    pool->u8_residual_addition_qparam,
                    pool->u8_residual_sum_qparam,
                    first_row, row_count, QBH_BLOCK_HIDDEN);
            }
            job->u8_final_residual_ticks +=
                HAP_perf_get_qtimer_count() - start;
            ++job->u8_final_residual_task_count;
        }
    }
}

static void qbh_u8_input_norm_pool_run_tasks(
    struct qbh_block_w4f16_pool *pool,
    struct qbh_block_w4f16_job *job) {
    for (;;) {
        const uint32_t task = qbh_atomic_fetch_increment(
            &pool->next_u8_input_norm_task);
        const uint32_t first_row =
            task * QBH_BLOCK_W4U8_INPUT_NORM_ROWS_PER_TASK;
        uint32_t row_count;
        uint64_t start;

        if (task >= pool->u8_input_norm_task_count ||
            first_row >= QBH_BLOCK_M) {
            break;
        }
        row_count = QBH_BLOCK_M - first_row;
        if (row_count > QBH_BLOCK_W4U8_INPUT_NORM_ROWS_PER_TASK) {
            row_count = QBH_BLOCK_W4U8_INPUT_NORM_ROWS_PER_TASK;
        }
        start = HAP_perf_get_qtimer_count();
        qbh_hvx_rms_norm_u8_native_activation_rows(
            pool->u8_input_norm_input,
            pool->u8_input_norm_input_qparam,
            pool->u8_input_norm_gamma,
            pool->u8_input_norm_output_tiles,
            pool->u8_input_norm_output_qparam,
            first_row, row_count, QBH_BLOCK_HIDDEN);
        job->u8_input_norm_ticks +=
            HAP_perf_get_qtimer_count() - start;
        ++job->u8_input_norm_task_count;
    }
}

static void qbh_w4u8_qkv_ring_expand_worker_run(
    struct qbh_block_w4f16_pool *pool,
    struct qbh_block_w4f16_job *job) {
    struct qbh_w4u8_qkv_ring_state *state =
        (struct qbh_w4u8_qkv_ring_state *)pool->qkv_ring_state;
    const uint32_t task_count =
        state != NULL
            ? state->batch_count *
                  QBH_BLOCK_W4U8_QKV_RING_TILES_PER_BATCH
            : 0U;

    while (state != NULL && state->abort_status == 0U) {
        const uint32_t task =
            qbh_atomic_fetch_increment(&state->next_expand_task);
        const uint32_t batch_index =
            task / QBH_BLOCK_W4U8_QKV_RING_TILES_PER_BATCH;
        const uint32_t tile =
            task % QBH_BLOCK_W4U8_QKV_RING_TILES_PER_BATCH;
        uint32_t slot;
        uint64_t start;
        uint32_t completed;

        if (task >= task_count) {
            break;
        }
        while (state->dma_ready[batch_index] != state->generation) {
            if (state->abort_status != 0U) {
                return;
            }
            asm volatile("pause(#8)" : : : "memory");
        }
        asm volatile("barrier" ::: "memory");
        slot = batch_index % QBH_BLOCK_W4U8_QKV_RING_SLOTS;
        start = HAP_perf_get_qtimer_count();
        qbh_unpack_w4_to_s8_hvx_relaxed(
            state->compressed_slots[slot] +
                (size_t)tile * state->k_tiles *
                    QBH_W4_PACKED_TILE_BYTES,
            (int8_t *)(state->expanded_slots[slot] +
                (size_t)tile * state->k_tiles *
                    QBH_HMX_WEIGHT_BYTES),
            state->k_tiles);
        job->expand_ticks +=
            HAP_perf_get_qtimer_count() - start;
        ++job->expand_count;
        completed = qbh_atomic_fetch_increment(
            &state->expanded_count[batch_index]) + 1U;
        if (completed ==
            QBH_BLOCK_W4U8_QKV_RING_TILES_PER_BATCH) {
            asm volatile("barrier" ::: "memory");
            qurt_sem_up(&state->compressed_free[slot]);
            qurt_sem_up(&state->expanded_ready[slot]);
        }
    }
    if (state != NULL && state->abort_status == 0U) {
        qbh_attention_u8_qk_prep_pool_run_tasks(pool, job);
    }
}

static void qbh_w4u8_generation_expand_worker_run(
    struct qbh_block_w4f16_pool *pool,
    struct qbh_block_w4f16_job *job) {
    for (;;) {
        const uint32_t tile = qbh_atomic_fetch_increment(
            &pool->next_region);
        uint64_t start;
        if (tile >= pool->region_count) {
            break;
        }
        start = HAP_perf_get_qtimer_count();
        qbh_unpack_w4_to_s8_hvx_relaxed(
            pool->compressed_weight +
                (size_t)tile * pool->region_tiles *
                    QBH_W4_PACKED_TILE_BYTES,
            (int8_t *)(pool->expanded_weight +
                (size_t)tile * pool->region_tiles *
                    QBH_HMX_WEIGHT_BYTES),
            pool->region_tiles);
        job->generation_expand_ticks +=
            HAP_perf_get_qtimer_count() - start;
        ++job->generation_expand_count;
    }
}

static void qbh_w4f16_hvx_worker_main(void *opaque) {
    struct qbh_block_w4f16_job *job =
        (struct qbh_block_w4f16_job *)opaque;
    struct qbh_block_w4f16_pool *pool = job->pool;

    job->lock_status = qurt_hvx_lock(QURT_HVX_MODE_128B);
    (void)qurt_sem_up(&pool->worker_started[job->worker_index]);
    if (job->lock_status != AEE_SUCCESS) {
        qurt_thread_exit(job->lock_status);
    }
    for (;;) {
        qurt_sem_down(&pool->command_ready[job->worker_index]);
        if (pool->stop != 0U) {
            break;
        }
        if (job->command_kind == QBH_BLOCK_HVX_POOL_W4_EXPAND) {
            if (pool->claim_regions == 3U) {
                const uint32_t worker = job->worker_index;
                const uint32_t base_regions =
                    pool->region_count / pool->active_worker_count;
                const uint32_t extra_regions =
                    pool->region_count % pool->active_worker_count;
                const uint32_t first_region =
                    worker * base_regions +
                    (worker < extra_regions ? worker : extra_regions);
                const uint32_t claimed_regions =
                    base_regions + (worker < extra_regions ? 1U : 0U);
                const uint64_t start = HAP_perf_get_qtimer_count();
                if (claimed_regions != 0U) {
                    qbh_unpack_w4_to_f16_hvx_relaxed(
                        pool->compressed_weight +
                            (size_t)first_region * pool->region_tiles *
                                QBH_W4_PACKED_TILE_BYTES,
                        pool->expanded_weight +
                            (size_t)first_region * pool->region_tiles *
                                QBH_HMX_FP16_TILE_BYTES,
                        claimed_regions * pool->region_tiles);
                }
                job->expand_ticks +=
                    HAP_perf_get_qtimer_count() - start;
                job->expand_count += claimed_regions;
            } else {
              for (;;) {
                uint32_t region = pool->claim_regions == 2U
                    ? qbh_atomic_fetch_add_two(&pool->next_region)
                    : qbh_atomic_fetch_increment(&pool->next_region);
                uint32_t claimed_regions;
                uint64_t start;
                if (region >= pool->region_count) {
                    break;
                }
                claimed_regions = pool->region_count - region;
                if (claimed_regions > pool->claim_regions) {
                    claimed_regions = pool->claim_regions;
                }
                start = HAP_perf_get_qtimer_count();
                if (pool->publish_ready != 0U) {
                    qbh_unpack_w4_to_f16_hvx(
                        pool->compressed_weight +
                            (size_t)region * pool->region_tiles *
                                QBH_W4_PACKED_TILE_BYTES,
                        pool->expanded_weight +
                            (size_t)region * pool->region_tiles *
                                QBH_HMX_FP16_TILE_BYTES,
                        pool->region_tiles);
                } else {
                    qbh_unpack_w4_to_f16_hvx_relaxed(
                        pool->compressed_weight +
                            (size_t)region * pool->region_tiles *
                                QBH_W4_PACKED_TILE_BYTES,
                        pool->expanded_weight +
                            (size_t)region * pool->region_tiles *
                                QBH_HMX_FP16_TILE_BYTES,
                        claimed_regions * pool->region_tiles);
                }
                job->expand_ticks +=
                    HAP_perf_get_qtimer_count() - start;
                job->expand_count += claimed_regions;
                if (pool->publish_ready != 0U) {
                    pool->ready_generations[region] =
                        pool->expected_generation;
                    asm volatile("release(%0):at"
                                 :
                                 : "r"(&pool->ready_generations[region])
                                 : "memory");
                }
            }
            }
        } else if (job->command_kind == QBH_BLOCK_HVX_POOL_SILU) {
            qbh_silu_pool_run_chunks(
                pool, &job->silu_ticks, &job->silu_chunk_count);
        } else if (job->command_kind ==
                   QBH_BLOCK_HVX_POOL_MLP_STREAM) {
            qbh_mlp_stream_worker_run(pool, job);
        } else if (job->command_kind ==
                   QBH_BLOCK_HVX_POOL_QK_NORM_ROPE) {
            qbh_attention_qk_norm_pool_run_tasks(pool, job);
        } else if (job->command_kind ==
                   QBH_BLOCK_HVX_POOL_SOFTMAX) {
            qbh_attention_softmax_pool_run_tasks(pool, job);
        } else if (job->command_kind ==
                   QBH_BLOCK_HVX_POOL_GQA_ATTENTION) {
            qbh_attention_gqa_pool_run_tasks(pool, job);
        } else if (job->command_kind ==
                   QBH_BLOCK_HVX_POOL_U8_GQA_ATTENTION) {
            qbh_attention_u8_pool_run_tasks(pool, job);
        } else if (job->command_kind ==
                   QBH_BLOCK_HVX_POOL_U8_QK_PREP) {
            qbh_attention_u8_qk_prep_pool_run_tasks(pool, job);
        } else if (job->command_kind ==
                   QBH_BLOCK_HVX_POOL_U8_RESIDUAL) {
            qbh_u8_residual_pool_run_tasks(pool, job);
        } else if (job->command_kind ==
                   QBH_BLOCK_HVX_POOL_U8_INPUT_NORM) {
            qbh_u8_input_norm_pool_run_tasks(pool, job);
        } else if (job->command_kind ==
                   QBH_BLOCK_HVX_POOL_FP16_INPUT_NORM) {
            qbh_fp16_input_norm_pool_run_tasks(pool, job);
        } else if (job->command_kind ==
                   QBH_BLOCK_HVX_POOL_FP16_POST_RESIDUAL_NORM) {
            qbh_fp16_post_residual_norm_pool_run_tasks(pool, job);
        } else if (job->command_kind ==
                   QBH_BLOCK_HVX_POOL_W4U8_QKV_RING) {
            qbh_w4u8_qkv_ring_expand_worker_run(pool, job);
        } else if (job->command_kind ==
                   QBH_BLOCK_HVX_POOL_U8_SEGMENTED_ATTENTION) {
            qbh_attention_u8_segmented_pool_run_tasks(pool, job);
        } else if (job->command_kind ==
                   QBH_BLOCK_HVX_POOL_W4U8_GENERATION_EXPAND) {
            qbh_w4u8_generation_expand_worker_run(pool, job);
        } else if (job->command_kind ==
                   QBH_BLOCK_HVX_POOL_W4U8_PIPELINE) {
            job->w4u8_pipeline_worker_status =
                qbh_run_chunked_w4_external_hvx_worker(
                    job->w4u8_pipeline_worker_context);
        }
        job->command_kind = QBH_BLOCK_HVX_POOL_NONE;
        (void)qurt_sem_up(&pool->command_done[job->worker_index]);
    }
    job->unlock_status = qurt_hvx_unlock();
    qurt_thread_exit(job->unlock_status);
}

static int qbh_w4f16_pool_create(
    struct qbh_block_w4f16_pool *pool, uint32_t worker_count) {
    int result = 0;

    if (worker_count == 0U ||
        worker_count > QBH_BLOCK_MAX_POOL_HVX_WORKERS) {
        return -1;
    }
    memset(pool, 0, sizeof(*pool));
    qurt_mutex_init(&pool->attention_hmx_mutex);
    pool->worker_count = worker_count;
    pool->extra_expand_worker_index = UINT32_MAX;
    for (uint32_t worker = 0; worker < pool->worker_count; ++worker) {
        qurt_sem_init_val(&pool->command_ready[worker], 0U);
        qurt_sem_init_val(&pool->command_done[worker], 0U);
        qurt_sem_init_val(&pool->worker_started[worker], 0U);
    }
    for (uint32_t worker = 0; worker < pool->worker_count; ++worker) {
        qurt_thread_attr_t attributes;
        char name[16] = "qbh-w4f16-hvx0";
        name[13] = (char)('0' + worker);
        pool->jobs[worker].pool = pool;
        pool->jobs[worker].worker_index = worker;
        qurt_thread_attr_init(&attributes);
        qurt_thread_attr_set_name(&attributes, name);
        qurt_thread_attr_set_stack_addr(
            &attributes, qbh_block_w4f16_hvx_stacks[worker]);
        qurt_thread_attr_set_stack_size(
            &attributes, QBH_BLOCK_W4F16_HVX_STACK_BYTES);
        qurt_thread_attr_set_priority(
            &attributes,
            qurt_thread_get_priority(qurt_thread_get_id()));
        qurt_thread_attr_set_detachstate(
            &attributes, QURT_THREAD_ATTR_CREATE_JOINABLE);
        if (qurt_thread_create(
                &pool->threads[worker], &attributes,
                qbh_w4f16_hvx_worker_main,
                &pool->jobs[worker]) != QURT_EOK) {
            result = -1;
            break;
        }
        ++pool->created_workers;
    }
    for (uint32_t worker = 0; worker < pool->created_workers;
         ++worker) {
        qurt_sem_down(&pool->worker_started[worker]);
        if (pool->jobs[worker].lock_status != AEE_SUCCESS) {
            result = -1;
        }
    }
    if (result != 0 ||
        pool->created_workers != pool->worker_count) {
        pool->stop = 1U;
        asm volatile("barrier" ::: "memory");
        for (uint32_t worker = 0; worker < pool->created_workers;
             ++worker) {
            int exit_status;
            (void)qurt_sem_up(&pool->command_ready[worker]);
            (void)qurt_thread_join(
                pool->threads[worker], &exit_status);
        }
        for (uint32_t worker = 0; worker < pool->worker_count;
             ++worker) {
            qurt_sem_destroy(&pool->worker_started[worker]);
            qurt_sem_destroy(&pool->command_done[worker]);
            qurt_sem_destroy(&pool->command_ready[worker]);
        }
        qurt_mutex_destroy(&pool->attention_hmx_mutex);
        return -1;
    }
    return 0;
}

static void qbh_w4f16_pool_start(
    struct qbh_block_w4f16_pool *pool,
    const uint8_t *compressed_weight, const float *channel_scale,
    uint8_t *expanded_weight, volatile uint32_t *ready_generations,
    uint32_t expected_generation, uint32_t region_count,
    uint32_t region_tiles, uint32_t active_worker_count,
    uint32_t publish_ready, uint32_t claim_regions,
    uint32_t extra_expand_worker_index) {
    pool->compressed_weight = compressed_weight;
    pool->channel_scale = channel_scale;
    pool->expanded_weight = expanded_weight;
    pool->ready_generations = ready_generations;
    pool->expected_generation = expected_generation;
    pool->region_count = region_count;
    pool->region_tiles = region_tiles;
    pool->publish_ready = publish_ready;
    pool->claim_regions = publish_ready != 0U ? 1U : claim_regions;
    pool->active_worker_count = active_worker_count;
    pool->extra_expand_worker_index = extra_expand_worker_index;
    pool->next_region = 0U;
    for (uint32_t worker = 0; worker < pool->active_worker_count;
         ++worker) {
        pool->jobs[worker].command_kind =
            QBH_BLOCK_HVX_POOL_W4_EXPAND;
    }
    if (pool->extra_expand_worker_index != UINT32_MAX) {
        pool->jobs[pool->extra_expand_worker_index].command_kind =
            QBH_BLOCK_HVX_POOL_W4_EXPAND;
    }
    asm volatile("barrier" ::: "memory");
    for (uint32_t worker = 0; worker < pool->active_worker_count;
         ++worker) {
        (void)qurt_sem_up(&pool->command_ready[worker]);
    }
    if (pool->extra_expand_worker_index != UINT32_MAX) {
        (void)qurt_sem_up(
            &pool->command_ready[pool->extra_expand_worker_index]);
    }
}

static void qbh_w4f16_pool_wait(
    struct qbh_block_w4f16_pool *pool) {
    for (uint32_t worker = 0; worker < pool->active_worker_count;
         ++worker) {
        qurt_sem_down(&pool->command_done[worker]);
    }
    if (pool->extra_expand_worker_index != UINT32_MAX) {
        qurt_sem_down(
            &pool->command_done[pool->extra_expand_worker_index]);
        pool->extra_expand_worker_index = UINT32_MAX;
    }
    asm volatile("barrier" ::: "memory");
}

static int qbh_w4u8_generation_expand_with_main(
    struct qbh_block_w4f16_pool *pool,
    const uint8_t *compressed_weight, uint8_t *expanded_weight,
    uint32_t k_tiles, uint32_t group_tiles,
    uint32_t active_worker_count) {
    struct qbh_block_w4f16_job main_job;

    if (pool == NULL || active_worker_count == 0U ||
        active_worker_count > pool->worker_count ||
        group_tiles == 0U) {
        return -1;
    }
    for (uint32_t worker = 0U; worker < pool->worker_count; ++worker) {
        if (pool->jobs[worker].command_kind !=
            QBH_BLOCK_HVX_POOL_NONE) {
            return -1;
        }
    }
    pool->compressed_weight = compressed_weight;
    pool->expanded_weight = expanded_weight;
    pool->region_count = group_tiles;
    pool->region_tiles = k_tiles;
    pool->next_region = 0U;
    pool->active_worker_count = active_worker_count;
    pool->extra_expand_worker_index = UINT32_MAX;
    for (uint32_t worker = 0U; worker < active_worker_count; ++worker) {
        pool->jobs[worker].command_kind =
            QBH_BLOCK_HVX_POOL_W4U8_GENERATION_EXPAND;
    }
    asm volatile("barrier" ::: "memory");
    for (uint32_t worker = 0U; worker < active_worker_count; ++worker) {
        (void)qurt_sem_up(&pool->command_ready[worker]);
    }
    memset(&main_job, 0, sizeof(main_job));
    qbh_w4u8_generation_expand_worker_run(pool, &main_job);
    qbh_w4f16_pool_wait(pool);
    return 0;
}

static int qbh_w4u8_pipeline_pool_start(
    void *context, void *const *worker_contexts,
    uint32_t worker_count) {
    struct qbh_block_w4f16_pool *pool =
        (struct qbh_block_w4f16_pool *)context;

    if (pool == NULL || worker_contexts == NULL ||
        worker_count == 0U || worker_count > pool->worker_count) {
        return AEE_EBADPARM;
    }
    for (uint32_t worker = 0U; worker < worker_count; ++worker) {
        if (worker_contexts[worker] == NULL ||
            pool->jobs[worker].command_kind !=
                QBH_BLOCK_HVX_POOL_NONE) {
            return AEE_EFAILED;
        }
    }
    pool->active_worker_count = worker_count;
    for (uint32_t worker = 0U; worker < worker_count; ++worker) {
        pool->jobs[worker].w4u8_pipeline_worker_context =
            worker_contexts[worker];
        pool->jobs[worker].w4u8_pipeline_worker_status =
            AEE_EFAILED;
        pool->jobs[worker].command_kind =
            QBH_BLOCK_HVX_POOL_W4U8_PIPELINE;
    }
    asm volatile("barrier" ::: "memory");
    for (uint32_t worker = 0U; worker < worker_count; ++worker) {
        (void)qurt_sem_up(&pool->command_ready[worker]);
    }
    return AEE_SUCCESS;
}

static int qbh_w4u8_pipeline_pool_wait(
    void *context, uint32_t worker_count) {
    struct qbh_block_w4f16_pool *pool =
        (struct qbh_block_w4f16_pool *)context;
    int result = AEE_SUCCESS;

    if (pool == NULL || worker_count == 0U ||
        worker_count != pool->active_worker_count) {
        return AEE_EBADPARM;
    }
    qbh_w4f16_pool_wait(pool);
    for (uint32_t worker = 0U; worker < worker_count; ++worker) {
        if (pool->jobs[worker].w4u8_pipeline_worker_status !=
            AEE_SUCCESS) {
            result = AEE_EFAILED;
        }
        pool->jobs[worker].w4u8_pipeline_worker_context = NULL;
    }
    pool->active_worker_count = 0U;
    return result;
}

static void qbh_w4u8_down_transient_hvx_worker_main(void *opaque) {
    qurt_thread_exit(
        qbh_run_chunked_w4_managed_hvx_worker(opaque));
}

static int qbh_w4u8_hybrid_down_runner_start(
    void *context, void *const *worker_contexts,
    uint32_t worker_count) {
    struct qbh_block_w4u8_hybrid_hvx_runner *runner =
        (struct qbh_block_w4u8_hybrid_hvx_runner *)context;
    struct qbh_block_w4f16_pool *pool;
    qurt_thread_attr_t attributes;

    if (runner == NULL || worker_contexts == NULL ||
        worker_count != QBH_BLOCK_W4U8_DOWN_HVX_WORKERS ||
        runner->transient_thread_created != 0U) {
        return AEE_EBADPARM;
    }
    pool = runner->pool;
    if (pool == NULL ||
        pool->worker_count <
            QBH_BLOCK_W4U8_DOWN_PERSISTENT_HVX_WORKERS) {
        return AEE_EBADPARM;
    }
    for (uint32_t worker = 0U;
         worker < QBH_BLOCK_W4U8_DOWN_PERSISTENT_HVX_WORKERS;
         ++worker) {
        if (worker_contexts[worker] == NULL ||
            pool->jobs[worker].command_kind !=
                QBH_BLOCK_HVX_POOL_NONE) {
            return AEE_EFAILED;
        }
    }
    if (worker_contexts[QBH_BLOCK_W4U8_DOWN_PERSISTENT_HVX_WORKERS]
        == NULL) {
        return AEE_EBADPARM;
    }

    qurt_thread_attr_init(&attributes);
    qurt_thread_attr_set_name(
        &attributes, "qbh-w4u8-down5");
    qurt_thread_attr_set_stack_addr(
        &attributes, qbh_block_w4u8_down_transient_hvx_stack);
    qurt_thread_attr_set_stack_size(
        &attributes, QBH_BLOCK_W4F16_HVX_STACK_BYTES);
    qurt_thread_attr_set_priority(
        &attributes,
        qurt_thread_get_priority(qurt_thread_get_id()));
    qurt_thread_attr_set_detachstate(
        &attributes, QURT_THREAD_ATTR_CREATE_JOINABLE);
    if (qurt_thread_create(
            &runner->transient_thread, &attributes,
            qbh_w4u8_down_transient_hvx_worker_main,
            worker_contexts[
                QBH_BLOCK_W4U8_DOWN_PERSISTENT_HVX_WORKERS]) !=
        QURT_EOK) {
        return AEE_EFAILED;
    }
    runner->transient_thread_created = 1U;

    pool->active_worker_count =
        QBH_BLOCK_W4U8_DOWN_PERSISTENT_HVX_WORKERS;
    for (uint32_t worker = 0U;
         worker < pool->active_worker_count; ++worker) {
        pool->jobs[worker].w4u8_pipeline_worker_context =
            worker_contexts[worker];
        pool->jobs[worker].w4u8_pipeline_worker_status =
            AEE_EFAILED;
        pool->jobs[worker].command_kind =
            QBH_BLOCK_HVX_POOL_W4U8_PIPELINE;
    }
    asm volatile("barrier" ::: "memory");
    for (uint32_t worker = 0U;
         worker < pool->active_worker_count; ++worker) {
        (void)qurt_sem_up(&pool->command_ready[worker]);
    }
    return AEE_SUCCESS;
}

static int qbh_w4u8_hybrid_down_runner_wait(
    void *context, uint32_t worker_count) {
    struct qbh_block_w4u8_hybrid_hvx_runner *runner =
        (struct qbh_block_w4u8_hybrid_hvx_runner *)context;
    struct qbh_block_w4f16_pool *pool;
    int transient_exit_status = AEE_EFAILED;
    int result = AEE_SUCCESS;

    if (runner == NULL ||
        worker_count != QBH_BLOCK_W4U8_DOWN_HVX_WORKERS ||
        runner->transient_thread_created == 0U) {
        return AEE_EBADPARM;
    }
    pool = runner->pool;
    if (pool == NULL || pool->active_worker_count !=
            QBH_BLOCK_W4U8_DOWN_PERSISTENT_HVX_WORKERS) {
        return AEE_EBADPARM;
    }
    qbh_w4f16_pool_wait(pool);
    for (uint32_t worker = 0U;
         worker < pool->active_worker_count; ++worker) {
        if (pool->jobs[worker].w4u8_pipeline_worker_status !=
            AEE_SUCCESS) {
            result = AEE_EFAILED;
        }
        pool->jobs[worker].w4u8_pipeline_worker_context = NULL;
    }
    pool->active_worker_count = 0U;
    if (qurt_thread_join(
            runner->transient_thread,
            &transient_exit_status) != QURT_EOK ||
        transient_exit_status != AEE_SUCCESS) {
        result = AEE_EFAILED;
    }
    runner->transient_thread_created = 0U;
    return result;
}

static int qbh_hvx_pool_u8_input_norm(
    struct qbh_block_header *header,
    struct qbh_block_w4f16_pool *pool,
    const uint8_t *input,
    const struct qbh_block_qparam *input_qparam,
    const __fp16 *gamma, uint8_t *output_tiles,
    const struct qbh_block_qparam *output_qparam) {
    struct qbh_block_w4f16_job main_job;
    uint32_t worker_task_count_before = 0U;
    uint32_t worker_task_count_after = 0U;
    uint64_t worker_ticks_before = 0U;
    uint64_t worker_ticks_after = 0U;
    uint64_t wait_start;

    if (pool == NULL ||
        pool->worker_count < QBH_BLOCK_MAX_POOL_HVX_WORKERS) {
        return -1;
    }
    memset(&main_job, 0, sizeof(main_job));
    pool->u8_input_norm_input = input;
    pool->u8_input_norm_input_qparam = input_qparam;
    pool->u8_input_norm_gamma = gamma;
    pool->u8_input_norm_output_tiles = output_tiles;
    pool->u8_input_norm_output_qparam = output_qparam;
    pool->next_u8_input_norm_task = 0U;
    pool->u8_input_norm_task_count =
        (QBH_BLOCK_M + QBH_BLOCK_W4U8_INPUT_NORM_ROWS_PER_TASK - 1U) /
        QBH_BLOCK_W4U8_INPUT_NORM_ROWS_PER_TASK;
    pool->active_worker_count = QBH_BLOCK_MAX_POOL_HVX_WORKERS;

    for (uint32_t worker = 0U; worker < pool->active_worker_count;
         ++worker) {
        worker_task_count_before +=
            pool->jobs[worker].u8_input_norm_task_count;
        worker_ticks_before += pool->jobs[worker].u8_input_norm_ticks;
        pool->jobs[worker].command_kind =
            QBH_BLOCK_HVX_POOL_U8_INPUT_NORM;
    }
    asm volatile("barrier" ::: "memory");
    for (uint32_t worker = 0U; worker < pool->active_worker_count;
         ++worker) {
        (void)qurt_sem_up(&pool->command_ready[worker]);
    }

    qbh_u8_input_norm_pool_run_tasks(pool, &main_job);
    wait_start = HAP_perf_get_qtimer_count();
    qbh_w4f16_pool_wait(pool);
    header->w4u8_input_norm_pool_wait_ticks +=
        HAP_perf_get_qtimer_count() - wait_start;

    for (uint32_t worker = 0U; worker < pool->active_worker_count;
         ++worker) {
        worker_task_count_after +=
            pool->jobs[worker].u8_input_norm_task_count;
        worker_ticks_after += pool->jobs[worker].u8_input_norm_ticks;
    }
    {
        const uint32_t task_count =
            main_job.u8_input_norm_task_count +
            worker_task_count_after - worker_task_count_before;
        header->w4u8_input_norm_task_count += task_count;
        header->w4u8_input_norm_main_work_ticks +=
            main_job.u8_input_norm_ticks;
        header->w4u8_input_norm_worker_work_ticks +=
            worker_ticks_after - worker_ticks_before;
        return task_count == pool->u8_input_norm_task_count ? 0 : -1;
    }
}

static int qbh_hvx_pool_fp16_input_norm(
    struct qbh_block_header *header,
    struct qbh_block_w4f16_pool *pool,
    const __fp16 *input, const __fp16 *gamma,
    __fp16 *output, uint32_t crouton_output) {
    struct qbh_block_w4f16_job main_job;
    uint32_t worker_task_count_before = 0U;
    uint32_t worker_task_count_after = 0U;
    uint64_t worker_ticks_before = 0U;
    uint64_t worker_ticks_after = 0U;
    uint64_t wait_start;
    uint32_t worker_count;

    if (header == NULL || pool == NULL) {
        return -1;
    }
    worker_count = header->fp16_norm_contexts - 1U;
    if (pool->worker_count < worker_count) {
        return -1;
    }
    memset(&main_job, 0, sizeof(main_job));
    pool->fp16_input_norm_input = input;
    pool->fp16_input_norm_gamma = gamma;
    pool->fp16_input_norm_output = output;
    pool->fp16_input_norm_crouton = crouton_output;
    pool->fp16_norm_rows_per_task =
        header->fp16_norm_rows_per_task;
    pool->next_fp16_input_norm_task = 0U;
    pool->fp16_input_norm_task_count =
        (QBH_BLOCK_M + pool->fp16_norm_rows_per_task - 1U) /
        pool->fp16_norm_rows_per_task;
    pool->active_worker_count = worker_count;
    header->fp16_input_norm_active_contexts = worker_count + 1U;

    for (uint32_t worker = 0U; worker < worker_count; ++worker) {
        worker_task_count_before +=
            pool->jobs[worker].fp16_input_norm_task_count;
        worker_ticks_before +=
            pool->jobs[worker].fp16_input_norm_ticks;
        pool->jobs[worker].command_kind =
            QBH_BLOCK_HVX_POOL_FP16_INPUT_NORM;
    }
    asm volatile("barrier" ::: "memory");
    for (uint32_t worker = 0U; worker < worker_count; ++worker) {
        (void)qurt_sem_up(&pool->command_ready[worker]);
    }

    qbh_fp16_input_norm_pool_run_tasks(pool, &main_job);
    wait_start = HAP_perf_get_qtimer_count();
    qbh_w4f16_pool_wait(pool);
    header->fp16_input_norm_pool_wait_ticks +=
        HAP_perf_get_qtimer_count() - wait_start;
    asm volatile("barrier" ::: "memory");

    for (uint32_t worker = 0U; worker < worker_count; ++worker) {
        worker_task_count_after +=
            pool->jobs[worker].fp16_input_norm_task_count;
        worker_ticks_after +=
            pool->jobs[worker].fp16_input_norm_ticks;
    }
    {
        const uint32_t task_count =
            main_job.fp16_input_norm_task_count +
            worker_task_count_after - worker_task_count_before;
        header->fp16_input_norm_task_count += task_count;
        header->fp16_input_norm_main_work_ticks +=
            main_job.fp16_input_norm_ticks;
        header->fp16_input_norm_worker_work_ticks +=
            worker_ticks_after - worker_ticks_before;
        return task_count == pool->fp16_input_norm_task_count ? 0 : -1;
    }
}

static int qbh_hvx_pool_fp16_post_residual_norm(
    struct qbh_block_header *header,
    struct qbh_block_w4f16_pool *pool,
    __fp16 *residual, const __fp16 *addition,
    const __fp16 *gamma, __fp16 *output,
    uint32_t crouton_output) {
    struct qbh_block_w4f16_job main_job;
    uint32_t worker_task_count_before = 0U;
    uint32_t worker_task_count_after = 0U;
    uint64_t worker_ticks_before = 0U;
    uint64_t worker_ticks_after = 0U;
    uint64_t wait_start;
    uint32_t worker_count;

    if (header == NULL || pool == NULL) {
        return -1;
    }
    worker_count = header->fp16_norm_contexts - 1U;
    if (pool->worker_count < worker_count) {
        return -1;
    }
    memset(&main_job, 0, sizeof(main_job));
    pool->fp16_post_residual = residual;
    pool->fp16_post_addition = addition;
    pool->fp16_post_gamma = gamma;
    pool->fp16_post_output = output;
    pool->fp16_post_residual_norm_crouton = crouton_output;
    pool->fp16_norm_rows_per_task =
        header->fp16_norm_rows_per_task;
    pool->next_fp16_post_residual_norm_task = 0U;
    pool->fp16_post_residual_norm_task_count =
        (QBH_BLOCK_M + pool->fp16_norm_rows_per_task - 1U) /
        pool->fp16_norm_rows_per_task;
    pool->active_worker_count = worker_count;
    header->fp16_post_residual_norm_active_contexts =
        worker_count + 1U;

    for (uint32_t worker = 0U; worker < worker_count; ++worker) {
        worker_task_count_before +=
            pool->jobs[worker].fp16_post_residual_norm_task_count;
        worker_ticks_before +=
            pool->jobs[worker].fp16_post_residual_norm_ticks;
        pool->jobs[worker].command_kind =
            QBH_BLOCK_HVX_POOL_FP16_POST_RESIDUAL_NORM;
    }
    asm volatile("barrier" ::: "memory");
    for (uint32_t worker = 0U; worker < worker_count; ++worker) {
        (void)qurt_sem_up(&pool->command_ready[worker]);
    }

    qbh_fp16_post_residual_norm_pool_run_tasks(pool, &main_job);
    wait_start = HAP_perf_get_qtimer_count();
    qbh_w4f16_pool_wait(pool);
    header->fp16_post_residual_norm_pool_wait_ticks +=
        HAP_perf_get_qtimer_count() - wait_start;
    asm volatile("barrier" ::: "memory");

    for (uint32_t worker = 0U; worker < worker_count; ++worker) {
        worker_task_count_after +=
            pool->jobs[worker].fp16_post_residual_norm_task_count;
        worker_ticks_after +=
            pool->jobs[worker].fp16_post_residual_norm_ticks;
    }
    {
        const uint32_t task_count =
            main_job.fp16_post_residual_norm_task_count +
            worker_task_count_after - worker_task_count_before;
        header->fp16_post_residual_norm_task_count += task_count;
        header->fp16_post_residual_norm_main_work_ticks +=
            main_job.fp16_post_residual_norm_ticks;
        header->fp16_post_residual_norm_worker_work_ticks +=
            worker_ticks_after - worker_ticks_before;
        return task_count == pool->fp16_post_residual_norm_task_count
            ? 0 : -1;
    }
}

static int qbh_hvx_pool_u8_native_residual(
    struct qbh_block_header *header,
    struct qbh_block_w4f16_pool *pool,
    uint8_t *residual,
    const struct qbh_block_qparam *residual_qparam,
    const uint8_t *addition_tiles,
    const struct qbh_block_qparam *addition_qparam,
    const struct qbh_block_qparam *sum_qparam,
    const __fp16 *gamma, uint8_t *normalized_tiles,
    const struct qbh_block_qparam *normalized_qparam,
    uint32_t residual_kind) {
    struct qbh_block_w4f16_job main_job;
    uint32_t worker_task_count_before = 0U;
    uint32_t worker_task_count_after = 0U;
    uint64_t worker_ticks_before = 0U;
    uint64_t worker_ticks_after = 0U;
    uint64_t wait_start;
    const uint32_t active_worker_count =
        (header->residual_mode ==
             QBH_BLOCK_RESIDUAL_HVX_FUSED_POST_NORM_POOL6 ||
         header->residual_mode ==
             QBH_BLOCK_RESIDUAL_HVX_FUSED_POST_NORM_POOL6_SHUFFLE4)
            ? QBH_BLOCK_MAX_POOL_HVX_WORKERS
            : 3U;

    if (pool == NULL || pool->worker_count < active_worker_count ||
        (residual_kind != QBH_BLOCK_U8_RESIDUAL_POST_NORM &&
         residual_kind != QBH_BLOCK_U8_RESIDUAL_FINAL)) {
        return -1;
    }
    memset(&main_job, 0, sizeof(main_job));
    pool->u8_residual = residual;
    pool->u8_residual_qparam = residual_qparam;
    pool->u8_residual_addition_tiles = addition_tiles;
    pool->u8_residual_addition_qparam = addition_qparam;
    pool->u8_residual_sum_qparam = sum_qparam;
    pool->u8_residual_gamma = gamma;
    pool->u8_residual_normalized_tiles = normalized_tiles;
    pool->u8_residual_normalized_qparam = normalized_qparam;
    pool->next_u8_residual_task = 0U;
    pool->u8_residual_task_count =
        (QBH_BLOCK_M + QBH_BLOCK_W4U8_RESIDUAL_ROWS_PER_TASK - 1U) /
        QBH_BLOCK_W4U8_RESIDUAL_ROWS_PER_TASK;
    pool->u8_residual_kind = residual_kind;
    pool->u8_residual_shuffle4 =
        header->residual_mode ==
            QBH_BLOCK_RESIDUAL_HVX_FUSED_POST_NORM_POOL6_SHUFFLE4;
    pool->active_worker_count = active_worker_count;
    header->w4u8_residual_active_contexts =
        active_worker_count + 1U;

    for (uint32_t worker = 0U; worker < pool->active_worker_count;
         ++worker) {
        if (residual_kind == QBH_BLOCK_U8_RESIDUAL_POST_NORM) {
            worker_task_count_before +=
                pool->jobs[worker].u8_post_residual_task_count;
            worker_ticks_before +=
                pool->jobs[worker].u8_post_residual_ticks;
        } else {
            worker_task_count_before +=
                pool->jobs[worker].u8_final_residual_task_count;
            worker_ticks_before +=
                pool->jobs[worker].u8_final_residual_ticks;
        }
        pool->jobs[worker].command_kind =
            QBH_BLOCK_HVX_POOL_U8_RESIDUAL;
    }
    asm volatile("barrier" ::: "memory");
    for (uint32_t worker = 0U; worker < pool->active_worker_count;
         ++worker) {
        (void)qurt_sem_up(&pool->command_ready[worker]);
    }

    qbh_u8_residual_pool_run_tasks(pool, &main_job);
    wait_start = HAP_perf_get_qtimer_count();
    qbh_w4f16_pool_wait(pool);
    if (residual_kind == QBH_BLOCK_U8_RESIDUAL_POST_NORM) {
        header->w4u8_post_residual_pool_wait_ticks +=
            HAP_perf_get_qtimer_count() - wait_start;
    } else {
        header->w4u8_final_residual_pool_wait_ticks +=
            HAP_perf_get_qtimer_count() - wait_start;
    }

    for (uint32_t worker = 0U; worker < pool->active_worker_count;
         ++worker) {
        if (residual_kind == QBH_BLOCK_U8_RESIDUAL_POST_NORM) {
            worker_task_count_after +=
                pool->jobs[worker].u8_post_residual_task_count;
            worker_ticks_after +=
                pool->jobs[worker].u8_post_residual_ticks;
        } else {
            worker_task_count_after +=
                pool->jobs[worker].u8_final_residual_task_count;
            worker_ticks_after +=
                pool->jobs[worker].u8_final_residual_ticks;
        }
    }
    if (residual_kind == QBH_BLOCK_U8_RESIDUAL_POST_NORM) {
        const uint32_t task_count =
            main_job.u8_post_residual_task_count +
            worker_task_count_after - worker_task_count_before;
        header->w4u8_post_residual_task_count += task_count;
        header->w4u8_post_residual_main_work_ticks +=
            main_job.u8_post_residual_ticks;
        header->w4u8_post_residual_worker_work_ticks +=
            worker_ticks_after - worker_ticks_before;
        return task_count == pool->u8_residual_task_count ? 0 : -1;
    }
    {
        const uint32_t task_count =
            main_job.u8_final_residual_task_count +
            worker_task_count_after - worker_task_count_before;
        header->w4u8_final_residual_task_count += task_count;
        header->w4u8_final_residual_main_work_ticks +=
            main_job.u8_final_residual_ticks;
        header->w4u8_final_residual_worker_work_ticks +=
            worker_ticks_after - worker_ticks_before;
        return task_count == pool->u8_residual_task_count ? 0 : -1;
    }
}

static int qbh_hvx_pool_silu(
    struct qbh_block_header *header,
    struct qbh_block_w4f16_pool *pool,
    const __fp16 *gate, const __fp16 *up, __fp16 *middle,
    uint32_t elements, struct qbh_hvx_check_metrics *check) {
    uint32_t main_chunks = 0U;
    uint64_t main_ticks = 0U;
    uint64_t wait_start;

    if (header->mlp_mode == QBH_BLOCK_MLP_CONTROL ||
        header->mlp_hvx_contexts == 1U) {
        qbh_hvx_silu_multiply_f16(
            gate, up, middle, elements, check);
        return 0;
    }
    if (pool == NULL ||
        header->mlp_hvx_contexts - 1U > pool->worker_count ||
        elements % QBH_BLOCK_HVX_F16_LANES != 0U) {
        return -1;
    }

    pool->silu_gate = gate;
    pool->silu_up = up;
    pool->silu_middle = middle;
    pool->silu_vector_count =
        elements / QBH_BLOCK_HVX_F16_LANES;
    pool->silu_chunk_vectors = header->mlp_chunk_vectors;
    pool->silu_chunk_count =
        (pool->silu_vector_count + pool->silu_chunk_vectors - 1U) /
        pool->silu_chunk_vectors;
    pool->next_silu_chunk = 0U;
    pool->active_worker_count = header->mlp_hvx_contexts - 1U;
    for (uint32_t worker = 0; worker < pool->active_worker_count;
         ++worker) {
        pool->jobs[worker].command_kind =
            QBH_BLOCK_HVX_POOL_SILU;
    }
    asm volatile("barrier" ::: "memory");
    for (uint32_t worker = 0; worker < pool->active_worker_count;
         ++worker) {
        (void)qurt_sem_up(&pool->command_ready[worker]);
    }

    qbh_silu_pool_run_chunks(pool, &main_ticks, &main_chunks);
    wait_start = HAP_perf_get_qtimer_count();
    qbh_w4f16_pool_wait(pool);
    header->mlp_silu_pool_wait_ticks +=
        HAP_perf_get_qtimer_count() - wait_start;
    header->mlp_silu_main_work_ticks += main_ticks;
    header->mlp_silu_chunk_count += main_chunks;
    asm volatile("barrier" ::: "memory");

    qbh_hvx_silu_multiply_f16_audit(
        gate, up, middle, elements, check);
    return 0;
}

static int qbh_hvx_pool_qk_norm_rope(
    struct qbh_block_header *header,
    struct qbh_block_w4f16_pool *pool,
    __fp16 *q, __fp16 *k, const __fp16 *q_gamma,
    const __fp16 *k_gamma, const __fp16 *cosine,
    const __fp16 *sine) {
    struct qbh_block_w4f16_job main_job;
    uint64_t wait_start;

    if (pool == NULL || header->attention_hvx_contexts != 4U ||
        header->attention_hvx_contexts - 1U > pool->worker_count) {
        return -1;
    }
    memset(&main_job, 0, sizeof(main_job));
    pool->attention_q = q;
    pool->attention_header = header;
    pool->attention_k = k;
    pool->attention_q_gamma = q_gamma;
    pool->attention_k_gamma = k_gamma;
    pool->attention_rope_cos = cosine;
    pool->attention_rope_sin = sine;
    pool->fp16_common_schedule_mode =
        header->fp16_common_schedule_mode;
    pool->attention_task_count =
        (header->fp16_common_schedule_mode &
         QBH_BLOCK_FP16_COMMON_SCHEDULE_QK_HEAD_PAIRS) != 0U
            ? QBH_BLOCK_HEADS / 2U + QBH_BLOCK_KV_HEADS / 2U
            : QBH_BLOCK_HEADS + QBH_BLOCK_KV_HEADS;
    pool->attention_task_base = 0U;
    pool->attention_qk_streaming = 0U;
    pool->next_attention_task = 0U;
    pool->active_worker_count =
        header->attention_hvx_contexts - 1U;
    for (uint32_t worker = 0U;
         worker < pool->active_worker_count; ++worker) {
        pool->jobs[worker].command_kind =
            QBH_BLOCK_HVX_POOL_QK_NORM_ROPE;
    }
    asm volatile("barrier" ::: "memory");
    for (uint32_t worker = 0U;
         worker < pool->active_worker_count; ++worker) {
        (void)qurt_sem_up(&pool->command_ready[worker]);
    }
    qbh_attention_qk_norm_pool_run_tasks(pool, &main_job);
    wait_start = HAP_perf_get_qtimer_count();
    qbh_w4f16_pool_wait(pool);
    header->attention_qk_norm_pool_wait_ticks +=
        HAP_perf_get_qtimer_count() - wait_start;
    header->attention_qk_norm_main_work_ticks +=
        main_job.attention_qk_norm_ticks;
    header->attention_qk_norm_task_count +=
        main_job.attention_qk_norm_task_count;
    header->fp16_qk_norm_pair_task_count +=
        main_job.fp16_qk_norm_pair_task_count;
    asm volatile("barrier" ::: "memory");
    return 0;
}

static int qbh_hvx_pool_qk_norm_rope_start_async(
    struct qbh_block_header *header,
    struct qbh_block_w4f16_pool *pool,
    __fp16 *q, __fp16 *k, __fp16 *q_destination,
    __fp16 *k_weight, uint32_t crouton_qkv,
    const __fp16 *q_gamma,
    const __fp16 *k_gamma, const __fp16 *cosine,
    const __fp16 *sine, uint32_t task_base,
    uint32_t task_count, uint32_t first_worker,
    uint32_t worker_count) {
    const uint32_t paired =
        header != NULL &&
        (header->fp16_common_schedule_mode &
         QBH_BLOCK_FP16_COMMON_SCHEDULE_QK_HEAD_PAIRS) != 0U;
    if (header == NULL || pool == NULL || worker_count == 0U ||
        first_worker + worker_count > pool->worker_count ||
        task_base + task_count >
            QBH_BLOCK_HEADS + QBH_BLOCK_KV_HEADS ||
        (paired != 0U &&
         (task_base != 0U ||
          task_count != QBH_BLOCK_HEADS + QBH_BLOCK_KV_HEADS))) {
        return -1;
    }
    pool->attention_q = q;
    pool->attention_header = header;
    pool->attention_k = k;
    pool->attention_q_destination = q_destination;
    pool->attention_k_weight = k_weight;
    pool->attention_crouton_qkv = crouton_qkv;
    pool->attention_q_gamma = q_gamma;
    pool->attention_k_gamma = k_gamma;
    pool->attention_rope_cos = cosine;
    pool->attention_rope_sin = sine;
    pool->fp16_common_schedule_mode =
        header->fp16_common_schedule_mode;
    pool->attention_task_base = task_base;
    pool->attention_task_count =
        header->qkv_schedule_mode ==
                QBH_BLOCK_QKV_SCHEDULE_Q_PREFIX4_K_ALL
            ? QBH_BLOCK_HEADS + QBH_BLOCK_KV_HEADS
            : (paired != 0U
                   ? QBH_BLOCK_HEADS / 2U +
                         QBH_BLOCK_KV_HEADS / 2U
                   : task_count);
    pool->next_attention_task = 0U;
    pool->attention_qk_stream_abort = 0U;
    pool->attention_qk_streaming = 1U;
    ++pool->attention_qk_generation;
    if (pool->attention_qk_generation == 0U) {
        memset((void *)pool->attention_qk_ready, 0,
               sizeof(pool->attention_qk_ready));
        pool->attention_qk_generation = 1U;
    }
    for (uint32_t index = 0U; index < worker_count; ++index) {
        pool->jobs[first_worker + index].command_kind =
            QBH_BLOCK_HVX_POOL_QK_NORM_ROPE;
    }
    asm volatile("barrier" ::: "memory");
    for (uint32_t index = 0U; index < worker_count; ++index) {
        (void)qurt_sem_up(&pool->command_ready[first_worker + index]);
    }
    return 0;
}

static void qbh_hvx_pool_qk_norm_rope_abort_async(
    struct qbh_block_w4f16_pool *pool) {
    if (pool != NULL) {
        pool->attention_qk_stream_abort = 1U;
        asm volatile("barrier" ::: "memory");
    }
}

static void qbh_hvx_pool_qk_norm_rope_publish(
    const struct qbh_block_header *header,
    const struct qbh_block_projection_desc *desc,
    struct qbh_block_w4f16_pool *pool,
    uint32_t first_n_tile, uint32_t n_tiles) {
    uint32_t end_tile;
    uint32_t task;

    if (header->attention_pipeline_mode !=
            QBH_BLOCK_ATTENTION_PIPELINE_GQA_QKV_OVERLAP ||
        pool == NULL ||
        (desc != &header->projections[QBH_BLOCK_PROJ_Q] &&
         desc != &header->projections[QBH_BLOCK_PROJ_K])) {
        return;
    }
    end_tile = first_n_tile + n_tiles;
    if (end_tile == 0U ||
        end_tile % (QBH_BLOCK_HEAD_DIM / QBH_HMX_FP16_COLS) != 0U) {
        return;
    }
    task = end_tile /
        (QBH_BLOCK_HEAD_DIM / QBH_HMX_FP16_COLS) - 1U;
    if (desc == &header->projections[QBH_BLOCK_PROJ_K]) {
        task += QBH_BLOCK_HEADS;
    }
    if (task >= QBH_BLOCK_HEADS + QBH_BLOCK_KV_HEADS) {
        return;
    }
    pool->attention_qk_ready[task] =
        pool->attention_qk_generation;
    asm volatile("release(%0):at"
                 :
                 : "r"(&pool->attention_qk_ready[task])
                 : "memory");
}

static int qbh_hvx_pool_qk_norm_rope_wait_async(
    struct qbh_block_header *header,
    struct qbh_block_w4f16_pool *pool,
    uint32_t first_worker, uint32_t worker_count) {
    uint64_t wait_start;

    if (pool == NULL || worker_count == 0U ||
        first_worker + worker_count > pool->worker_count) {
        return -1;
    }
    wait_start = HAP_perf_get_qtimer_count();
    for (uint32_t index = 0U; index < worker_count; ++index) {
        qurt_sem_down(&pool->command_done[first_worker + index]);
    }
    header->attention_qk_norm_pool_wait_ticks +=
        HAP_perf_get_qtimer_count() - wait_start;
    asm volatile("barrier" ::: "memory");
    return 0;
}

static int qbh_hvx_pool_softmax(
    struct qbh_block_header *header,
    struct qbh_block_w4f16_pool *pool,
    __fp16 *scores, __fp16 *probability) {
    struct qbh_block_w4f16_job main_job;
    uint64_t wait_start;

    if (pool == NULL || header->attention_hvx_contexts != 4U ||
        header->attention_hvx_contexts - 1U > pool->worker_count) {
        return -1;
    }
    memset(&main_job, 0, sizeof(main_job));
    pool->attention_scores = scores;
    pool->attention_probability = probability;
    pool->attention_task_count = QBH_BLOCK_HEADS;
    pool->next_attention_task = 0U;
    pool->active_worker_count =
        header->attention_hvx_contexts - 1U;
    for (uint32_t worker = 0U;
         worker < pool->active_worker_count; ++worker) {
        pool->jobs[worker].command_kind =
            QBH_BLOCK_HVX_POOL_SOFTMAX;
    }
    asm volatile("barrier" ::: "memory");
    for (uint32_t worker = 0U;
         worker < pool->active_worker_count; ++worker) {
        (void)qurt_sem_up(&pool->command_ready[worker]);
    }
    qbh_attention_softmax_pool_run_tasks(pool, &main_job);
    wait_start = HAP_perf_get_qtimer_count();
    qbh_w4f16_pool_wait(pool);
    header->attention_softmax_pool_wait_ticks +=
        HAP_perf_get_qtimer_count() - wait_start;
    header->attention_softmax_main_work_ticks +=
        main_job.attention_softmax_ticks;
    header->attention_softmax_task_count +=
        main_job.attention_softmax_task_count;
    asm volatile("barrier" ::: "memory");
    return 0;
}

static int qbh_mlp_stream_pipeline_start(
    struct qbh_block_header *header,
    struct qbh_block_w4f16_pool *pool,
    const __fp16 *gate, const __fp16 *up, __fp16 *middle,
    __fp16 *hmx_activation) {
    uint32_t first_worker;
    uint32_t worker_count;

    if (header->mlp_mode != QBH_BLOCK_MLP_STREAMING &&
        header->mlp_mode != QBH_BLOCK_MLP_CROUTON_NATIVE &&
        header->mlp_mode != QBH_BLOCK_MLP_CROUTON_NATIVE_BATCH8) {
        return 0;
    }
    if (pool == NULL || header->mlp_hvx_contexts != 4U) {
        return -1;
    }
    if (header->variant == QBH_BLOCK_W4F16) {
        first_worker = 2U;
        worker_count = 1U;
    } else if (header->variant == QBH_BLOCK_F16F16) {
        first_worker = 0U;
        worker_count = header->mlp_hvx_contexts - 1U;
    } else {
        return -1;
    }
    if (first_worker + worker_count > pool->worker_count) {
        return -1;
    }

    pool->mlp_gate = gate;
    pool->mlp_up = up;
    pool->mlp_middle = middle;
    pool->mlp_hmx_activation = hmx_activation;
    pool->next_mlp_group = 0U;
    pool->mlp_stream_abort = 0U;
    pool->mlp_crouton_native =
        header->mlp_mode == QBH_BLOCK_MLP_CROUTON_NATIVE ||
        header->mlp_mode == QBH_BLOCK_MLP_CROUTON_NATIVE_BATCH8;
    pool->mlp_crouton_group_tiles =
        header->mlp_mode == QBH_BLOCK_MLP_CROUTON_NATIVE_BATCH8
            ? 8U : QBH_BLOCK_MLP_CROUTON_GROUP_TILES;
    pool->mlp_crouton_stream_group_tiles =
        header->variant == QBH_BLOCK_W4F16 &&
        header->mlp_mode == QBH_BLOCK_MLP_CROUTON_NATIVE_BATCH8
            ? header->w4f16_gate_up_stream_group_tiles
            : pool->mlp_crouton_group_tiles;
    pool->mlp_stream_group_limit = pool->mlp_crouton_native != 0U
        ? QBH_BLOCK_INTERMEDIATE / QBH_HMX_FP16_COLS /
              pool->mlp_crouton_stream_group_tiles
        : QBH_BLOCK_MLP_STREAM_GROUPS;
    pool->mlp_crouton_slot_elements =
        (QBH_BLOCK_M / QBH_HMX_FP16_ROWS) *
        pool->mlp_crouton_group_tiles *
        QBH_HMX_FP16_TILE_ELEMENTS;
    memset((void *)pool->mlp_crouton_slot_consumed, 0,
           sizeof(pool->mlp_crouton_slot_consumed));
    memset((void *)pool->mlp_crouton_subgroup_done, 0,
           sizeof(pool->mlp_crouton_subgroup_done));
    ++pool->mlp_stream_generation;
    if (pool->mlp_stream_generation == 0U) {
        memset((void *)pool->mlp_up_group_ready, 0,
               sizeof(pool->mlp_up_group_ready));
        pool->mlp_stream_generation = 1U;
    }
    pool->mlp_stream_first_worker = first_worker;
    pool->mlp_stream_worker_count = worker_count;
    for (uint32_t index = 0U; index < worker_count; ++index) {
        pool->jobs[first_worker + index].command_kind =
            QBH_BLOCK_HVX_POOL_MLP_STREAM;
    }
    asm volatile("barrier" ::: "memory");
    for (uint32_t index = 0U; index < worker_count; ++index) {
        (void)qurt_sem_up(
            &pool->command_ready[first_worker + index]);
    }
    return 0;
}

static int qbh_mlp_stream_publish_up_group(
    const struct qbh_block_header *header,
    const struct qbh_block_projection_desc *desc,
    struct qbh_block_w4f16_pool *pool,
    uint32_t first_n_tile, uint32_t n_tiles) {
    uint32_t group;

    if (header->mlp_mode != QBH_BLOCK_MLP_STREAMING ||
        desc != &header->projections[QBH_BLOCK_PROJ_UP]) {
        return 0;
    }
    if (pool == NULL ||
        (n_tiles != 2U && n_tiles != 4U) ||
        (first_n_tile & 1U) != 0U) {
        return -1;
    }
    for (uint32_t tile = 0U; tile < n_tiles; tile += 2U) {
        group = (first_n_tile + tile) / 2U;
        if (group >= QBH_BLOCK_MLP_STREAM_GROUPS) {
            return -1;
        }
        pool->mlp_up_group_ready[group] =
            pool->mlp_stream_generation;
        asm volatile("release(%0):at"
                     :
                     : "r"(&pool->mlp_up_group_ready[group])
                     : "memory");
    }
    return 0;
}

static int qbh_mlp_crouton_publish_up_group(
    const struct qbh_block_header *header,
    const struct qbh_block_projection_desc *desc,
    struct qbh_block_w4f16_pool *pool, uint32_t group) {
    if ((header->mlp_mode != QBH_BLOCK_MLP_CROUTON_NATIVE &&
         header->mlp_mode != QBH_BLOCK_MLP_CROUTON_NATIVE_BATCH8) ||
        desc != &header->projections[QBH_BLOCK_PROJ_UP]) {
        return 0;
    }
    uint32_t subgroups_per_hmx;
    uint32_t first_stream_group;

    if (pool == NULL || pool->mlp_crouton_stream_group_tiles == 0U) {
        return -1;
    }
    subgroups_per_hmx = pool->mlp_crouton_group_tiles /
        pool->mlp_crouton_stream_group_tiles;
    if (subgroups_per_hmx == 0U ||
        group >= pool->mlp_stream_group_limit / subgroups_per_hmx) {
        return -1;
    }
    first_stream_group = group * subgroups_per_hmx;
    for (uint32_t subgroup = 0U;
         subgroup < subgroups_per_hmx; ++subgroup) {
        const uint32_t stream_group = first_stream_group + subgroup;
        pool->mlp_up_group_ready[stream_group] =
            pool->mlp_stream_generation;
        asm volatile("release(%0):at"
                     :
                     : "r"(&pool->mlp_up_group_ready[stream_group])
                     : "memory");
    }
    return 0;
}

static int qbh_mlp_crouton_wait_slot(
    struct qbh_block_header *header,
    struct qbh_block_w4f16_pool *pool, uint32_t group) {
    uint64_t wait_start;
    uint32_t slot;
    uint32_t expected;

    if ((header->mlp_mode != QBH_BLOCK_MLP_CROUTON_NATIVE &&
         header->mlp_mode != QBH_BLOCK_MLP_CROUTON_NATIVE_BATCH8) ||
        group < QBH_BLOCK_MLP_CROUTON_RING_SLOTS) {
        return 0;
    }
    if (pool == NULL || pool->mlp_crouton_native == 0U) {
        return -1;
    }
    slot = group % QBH_BLOCK_MLP_CROUTON_RING_SLOTS;
    expected = group - QBH_BLOCK_MLP_CROUTON_RING_SLOTS + 1U;
    wait_start = HAP_perf_get_qtimer_count();
    while (pool->mlp_crouton_slot_consumed[slot] != expected) {
        if (pool->mlp_stream_abort != 0U ||
            HAP_perf_get_qtimer_count() - wait_start >
                QBH_BLOCK_DMA_DESCRIPTOR_TIMEOUT_TICKS) {
            pool->mlp_stream_abort = 1U;
            header->mlp_stream_ready_wait_ticks +=
                HAP_perf_get_qtimer_count() - wait_start;
            return -1;
        }
        asm volatile("pause(#8)" : : : "memory");
    }
    asm volatile("barrier" ::: "memory");
    header->mlp_stream_ready_wait_ticks +=
        HAP_perf_get_qtimer_count() - wait_start;
    return 0;
}

static int qbh_mlp_stream_pipeline_wait(
    struct qbh_block_header *header,
    struct qbh_block_w4f16_pool *pool, uint32_t abort_pipeline) {
    struct qbh_block_w4f16_job main_job;
    uint64_t wait_start;

    if (header->mlp_mode != QBH_BLOCK_MLP_STREAMING &&
        header->mlp_mode != QBH_BLOCK_MLP_CROUTON_NATIVE &&
        header->mlp_mode != QBH_BLOCK_MLP_CROUTON_NATIVE_BATCH8) {
        return 0;
    }
    if (pool == NULL || pool->mlp_stream_worker_count == 0U) {
        return -1;
    }
    if (abort_pipeline != 0U) {
        pool->mlp_stream_abort = 1U;
        asm volatile("barrier" ::: "memory");
    } else if (header->variant == QBH_BLOCK_W4F16 &&
               pool->mlp_stream_first_worker == 2U &&
               pool->mlp_stream_worker_count == 1U) {
        const uint32_t extra_stream_worker =
            header->w4f16_gate_up_extra_stream_worker != 0U
                ? 3U : UINT32_MAX;
        memset(&main_job, 0, sizeof(main_job));
        pool->jobs[0].command_kind =
            QBH_BLOCK_HVX_POOL_MLP_STREAM;
        pool->jobs[1].command_kind =
            QBH_BLOCK_HVX_POOL_MLP_STREAM;
        if (extra_stream_worker != UINT32_MAX) {
            pool->jobs[extra_stream_worker].command_kind =
                QBH_BLOCK_HVX_POOL_MLP_STREAM;
        }
        asm volatile("barrier" ::: "memory");
        (void)qurt_sem_up(&pool->command_ready[0]);
        (void)qurt_sem_up(&pool->command_ready[1]);
        if (extra_stream_worker != UINT32_MAX) {
            (void)qurt_sem_up(
                &pool->command_ready[extra_stream_worker]);
        }
        pool->mlp_stream_first_worker = 0U;
        pool->mlp_stream_worker_count =
            extra_stream_worker != UINT32_MAX ? 4U : 3U;
        qbh_mlp_stream_worker_run(pool, &main_job);
        header->mlp_stream_main_work_ticks +=
            main_job.stream_ticks;
        header->mlp_stream_ready_wait_ticks +=
            main_job.stream_ready_wait_ticks;
        header->mlp_stream_group_count +=
            main_job.stream_group_count;
    }
    wait_start = HAP_perf_get_qtimer_count();
    for (uint32_t index = 0U;
         index < pool->mlp_stream_worker_count; ++index) {
        qurt_sem_down(&pool->command_done[
            pool->mlp_stream_first_worker + index]);
    }
    header->mlp_stream_join_wait_ticks +=
        HAP_perf_get_qtimer_count() - wait_start;
    asm volatile("barrier" ::: "memory");
    pool->mlp_stream_worker_count = 0U;
    return abort_pipeline == 0U ? 0 : -1;
}

static void qbh_w4f16_expand_with_main(
    struct qbh_block_header *header,
    struct qbh_block_w4f16_pool *pool,
    const uint8_t *compressed_weight, const float *channel_scale,
    uint8_t *expanded_weight, volatile uint32_t *ready_generations,
    uint32_t expected_generation, uint32_t k_tiles,
    uint32_t region_tiles, uint32_t active_worker_count,
    uint32_t publish_ready, uint32_t relaxed_group_fence) {
    uint32_t total_regions = k_tiles / region_tiles;
    const uint32_t extra_expand_worker =
        relaxed_group_fence != 0U &&
                header->w4f16_gate_up_extra_expand_worker != 0U
            ? 3U : UINT32_MAX;
    const uint32_t expansion_worker_count =
        active_worker_count +
        (extra_expand_worker != UINT32_MAX ? 1U : 0U);
    uint32_t main_regions;
    uint32_t main_tiles;
    uint32_t pool_regions;
    uint64_t main_start;

    if (header->w4f16_pipeline_mode ==
            QBH_BLOCK_W4F16_PIPELINE_MAIN_HALF) {
        main_regions = (total_regions + 1U) / 2U;
    } else if (header->w4f16_pipeline_mode ==
                   QBH_BLOCK_W4F16_PIPELINE_MAIN_TWO_THIRDS) {
        main_regions = (2U * total_regions + 2U) / 3U;
    } else {
        main_regions =
            (total_regions + expansion_worker_count) /
            (expansion_worker_count + 1U);
    }
    main_tiles = main_regions * region_tiles;
    pool_regions = total_regions - main_regions;

    qbh_w4f16_pool_start(
        pool,
        compressed_weight +
            (size_t)main_tiles * QBH_W4_PACKED_TILE_BYTES,
        channel_scale,
        expanded_weight +
            (size_t)main_tiles * QBH_HMX_FP16_TILE_BYTES,
        ready_generations + main_regions, expected_generation,
        pool_regions, region_tiles,
        active_worker_count,
        publish_ready != 0U || relaxed_group_fence == 0U,
        relaxed_group_fence != 0U
            ? header->w4f16_expand_claim_regions : 1U,
        extra_expand_worker);
    main_start = HAP_perf_get_qtimer_count();
    if (publish_ready == 0U && relaxed_group_fence != 0U) {
        qbh_unpack_w4_to_f16_hvx_relaxed(
            compressed_weight, expanded_weight, main_tiles);
    } else if (publish_ready == 0U) {
        qbh_unpack_w4_to_f16_hvx(
            compressed_weight, expanded_weight, main_tiles);
    } else {
        for (uint32_t region = 0; region < main_regions; ++region) {
            qbh_unpack_w4_to_f16_hvx(
                compressed_weight +
                    (size_t)region * region_tiles *
                        QBH_W4_PACKED_TILE_BYTES,
                expanded_weight +
                    (size_t)region * region_tiles *
                        QBH_HMX_FP16_TILE_BYTES,
                region_tiles);
            ready_generations[region] = expected_generation;
            asm volatile("release(%0):at"
                         :
                         : "r"(&ready_generations[region])
                         : "memory");
        }
    }
    header->w4f16_expand_work_ticks +=
        HAP_perf_get_qtimer_count() - main_start;
    header->w4f16_expand_region_count += main_regions;
    {
        uint64_t wait_start = HAP_perf_get_qtimer_count();
        qbh_w4f16_pool_wait(pool);
        header->w4f16_expand_pool_wait_ticks +=
            HAP_perf_get_qtimer_count() - wait_start;
    }
}

static int qbh_w4f16_pool_destroy(
    struct qbh_block_w4f16_pool *pool) {
    int result = 0;
    pool->stop = 1U;
    pool->attention_qk_stream_abort = 1U;
    asm volatile("barrier" ::: "memory");
    for (uint32_t worker = 0; worker < pool->created_workers;
         ++worker) {
        int exit_status = 0;
        (void)qurt_sem_up(&pool->command_ready[worker]);
        if (qurt_thread_join(pool->threads[worker], &exit_status) !=
                QURT_EOK ||
            exit_status != AEE_SUCCESS ||
            pool->jobs[worker].unlock_status != AEE_SUCCESS) {
            result = -1;
        }
    }
    for (uint32_t worker = 0; worker < pool->worker_count; ++worker) {
        qurt_sem_destroy(&pool->worker_started[worker]);
        qurt_sem_destroy(&pool->command_done[worker]);
        qurt_sem_destroy(&pool->command_ready[worker]);
    }
    qurt_mutex_destroy(&pool->attention_hmx_mutex);
    return result;
}

static void qbh_pack_fp16_activation(const __fp16 *source,
                                     uint32_t source_stride,
                                     uint32_t k, __fp16 *destination) {
    uint32_t k_tiles = k / QBH_HMX_FP16_COLS;
    for (uint32_t row = 0; row < QBH_BLOCK_M; row += 2U) {
        uint32_t row_tile = row / QBH_HMX_FP16_ROWS;
        uint32_t row_pair = (row % QBH_HMX_FP16_ROWS) / 2U;
        const HVX_Vector *source0 = (const HVX_Vector *)(
            source + (size_t)row * source_stride);
        const HVX_Vector *source1 = (const HVX_Vector *)(
            source + (size_t)(row + 1U) * source_stride);

        /* Adapted from htp-ops-lib@85eb88e FP16 FlashAttention packing.
         * Two 64-column row vectors become two adjacent Crouton tiles. */
        for (uint32_t channel = 0; channel < k; channel += 64U) {
            HVX_VectorPair packed = Q6_W_vshuff_VVR(
                *source1++, *source0++, -2);
            size_t tile = qbh_hmx_fp16_matrix_tile_offset(
                row_tile, channel / QBH_HMX_FP16_COLS, k_tiles);
            HVX_Vector *output0 = (HVX_Vector *)(
                destination + tile) + row_pair;
            HVX_Vector *output1 = output0 +
                QBH_HMX_FP16_TILE_BYTES / sizeof(HVX_Vector);
            *output0 = Q6_V_lo_W(packed);
            *output1 = Q6_V_hi_W(packed);
        }
    }
    asm volatile("barrier" ::: "memory");
}

/* EXP-0164 only consumes one logical row in the LM head.  Pack that row as
 * HMX row zero and explicitly zero its paired row; the remaining physical
 * rows are outside the selected-logit contract and are never inspected. */
static void qbh_pack_fp16_activation_row0(
    const __fp16 *source, uint32_t k, __fp16 *destination) {
    const uint32_t k_tiles = k / QBH_HMX_FP16_COLS;
    const HVX_Vector zero = Q6_V_vzero();
    const HVX_Vector *input = (const HVX_Vector *)source;

    for (uint32_t channel = 0U; channel < k; channel += 64U) {
        HVX_VectorPair packed = Q6_W_vshuff_VVR(zero, *input++, -2);
        const size_t tile = qbh_hmx_fp16_matrix_tile_offset(
            0U, channel / QBH_HMX_FP16_COLS, k_tiles);
        HVX_Vector *output0 = (HVX_Vector *)(destination + tile);
        HVX_Vector *output1 = output0 +
            QBH_HMX_FP16_TILE_BYTES / sizeof(HVX_Vector);
        *output0 = Q6_V_lo_W(packed);
        *output1 = Q6_V_hi_W(packed);
    }
    asm volatile("barrier" ::: "memory");
}

static float qbh_hvx_reduce_max_f16(HVX_Vector value) {
    const HVX_Vector negative_max = Q6_Vh_vsplat_R(0xfbff);
    __fp16 lanes[QBH_BLOCK_HVX_F16_LANES]
        __attribute__((aligned(QBH_BLOCK_ALIGNMENT)));

    for (int shift = 64; shift >= 2; shift >>= 1) {
        value = Q6_Vhf_vmax_VhfVhf(
            value, Q6_V_vlalign_VVR(value, negative_max, shift));
    }
    *(HVX_Vector *)lanes = value;
    return (float)lanes[QBH_BLOCK_HVX_F16_LANES - 1U];
}

static int qbh_stage_generation_embedding(
    struct qbh_block_header *header, uint8_t *shared,
    struct qbh_block_buffers *buffers) {
    const uint32_t token_bytes =
        QBH_BLOCK_M * (uint32_t)sizeof(uint32_t);
    const uint32_t embedding_row_bytes = QBH_BLOCK_HIDDEN *
        (qbh_generation_w4u8_enabled(header->generation_mode)
             ? (uint32_t)sizeof(uint8_t)
             : (uint32_t)sizeof(uint16_t));
    const uint64_t start = HAP_perf_get_qtimer_count();
    uint32_t *token_ids = (uint32_t *)buffers->scale_or_bias;

    if (!qbh_generation_enabled(header->generation_mode) ||
        qbh_dma_copy(header, token_ids,
                     shared + header->generation_token_ids_offset,
                     token_bytes, 1U) != 0) {
        return -1;
    }
    header->generation_input_token_count_observed =
        header->generation_token_count;
    header->generation_embedding_ddr_read_bytes += token_bytes;
    header->boundary_ddr_read_bytes += token_bytes;
    ++header->boundary_dma_descriptor_count;
    for (uint32_t row = 0U; row < header->generation_token_count; ++row) {
        const uint32_t token = token_ids[row];
        if (token >= QBH_QWEN3_VOCAB_SIZE ||
            qbh_dma_copy(
                header,
                buffers->residual + (size_t)row * embedding_row_bytes,
                shared + header->generation_embedding_offset +
                    (size_t)token * embedding_row_bytes,
                embedding_row_bytes, 1U) != 0) {
            return -1;
        }
        header->generation_embedding_ddr_read_bytes +=
            embedding_row_bytes;
        header->boundary_ddr_read_bytes += embedding_row_bytes;
        ++header->boundary_dma_descriptor_count;
    }
    header->generation_embedding_ticks +=
        HAP_perf_get_qtimer_count() - start;
    return 0;
}

static void qbh_generation_hvx_argmax_group(
    const __fp16 *hmx_output, uint32_t n_tile, uint32_t group_tiles,
    float *best_value, uint32_t *best_token, uint16_t *best_bits) {
    for (uint32_t tile_pair = 0U; tile_pair < group_tiles;
         tile_pair += 2U) {
        const size_t matrix_tile = qbh_hmx_fp16_matrix_tile_offset(
            0U, tile_pair, group_tiles);
        const HVX_Vector *tile0 =
            (const HVX_Vector *)(hmx_output + matrix_tile);
        const HVX_Vector *tile1 = tile0 +
            QBH_HMX_FP16_TILE_BYTES / sizeof(HVX_Vector);
        const HVX_VectorPair rows = Q6_W_vdeal_VVR(
            *tile1, *tile0, -2);
        const HVX_Vector row0 = Q6_V_lo_W(rows);
        const float group_max = qbh_hvx_reduce_max_f16(row0);

        if (group_max > *best_value) {
            __fp16 lanes[QBH_BLOCK_HVX_F16_LANES]
                __attribute__((aligned(QBH_BLOCK_ALIGNMENT)));
            *(HVX_Vector *)lanes = row0;
            for (uint32_t lane = 0U;
                 lane < QBH_BLOCK_HVX_F16_LANES; ++lane) {
                const __fp16 candidate = lanes[lane];
                const float value = (float)candidate;
                if (value > *best_value) {
                    *best_value = value;
                    *best_token =
                        (n_tile + tile_pair) * QBH_HMX_FP16_COLS + lane;
                    memcpy(best_bits, &candidate, sizeof(*best_bits));
                }
            }
        }
    }
}

/* EXP-0166 keeps EXP-0165's batch-eight HMX and exact argmax math. Its
 * diagnostic modes compare region-streamed HMX, DMA/HVX overlap, and a
 * coarse double-buffer pipeline. The selected coarse pipeline overlaps the
 * next W4 DMA, current HVX expansion, previous HMX command, and previous
 * argmax without per-region synchronization. Raw scales are resident in a
 * dead Up buffer; HMX scale carriers reuse dead attention-phase VTCM. */
static int qbh_run_generation_head_w4f16_overlap(
    struct qbh_block_header *header, uint8_t *shared,
    struct qbh_block_buffers *buffers,
    struct qbh_block_hmx_worker *worker,
    struct qbh_block_w4f16_pool *pool,
    uint32_t logical_rows, uint32_t generation_step) {
    const struct qbh_block_projection_desc *head =
        &header->generation_lm_head;
    const uint32_t k_tiles = QBH_BLOCK_HIDDEN / QBH_HMX_FP16_COLS;
    const uint32_t n_tiles = QBH_QWEN3_VOCAB_SIZE /
        QBH_HMX_FP16_COLS;
    const uint32_t group_limit = 8U;
    const uint32_t dma_batch_tiles =
        header->generation_mode ==
                QBH_BLOCK_GENERATION_GREEDY_W4F16_LM_HEAD_OVERLAP
            ? 16U : 8U;
    const uint32_t compressed_tile_bytes =
        k_tiles * QBH_W4_PACKED_TILE_BYTES;
    const uint32_t compressed_batch_bytes =
        dma_batch_tiles * compressed_tile_bytes;
    const uint32_t expanded_group_bytes =
        group_limit * k_tiles * QBH_HMX_FP16_TILE_BYTES;
    const uint32_t hidden_bytes =
        QBH_BLOCK_M * QBH_BLOCK_HIDDEN * sizeof(uint16_t);
    const uint32_t intermediate_bytes =
        QBH_BLOCK_M * QBH_BLOCK_INTERMEDIATE * sizeof(uint16_t);
    const uint64_t head_start = HAP_perf_get_qtimer_count();
    uint8_t *compressed_slots[2] = {
        buffers->gate,
        header->generation_mode ==
                QBH_BLOCK_GENERATION_GREEDY_W4F16_COARSE_PIPELINE
            ? buffers->attention_projection
            : buffers->attention_concat};
    uint8_t *expanded_slots[2] = {
        buffers->expanded_weight, buffers->expanded_weight_alt};
    uint8_t *scale_blocks = buffers->middle;
    uint8_t *scale_table = buffers->residual +
        4U * QBH_HMX_FP16_SCALE_BYTES;
    float *resident_scales = (float *)buffers->up;
    volatile uint32_t ready[QBH_BLOCK_W4F16_MAX_REGIONS];
    struct qbh_dma_aligned_desc_1d prefetch_descriptor
        __attribute__((aligned(64)));
    uint64_t prefetch_start = 0U;
    uint32_t prefetch_active = 0U;
    const uint32_t stream_hmx =
        header->generation_mode ==
            QBH_BLOCK_GENERATION_GREEDY_W4F16_LM_HEAD_OVERLAP;
    const uint32_t coarse_pipeline =
        header->generation_mode ==
            QBH_BLOCK_GENERATION_GREEDY_W4F16_COARSE_PIPELINE;
    float best_value = -INFINITY;
    uint32_t best_token = 0U;
    uint16_t best_bits = 0U;
    const __fp16 *last_hmx_output = NULL;
    uint32_t last_n_tile = 0U;
    uint32_t last_group_tiles = 0U;
    int result;

    if (pool == NULL || logical_rows == 0U ||
        generation_step >= header->generation_expected_token_count ||
        (uint8_t *)buffers->attention_projection !=
            buffers->attention_concat + hidden_bytes ||
        head->scale_bytes > intermediate_bytes ||
        compressed_batch_bytes > intermediate_bytes ||
        expanded_group_bytes >
            QBH_BLOCK_HIDDEN * group_limit * QBH_HMX_FP16_COLS *
                sizeof(uint16_t) ||
        (coarse_pipeline != 0U &&
         (size_t)(buffers->attention_projection - scale_table) <
             (size_t)n_tiles * QBH_HMX_FP16_SCALE_BYTES)) {
        return -1;
    }
    {
        const uint64_t norm_start = HAP_perf_get_qtimer_count();
        if (qbh_dma_copy(
                header, buffers->input_norm_weight,
                shared + header->generation_final_norm_offset,
                header->generation_final_norm_bytes, 1U) != 0) {
            return -2;
        }
        header->weight_ddr_read_bytes +=
            header->generation_final_norm_bytes;
        qbh_hvx_rms_norm_f16(
            (const __fp16 *)buffers->residual,
            (const __fp16 *)buffers->input_norm_weight,
            (__fp16 *)buffers->normalized, logical_rows,
            QBH_BLOCK_HIDDEN, NULL);
        header->generation_final_norm_ticks +=
            HAP_perf_get_qtimer_count() - norm_start;
    }
    qbh_pack_fp16_activation_row0(
        (const __fp16 *)buffers->normalized +
            (size_t)(logical_rows - 1U) * QBH_BLOCK_HIDDEN,
        QBH_BLOCK_HIDDEN, (__fp16 *)buffers->hmx_activation);

    header->generation_lm_head_batch_n_tiles = group_limit;
    header->generation_lm_head_n_tiles = n_tiles;
    memset((void *)ready, 0, sizeof(ready));
    {
        const uint64_t scale_start = HAP_perf_get_qtimer_count();
        if (qbh_dma_copy(
                header, resident_scales,
                shared + head->scale_offset, head->scale_bytes, 1U) != 0) {
            return -3;
        }
        header->generation_lm_head_scale_dma_ticks +=
            HAP_perf_get_qtimer_count() - scale_start;
        header->generation_lm_head_scale_resident_bytes =
            head->scale_bytes;
        header->generation_lm_head_ddr_read_bytes += head->scale_bytes;
        header->weight_ddr_read_bytes += head->scale_bytes;
        ++header->weight_dma_descriptor_count;
    }
    if (coarse_pipeline != 0U) {
        const uint64_t scale_init_start =
            HAP_perf_get_qtimer_count();
        qbh_hmx_fp16_init_channel_scale_tiles(
            scale_table, resident_scales, n_tiles);
        header->generation_lm_head_scale_init_ticks +=
            HAP_perf_get_qtimer_count() - scale_init_start;
    }
    {
        uint32_t first_tiles = n_tiles;
        uint64_t dma_start;
        if (first_tiles > dma_batch_tiles) {
            first_tiles = dma_batch_tiles;
        }
        dma_start = HAP_perf_get_qtimer_count();
        if (qbh_dma_copy(
                header, compressed_slots[0],
                shared + head->weight_offset,
                first_tiles * compressed_tile_bytes, 1U) != 0) {
            return -4;
        }
        header->generation_lm_head_weight_dma_ticks +=
            HAP_perf_get_qtimer_count() - dma_start;
        header->generation_lm_head_weight_dma_wait_ticks +=
            HAP_perf_get_qtimer_count() - dma_start;
        header->generation_lm_head_ddr_read_bytes +=
            first_tiles * compressed_tile_bytes;
        header->weight_ddr_read_bytes +=
            first_tiles * compressed_tile_bytes;
        ++header->weight_dma_descriptor_count;
    }
    if (n_tiles > dma_batch_tiles) {
        uint32_t next_tiles = n_tiles - dma_batch_tiles;
        if (next_tiles > dma_batch_tiles) {
            next_tiles = dma_batch_tiles;
        }
        prefetch_start = HAP_perf_get_qtimer_count();
        result = qbh_dma_start_weight_prefetch(
            &prefetch_descriptor, compressed_slots[1],
            shared + head->weight_offset +
                (size_t)dma_batch_tiles * compressed_tile_bytes,
            next_tiles * compressed_tile_bytes);
        if (result != 0) {
            return -5;
        }
        prefetch_active = 1U;
        ++header->generation_lm_head_prefetch_count;
        header->generation_lm_head_ddr_read_bytes +=
            next_tiles * compressed_tile_bytes;
        header->weight_ddr_read_bytes +=
            next_tiles * compressed_tile_bytes;
        ++header->weight_dma_descriptor_count;
    }

    if (coarse_pipeline != 0U) {
        uint64_t hmx_command_start = 0U;
        uint32_t hmx_active = 0U;
        const __fp16 *previous_output = NULL;
        uint32_t previous_n_tile = 0U;
        uint32_t previous_group_tiles = 0U;

        for (uint32_t n_tile = 0U; n_tile < n_tiles;
             n_tile += group_limit) {
            const uint32_t group_index = n_tile / group_limit;
            const uint32_t slot = group_index & 1U;
            uint32_t group_tiles = n_tiles - n_tile;
            uint8_t *group_scales = scale_blocks +
                (size_t)slot * group_limit *
                    QBH_HMX_FP16_SCALE_BYTES;
            uint8_t *group_output = slot == 0U
                ? buffers->hmx_output
                : scale_blocks + 2U * group_limit *
                    QBH_HMX_FP16_SCALE_BYTES;
            uint64_t stage_start;

            if (group_tiles > group_limit) {
                group_tiles = group_limit;
            }
            if (n_tile != 0U) {
                const uint64_t wait_start =
                    HAP_perf_get_qtimer_count();
                result = qbh_dma_wait_weight_prefetch(
                    &prefetch_descriptor);
                header->generation_lm_head_weight_dma_wait_ticks +=
                    HAP_perf_get_qtimer_count() - wait_start;
                header->generation_lm_head_weight_dma_ticks +=
                    HAP_perf_get_qtimer_count() - prefetch_start;
                header->weight_dma_ticks +=
                    HAP_perf_get_qtimer_count() - prefetch_start;
                prefetch_active = 0U;
                if (result != 0) {
                    if (hmx_active != 0U) {
                        (void)qbh_hmx_wait(worker);
                    }
                    return -10;
                }
            }
            if (prefetch_active == 0U &&
                n_tile + group_limit < n_tiles) {
                uint32_t next_tiles = n_tiles -
                    (n_tile + group_limit);
                if (next_tiles > group_limit) {
                    next_tiles = group_limit;
                }
                prefetch_start = HAP_perf_get_qtimer_count();
                result = qbh_dma_start_weight_prefetch(
                    &prefetch_descriptor,
                    compressed_slots[(group_index + 1U) & 1U],
                    shared + head->weight_offset +
                        (size_t)(n_tile + group_limit) *
                            compressed_tile_bytes,
                    next_tiles * compressed_tile_bytes);
                if (result != 0) {
                    if (hmx_active != 0U) {
                        (void)qbh_hmx_wait(worker);
                    }
                    return -11;
                }
                prefetch_active = 1U;
                ++header->generation_lm_head_prefetch_count;
                header->generation_lm_head_ddr_read_bytes +=
                    next_tiles * compressed_tile_bytes;
                header->weight_ddr_read_bytes +=
                    next_tiles * compressed_tile_bytes;
                ++header->weight_dma_descriptor_count;
            }

            stage_start = HAP_perf_get_qtimer_count();
            {
                const HVX_Vector *scale_source =
                    (const HVX_Vector *)(scale_table +
                        (size_t)n_tile *
                            QBH_HMX_FP16_SCALE_BYTES);
                HVX_Vector *scale_destination =
                    (HVX_Vector *)group_scales;
                const uint32_t scale_vectors = group_tiles *
                    (QBH_HMX_FP16_SCALE_BYTES /
                     sizeof(HVX_Vector));
                for (uint32_t vector = 0U;
                     vector < scale_vectors; ++vector) {
                    scale_destination[vector] = scale_source[vector];
                }
            }
            header->generation_lm_head_scale_init_ticks +=
                HAP_perf_get_qtimer_count() - stage_start;

            stage_start = HAP_perf_get_qtimer_count();
            qbh_w4f16_expand_with_main(
                header, pool, compressed_slots[slot],
                resident_scales +
                    (size_t)n_tile * QBH_HMX_FP16_COLS,
                expanded_slots[slot], ready, group_index + 1U,
                k_tiles * group_tiles, header->w4f16_region_tiles,
                pool->worker_count, 0U, 0U);
            header->generation_lm_head_expand_ticks +=
                HAP_perf_get_qtimer_count() - stage_start;

            if (hmx_active != 0U) {
                const uint64_t tail_start =
                    HAP_perf_get_qtimer_count();
                result = qbh_hmx_wait(worker);
                header->generation_lm_head_hmx_tail_wait_ticks +=
                    HAP_perf_get_qtimer_count() - tail_start;
                header->generation_lm_head_hmx_ticks +=
                    HAP_perf_get_qtimer_count() - hmx_command_start;
                hmx_active = 0U;
                if (result != 0) {
                    if (prefetch_active != 0U) {
                        (void)qbh_dma_wait_weight_prefetch(
                            &prefetch_descriptor);
                    }
                    return -12;
                }
            }

            hmx_command_start = HAP_perf_get_qtimer_count();
            qbh_hmx_start_fp16_tile_scales(
                worker, buffers->hmx_activation,
                expanded_slots[slot], group_scales, group_output,
                1U, k_tiles, group_tiles);
            hmx_active = 1U;
            ++header->generation_lm_head_command_count;
            ++header->hmx_command_count;
            header->hmx_fp16_tile_pair_count +=
                k_tiles * group_tiles;

            if (previous_output != NULL) {
                stage_start = HAP_perf_get_qtimer_count();
                qbh_generation_hvx_argmax_group(
                    previous_output, previous_n_tile,
                    previous_group_tiles, &best_value,
                    &best_token, &best_bits);
                header->generation_lm_head_argmax_ticks +=
                    HAP_perf_get_qtimer_count() - stage_start;
            }
            previous_output = (const __fp16 *)group_output;
            previous_n_tile = n_tile;
            previous_group_tiles = group_tiles;
        }
        if (hmx_active != 0U) {
            const uint64_t tail_start = HAP_perf_get_qtimer_count();
            result = qbh_hmx_wait(worker);
            header->generation_lm_head_hmx_tail_wait_ticks +=
                HAP_perf_get_qtimer_count() - tail_start;
            header->generation_lm_head_hmx_ticks +=
                HAP_perf_get_qtimer_count() - hmx_command_start;
            if (result != 0) {
                return -13;
            }
        }
        if (previous_output != NULL) {
            const uint64_t argmax_start =
                HAP_perf_get_qtimer_count();
            qbh_generation_hvx_argmax_group(
                previous_output, previous_n_tile,
                previous_group_tiles, &best_value,
                &best_token, &best_bits);
            header->generation_lm_head_argmax_ticks +=
                HAP_perf_get_qtimer_count() - argmax_start;
        }
        goto generation_complete;
    }

    for (uint32_t n_tile = 0U; n_tile < n_tiles;
         n_tile += group_limit) {
        const uint32_t group_index = n_tile / group_limit;
        const uint32_t batch_index = n_tile / dma_batch_tiles;
        const uint32_t in_batch = n_tile % dma_batch_tiles;
        const uint32_t compressed_slot = batch_index & 1U;
        const uint32_t expanded_slot =
            stream_hmx != 0U ? group_index & 1U : 0U;
        uint8_t *hmx_output = stream_hmx != 0U ||
                (group_index & 1U) == 0U
            ? buffers->hmx_output : buffers->expanded_weight_alt;
        uint32_t group_tiles = n_tiles - n_tile;
        uint64_t stage_start;
        uint64_t command_start;
        uint64_t tail_wait_start;

        if (group_tiles > group_limit) {
            group_tiles = group_limit;
        }
        if (n_tile != 0U && in_batch == 0U) {
            uint64_t wait_start = HAP_perf_get_qtimer_count();
            result = qbh_dma_wait_weight_prefetch(
                &prefetch_descriptor);
            header->generation_lm_head_weight_dma_wait_ticks +=
                HAP_perf_get_qtimer_count() - wait_start;
            header->generation_lm_head_weight_dma_ticks +=
                HAP_perf_get_qtimer_count() - prefetch_start;
            header->weight_dma_ticks +=
                HAP_perf_get_qtimer_count() - prefetch_start;
            prefetch_active = 0U;
            if (result != 0) {
                return -6;
            }
            if (n_tile + dma_batch_tiles < n_tiles) {
                const uint32_t following_first =
                    n_tile + dma_batch_tiles;
                uint32_t following_tiles = n_tiles - following_first;
                if (following_tiles > dma_batch_tiles) {
                    following_tiles = dma_batch_tiles;
                }
                prefetch_start = HAP_perf_get_qtimer_count();
                result = qbh_dma_start_weight_prefetch(
                    &prefetch_descriptor,
                    compressed_slots[(batch_index + 1U) & 1U],
                    shared + head->weight_offset +
                        (size_t)following_first * compressed_tile_bytes,
                    following_tiles * compressed_tile_bytes);
                if (result != 0) {
                    return -7;
                }
                prefetch_active = 1U;
                ++header->generation_lm_head_prefetch_count;
                header->generation_lm_head_ddr_read_bytes +=
                    following_tiles * compressed_tile_bytes;
                header->weight_ddr_read_bytes +=
                    following_tiles * compressed_tile_bytes;
                ++header->weight_dma_descriptor_count;
            }
        }

        stage_start = HAP_perf_get_qtimer_count();
        qbh_hmx_fp16_init_channel_scale_tiles(
            scale_blocks,
            resident_scales +
                (size_t)n_tile * QBH_HMX_FP16_COLS,
            group_tiles);
        header->generation_lm_head_scale_init_ticks +=
            HAP_perf_get_qtimer_count() - stage_start;

        if (stream_hmx != 0U) {
            command_start = HAP_perf_get_qtimer_count();
            qbh_hmx_start_fp16_tile_scales_streaming(
                worker, buffers->hmx_activation,
                expanded_slots[expanded_slot], scale_blocks,
                hmx_output, 1U, k_tiles, group_tiles,
                header->w4f16_region_tiles, ready, group_index + 1U);
        }

        stage_start = HAP_perf_get_qtimer_count();
        qbh_w4f16_expand_with_main(
            header, pool,
            compressed_slots[compressed_slot] +
                (size_t)in_batch * compressed_tile_bytes,
            resident_scales +
                (size_t)n_tile * QBH_HMX_FP16_COLS,
            expanded_slots[expanded_slot], ready, group_index + 1U,
            k_tiles * group_tiles, header->w4f16_region_tiles,
            pool->worker_count, stream_hmx, 0U);
        header->generation_lm_head_expand_ticks +=
            HAP_perf_get_qtimer_count() - stage_start;

        if (stream_hmx == 0U) {
            command_start = HAP_perf_get_qtimer_count();
            qbh_hmx_start_fp16_tile_scales(
                worker, buffers->hmx_activation,
                expanded_slots[expanded_slot], scale_blocks,
                hmx_output, 1U, k_tiles, group_tiles);
            if (last_hmx_output != NULL) {
                stage_start = HAP_perf_get_qtimer_count();
                qbh_generation_hvx_argmax_group(
                    last_hmx_output, last_n_tile, last_group_tiles,
                    &best_value, &best_token, &best_bits);
                header->generation_lm_head_argmax_ticks +=
                    HAP_perf_get_qtimer_count() - stage_start;
            }
        }

        tail_wait_start = HAP_perf_get_qtimer_count();
        result = qbh_hmx_wait(worker);
        header->generation_lm_head_hmx_tail_wait_ticks +=
            HAP_perf_get_qtimer_count() - tail_wait_start;
        header->generation_lm_head_hmx_ticks +=
            HAP_perf_get_qtimer_count() - command_start;
        ++header->generation_lm_head_command_count;
        ++header->hmx_command_count;
        header->hmx_fp16_tile_pair_count += k_tiles * group_tiles;
        if (result != 0) {
            if (prefetch_active != 0U) {
                (void)qbh_dma_wait_weight_prefetch(
                    &prefetch_descriptor);
            }
            return -8;
        }

        if (stream_hmx != 0U) {
            stage_start = HAP_perf_get_qtimer_count();
            qbh_generation_hvx_argmax_group(
                (const __fp16 *)hmx_output, n_tile, group_tiles,
                &best_value, &best_token, &best_bits);
            header->generation_lm_head_argmax_ticks +=
                HAP_perf_get_qtimer_count() - stage_start;
        } else {
            last_hmx_output = (const __fp16 *)hmx_output;
            last_n_tile = n_tile;
            last_group_tiles = group_tiles;
        }
    }
    if (stream_hmx == 0U && last_hmx_output != NULL) {
        const uint64_t argmax_start = HAP_perf_get_qtimer_count();
        qbh_generation_hvx_argmax_group(
            last_hmx_output, last_n_tile, last_group_tiles,
            &best_value, &best_token, &best_bits);
        header->generation_lm_head_argmax_ticks +=
            HAP_perf_get_qtimer_count() - argmax_start;
    }
    if (prefetch_active != 0U) {
        uint64_t wait_start = HAP_perf_get_qtimer_count();
        result = qbh_dma_wait_weight_prefetch(&prefetch_descriptor);
        header->generation_lm_head_weight_dma_wait_ticks +=
            HAP_perf_get_qtimer_count() - wait_start;
        header->generation_lm_head_weight_dma_ticks +=
            HAP_perf_get_qtimer_count() - prefetch_start;
        header->weight_dma_ticks +=
            HAP_perf_get_qtimer_count() - prefetch_start;
        if (result != 0) {
            return -9;
        }
    }
generation_complete:
    header->generation_selected_token_id = best_token;
    header->generation_selected_logit_half_bits = best_bits;
    header->generation_expected_token_id =
        ((const uint32_t *)(shared +
          header->generation_expected_token_ids_offset))[generation_step];
    header->generation_token_match =
        best_token == header->generation_expected_token_id;
    header->generation_lm_head_ticks +=
        HAP_perf_get_qtimer_count() - head_start;
    return 0;
}

static int qbh_run_generation_head_w4f16(
    struct qbh_block_header *header, uint8_t *shared,
    struct qbh_block_buffers *buffers,
    struct qbh_block_hmx_worker *worker,
    struct qbh_block_w4f16_pool *pool,
    uint32_t logical_rows, uint32_t generation_step) {
    const struct qbh_block_projection_desc *head =
        &header->generation_lm_head;
    const uint32_t k_tiles = QBH_BLOCK_HIDDEN / QBH_HMX_FP16_COLS;
    const uint32_t n_tiles = QBH_QWEN3_VOCAB_SIZE /
        QBH_HMX_FP16_COLS;
    const uint32_t group_limit =
        header->generation_mode ==
                QBH_BLOCK_GENERATION_GREEDY_W4F16_HVX_ARGMAX_BATCH8
            ? 8U
            : header->generation_mode ==
                QBH_BLOCK_GENERATION_GREEDY_W4F16_HVX_ARGMAX_BATCH4
            ? QBH_BLOCK_W4F16_DMA_BATCH_N_TILES
            : QBH_BLOCK_W4F16_HMX_BATCH_N_TILES;
    const uint32_t compressed_tile_bytes =
        k_tiles * QBH_W4_PACKED_TILE_BYTES;
    const uint32_t scale_tile_bytes =
        QBH_HMX_FP16_COLS * (uint32_t)sizeof(float);
    const uint64_t head_start = HAP_perf_get_qtimer_count();
    float best_value = -INFINITY;
    uint32_t best_token = 0U;
    uint16_t best_bits = 0U;
    uint8_t *head_compressed_weight =
        header->generation_mode ==
                QBH_BLOCK_GENERATION_GREEDY_W4F16_HVX_ARGMAX_BATCH8
            ? buffers->gate : buffers->compressed_weight;
    uint8_t *head_scale_blocks =
        header->generation_mode ==
                QBH_BLOCK_GENERATION_GREEDY_W4F16_HVX_ARGMAX_BATCH8
            ? buffers->middle : buffers->scale_or_bias;

    if (header->generation_mode ==
            QBH_BLOCK_GENERATION_GREEDY_W4F16_LM_HEAD_OVERLAP ||
        header->generation_mode ==
            QBH_BLOCK_GENERATION_GREEDY_W4F16_DMA_HVX_OVERLAP ||
        header->generation_mode ==
            QBH_BLOCK_GENERATION_GREEDY_W4F16_COARSE_PIPELINE) {
        return qbh_run_generation_head_w4f16_overlap(
            header, shared, buffers, worker, pool,
            logical_rows, generation_step);
    }

    if (pool == NULL || logical_rows == 0U ||
        generation_step >= header->generation_expected_token_count) {
        return -1;
    }
    {
        const uint64_t norm_start = HAP_perf_get_qtimer_count();
        if (qbh_dma_copy(
                header, buffers->input_norm_weight,
                shared + header->generation_final_norm_offset,
                header->generation_final_norm_bytes, 1U) != 0) {
            return -2;
        }
        header->weight_ddr_read_bytes +=
            header->generation_final_norm_bytes;
        qbh_hvx_rms_norm_f16(
            (const __fp16 *)buffers->residual,
            (const __fp16 *)buffers->input_norm_weight,
            (__fp16 *)buffers->normalized, logical_rows,
            QBH_BLOCK_HIDDEN, NULL);
        header->generation_final_norm_ticks +=
            HAP_perf_get_qtimer_count() - norm_start;
    }
    qbh_pack_fp16_activation_row0(
        (const __fp16 *)buffers->normalized +
            (size_t)(logical_rows - 1U) * QBH_BLOCK_HIDDEN,
        QBH_BLOCK_HIDDEN, (__fp16 *)buffers->hmx_activation);

    header->generation_lm_head_batch_n_tiles = group_limit;
    header->generation_lm_head_n_tiles = n_tiles;
    for (uint32_t n_tile = 0U; n_tile < n_tiles;
         n_tile += group_limit) {
        uint32_t group_tiles = n_tiles - n_tile;
        const uint32_t generation = n_tile / group_limit + 1U;
        volatile uint32_t ready[QBH_BLOCK_W4F16_MAX_REGIONS];
        uint64_t stage_start;
        uint64_t hmx_start;

        if (group_tiles > group_limit) {
            group_tiles = group_limit;
        }
        memset((void *)ready, 0, sizeof(ready));
        stage_start = HAP_perf_get_qtimer_count();
        if (qbh_dma_copy(
                header, head_compressed_weight,
                shared + head->weight_offset +
                    (size_t)n_tile * compressed_tile_bytes,
                group_tiles * compressed_tile_bytes, 1U) != 0) {
            return -3;
        }
        header->generation_lm_head_weight_dma_ticks +=
            HAP_perf_get_qtimer_count() - stage_start;
        header->generation_lm_head_ddr_read_bytes +=
            group_tiles * compressed_tile_bytes;
        header->weight_ddr_read_bytes +=
            group_tiles * compressed_tile_bytes;
        ++header->weight_dma_descriptor_count;

        stage_start = HAP_perf_get_qtimer_count();
        if (qbh_dma_copy(
                header, buffers->projection_scales,
                shared + head->scale_offset +
                    (size_t)n_tile * scale_tile_bytes,
                group_tiles * scale_tile_bytes, 1U) != 0) {
            return -4;
        }
        header->generation_lm_head_scale_dma_ticks +=
            HAP_perf_get_qtimer_count() - stage_start;
        header->generation_lm_head_ddr_read_bytes +=
            group_tiles * scale_tile_bytes;
        header->weight_ddr_read_bytes +=
            group_tiles * scale_tile_bytes;
        ++header->weight_dma_descriptor_count;

        for (uint32_t tile = 0U; tile < group_tiles; ++tile) {
            qbh_hmx_fp16_init_channel_scales(
                head_scale_blocks +
                    (size_t)tile * QBH_HMX_FP16_SCALE_BYTES,
                (const float *)buffers->projection_scales +
                    (size_t)tile * QBH_HMX_FP16_COLS);
        }
        stage_start = HAP_perf_get_qtimer_count();
        qbh_w4f16_expand_with_main(
            header, pool, head_compressed_weight,
            (const float *)buffers->projection_scales,
            buffers->expanded_weight, ready, generation,
            k_tiles * group_tiles, header->w4f16_region_tiles,
            pool->worker_count, 0U, 0U);
        header->generation_lm_head_expand_ticks +=
            HAP_perf_get_qtimer_count() - stage_start;

        hmx_start = HAP_perf_get_qtimer_count();
        if (qbh_hmx_submit(
                worker, QBH_BLOCK_HMX_FP16_TILE_SCALES,
                buffers->hmx_activation, buffers->expanded_weight,
                head_scale_blocks, buffers->hmx_output,
                1U, k_tiles, group_tiles) != 0) {
            return -5;
        }
        header->generation_lm_head_hmx_ticks +=
            HAP_perf_get_qtimer_count() - hmx_start;
        ++header->generation_lm_head_command_count;
        ++header->hmx_command_count;
        header->hmx_fp16_tile_pair_count += k_tiles * group_tiles;

        stage_start = HAP_perf_get_qtimer_count();
        if (header->generation_mode !=
                QBH_BLOCK_GENERATION_GREEDY_W4F16 &&
            group_tiles >= 2U && group_tiles <= 8U &&
            group_tiles % 2U == 0U) {
            for (uint32_t tile_pair = 0U; tile_pair < group_tiles;
                 tile_pair += 2U) {
                const size_t matrix_tile = qbh_hmx_fp16_matrix_tile_offset(
                    0U, tile_pair, group_tiles);
                const HVX_Vector *tile0 =
                    (const HVX_Vector *)((const __fp16 *)
                        buffers->hmx_output + matrix_tile);
                const HVX_Vector *tile1 = tile0 +
                    QBH_HMX_FP16_TILE_BYTES / sizeof(HVX_Vector);
                const HVX_VectorPair rows = Q6_W_vdeal_VVR(
                    *tile1, *tile0, -2);
                const HVX_Vector row0 = Q6_V_lo_W(rows);
                const float group_max = qbh_hvx_reduce_max_f16(row0);

                if (group_max > best_value) {
                    __fp16 lanes[QBH_BLOCK_HVX_F16_LANES]
                        __attribute__((aligned(QBH_BLOCK_ALIGNMENT)));
                    *(HVX_Vector *)lanes = row0;
                    for (uint32_t lane = 0U;
                         lane < QBH_BLOCK_HVX_F16_LANES; ++lane) {
                        const __fp16 candidate = lanes[lane];
                        const float value = (float)candidate;
                        if (value > best_value) {
                            best_value = value;
                            best_token =
                                (n_tile + tile_pair) *
                                    QBH_HMX_FP16_COLS + lane;
                            memcpy(&best_bits, &candidate,
                                   sizeof(best_bits));
                        }
                    }
                }
            }
        } else {
            for (uint32_t tile = 0U; tile < group_tiles; ++tile) {
                const size_t matrix_tile = qbh_hmx_fp16_matrix_tile_offset(
                    0U, tile, group_tiles);
                for (uint32_t channel = 0U;
                     channel < QBH_HMX_FP16_COLS; ++channel) {
                    const __fp16 candidate =
                        ((__fp16 *)buffers->hmx_output)[
                            matrix_tile +
                            qbh_hmx_fp16_tile_offset(0U, channel)];
                    const float value = (float)candidate;
                    const uint32_t token =
                        (n_tile + tile) * QBH_HMX_FP16_COLS + channel;
                    if (value > best_value) {
                        best_value = value;
                        best_token = token;
                        memcpy(&best_bits, &candidate, sizeof(best_bits));
                    }
                }
            }
        }
        header->generation_lm_head_argmax_ticks +=
            HAP_perf_get_qtimer_count() - stage_start;
    }
    header->generation_selected_token_id = best_token;
    header->generation_selected_logit_half_bits = best_bits;
    header->generation_expected_token_id =
        ((const uint32_t *)(shared +
          header->generation_expected_token_ids_offset))[generation_step];
    header->generation_token_match =
        best_token == header->generation_expected_token_id;
    header->generation_lm_head_ticks +=
        HAP_perf_get_qtimer_count() - head_start;
    return 0;
}

#define QBH_GENERATION_W4U8_CONTROL_GROUP_TILES UINT32_C(4)
#define QBH_GENERATION_W4U8_BATCH8_GROUP_TILES UINT32_C(8)

static uint8_t qbh_generation_hvx_reduce_max_u8(HVX_Vector value) {
    const HVX_Vector zero = Q6_V_vzero();
    for (uint32_t shift = 64U; shift != 0U; shift >>= 1U) {
        value = Q6_Vub_vmax_VubVub(
            value, Q6_V_vlalign_VVR(value, zero, shift));
    }
    return (uint8_t)(Q6_R_vextract_VR(value, 124) >> 24U);
}

static void qbh_generation_hvx_argmax_u8_group(
    const uint8_t *hmx_output, uint32_t first_n_tile,
    uint32_t group_tiles, uint8_t *scratch,
    uint8_t *best_code, uint32_t *best_token) {
    for (uint32_t tile = 0U; tile < group_tiles; ++tile) {
        memcpy(scratch + tile * QBH_HMX_OUTPUT_CHANNELS,
               hmx_output + (size_t)tile * QBH_HMX_OUTPUT_BYTES,
               QBH_HMX_OUTPUT_CHANNELS);
    }
    {
        const uint32_t valid =
            group_tiles * QBH_HMX_OUTPUT_CHANNELS;
        const uint32_t vector_count =
            (valid + sizeof(HVX_Vector) - 1U) / sizeof(HVX_Vector);
        const uint32_t padded = vector_count * sizeof(HVX_Vector);
        if (valid < padded) {
            memset(scratch + valid, 0, padded - valid);
        }
        for (uint32_t half = 0U; half < vector_count; ++half) {
            const uint32_t first_lane =
                half * sizeof(HVX_Vector);
            const uint8_t group_max = qbh_generation_hvx_reduce_max_u8(
                *(const HVX_Vector *)(scratch + first_lane));
            if (group_max > *best_code) {
                const uint32_t limit =
                    valid - first_lane < sizeof(HVX_Vector)
                        ? valid : first_lane + sizeof(HVX_Vector);
                for (uint32_t lane = first_lane; lane < limit; ++lane) {
                    if (scratch[lane] == group_max) {
                        *best_code = group_max;
                        *best_token =
                            first_n_tile * QBH_HMX_OUTPUT_CHANNELS + lane;
                        break;
                    }
                }
            }
        }
    }
}

static int qbh_stage_generation_w4u8_group(
    struct qbh_block_header *header, const uint8_t *shared,
    const struct qbh_block_projection_desc *head,
    struct qbh_block_w4f16_pool *pool,
    uint32_t first_n_tile, uint32_t group_tiles,
    uint8_t *compressed, uint8_t *expanded, uint8_t *bias,
    uint32_t bias_is_resident) {
    const uint32_t k_tiles = head->k / QBH_HMX_INPUT_CHANNELS;
    const uint32_t compressed_tile_bytes =
        k_tiles * QBH_W4_PACKED_TILE_BYTES;
    const uint32_t compressed_bytes =
        group_tiles * compressed_tile_bytes;
    const uint32_t bias_bytes = group_tiles * QBH_HMX_BIAS_BYTES;
    uint64_t start = HAP_perf_get_qtimer_count();

    if (qbh_dma_copy(
            header, compressed,
            shared + head->weight_offset +
                (size_t)first_n_tile * compressed_tile_bytes,
            compressed_bytes, 1U) != 0) {
        return -1;
    }
    header->generation_lm_head_weight_dma_ticks +=
        HAP_perf_get_qtimer_count() - start;
    header->generation_lm_head_ddr_read_bytes += compressed_bytes;
    header->weight_ddr_read_bytes += compressed_bytes;
    ++header->weight_dma_descriptor_count;

    if (bias_is_resident == 0U) {
        start = HAP_perf_get_qtimer_count();
        if (qbh_dma_copy(
                header, bias,
                shared + head->bias_offset +
                    (size_t)first_n_tile * QBH_HMX_BIAS_BYTES,
                bias_bytes, 1U) != 0) {
            return -2;
        }
        header->generation_lm_head_scale_dma_ticks +=
            HAP_perf_get_qtimer_count() - start;
        header->generation_lm_head_ddr_read_bytes += bias_bytes;
        header->weight_ddr_read_bytes += bias_bytes;
        ++header->weight_dma_descriptor_count;
    }

    start = HAP_perf_get_qtimer_count();
    if (bias_is_resident != 0U) {
        if (qbh_w4u8_generation_expand_with_main(
                pool, compressed, expanded, k_tiles,
                group_tiles,
                QBH_BLOCK_W4U8_GATE_UP_HVX_WORKERS) != 0) {
            return -3;
        }
    } else {
        for (uint32_t tile = 0U; tile < group_tiles; ++tile) {
            qbh_unpack_w4_to_s8_hvx_relaxed(
                compressed + (size_t)tile * compressed_tile_bytes,
                (int8_t *)(expanded +
                    (size_t)tile * k_tiles * QBH_HMX_WEIGHT_BYTES),
                k_tiles);
        }
    }
    asm volatile("barrier" ::: "memory");
    header->generation_lm_head_expand_ticks +=
        HAP_perf_get_qtimer_count() - start;
    return 0;
}

/* EXP-0167 adds the real token boundary to the selected cache-native W4U8
 * stack.  Final RMSNorm writes the last logical row directly into integer
 * HMX activation tiles.  The LM head keeps W4 in DDR, expands four output
 * tiles at a time in VTCM, and uses precomputed HMX bias/requant carriers to
 * map every vocabulary channel into one shared U8 logit domain.  Only the
 * selected token leaves the DSP; full logits are never materialized in DDR.
 * EXP-0168 keeps mode 8 as the direct control and adds mode 9: eight output
 * tiles per HMX command plus a phase-overlaid, VTCM-resident bias/requant
 * table.  Transformer buffers are dead after final RMSNorm, so the table can
 * reuse that storage without increasing the arena peak. */
static int qbh_run_generation_head_w4u8(
    struct qbh_block_header *header, uint8_t *shared,
    struct qbh_block_buffers *buffers,
    struct qbh_block_hmx_worker *worker,
    struct qbh_block_w4f16_pool *pool,
    uint32_t logical_rows, uint32_t generation_step) {
    const struct qbh_block_projection_desc *head =
        &header->generation_lm_head;
    const uint32_t k_tiles = head->k / QBH_HMX_INPUT_CHANNELS;
    const uint32_t n_tiles = head->n / QBH_HMX_OUTPUT_CHANNELS;
    const uint32_t resident_bias =
        header->generation_mode ==
            QBH_BLOCK_GENERATION_GREEDY_W4U8_BATCH8_RESIDENT_BIAS;
    const uint32_t group_limit = resident_bias != 0U
        ? QBH_GENERATION_W4U8_BATCH8_GROUP_TILES
        : QBH_GENERATION_W4U8_CONTROL_GROUP_TILES;
    const uint32_t group_count =
        (n_tiles + group_limit - 1U) / group_limit;
    const uint32_t expanded_capacity =
        header->mlp_mode == QBH_BLOCK_MLP_CROUTON_NATIVE_BATCH8 ||
                header->generation_mode ==
                    QBH_BLOCK_GENERATION_GREEDY_W4U8_BATCH8_RESIDENT_BIAS
            ? QBH_BLOCK_HIDDEN * 8U * QBH_HMX_FP16_COLS *
                  sizeof(uint16_t)
            : QBH_BLOCK_MAX_K * QBH_HMX_OUTPUT_CHANNELS *
                  sizeof(uint16_t);
    uint8_t *compressed_slots[2] = {
        buffers->compressed_weight, buffers->compressed_weight_alt};
    uint8_t *expanded_slots[2] = {
        buffers->expanded_weight, buffers->expanded_weight_alt};
    uint8_t *resident_bias_table = buffers->input_norm_weight;
    uint8_t *bias_slots[2] = {buffers->scale_or_bias,
        buffers->scale_or_bias +
            QBH_GENERATION_W4U8_CONTROL_GROUP_TILES *
                QBH_HMX_BIAS_BYTES};
    uint8_t *argmax_scratch = buffers->channel_scale;
    uint8_t best_code = 0U;
    uint32_t best_token = 0U;
    const uint64_t head_start = HAP_perf_get_qtimer_count();
    int result;

    if (!qbh_generation_w4u8_enabled(header->generation_mode) ||
        (resident_bias != 0U && pool == NULL) ||
        logical_rows == 0U ||
        n_tiles == 0U || group_count == 0U ||
        group_limit * k_tiles * QBH_HMX_WEIGHT_BYTES >
            expanded_capacity ||
        (resident_bias == 0U &&
         2U * group_limit * QBH_HMX_BIAS_BYTES >
            2U * QBH_BLOCK_W4U8_QKVO_MAX_BATCH_N_TILES *
                QBH_HMX_FP16_SCALE_BYTES) ||
        (resident_bias != 0U &&
         ((uintptr_t)resident_bias_table >=
              (uintptr_t)buffers->hmx_activation ||
          (uintptr_t)buffers->hmx_activation -
                  (uintptr_t)resident_bias_table <
              head->bias_bytes))) {
        return -1;
    }

    if (header->generation_boundary_audit_enabled != 0U) {
        if (qbh_dma_copy(
                header, shared + header->output_offset,
                buffers->residual +
                    (size_t)(logical_rows - 1U) * QBH_BLOCK_HIDDEN,
                QBH_BLOCK_HIDDEN, 0U) != 0) {
            return -2;
        }
        header->boundary_ddr_write_bytes += QBH_BLOCK_HIDDEN;
        ++header->boundary_dma_descriptor_count;
    }

    {
        const uint64_t norm_start = HAP_perf_get_qtimer_count();
        if (qbh_dma_copy(
                header, buffers->input_norm_weight,
                shared + header->generation_final_norm_offset,
                header->generation_final_norm_bytes, 1U) != 0) {
            return -2;
        }
        header->weight_ddr_read_bytes +=
            header->generation_final_norm_bytes;
        qbh_hvx_rms_norm_u8_native_activation(
            buffers->residual +
                (size_t)(logical_rows - 1U) * QBH_BLOCK_HIDDEN,
            &header->qparams[QBH_BLOCK_QP_BLOCK_OUTPUT],
            (const __fp16 *)buffers->input_norm_weight,
            buffers->hmx_activation,
            &header->generation_final_norm_output_qparam,
            1U, QBH_BLOCK_HIDDEN);
        header->generation_final_norm_ticks +=
            HAP_perf_get_qtimer_count() - norm_start;
    }

    if (resident_bias != 0U) {
        const uint64_t bias_start = HAP_perf_get_qtimer_count();
        if (qbh_dma_copy(
                header, resident_bias_table,
                shared + head->bias_offset, head->bias_bytes, 1U) != 0) {
            return -3;
        }
        header->generation_lm_head_scale_dma_ticks +=
            HAP_perf_get_qtimer_count() - bias_start;
        header->generation_lm_head_scale_resident_bytes =
            head->bias_bytes;
        header->generation_lm_head_ddr_read_bytes += head->bias_bytes;
        header->weight_ddr_read_bytes += head->bias_bytes;
        ++header->weight_dma_descriptor_count;
    }

    header->generation_lm_head_batch_n_tiles = group_limit;
    header->generation_lm_head_n_tiles = n_tiles;
    if (resident_bias != 0U) {
        struct qbh_dma_aligned_desc_1d prefetch_descriptor
            __attribute__((aligned(64)));
        uint64_t prefetch_start = 0U;
        uint64_t hmx_start = 0U;
        uint32_t prefetch_active = 0U;
        uint32_t hmx_active = 0U;
        uint32_t previous_first_n_tile = 0U;
        uint32_t previous_group_tiles = 0U;
        uint32_t first_tiles = n_tiles < group_limit
            ? n_tiles : group_limit;
        const uint32_t compressed_tile_bytes =
            k_tiles * QBH_W4_PACKED_TILE_BYTES;
        uint64_t start = HAP_perf_get_qtimer_count();

        if (qbh_dma_copy(
                header, compressed_slots[0],
                shared + head->weight_offset,
                first_tiles * compressed_tile_bytes, 1U) != 0) {
            return -4;
        }
        header->generation_lm_head_weight_dma_ticks +=
            HAP_perf_get_qtimer_count() - start;
        header->generation_lm_head_ddr_read_bytes +=
            first_tiles * compressed_tile_bytes;
        header->weight_ddr_read_bytes +=
            first_tiles * compressed_tile_bytes;
        ++header->weight_dma_descriptor_count;

        if (group_count > 1U) {
            uint32_t next_tiles = n_tiles - group_limit;
            if (next_tiles > group_limit) {
                next_tiles = group_limit;
            }
            prefetch_start = HAP_perf_get_qtimer_count();
            result = qbh_dma_start_weight_prefetch(
                &prefetch_descriptor, compressed_slots[1],
                shared + head->weight_offset +
                    (size_t)group_limit * compressed_tile_bytes,
                next_tiles * compressed_tile_bytes);
            if (result != 0) {
                return -5;
            }
            prefetch_active = 1U;
            ++header->generation_lm_head_prefetch_count;
            header->generation_lm_head_ddr_read_bytes +=
                next_tiles * compressed_tile_bytes;
            header->weight_ddr_read_bytes +=
                next_tiles * compressed_tile_bytes;
            ++header->weight_dma_descriptor_count;
        }

        for (uint32_t group = 0U; group < group_count; ++group) {
            const uint32_t slot = group & 1U;
            const uint32_t first_n_tile = group * group_limit;
            uint32_t group_tiles = n_tiles - first_n_tile;
            if (group_tiles > group_limit) {
                group_tiles = group_limit;
            }

            if (group != 0U) {
                const uint64_t wait_start =
                    HAP_perf_get_qtimer_count();
                result = qbh_dma_wait_weight_prefetch(
                    &prefetch_descriptor);
                header->generation_lm_head_weight_dma_wait_ticks +=
                    HAP_perf_get_qtimer_count() - wait_start;
                header->generation_lm_head_weight_dma_ticks +=
                    HAP_perf_get_qtimer_count() - prefetch_start;
                header->weight_dma_ticks +=
                    HAP_perf_get_qtimer_count() - prefetch_start;
                prefetch_active = 0U;
                if (result != 0) {
                    if (hmx_active != 0U) {
                        (void)qbh_hmx_wait(worker);
                    }
                    return -6;
                }
            }

            if (group != 0U && group + 1U < group_count) {
                const uint32_t next_group = group + 1U;
                const uint32_t next_slot = next_group & 1U;
                const uint32_t next_first = next_group * group_limit;
                uint32_t next_tiles = n_tiles - next_first;
                if (next_tiles > group_limit) {
                    next_tiles = group_limit;
                }
                prefetch_start = HAP_perf_get_qtimer_count();
                result = qbh_dma_start_weight_prefetch(
                    &prefetch_descriptor,
                    compressed_slots[next_slot],
                    shared + head->weight_offset +
                        (size_t)next_first * compressed_tile_bytes,
                    next_tiles * compressed_tile_bytes);
                if (result != 0) {
                    if (hmx_active != 0U) {
                        (void)qbh_hmx_wait(worker);
                    }
                    return -7;
                }
                prefetch_active = 1U;
                ++header->generation_lm_head_prefetch_count;
                header->generation_lm_head_ddr_read_bytes +=
                    next_tiles * compressed_tile_bytes;
                header->weight_ddr_read_bytes +=
                    next_tiles * compressed_tile_bytes;
                ++header->weight_dma_descriptor_count;
            }

            start = HAP_perf_get_qtimer_count();
            result = qbh_w4u8_generation_expand_with_main(
                pool, compressed_slots[slot], expanded_slots[slot],
                k_tiles, group_tiles,
                QBH_BLOCK_W4U8_GATE_UP_HVX_WORKERS);
            header->generation_lm_head_expand_ticks +=
                HAP_perf_get_qtimer_count() - start;
            if (result != 0) {
                if (hmx_active != 0U) {
                    (void)qbh_hmx_wait(worker);
                }
                if (prefetch_active != 0U) {
                    (void)qbh_dma_wait_weight_prefetch(
                        &prefetch_descriptor);
                }
                return -8;
            }

            if (hmx_active != 0U) {
                const uint64_t wait_start =
                    HAP_perf_get_qtimer_count();
                result = qbh_hmx_wait(worker);
                header->generation_lm_head_hmx_tail_wait_ticks +=
                    HAP_perf_get_qtimer_count() - wait_start;
                header->generation_lm_head_hmx_ticks +=
                    HAP_perf_get_qtimer_count() - hmx_start;
                hmx_active = 0U;
                if (result != 0) {
                    if (prefetch_active != 0U) {
                        (void)qbh_dma_wait_weight_prefetch(
                            &prefetch_descriptor);
                    }
                    return -9;
                }
                start = HAP_perf_get_qtimer_count();
                qbh_generation_hvx_argmax_u8_group(
                    buffers->hmx_output, previous_first_n_tile,
                    previous_group_tiles, argmax_scratch,
                    &best_code, &best_token);
                header->generation_lm_head_argmax_ticks +=
                    HAP_perf_get_qtimer_count() - start;
            }

            hmx_start = HAP_perf_get_qtimer_count();
            qbh_hmx_start(
                worker, QBH_BLOCK_HMX_U8S8,
                buffers->hmx_activation, expanded_slots[slot],
                resident_bias_table +
                    (size_t)first_n_tile * QBH_HMX_BIAS_BYTES,
                buffers->hmx_output, 1U, k_tiles, group_tiles);
            hmx_active = 1U;
            previous_first_n_tile = first_n_tile;
            previous_group_tiles = group_tiles;
            ++header->generation_lm_head_command_count;
            ++header->hmx_command_count;
            header->hmx_u8s8_tile_pair_count +=
                k_tiles * group_tiles;
        }

        if (hmx_active != 0U) {
            const uint64_t wait_start = HAP_perf_get_qtimer_count();
            result = qbh_hmx_wait(worker);
            header->generation_lm_head_hmx_tail_wait_ticks +=
                HAP_perf_get_qtimer_count() - wait_start;
            header->generation_lm_head_hmx_ticks +=
                HAP_perf_get_qtimer_count() - hmx_start;
            if (result != 0) {
                return -10;
            }
            start = HAP_perf_get_qtimer_count();
            qbh_generation_hvx_argmax_u8_group(
                buffers->hmx_output, previous_first_n_tile,
                previous_group_tiles, argmax_scratch,
                &best_code, &best_token);
            header->generation_lm_head_argmax_ticks +=
                HAP_perf_get_qtimer_count() - start;
        }
        goto w4u8_generation_complete;
    }

    result = qbh_stage_generation_w4u8_group(
        header, shared, head, pool, 0U,
        n_tiles < group_limit ? n_tiles : group_limit,
        compressed_slots[0], expanded_slots[0],
        resident_bias != 0U ? resident_bias_table : bias_slots[0],
        resident_bias);
    if (result != 0) {
        return -3;
    }

    for (uint32_t group = 0U; group < group_count; ++group) {
        const uint32_t slot = group & 1U;
        const uint32_t first_n_tile = group * group_limit;
        uint32_t group_tiles = n_tiles - first_n_tile;
        uint64_t hmx_start;
        uint64_t wait_start;
        if (group_tiles > group_limit) {
            group_tiles = group_limit;
        }

        hmx_start = HAP_perf_get_qtimer_count();
        qbh_hmx_start(
            worker, QBH_BLOCK_HMX_U8S8,
            buffers->hmx_activation, expanded_slots[slot],
            resident_bias != 0U
                ? resident_bias_table +
                    (size_t)first_n_tile * QBH_HMX_BIAS_BYTES
                : bias_slots[slot],
            buffers->hmx_output,
            1U, k_tiles, group_tiles);
        if (group + 1U < group_count) {
            const uint32_t next_group = group + 1U;
            const uint32_t next_slot = next_group & 1U;
            const uint32_t next_first = next_group * group_limit;
            uint32_t next_tiles = n_tiles - next_first;
            if (next_tiles > group_limit) {
                next_tiles = group_limit;
            }
            result = qbh_stage_generation_w4u8_group(
                header, shared, head, pool, next_first, next_tiles,
                compressed_slots[next_slot], expanded_slots[next_slot],
                resident_bias != 0U
                    ? resident_bias_table +
                        (size_t)next_first * QBH_HMX_BIAS_BYTES
                    : bias_slots[next_slot],
                resident_bias);
            if (result != 0) {
                (void)qbh_hmx_wait(worker);
                return -4;
            }
            ++header->generation_lm_head_prefetch_count;
        }
        wait_start = HAP_perf_get_qtimer_count();
        result = qbh_hmx_wait(worker);
        header->generation_lm_head_hmx_tail_wait_ticks +=
            HAP_perf_get_qtimer_count() - wait_start;
        header->generation_lm_head_hmx_ticks +=
            HAP_perf_get_qtimer_count() - hmx_start;
        if (result != 0) {
            return -5;
        }
        ++header->generation_lm_head_command_count;
        ++header->hmx_command_count;
        header->hmx_u8s8_tile_pair_count += k_tiles * group_tiles;

        {
            const uint64_t argmax_start = HAP_perf_get_qtimer_count();
            qbh_generation_hvx_argmax_u8_group(
                buffers->hmx_output, first_n_tile, group_tiles,
                argmax_scratch, &best_code, &best_token);
            header->generation_lm_head_argmax_ticks +=
                HAP_perf_get_qtimer_count() - argmax_start;
        }
    }

w4u8_generation_complete:
    header->generation_selected_token_id = best_token;
    header->generation_selected_logit_half_bits = best_code;
    if (generation_step < header->generation_expected_token_count) {
        header->generation_expected_token_id =
            ((const uint32_t *)(shared +
              header->generation_expected_token_ids_offset))[generation_step];
        header->generation_token_match =
            best_token == header->generation_expected_token_id;
    } else {
        header->generation_expected_token_id = UINT32_MAX;
        header->generation_token_match = 0U;
    }
    header->generation_lm_head_ticks +=
        HAP_perf_get_qtimer_count() - head_start;
    return 0;
}

static void qbh_pack_u8_activation(const uint8_t *source,
                                   uint32_t source_stride, uint32_t k,
                                   uint8_t *destination) {
    uint32_t k_tiles = k / QBH_HMX_INPUT_CHANNELS;
    for (uint32_t k_tile = 0; k_tile < k_tiles; ++k_tile) {
        for (uint32_t row = 0; row < QBH_BLOCK_M; ++row) {
            memcpy(destination +
                       (size_t)k_tile * QBH_HMX_ACTIVATION_BYTES +
                       row * QBH_HMX_INPUT_CHANNELS,
                   source + (size_t)row * source_stride +
                       k_tile * QBH_HMX_INPUT_CHANNELS,
                   QBH_HMX_INPUT_CHANNELS);
        }
    }
}

static void qbh_unpack_fp16_output(const __fp16 *source,
                                   uint32_t n_tiles,
                                   __fp16 *destination,
                                   uint32_t destination_stride,
                                   uint32_t destination_column) {
    for (uint32_t row = 0; row < QBH_BLOCK_M; row += 2U) {
        uint32_t row_tile = row / QBH_HMX_FP16_ROWS;
        uint32_t row_pair = (row % QBH_HMX_FP16_ROWS) / 2U;
        uint32_t column_tile = 0U;
        __fp16 *destination0 = destination +
            (size_t)row * destination_stride + destination_column;
        __fp16 *destination1 = destination +
            (size_t)(row + 1U) * destination_stride +
            destination_column;

        for (; column_tile + 1U < n_tiles; column_tile += 2U) {
            size_t tile = qbh_hmx_fp16_matrix_tile_offset(
                row_tile, column_tile, n_tiles);
            const HVX_Vector *input0 = (const HVX_Vector *)(
                source + tile) + row_pair;
            const HVX_Vector *input1 = input0 +
                QBH_HMX_FP16_TILE_BYTES / sizeof(HVX_Vector);
            HVX_VectorPair rows = Q6_W_vdeal_VVR(
                *input1, *input0, -2);
            *(HVX_Vector *)(destination0 +
                (size_t)column_tile * QBH_HMX_FP16_COLS) =
                    Q6_V_lo_W(rows);
            *(HVX_Vector *)(destination1 +
                (size_t)column_tile * QBH_HMX_FP16_COLS) =
                    Q6_V_hi_W(rows);
        }
        if (column_tile < n_tiles) {
            size_t tile = qbh_hmx_fp16_matrix_tile_offset(
                row_tile, column_tile, n_tiles);
            const HVX_Vector *input = (const HVX_Vector *)(
                source + tile) + row_pair;
            HVX_VectorPair rows = Q6_W_vdeal_VVR(
                Q6_V_vzero(), *input, -2);
            HVX_Vector row0 = Q6_V_lo_W(rows);
            HVX_Vector row1 = Q6_V_hi_W(rows);
            memcpy(destination0 +
                       (size_t)column_tile * QBH_HMX_FP16_COLS,
                   &row0, 64U);
            memcpy(destination1 +
                       (size_t)column_tile * QBH_HMX_FP16_COLS,
                   &row1, 64U);
        }
    }
    asm volatile("barrier" ::: "memory");
}

static void qbh_unpack_u8_output(const uint8_t *source,
                                 uint8_t *destination,
                                 uint32_t destination_stride,
                                 uint32_t destination_column) {
    for (uint32_t row = 0; row < QBH_BLOCK_M; ++row) {
        memcpy(destination + (size_t)row * destination_stride +
                   destination_column,
               source + row * QBH_HMX_OUTPUT_CHANNELS,
               QBH_HMX_OUTPUT_CHANNELS);
    }
}

static void qbh_record_projection_failure(
    struct qbh_block_header *header,
    const struct qbh_block_projection_desc *desc,
    uint32_t n_tile, uint32_t step, int32_t result) {
    header->projection_failure_result = result;
    header->projection_failure_index =
        (uint32_t)(desc - header->projections);
    header->projection_failure_n_tile = n_tile;
    header->projection_failure_step = step;
}

static const float *qbh_w4f16_projection_scales(
    const struct qbh_block_header *header,
    const struct qbh_block_buffers *buffers,
    const struct qbh_block_projection_desc *desc) {
    uint32_t projection_index =
        (uint32_t)(desc - header->projections);
    uint32_t channel_offset = 0U;

    for (uint32_t index = 0; index < projection_index; ++index) {
        channel_offset += header->projections[index].n;
    }
    return (const float *)(buffers->projection_scales +
                           (size_t)channel_offset * sizeof(float));
}

static int qbh_crouton_qkv_enabled(
    const struct qbh_block_header *header) {
    return header->variant != QBH_BLOCK_W4U8 &&
           (header->crouton_boundary_mode &
            QBH_BLOCK_CROUTON_BOUNDARY_QKV) != 0U;
}

static int qbh_projection_direct_qkv_crouton(
    const struct qbh_block_header *header,
    const struct qbh_block_projection_desc *desc) {
    return qbh_crouton_qkv_enabled(header) &&
           (desc == &header->projections[QBH_BLOCK_PROJ_Q] ||
            desc == &header->projections[QBH_BLOCK_PROJ_K] ||
            desc == &header->projections[QBH_BLOCK_PROJ_V]);
}

static void qbh_capture_row_major_qkv_reference(
    const struct qbh_block_header *header,
    const struct qbh_block_projection_desc *desc,
    struct qbh_block_buffers *buffers, const __fp16 *output,
    uint32_t first_n_tile, uint32_t n_tiles) {
    __fp16 *reference;
    uint32_t stride;
    const uint32_t first_column =
        first_n_tile * QBH_HMX_FP16_COLS;
    const uint32_t columns = n_tiles * QBH_HMX_FP16_COLS;

    if (header->numerical_audit_enabled == 0U ||
        header->variant != QBH_BLOCK_W4F16) {
        return;
    }
    if (desc == &header->projections[QBH_BLOCK_PROJ_Q]) {
        reference = (__fp16 *)buffers->gate;
        stride = QBH_BLOCK_HIDDEN;
    } else if (desc == &header->projections[QBH_BLOCK_PROJ_K]) {
        reference = (__fp16 *)buffers->up;
        stride = QBH_BLOCK_KV_HIDDEN;
    } else if (desc == &header->projections[QBH_BLOCK_PROJ_V]) {
        reference = (__fp16 *)buffers->gate +
            QBH_BLOCK_M * QBH_BLOCK_HIDDEN;
        stride = QBH_BLOCK_KV_HIDDEN;
    } else {
        return;
    }
    for (uint32_t row = 0U; row < QBH_BLOCK_M; ++row) {
        memcpy(reference + (size_t)row * stride + first_column,
               output + (size_t)row * desc->n + first_column,
               (size_t)columns * sizeof(__fp16));
    }
}

static __attribute__((noinline)) int qbh_run_f16f16_pipelined_projection(
    struct qbh_block_header *header, const uint8_t *shared,
    const struct qbh_block_projection_desc *desc,
    struct qbh_block_buffers *buffers,
    struct qbh_block_hmx_worker *worker,
    struct qbh_block_w4f16_pool *hvx_pool,
    const void *activation_tiles, void *output) {
    uint32_t k_tiles = desc->k / QBH_HMX_FP16_COLS;
    uint32_t n_tiles = desc->n / QBH_HMX_FP16_COLS;
    uint32_t batch_n_tiles = 1U;
    uint32_t weight_bytes_per_tile =
        k_tiles * QBH_HMX_FP16_TILE_BYTES;
    uint8_t *weight_slots[2] = {
        buffers->expanded_weight, buffers->expanded_weight_alt};
    const uint32_t direct_qkv =
        qbh_projection_direct_qkv_crouton(header, desc);
    const uint32_t direct_crouton = direct_qkv ||
        (header->mlp_mode == QBH_BLOCK_MLP_CROUTON_NATIVE_BATCH8 &&
         (desc == &header->projections[QBH_BLOCK_PROJ_GATE] ||
          desc == &header->projections[QBH_BLOCK_PROJ_UP]));
    struct qbh_dma_aligned_desc_1d prefetch_descriptor
        __attribute__((aligned(64)));
    int prefetch_active = 0;

    if (header->f16f16_projection_mode ==
            QBH_BLOCK_F16F16_PROJECTION_BATCH2 ||
        header->f16f16_projection_mode ==
            QBH_BLOCK_F16F16_PROJECTION_GATE4 ||
        header->f16f16_projection_mode ==
            QBH_BLOCK_F16F16_PROJECTION_GATE8 ||
        header->f16f16_projection_mode ==
            QBH_BLOCK_F16F16_PROJECTION_GATE8_INTERLEAVED) {
        batch_n_tiles = QBH_BLOCK_F16F16_BATCH_N_TILES;
    }
    if (header->f16f16_projection_mode ==
            QBH_BLOCK_F16F16_PROJECTION_GATE4 &&
        (desc == &header->projections[QBH_BLOCK_PROJ_GATE] ||
         desc == &header->projections[QBH_BLOCK_PROJ_UP])) {
        batch_n_tiles = 4U;
    }
    if (header->f16f16_projection_mode ==
            QBH_BLOCK_F16F16_PROJECTION_GATE8 &&
        (desc == &header->projections[QBH_BLOCK_PROJ_GATE] ||
         desc == &header->projections[QBH_BLOCK_PROJ_UP])) {
        batch_n_tiles = 8U;
    }
    if (header->f16f16_projection_mode ==
            QBH_BLOCK_F16F16_PROJECTION_GATE8_INTERLEAVED &&
        (desc == &header->projections[QBH_BLOCK_PROJ_GATE] ||
         desc == &header->projections[QBH_BLOCK_PROJ_UP])) {
        batch_n_tiles = 8U;
    }

    header->f16f16_weight_batch_n_tiles = batch_n_tiles;
    {
        uint32_t first_group_tiles =
            n_tiles < batch_n_tiles ? n_tiles : batch_n_tiles;
        uint32_t first_group_bytes =
            first_group_tiles * weight_bytes_per_tile;
        if (qbh_dma_copy(header, weight_slots[0],
                         shared + desc->weight_offset,
                         first_group_bytes, 1U) != 0) {
            qbh_record_projection_failure(header, desc, 0U, 21U, -1);
            return -1;
        }
        header->weight_ddr_read_bytes += first_group_bytes;
        ++header->weight_dma_descriptor_count;
    }

    for (uint32_t n_tile = 0; n_tile < n_tiles;
         n_tile += batch_n_tiles) {
        uint32_t group_index = n_tile / batch_n_tiles;
        uint32_t group_tiles = n_tiles - n_tile;
        uint32_t next_tile;
        uint64_t hmx_start;
        uint64_t prefetch_start = 0U;
        __fp16 *hmx_output = (__fp16 *)buffers->hmx_output;
        int hmx_result;
        int prefetch_result = 0;

        if (group_tiles > batch_n_tiles) {
            group_tiles = batch_n_tiles;
        }
        next_tile = n_tile + group_tiles;
        if (direct_crouton != 0U) {
            const uint32_t output_group_elements =
                (QBH_BLOCK_M / QBH_HMX_FP16_ROWS) *
                group_tiles * QBH_HMX_FP16_TILE_ELEMENTS;
            hmx_output = (__fp16 *)output +
                (size_t)group_index * output_group_elements;
        }

        hmx_start = HAP_perf_get_qtimer_count();
        qbh_hmx_start(
            worker, QBH_BLOCK_HMX_FP16, activation_tiles,
            weight_slots[group_index & 1U], buffers->scale_or_bias,
            hmx_output, 2U, k_tiles, group_tiles);

        if (next_tile < n_tiles) {
            uint32_t next_group_tiles = n_tiles - next_tile;
            uint32_t next_group_index = next_tile / batch_n_tiles;
            uint32_t next_group_bytes;
            if (next_group_tiles > batch_n_tiles) {
                next_group_tiles = batch_n_tiles;
            }
            next_group_bytes =
                next_group_tiles * weight_bytes_per_tile;
            prefetch_start = HAP_perf_get_qtimer_count();
            prefetch_result = qbh_dma_start_weight_prefetch(
                &prefetch_descriptor,
                weight_slots[next_group_index & 1U],
                shared + desc->weight_offset +
                    (size_t)next_tile * weight_bytes_per_tile,
                next_group_bytes);
            if (prefetch_result == 0) {
                prefetch_active = 1;
                ++header->f16f16_prefetch_count;
                header->weight_ddr_read_bytes += next_group_bytes;
                ++header->weight_dma_descriptor_count;
            }
        }

        hmx_result = qbh_hmx_wait(worker);
        header->projection_hmx_wait_ticks +=
            HAP_perf_get_qtimer_count() - hmx_start;
        header->hmx_fp16_tile_pair_count +=
            2U * k_tiles * group_tiles;
        ++header->hmx_command_count;

        if (hmx_result == 0 && direct_crouton != 0U) {
            if (direct_qkv != 0U) {
                header->crouton_qkv_unpack_skipped += group_tiles;
                qbh_hvx_pool_qk_norm_rope_publish(
                    header, desc, hvx_pool, n_tile, group_tiles);
            }
            if (qbh_mlp_crouton_publish_up_group(
                    header, desc, hvx_pool, group_index) != 0) {
                qbh_record_projection_failure(
                    header, desc, n_tile, 24U, -1);
                return -1;
            }
        } else if (hmx_result == 0) {
            uint64_t unpack_start = HAP_perf_get_qtimer_count();
            qbh_unpack_fp16_output(
                (const __fp16 *)buffers->hmx_output, group_tiles,
                (__fp16 *)output, desc->n, n_tile * QBH_HMX_FP16_COLS);
            header->projection_unpack_ticks +=
                HAP_perf_get_qtimer_count() - unpack_start;
            qbh_hvx_pool_qk_norm_rope_publish(
                header, desc, hvx_pool, n_tile, group_tiles);
            if (qbh_mlp_stream_publish_up_group(
                    header, desc, hvx_pool, n_tile,
                    group_tiles) != 0) {
                qbh_record_projection_failure(
                    header, desc, n_tile, 24U, -1);
                return -1;
            }
        }

        if (prefetch_active != 0) {
            uint64_t wait_start = HAP_perf_get_qtimer_count();
            prefetch_result = qbh_dma_wait_weight_prefetch(
                &prefetch_descriptor);
            header->f16f16_prefetch_wait_ticks +=
                HAP_perf_get_qtimer_count() - wait_start;
            header->weight_dma_ticks +=
                HAP_perf_get_qtimer_count() - prefetch_start;
            prefetch_active = 0;
        }
        if (hmx_result != 0 || prefetch_result != 0) {
            qbh_record_projection_failure(
                header, desc, n_tile,
                hmx_result != 0 ? 22U : 23U,
                hmx_result != 0 ? hmx_result : prefetch_result);
            return -1;
        }
    }
    return 0;
}

/*
 * EXP-0109 keeps the accepted FP16 HMX/Crouton contract, but presents Gate
 * and Up as one producer stream.  A command consumes one eight-output-tile
 * weight slot while DMA fills the other slot for the matched projection.
 * Publishing an Up group releases the already-complete Gate/Up pair to the
 * existing streaming SwiGLU workers.  There is still exactly one HMX owner.
 */
static __attribute__((noinline)) int
qbh_run_f16f16_interleaved_gate_up(
    struct qbh_block_header *header, const uint8_t *shared,
    struct qbh_block_buffers *buffers,
    struct qbh_block_hmx_worker *worker,
    struct qbh_block_w4f16_pool *pool,
    const void *activation_tiles) {
    const struct qbh_block_projection_desc *descs[2] = {
        &header->projections[QBH_BLOCK_PROJ_GATE],
        &header->projections[QBH_BLOCK_PROJ_UP]};
    void *outputs[2] = {buffers->gate, buffers->up};
    uint8_t *weight_slots[2] = {
        buffers->expanded_weight, buffers->expanded_weight_alt};
    struct qbh_dma_aligned_desc_1d prefetch_descriptor
        __attribute__((aligned(64)));
    const uint32_t group_tiles = 8U;
    uint32_t k_tiles = descs[0]->k / QBH_HMX_FP16_COLS;
    uint32_t n_tiles = descs[0]->n / QBH_HMX_FP16_COLS;
    uint32_t group_count = n_tiles / group_tiles;
    uint32_t weight_bytes =
        group_tiles * k_tiles * QBH_HMX_FP16_TILE_BYTES;
    uint32_t output_group_elements =
        (QBH_BLOCK_M / QBH_HMX_FP16_ROWS) * group_tiles *
        QBH_HMX_FP16_TILE_ELEMENTS;
    uint32_t command_count = group_count * 2U;

    if (header->variant != QBH_BLOCK_F16F16 || pool == NULL ||
        header->mlp_mode != QBH_BLOCK_MLP_CROUTON_NATIVE_BATCH8 ||
        descs[0]->k != descs[1]->k ||
        descs[0]->n != descs[1]->n ||
        n_tiles == 0U || n_tiles % group_tiles != 0U) {
        return -1;
    }
    header->f16f16_weight_batch_n_tiles = group_tiles;
    if (qbh_dma_copy(
            header, weight_slots[0],
            shared + descs[0]->weight_offset,
            weight_bytes, 1U) != 0) {
        qbh_record_projection_failure(
            header, descs[0], 0U, 60U, -1);
        return -1;
    }
    header->weight_ddr_read_bytes += weight_bytes;
    ++header->weight_dma_descriptor_count;

    for (uint32_t command = 0U; command < command_count; ++command) {
        uint32_t projection = command & 1U;
        uint32_t group = command >> 1U;
        uint32_t slot = command & 1U;
        uint64_t hmx_start;
        uint64_t prefetch_start = 0U;
        int hmx_result;
        int prefetch_result = 0;
        int prefetch_active = 0;

        if (projection == 0U &&
            qbh_mlp_crouton_wait_slot(header, pool, group) != 0) {
            qbh_record_projection_failure(
                header, descs[projection], group * group_tiles,
                61U, -1);
            return -1;
        }

        hmx_start = HAP_perf_get_qtimer_count();
        qbh_hmx_start(
            worker, QBH_BLOCK_HMX_FP16, activation_tiles,
            weight_slots[slot], buffers->scale_or_bias,
            (__fp16 *)outputs[projection] +
                (size_t)group * output_group_elements,
            2U, k_tiles, group_tiles);

        if (command + 1U < command_count) {
            uint32_t next_command = command + 1U;
            uint32_t next_projection = next_command & 1U;
            uint32_t next_group = next_command >> 1U;
            uint32_t next_slot = next_command & 1U;
            prefetch_start = HAP_perf_get_qtimer_count();
            prefetch_result = qbh_dma_start_weight_prefetch(
                &prefetch_descriptor, weight_slots[next_slot],
                shared + descs[next_projection]->weight_offset +
                    (size_t)next_group * weight_bytes,
                weight_bytes);
            if (prefetch_result == 0) {
                prefetch_active = 1;
                ++header->f16f16_prefetch_count;
                header->weight_ddr_read_bytes += weight_bytes;
                ++header->weight_dma_descriptor_count;
            }
        }

        hmx_result = qbh_hmx_wait(worker);
        header->projection_hmx_wait_ticks +=
            HAP_perf_get_qtimer_count() - hmx_start;
        header->hmx_fp16_tile_pair_count +=
            2U * k_tiles * group_tiles;
        ++header->hmx_command_count;

        if (hmx_result == 0 && projection == 1U &&
            qbh_mlp_crouton_publish_up_group(
                header, descs[projection], pool, group) != 0) {
            hmx_result = -1;
        }
        if (prefetch_active != 0) {
            uint64_t wait_start = HAP_perf_get_qtimer_count();
            prefetch_result = qbh_dma_wait_weight_prefetch(
                &prefetch_descriptor);
            header->f16f16_prefetch_wait_ticks +=
                HAP_perf_get_qtimer_count() - wait_start;
            header->weight_dma_ticks +=
                HAP_perf_get_qtimer_count() - prefetch_start;
        }
        if (hmx_result != 0 || prefetch_result != 0) {
            qbh_record_projection_failure(
                header, descs[projection], group * group_tiles,
                hmx_result != 0 ? 62U : 63U,
                hmx_result != 0 ? hmx_result : prefetch_result);
            return -1;
        }
    }
    return 0;
}

static uint32_t qbh_w4f16_projection_worker_count(
    const struct qbh_block_header *header,
    const struct qbh_block_projection_desc *desc,
    const struct qbh_block_w4f16_pool *pool) {
    if ((header->w4f16_pipeline_mode ==
             QBH_BLOCK_W4F16_PIPELINE_HYBRID_WORKERS ||
         header->w4f16_pipeline_mode ==
             QBH_BLOCK_W4F16_PIPELINE_HYBRID_CROSS_PREFETCH ||
         header->w4f16_pipeline_mode ==
             QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN64_CROSS_PREFETCH ||
         header->w4f16_pipeline_mode ==
             QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN48_CROSS_PREFETCH ||
         header->w4f16_pipeline_mode ==
             QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_CROSS_PREFETCH ||
         header->w4f16_pipeline_mode ==
             QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_GATE16_CROSS_PREFETCH ||
         header->w4f16_pipeline_mode ==
             QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_GATE8_CROSS_PREFETCH ||
         header->w4f16_pipeline_mode ==
             QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_GATE4_CROSS_PREFETCH ||
         header->w4f16_pipeline_mode ==
             QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_GATE4_DMA8_CROSS_PREFETCH) &&
        desc->k != QBH_BLOCK_INTERMEDIATE &&
        pool->worker_count > 2U) {
        return 2U;
    }
    return pool->worker_count;
}

static uint32_t qbh_w4f16_projection_region_tiles(
    const struct qbh_block_header *header,
    const struct qbh_block_projection_desc *desc) {
    if (desc->k == QBH_BLOCK_INTERMEDIATE) {
        if (header->w4f16_pipeline_mode ==
            QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN48_CROSS_PREFETCH) {
            return 48U;
        }
        if (header->w4f16_pipeline_mode ==
            QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN64_CROSS_PREFETCH) {
            return 64U;
        }
        if (header->w4f16_pipeline_mode ==
                QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_CROSS_PREFETCH ||
            header->w4f16_pipeline_mode ==
                QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_GATE16_CROSS_PREFETCH ||
            header->w4f16_pipeline_mode ==
                QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_GATE8_CROSS_PREFETCH ||
            header->w4f16_pipeline_mode ==
                QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_GATE4_CROSS_PREFETCH ||
            header->w4f16_pipeline_mode ==
                QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_GATE4_DMA8_CROSS_PREFETCH) {
            return 96U;
        }
    }
    return header->w4f16_region_tiles;
}

static uint32_t qbh_w4f16_gate_up_region_tiles(
    const struct qbh_block_header *header) {
    if (header->w4f16_pipeline_mode ==
        QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_GATE16_CROSS_PREFETCH) {
        return 16U;
    }
    if (header->w4f16_pipeline_mode ==
        QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_GATE8_CROSS_PREFETCH) {
        return 8U;
    }
    return header->w4f16_region_tiles;
}

static uint32_t qbh_w4f16_gate_up_group_tiles(
    const struct qbh_block_header *header) {
    if (header->mlp_mode ==
        QBH_BLOCK_MLP_CROUTON_NATIVE_BATCH8) {
        return 8U;
    }
    return (header->w4f16_pipeline_mode ==
                QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_GATE4_CROSS_PREFETCH ||
            header->w4f16_pipeline_mode ==
                QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_GATE4_DMA8_CROSS_PREFETCH)
               ? 4U : QBH_BLOCK_W4F16_HMX_BATCH_N_TILES;
}

/* EXP-0127: a Qwen3 Q/K/V head is four 32-channel HMX output tiles.
 * Keep Q at batch two for selective modes so its first complete head can be
 * published without waiting for a four-tile weight/expand group. */
static uint32_t qbh_w4f16_projection_group_tiles(
    const struct qbh_block_header *header,
    const struct qbh_block_projection_desc *desc) {
    const int is_q = desc == &header->projections[QBH_BLOCK_PROJ_Q];
    const int is_k = desc == &header->projections[QBH_BLOCK_PROJ_K];
    const int is_v = desc == &header->projections[QBH_BLOCK_PROJ_V];
    const uint32_t mode = header->qkv_schedule_mode;
    if ((mode == QBH_BLOCK_QKV_SCHEDULE_HEAD_ALIGNED_BATCH4 &&
         (is_q || is_k || is_v)) ||
        (mode == QBH_BLOCK_QKV_SCHEDULE_V_BATCH4 && is_v) ||
        (mode == QBH_BLOCK_QKV_SCHEDULE_KV_BATCH4 && (is_k || is_v))) {
        return QBH_BLOCK_HEAD_DIM / QBH_HMX_FP16_COLS;
    }
    return QBH_BLOCK_W4F16_HMX_BATCH_N_TILES;
}

static void qbh_w4f16_note_effective_region(
    struct qbh_block_header *header, uint32_t region_tiles) {
    if (header->w4f16_effective_region_min == 0U ||
        region_tiles < header->w4f16_effective_region_min) {
        header->w4f16_effective_region_min = region_tiles;
    }
    if (region_tiles > header->w4f16_effective_region_max) {
        header->w4f16_effective_region_max = region_tiles;
    }
}

static void qbh_w4f16_note_active_workers(
    struct qbh_block_header *header, uint32_t active_workers) {
    if (header->w4f16_active_worker_min == 0U ||
        active_workers < header->w4f16_active_worker_min) {
        header->w4f16_active_worker_min = active_workers;
    }
    if (active_workers > header->w4f16_active_worker_max) {
        header->w4f16_active_worker_max = active_workers;
    }
}

static int qbh_run_w4f16_projection_early_region(
    struct qbh_block_header *header, const uint8_t *shared,
    const struct qbh_block_projection_desc *desc,
    struct qbh_block_buffers *buffers,
    struct qbh_block_hmx_worker *worker,
    struct qbh_block_w4f16_pool *pool,
    const void *activation_tiles, void *output) {
    uint32_t k_tiles = desc->k / QBH_HMX_FP16_COLS;
    uint32_t n_tiles = desc->n / QBH_HMX_FP16_COLS;
    uint32_t compressed_bytes = k_tiles * QBH_W4_PACKED_TILE_BYTES;
    uint32_t batch_count =
        (n_tiles + QBH_BLOCK_W4F16_DMA_BATCH_N_TILES - 1U) /
        QBH_BLOCK_W4F16_DMA_BATCH_N_TILES;
    uint32_t active_workers = qbh_w4f16_projection_worker_count(
        header, desc, pool);
    uint8_t *compressed_slots[2] = {
        buffers->compressed_weight, buffers->compressed_weight_alt};
    uint8_t *expanded_slots[2] = {
        buffers->expanded_weight, buffers->expanded_weight_alt};
    const float *projection_scales = qbh_w4f16_projection_scales(
        header, buffers, desc);
    volatile uint32_t ready[QBH_BLOCK_W4F16_MAX_REGIONS];
    struct qbh_dma_aligned_desc_1d prefetch_descriptor
        __attribute__((aligned(64)));
    uint64_t prefetch_start = 0U;
    int prefetch_active = 0;
    int result;

    memset((void *)ready, 0, sizeof(ready));
    qbh_w4f16_note_active_workers(header, active_workers);
    qbh_w4f16_note_effective_region(
        header, header->w4f16_region_tiles);
    {
        uint32_t first_batch_tiles =
            n_tiles < QBH_BLOCK_W4F16_DMA_BATCH_N_TILES
                ? n_tiles : QBH_BLOCK_W4F16_DMA_BATCH_N_TILES;
        uint32_t first_batch_bytes = first_batch_tiles * compressed_bytes;
        if (qbh_dma_copy(header, compressed_slots[0],
                         shared + desc->weight_offset,
                         first_batch_bytes, 1U) != 0) {
            qbh_record_projection_failure(header, desc, 0U, 31U, -1);
            return -1;
        }
        header->weight_ddr_read_bytes += first_batch_bytes;
        ++header->weight_dma_descriptor_count;
    }
    if (batch_count > 1U) {
        uint32_t second_batch_tiles =
            n_tiles - QBH_BLOCK_W4F16_DMA_BATCH_N_TILES;
        uint32_t second_batch_bytes;
        if (second_batch_tiles > QBH_BLOCK_W4F16_DMA_BATCH_N_TILES) {
            second_batch_tiles = QBH_BLOCK_W4F16_DMA_BATCH_N_TILES;
        }
        second_batch_bytes = second_batch_tiles * compressed_bytes;
        prefetch_start = HAP_perf_get_qtimer_count();
        result = qbh_dma_start_weight_prefetch(
            &prefetch_descriptor, compressed_slots[1],
            shared + desc->weight_offset +
                (size_t)QBH_BLOCK_W4F16_DMA_BATCH_N_TILES *
                    compressed_bytes,
            second_batch_bytes);
        if (result != 0) {
            qbh_record_projection_failure(header, desc, 0U, 32U, result);
            return -1;
        }
        prefetch_active = 1;
        ++header->w4f16_prefetch_count;
        header->weight_ddr_read_bytes += second_batch_bytes;
        ++header->weight_dma_descriptor_count;
    }

    for (uint32_t n_tile = 0; n_tile < n_tiles;
         n_tile += QBH_BLOCK_W4F16_HMX_BATCH_N_TILES) {
        uint32_t group_tiles = n_tiles - n_tile;
        uint32_t group_index =
            n_tile / QBH_BLOCK_W4F16_HMX_BATCH_N_TILES;
        uint32_t batch_index =
            n_tile / QBH_BLOCK_W4F16_DMA_BATCH_N_TILES;
        uint32_t in_batch =
            n_tile % QBH_BLOCK_W4F16_DMA_BATCH_N_TILES;
        uint32_t compressed_slot = batch_index & 1U;
        uint32_t expanded_slot = group_index & 1U;
        uint64_t command_start;
        uint64_t expand_start;
        uint64_t tail_wait_start;

        if (group_tiles > QBH_BLOCK_W4F16_HMX_BATCH_N_TILES) {
            group_tiles = QBH_BLOCK_W4F16_HMX_BATCH_N_TILES;
        }
        if (n_tile != 0U && in_batch == 0U) {
            uint64_t wait_start = HAP_perf_get_qtimer_count();
            result = qbh_dma_wait_weight_prefetch(&prefetch_descriptor);
            header->w4f16_prefetch_wait_ticks +=
                HAP_perf_get_qtimer_count() - wait_start;
            header->weight_dma_ticks +=
                HAP_perf_get_qtimer_count() - prefetch_start;
            prefetch_active = 0;
            if (result != 0) {
                qbh_record_projection_failure(
                    header, desc, n_tile, 33U, result);
                return -1;
            }
            if (batch_index + 1U < batch_count) {
                uint32_t following_batch = batch_index + 1U;
                uint32_t following_first_tile = following_batch *
                    QBH_BLOCK_W4F16_DMA_BATCH_N_TILES;
                uint32_t following_tiles = n_tiles - following_first_tile;
                uint32_t following_bytes;
                uint32_t following_slot = following_batch & 1U;
                if (following_tiles > QBH_BLOCK_W4F16_DMA_BATCH_N_TILES) {
                    following_tiles = QBH_BLOCK_W4F16_DMA_BATCH_N_TILES;
                }
                following_bytes = following_tiles * compressed_bytes;
                prefetch_start = HAP_perf_get_qtimer_count();
                result = qbh_dma_start_weight_prefetch(
                    &prefetch_descriptor, compressed_slots[following_slot],
                    shared + desc->weight_offset +
                        (size_t)following_first_tile * compressed_bytes,
                    following_bytes);
                if (result != 0) {
                    qbh_record_projection_failure(
                        header, desc, n_tile, 34U, result);
                    return -1;
                }
                prefetch_active = 1;
                ++header->w4f16_prefetch_count;
                header->weight_ddr_read_bytes += following_bytes;
                ++header->weight_dma_descriptor_count;
            }
        }

        for (uint32_t tile = 0; tile < group_tiles; ++tile) {
            qbh_hmx_fp16_init_channel_scales(
                buffers->scale_or_bias +
                    (size_t)tile * QBH_HMX_FP16_SCALE_BYTES,
                projection_scales +
                    (size_t)(n_tile + tile) * 32U);
        }
        command_start = HAP_perf_get_qtimer_count();
        qbh_hmx_start_fp16_tile_scales_streaming(
            worker, activation_tiles,
            expanded_slots[expanded_slot], buffers->scale_or_bias,
            buffers->hmx_output, 2U, k_tiles, group_tiles,
            header->w4f16_region_tiles, ready, group_index + 1U);
        ++header->w4f16_streamed_command_count;
        ++header->w4f16_early_region_command_count;

        expand_start = HAP_perf_get_qtimer_count();
        qbh_w4f16_expand_with_main(
            header, pool,
            compressed_slots[compressed_slot] +
                (size_t)in_batch * compressed_bytes,
            projection_scales + (size_t)n_tile * 32U,
            expanded_slots[expanded_slot], ready, group_index + 1U,
            k_tiles * group_tiles, header->w4f16_region_tiles,
            active_workers, 1U, 0U);
        if (group_index == 0U) {
            header->w4f16_first_expand_ticks +=
                HAP_perf_get_qtimer_count() - expand_start;
        } else {
            header->w4f16_steady_expand_ticks +=
                HAP_perf_get_qtimer_count() - expand_start;
        }
        header->w4f16_expand_ticks +=
            HAP_perf_get_qtimer_count() - expand_start;

        if (desc == &header->projections[0] && n_tile == 0U) {
            header->w4f16_expand_mismatch_count =
                qbh_audit_unscaled_w4_to_f16_tile(
                    compressed_slots[compressed_slot],
                    expanded_slots[expanded_slot],
                    &header->w4f16_expand_first_logical_index,
                    &header->w4f16_expand_expected_half_bits,
                    &header->w4f16_expand_actual_half_bits);
            if (header->w4f16_expand_mismatch_count != 0U) {
                (void)qbh_hmx_wait(worker);
                qbh_record_projection_failure(
                    header, desc, n_tile, 35U, -1);
                return -1;
            }
        }
        tail_wait_start = HAP_perf_get_qtimer_count();
        result = qbh_hmx_wait(worker);
        header->w4f16_hmx_tail_wait_ticks +=
            HAP_perf_get_qtimer_count() - tail_wait_start;
        header->projection_hmx_wait_ticks +=
            HAP_perf_get_qtimer_count() - command_start;
        header->hmx_fp16_tile_pair_count +=
            2U * k_tiles * group_tiles;
        ++header->hmx_command_count;
        if (result != 0) {
            qbh_record_projection_failure(
                header, desc, n_tile, 36U, result);
            return -1;
        }
        {
            uint64_t unpack_start = HAP_perf_get_qtimer_count();
            qbh_unpack_fp16_output(
                (const __fp16 *)buffers->hmx_output, group_tiles,
                (__fp16 *)output, desc->n,
                n_tile * QBH_HMX_FP16_COLS);
            header->projection_unpack_ticks +=
                HAP_perf_get_qtimer_count() - unpack_start;
            qbh_hvx_pool_qk_norm_rope_publish(
                header, desc, pool, n_tile, group_tiles);
            if (qbh_mlp_stream_publish_up_group(
                    header, desc, pool, n_tile,
                    group_tiles) != 0) {
                qbh_record_projection_failure(
                    header, desc, n_tile, 40U, -1);
                return -1;
            }
        }
    }
    if (prefetch_active != 0) {
        qbh_record_projection_failure(header, desc, n_tiles, 37U, -1);
        return -1;
    }
    return 0;
}

static int qbh_run_w4f16_projection(
    struct qbh_block_header *header, const uint8_t *shared,
    const struct qbh_block_projection_desc *desc,
    struct qbh_block_buffers *buffers,
    struct qbh_block_hmx_worker *worker,
    struct qbh_block_w4f16_pool *pool,
    const void *activation_tiles, void *output,
    const struct qbh_block_projection_desc *next_desc,
    struct qbh_block_w4f16_cross_prefetch *cross_prefetch) {
    uint32_t k_tiles = desc->k / QBH_HMX_FP16_COLS;
    uint32_t n_tiles = desc->n / QBH_HMX_FP16_COLS;
    uint32_t compressed_bytes = k_tiles * QBH_W4_PACKED_TILE_BYTES;
    uint32_t batch_count =
        (n_tiles + QBH_BLOCK_W4F16_DMA_BATCH_N_TILES - 1U) /
        QBH_BLOCK_W4F16_DMA_BATCH_N_TILES;
    uint8_t *compressed_slots[2] = {
        buffers->compressed_weight, buffers->compressed_weight_alt};
    uint8_t *expanded_slots[2] = {
        buffers->expanded_weight, buffers->expanded_weight_alt};
    const float *projection_scales = qbh_w4f16_projection_scales(
        header, buffers, desc);
    volatile uint32_t ready[QBH_BLOCK_W4F16_MAX_REGIONS];
    struct qbh_dma_aligned_desc_1d prefetch_descriptor
        __attribute__((aligned(64)));
    uint64_t phase_start;
    uint64_t prefetch_start = 0U;
    int prefetch_active = 0;
    int result;
    int first_batch_state;
    int cross_prefetch_result = 0;
    uint32_t cross_prefetch_attempted = 0U;
    uint32_t active_workers;
    uint32_t region_tiles;
    const uint32_t hmx_batch_tiles =
        qbh_w4f16_projection_group_tiles(header, desc);
    const uint32_t relaxed_down_group_fence =
        desc == &header->projections[QBH_BLOCK_PROJ_DOWN] &&
        header->w4f16_group_fence_mode ==
            QBH_BLOCK_W4F16_GROUP_FENCE_JOIN_ONLY_DOWN;
    const uint32_t direct_qkv =
        qbh_projection_direct_qkv_crouton(header, desc);

    if (pool == NULL) {
        qbh_record_projection_failure(header, desc, 0U, 10U, -1);
        return -1;
    }
    if (header->w4f16_pipeline_mode ==
            QBH_BLOCK_W4F16_PIPELINE_EARLY_REGION) {
        return qbh_run_w4f16_projection_early_region(
            header, shared, desc, buffers, worker, pool,
            activation_tiles, output);
    }
    active_workers = qbh_w4f16_projection_worker_count(
        header, desc, pool);
    region_tiles = qbh_w4f16_projection_region_tiles(header, desc);
    qbh_w4f16_note_active_workers(header, active_workers);
    qbh_w4f16_note_effective_region(header, region_tiles);
    memset((void *)ready, 0, sizeof(ready));

    first_batch_state = qbh_w4f16_consume_cross_prefetch(
        header, desc, cross_prefetch);
    if (first_batch_state < 0) {
        qbh_record_projection_failure(
            header, desc, 0U, 38U, first_batch_state);
        return -1;
    }
    if (first_batch_state == 0) {
        uint32_t first_batch_tiles =
            n_tiles < QBH_BLOCK_W4F16_DMA_BATCH_N_TILES
                ? n_tiles : QBH_BLOCK_W4F16_DMA_BATCH_N_TILES;
        uint32_t first_batch_bytes = first_batch_tiles * compressed_bytes;
        if (qbh_dma_copy(header, compressed_slots[0],
                         shared + desc->weight_offset,
                         first_batch_bytes, 1U) != 0) {
            qbh_record_projection_failure(header, desc, 0U, 11U, -1);
            return -1;
        }
        header->weight_ddr_read_bytes += first_batch_bytes;
        ++header->weight_dma_descriptor_count;
    }

    if (batch_count > 1U) {
        uint32_t second_batch_tiles =
            n_tiles - QBH_BLOCK_W4F16_DMA_BATCH_N_TILES;
        uint32_t second_batch_bytes;
        if (second_batch_tiles > QBH_BLOCK_W4F16_DMA_BATCH_N_TILES) {
            second_batch_tiles = QBH_BLOCK_W4F16_DMA_BATCH_N_TILES;
        }
        second_batch_bytes = second_batch_tiles * compressed_bytes;
        prefetch_start = HAP_perf_get_qtimer_count();
        result = qbh_dma_start_weight_prefetch(
            &prefetch_descriptor, compressed_slots[1],
            shared + desc->weight_offset +
                (size_t)QBH_BLOCK_W4F16_DMA_BATCH_N_TILES *
                    compressed_bytes,
            second_batch_bytes);
        if (result != 0) {
            qbh_record_projection_failure(
                header, desc, 0U, 14U, result);
            return -1;
        }
        prefetch_active = 1;
        ++header->w4f16_prefetch_count;
        header->weight_ddr_read_bytes += second_batch_bytes;
        ++header->weight_dma_descriptor_count;
    }

    phase_start = HAP_perf_get_qtimer_count();
    qbh_w4f16_expand_with_main(
        header, pool, compressed_slots[0], projection_scales,
        expanded_slots[0], ready, 1U,
        k_tiles * hmx_batch_tiles,
        region_tiles, active_workers, 0U,
        relaxed_down_group_fence);
    header->w4f16_expand_ticks +=
        HAP_perf_get_qtimer_count() - phase_start;
    header->w4f16_first_expand_ticks +=
        HAP_perf_get_qtimer_count() - phase_start;
    if (desc == &header->projections[0]) {
        header->w4f16_expand_mismatch_count =
            qbh_audit_unscaled_w4_to_f16_tile(
                compressed_slots[0], expanded_slots[0],
                &header->w4f16_expand_first_logical_index,
                &header->w4f16_expand_expected_half_bits,
                &header->w4f16_expand_actual_half_bits);
        if (header->w4f16_expand_mismatch_count != 0U) {
            qbh_record_projection_failure(header, desc, 0U, 12U, -1);
            return -1;
        }
    }

    for (uint32_t n_tile = 0; n_tile < n_tiles;
         n_tile += hmx_batch_tiles) {
        uint32_t group_tiles = n_tiles - n_tile;
        uint32_t group_index = n_tile / hmx_batch_tiles;
        uint32_t current_expanded_slot = group_index & 1U;
        uint32_t next_tile;
        __fp16 *hmx_output = (__fp16 *)buffers->hmx_output;

        if (group_tiles > hmx_batch_tiles) {
            group_tiles = hmx_batch_tiles;
        }
        if (direct_qkv != 0U) {
            const uint32_t output_group_elements =
                (QBH_BLOCK_M / QBH_HMX_FP16_ROWS) *
                group_tiles * QBH_HMX_FP16_TILE_ELEMENTS;
            hmx_output = (__fp16 *)output +
                (size_t)group_index * output_group_elements;
        }
        next_tile = n_tile + group_tiles;
        for (uint32_t tile = 0; tile < group_tiles; ++tile) {
            qbh_hmx_fp16_init_channel_scales(
                buffers->scale_or_bias +
                    (size_t)tile * QBH_HMX_FP16_SCALE_BYTES,
                projection_scales +
                    (size_t)(n_tile + tile) * 32U);
        }
        if (desc == &header->projections[0] && n_tile == 0U) {
            header->w4f16_expand_mismatch_count =
                qbh_hmx_fp16_audit_channel_scales(
                    buffers->scale_or_bias,
                    projection_scales,
                    &header->w4f16_expand_first_logical_index,
                    &header->w4f16_expand_expected_half_bits,
                    &header->w4f16_expand_actual_half_bits);
            if (header->w4f16_expand_mismatch_count != 0U) {
                qbh_record_projection_failure(
                    header, desc, n_tile, 13U, -1);
                return -1;
            }
        }

        phase_start = HAP_perf_get_qtimer_count();
        qbh_hmx_start_fp16_tile_scales(
            worker, activation_tiles,
            expanded_slots[current_expanded_slot],
            buffers->scale_or_bias,
            hmx_output, 2U, k_tiles, group_tiles);
        ++header->w4f16_streamed_command_count;
        if (next_tile < n_tiles) {
            uint32_t next_group_tiles = n_tiles - next_tile;
            uint32_t next_batch =
                next_tile / QBH_BLOCK_W4F16_DMA_BATCH_N_TILES;
            uint32_t next_in_batch =
                next_tile % QBH_BLOCK_W4F16_DMA_BATCH_N_TILES;
            uint32_t next_compressed_slot = next_batch & 1U;
            uint32_t next_expanded_slot =
                (next_tile / hmx_batch_tiles) & 1U;

            if (next_group_tiles > hmx_batch_tiles) {
                next_group_tiles = hmx_batch_tiles;
            }

            if (next_in_batch == 0U) {
                uint64_t wait_start = HAP_perf_get_qtimer_count();
                result = qbh_dma_wait_weight_prefetch(
                    &prefetch_descriptor);
                header->w4f16_prefetch_wait_ticks +=
                    HAP_perf_get_qtimer_count() - wait_start;
                header->weight_dma_ticks +=
                    HAP_perf_get_qtimer_count() - prefetch_start;
                prefetch_active = 0;
                if (result != 0) {
                    (void)qbh_hmx_wait(worker);
                    qbh_record_projection_failure(
                        header, desc, n_tile, 15U, result);
                    return -1;
                }

                if (next_batch + 1U < batch_count) {
                    uint32_t following_batch = next_batch + 1U;
                    uint32_t following_first_tile =
                        following_batch *
                        QBH_BLOCK_W4F16_DMA_BATCH_N_TILES;
                    uint32_t following_batch_tiles =
                        n_tiles - following_first_tile;
                    uint32_t following_batch_bytes;
                    uint32_t following_slot = following_batch & 1U;
                    if (following_batch_tiles >
                        QBH_BLOCK_W4F16_DMA_BATCH_N_TILES) {
                        following_batch_tiles =
                            QBH_BLOCK_W4F16_DMA_BATCH_N_TILES;
                    }
                    following_batch_bytes =
                        following_batch_tiles * compressed_bytes;
                    prefetch_start = HAP_perf_get_qtimer_count();
                    result = qbh_dma_start_weight_prefetch(
                        &prefetch_descriptor,
                        compressed_slots[following_slot],
                        shared + desc->weight_offset +
                            (size_t)following_first_tile *
                                compressed_bytes,
                        following_batch_bytes);
                    if (result != 0) {
                        (void)qbh_hmx_wait(worker);
                        qbh_record_projection_failure(
                            header, desc, n_tile, 14U, result);
                        return -1;
                    }
                    prefetch_active = 1;
                    ++header->w4f16_prefetch_count;
                    header->weight_ddr_read_bytes +=
                        following_batch_bytes;
                    ++header->weight_dma_descriptor_count;
                }
            }

            {
                uint64_t expand_start = HAP_perf_get_qtimer_count();
                qbh_w4f16_expand_with_main(
                    header, pool,
                    compressed_slots[next_compressed_slot] +
                        (size_t)next_in_batch * compressed_bytes,
                    projection_scales + (size_t)next_tile * 32U,
                    expanded_slots[next_expanded_slot], ready,
                    group_index + 2U,
                    k_tiles * next_group_tiles,
                    region_tiles, active_workers, 0U,
                    relaxed_down_group_fence);
                header->w4f16_expand_ticks +=
                    HAP_perf_get_qtimer_count() - expand_start;
                header->w4f16_steady_expand_ticks +=
                    HAP_perf_get_qtimer_count() - expand_start;
            }
            if (next_tile + next_group_tiles >= n_tiles &&
                cross_prefetch_attempted == 0U) {
                cross_prefetch_attempted = 1U;
                cross_prefetch_result =
                    qbh_w4f16_start_cross_prefetch(
                        header, shared, next_desc, buffers,
                        cross_prefetch);
            }
        } else if (cross_prefetch_attempted == 0U) {
            cross_prefetch_attempted = 1U;
            cross_prefetch_result = qbh_w4f16_start_cross_prefetch(
                header, shared, next_desc, buffers, cross_prefetch);
        }

        {
            uint64_t tail_wait_start = HAP_perf_get_qtimer_count();
            result = qbh_hmx_wait(worker);
            header->w4f16_hmx_tail_wait_ticks +=
                HAP_perf_get_qtimer_count() - tail_wait_start;
        }
        header->projection_hmx_wait_ticks +=
            HAP_perf_get_qtimer_count() - phase_start;
        header->hmx_fp16_tile_pair_count +=
            2U * k_tiles * group_tiles;
        ++header->hmx_command_count;
        if (result != 0 || cross_prefetch_result != 0) {
            qbh_w4f16_drain_cross_prefetch(header, cross_prefetch);
            qbh_record_projection_failure(
                header, desc, n_tile,
                result != 0 ? 17U : 39U,
                result != 0 ? result : cross_prefetch_result);
            return -1;
        }
        if (direct_qkv != 0U) {
            header->crouton_qkv_unpack_skipped += group_tiles;
            qbh_hvx_pool_qk_norm_rope_publish(
                header, desc, pool, n_tile, group_tiles);
        } else {
            uint64_t unpack_start = HAP_perf_get_qtimer_count();
            qbh_unpack_fp16_output(
                (const __fp16 *)buffers->hmx_output, group_tiles,
                (__fp16 *)output, desc->n,
                n_tile * QBH_HMX_FP16_COLS);
            header->projection_unpack_ticks +=
                HAP_perf_get_qtimer_count() - unpack_start;
            qbh_capture_row_major_qkv_reference(
                header, desc, buffers, (const __fp16 *)output,
                n_tile, group_tiles);
            qbh_hvx_pool_qk_norm_rope_publish(
                header, desc, pool, n_tile, group_tiles);
            if (qbh_mlp_stream_publish_up_group(
                    header, desc, pool, n_tile,
                    group_tiles) != 0) {
                qbh_record_projection_failure(
                    header, desc, n_tile, 40U, -1);
                return -1;
            }
        }
    }
    if (prefetch_active != 0) {
        qbh_record_projection_failure(header, desc, n_tiles, 18U, -1);
        return -1;
    }
    return 0;
}

struct qbh_w4f16_mlp_projection_state {
    const struct qbh_block_projection_desc *desc;
    uint8_t *compressed_slots[2];
    uint8_t *expanded_slots[2];
    const float *scales;
    const uint8_t *scale_blocks;
    __fp16 *output;
    struct qbh_dma_aligned_desc_1d prefetch_descriptor
        __attribute__((aligned(64)));
    uint64_t prefetch_start;
    uint32_t prefetched_batch;
    uint32_t prefetch_active;
    uint32_t k_tiles;
    uint32_t n_tiles;
    uint32_t compressed_bytes_per_tile;
    uint32_t batch_count;
    uint32_t group_tiles;
    uint32_t dma_batch_tiles;
    volatile uint32_t ready[QBH_BLOCK_W4F16_MAX_REGIONS];
};

static int qbh_w4f16_mlp_start_prefetch(
    struct qbh_block_header *header, const uint8_t *shared,
    struct qbh_w4f16_mlp_projection_state *state,
    uint32_t batch) {
    uint32_t first_tile;
    uint32_t batch_tiles;
    uint32_t bytes;
    uint32_t slot;
    int result;

    if (batch >= state->batch_count) {
        return 0;
    }
    if (state->prefetch_active != 0U) {
        return -1;
    }
    first_tile = batch * state->dma_batch_tiles;
    batch_tiles = state->n_tiles - first_tile;
    if (batch_tiles > state->dma_batch_tiles) {
        batch_tiles = state->dma_batch_tiles;
    }
    bytes = batch_tiles * state->compressed_bytes_per_tile;
    slot = batch & 1U;
    state->prefetch_start = HAP_perf_get_qtimer_count();
    result = qbh_dma_start_weight_prefetch(
        &state->prefetch_descriptor, state->compressed_slots[slot],
        shared + state->desc->weight_offset +
            (size_t)first_tile * state->compressed_bytes_per_tile,
        bytes);
    if (result != 0) {
        state->prefetch_start = 0U;
        return result;
    }
    state->prefetched_batch = batch;
    state->prefetch_active = 1U;
    ++header->w4f16_prefetch_count;
    header->weight_ddr_read_bytes += bytes;
    ++header->weight_dma_descriptor_count;
    return 0;
}

static int qbh_w4f16_mlp_wait_batch(
    struct qbh_block_header *header, const uint8_t *shared,
    struct qbh_w4f16_mlp_projection_state *state,
    uint32_t batch) {
    uint64_t wait_start;
    uint64_t end;
    int result;

    if (batch == 0U) {
        return 0;
    }
    if (state->prefetch_active == 0U ||
        state->prefetched_batch != batch) {
        return -1;
    }
    wait_start = HAP_perf_get_qtimer_count();
    result = qbh_dma_wait_weight_prefetch(
        &state->prefetch_descriptor);
    end = HAP_perf_get_qtimer_count();
    header->w4f16_prefetch_wait_ticks += end - wait_start;
    header->weight_dma_ticks += end - state->prefetch_start;
    state->prefetch_active = 0U;
    state->prefetch_start = 0U;
    if (result != 0) {
        return result;
    }
    return qbh_w4f16_mlp_start_prefetch(
        header, shared, state, batch + 1U);
}

static void qbh_w4f16_mlp_drain_prefetch(
    struct qbh_block_header *header,
    struct qbh_w4f16_mlp_projection_state *state) {
    uint64_t wait_start;
    uint64_t end;

    if (state->prefetch_active == 0U) {
        return;
    }
    wait_start = HAP_perf_get_qtimer_count();
    (void)qbh_dma_wait_weight_prefetch(&state->prefetch_descriptor);
    end = HAP_perf_get_qtimer_count();
    header->w4f16_prefetch_wait_ticks += end - wait_start;
    header->weight_dma_ticks += end - state->prefetch_start;
    state->prefetch_active = 0U;
    state->prefetch_start = 0U;
}

static int qbh_w4f16_mlp_prepare_group(
    struct qbh_block_header *header, const uint8_t *shared,
    struct qbh_block_w4f16_pool *pool,
    struct qbh_w4f16_mlp_projection_state *state,
    uint32_t group) {
    uint32_t first_tile = group * state->group_tiles;
    uint32_t batch =
        first_tile / state->dma_batch_tiles;
    uint32_t in_batch =
        first_tile % state->dma_batch_tiles;
    uint32_t compressed_slot = batch & 1U;
    uint32_t expanded_slot = group & 1U;
    uint32_t region_tiles =
        qbh_w4f16_gate_up_region_tiles(header);
    uint64_t expand_start;
    int result;

    if (in_batch == 0U && batch != 0U) {
        result = qbh_w4f16_mlp_wait_batch(
            header, shared, state, batch);
        if (result != 0) {
            return result;
        }
    }
    expand_start = HAP_perf_get_qtimer_count();
    qbh_w4f16_expand_with_main(
        header, pool,
        state->compressed_slots[compressed_slot] +
            (size_t)in_batch * state->compressed_bytes_per_tile,
        state->scales + (size_t)first_tile * 32U,
        state->expanded_slots[expanded_slot], state->ready,
        group + 1U,
        state->k_tiles * state->group_tiles,
        region_tiles, 2U, 0U,
        header->w4f16_group_fence_mode !=
            QBH_BLOCK_W4F16_GROUP_FENCE_CONTROL);
    header->w4f16_expand_ticks +=
        HAP_perf_get_qtimer_count() - expand_start;
    if (group == 0U) {
        header->w4f16_first_expand_ticks +=
            HAP_perf_get_qtimer_count() - expand_start;
    } else {
        header->w4f16_steady_expand_ticks +=
            HAP_perf_get_qtimer_count() - expand_start;
    }
    return 0;
}

static void qbh_w4f16_mlp_start_group(
    struct qbh_block_header *header,
    struct qbh_block_hmx_worker *worker,
    struct qbh_block_buffers *buffers,
    const __fp16 *activation_tiles,
    struct qbh_w4f16_mlp_projection_state *state,
    uint32_t group) {
    uint32_t first_tile = group * state->group_tiles;
    uint32_t expanded_slot = group & 1U;
    uint64_t scale_start = HAP_perf_get_qtimer_count();
    const void *scale_blocks;
    __fp16 *output_tiles = (__fp16 *)buffers->hmx_output;

    if (state->scale_blocks != NULL) {
        scale_blocks = state->scale_blocks +
            (size_t)first_tile * QBH_HMX_FP16_SCALE_BYTES;
    } else {
        for (uint32_t tile = 0U;
             tile < state->group_tiles; ++tile) {
            qbh_hmx_fp16_init_channel_scales(
                buffers->scale_or_bias +
                    (size_t)tile * QBH_HMX_FP16_SCALE_BYTES,
                state->scales + (size_t)(first_tile + tile) * 32U);
        }
        scale_blocks = buffers->scale_or_bias;
    }
    header->w4f16_gate_up_scale_init_ticks +=
        HAP_perf_get_qtimer_count() - scale_start;
    if (header->mlp_mode == QBH_BLOCK_MLP_CROUTON_NATIVE ||
        header->mlp_mode == QBH_BLOCK_MLP_CROUTON_NATIVE_BATCH8) {
        const uint32_t output_group_elements =
            (QBH_BLOCK_M / QBH_HMX_FP16_ROWS) *
            state->group_tiles * QBH_HMX_FP16_TILE_ELEMENTS;
        output_tiles = state->output +
            (size_t)(group % QBH_BLOCK_MLP_CROUTON_RING_SLOTS) *
                output_group_elements;
    }
    qbh_hmx_start_fp16_tile_scales(
        worker, activation_tiles,
        state->expanded_slots[expanded_slot],
        scale_blocks, output_tiles, 2U,
        state->k_tiles, state->group_tiles);
    ++header->w4f16_streamed_command_count;
}

static int qbh_w4f16_mlp_finish_group(
    struct qbh_block_header *header,
    struct qbh_block_hmx_worker *worker,
    struct qbh_block_buffers *buffers,
    struct qbh_block_w4f16_pool *pool,
    struct qbh_w4f16_mlp_projection_state *state,
    uint32_t group, uint64_t command_start) {
    uint32_t first_tile = group * state->group_tiles;
    uint64_t wait_start = HAP_perf_get_qtimer_count();
    int result = qbh_hmx_wait(worker);

    header->w4f16_hmx_tail_wait_ticks +=
        HAP_perf_get_qtimer_count() - wait_start;
    header->projection_hmx_wait_ticks +=
        HAP_perf_get_qtimer_count() - command_start;
    header->hmx_fp16_tile_pair_count +=
        2U * state->k_tiles * state->group_tiles;
    ++header->hmx_command_count;
    if (result != 0) {
        return result;
    }
    if (header->mlp_mode == QBH_BLOCK_MLP_CROUTON_NATIVE ||
        header->mlp_mode == QBH_BLOCK_MLP_CROUTON_NATIVE_BATCH8) {
        return qbh_mlp_crouton_publish_up_group(
            header, state->desc, pool, group);
    }
    {
        uint64_t unpack_start = HAP_perf_get_qtimer_count();
        qbh_unpack_fp16_output(
            (const __fp16 *)buffers->hmx_output,
            state->group_tiles, state->output,
            state->desc->n, first_tile * QBH_HMX_FP16_COLS);
        header->projection_unpack_ticks +=
            HAP_perf_get_qtimer_count() - unpack_start;
    }
    return qbh_mlp_stream_publish_up_group(
        header, state->desc, pool, first_tile,
        state->group_tiles);
}

static int qbh_run_w4f16_interleaved_gate_up(
    struct qbh_block_header *header, const uint8_t *shared,
    struct qbh_block_buffers *buffers,
    struct qbh_block_hmx_worker *worker,
    struct qbh_block_w4f16_pool *pool,
    struct qbh_block_w4f16_cross_prefetch *cross_prefetch) {
    struct qbh_w4f16_mlp_projection_state gate;
    struct qbh_w4f16_mlp_projection_state up;
    struct qbh_w4f16_mlp_projection_state *states[2] = {
        &gate, &up};
    const uint32_t compressed_capacity =
        QBH_BLOCK_MAX_K * QBH_HMX_OUTPUT_CHANNELS / 2U *
        QBH_BLOCK_W4F16_DMA_BATCH_N_TILES;
    const uint32_t expanded_capacity =
        header->mlp_mode == QBH_BLOCK_MLP_CROUTON_NATIVE_BATCH8
            ? QBH_BLOCK_HIDDEN * 8U * QBH_HMX_FP16_COLS *
                  sizeof(uint16_t) +
              8U * (QBH_BLOCK_HIDDEN / QBH_HMX_FP16_COLS) *
                  QBH_W4_PACKED_TILE_BYTES
            : QBH_BLOCK_MAX_K * QBH_HMX_OUTPUT_CHANNELS *
                  sizeof(uint16_t) *
                  QBH_BLOCK_W4F16_HMX_BATCH_N_TILES;
    uint32_t compressed_batch_bytes;
    uint32_t expanded_group_bytes;
    uint32_t group_count;
    uint32_t group_tiles;
    uint32_t dma_batch_tiles;
    uint64_t pack_start;
    int result = 0;

    if (pool == NULL || header->variant != QBH_BLOCK_W4F16 ||
        (header->mlp_mode != QBH_BLOCK_MLP_STREAMING &&
         header->mlp_mode != QBH_BLOCK_MLP_CROUTON_NATIVE &&
         header->mlp_mode != QBH_BLOCK_MLP_CROUTON_NATIVE_BATCH8)) {
        return -1;
    }
    memset(&gate, 0, sizeof(gate));
    memset(&up, 0, sizeof(up));
    gate.desc = &header->projections[QBH_BLOCK_PROJ_GATE];
    up.desc = &header->projections[QBH_BLOCK_PROJ_UP];
    gate.k_tiles = gate.desc->k / QBH_HMX_FP16_COLS;
    up.k_tiles = up.desc->k / QBH_HMX_FP16_COLS;
    gate.n_tiles = gate.desc->n / QBH_HMX_FP16_COLS;
    up.n_tiles = up.desc->n / QBH_HMX_FP16_COLS;
    group_tiles = qbh_w4f16_gate_up_group_tiles(header);
    dma_batch_tiles = qbh_w4f16_dma_batch_tiles(header, gate.desc);
    gate.group_tiles = group_tiles;
    up.group_tiles = group_tiles;
    gate.dma_batch_tiles = dma_batch_tiles;
    up.dma_batch_tiles = dma_batch_tiles;
    if (gate.k_tiles != up.k_tiles ||
        gate.n_tiles != up.n_tiles ||
        gate.n_tiles % group_tiles != 0U ||
        ((header->mlp_mode == QBH_BLOCK_MLP_CROUTON_NATIVE &&
          group_tiles != QBH_BLOCK_MLP_CROUTON_GROUP_TILES) ||
         (header->mlp_mode == QBH_BLOCK_MLP_CROUTON_NATIVE_BATCH8 &&
          group_tiles != 8U))) {
        return -1;
    }
    gate.compressed_bytes_per_tile =
        gate.k_tiles * QBH_W4_PACKED_TILE_BYTES;
    up.compressed_bytes_per_tile =
        up.k_tiles * QBH_W4_PACKED_TILE_BYTES;
    gate.batch_count =
        (gate.n_tiles + dma_batch_tiles - 1U) / dma_batch_tiles;
    up.batch_count = gate.batch_count;
    compressed_batch_bytes =
        dma_batch_tiles *
        gate.compressed_bytes_per_tile;
    expanded_group_bytes =
        group_tiles * gate.k_tiles *
        QBH_HMX_FP16_TILE_BYTES;
    if ((dma_batch_tiles == 8U
             ? compressed_batch_bytes > compressed_capacity ||
                   expanded_group_bytes + compressed_batch_bytes >
                       expanded_capacity
             : 2U * compressed_batch_bytes > compressed_capacity) ||
        (group_tiles >= 4U
             ? expanded_group_bytes > expanded_capacity
             : 2U * expanded_group_bytes > expanded_capacity)) {
        return -1;
    }
    gate.compressed_slots[0] = buffers->compressed_weight;
    gate.compressed_slots[1] = dma_batch_tiles == 8U
        ? buffers->expanded_weight + expanded_group_bytes
        : buffers->compressed_weight + compressed_batch_bytes;
    up.compressed_slots[0] = buffers->compressed_weight_alt;
    up.compressed_slots[1] = dma_batch_tiles == 8U
        ? buffers->expanded_weight_alt + expanded_group_bytes
        : buffers->compressed_weight_alt + compressed_batch_bytes;
    if (dma_batch_tiles == 8U) {
        uint32_t projection_scale_cache_bytes =
            gate.n_tiles * QBH_HMX_FP16_SCALE_BYTES;
        gate.scale_blocks = buffers->gate_up_scale_cache;
        up.scale_blocks =
            buffers->gate_up_scale_cache + projection_scale_cache_bytes;
    }
    gate.expanded_slots[0] = buffers->expanded_weight;
    up.expanded_slots[0] = buffers->expanded_weight_alt;
    if (group_tiles >= 4U) {
        gate.expanded_slots[1] = buffers->expanded_weight;
        up.expanded_slots[1] = buffers->expanded_weight_alt;
    } else {
        gate.expanded_slots[1] =
            buffers->expanded_weight + expanded_group_bytes;
        up.expanded_slots[1] =
            buffers->expanded_weight_alt + expanded_group_bytes;
    }
    gate.scales = qbh_w4f16_projection_scales(
        header, buffers, gate.desc);
    up.scales = qbh_w4f16_projection_scales(
        header, buffers, up.desc);
    gate.output = (__fp16 *)buffers->gate;
    up.output = (__fp16 *)buffers->up;
    group_count = gate.n_tiles / group_tiles;
    qbh_w4f16_note_active_workers(
        header, 2U + header->w4f16_gate_up_extra_expand_worker);
    qbh_w4f16_note_effective_region(
        header, qbh_w4f16_gate_up_region_tiles(header));
    header->w4f16_gate_up_effective_region_tiles =
        qbh_w4f16_gate_up_region_tiles(header);

    if ((header->crouton_boundary_mode &
         QBH_BLOCK_CROUTON_BOUNDARY_POST_NORM) == 0U) {
        pack_start = HAP_perf_get_qtimer_count();
        qbh_pack_fp16_activation(
            (const __fp16 *)buffers->normalized, QBH_BLOCK_HIDDEN,
            QBH_BLOCK_HIDDEN, (__fp16 *)buffers->hmx_activation);
        header->projection_pack_ticks +=
            HAP_perf_get_qtimer_count() - pack_start;
    }

    result = qbh_w4f16_consume_cross_prefetch(
        header, gate.desc, cross_prefetch);
    if (result < 0) {
        qbh_record_projection_failure(
            header, gate.desc, 0U, 50U, result);
        return -1;
    }
    if (result == 0) {
        if (qbh_dma_copy(
                header, gate.compressed_slots[0],
                shared + gate.desc->weight_offset,
                compressed_batch_bytes, 1U) != 0) {
            qbh_record_projection_failure(
                header, gate.desc, 0U, 51U, -1);
            return -1;
        }
        header->weight_ddr_read_bytes += compressed_batch_bytes;
        ++header->weight_dma_descriptor_count;
    }
    if (qbh_dma_copy(
            header, up.compressed_slots[0],
            shared + up.desc->weight_offset,
            compressed_batch_bytes, 1U) != 0) {
        qbh_record_projection_failure(
            header, up.desc, 0U, 52U, -1);
        return -1;
    }
    header->weight_ddr_read_bytes += compressed_batch_bytes;
    ++header->weight_dma_descriptor_count;
    if (qbh_w4f16_mlp_start_prefetch(
            header, shared, &gate, 1U) != 0 ||
        qbh_w4f16_mlp_start_prefetch(
            header, shared, &up, 1U) != 0) {
        qbh_w4f16_mlp_drain_prefetch(header, &gate);
        qbh_w4f16_mlp_drain_prefetch(header, &up);
        qbh_record_projection_failure(
            header, up.desc, 0U, 53U, -1);
        return -1;
    }
    if (qbh_w4f16_mlp_prepare_group(
            header, shared, pool, &gate, 0U) != 0) {
        result = -1;
        goto fused_gate_up_failed;
    }

    for (uint32_t group = 0U; group < group_count; ++group) {
        if (qbh_mlp_crouton_wait_slot(
                header, pool, group) != 0) {
            qbh_record_projection_failure(
                header, gate.desc, group * group_tiles, 55U, -1);
            result = -1;
            goto fused_gate_up_failed;
        }
        for (uint32_t projection = 0U; projection < 2U;
             ++projection) {
            struct qbh_w4f16_mlp_projection_state *current =
                states[projection];
            uint32_t next_group = group;
            uint32_t next_projection = projection + 1U;
            uint64_t command_start = HAP_perf_get_qtimer_count();

            if (next_projection == 2U) {
                next_projection = 0U;
                ++next_group;
            }
            qbh_w4f16_mlp_start_group(
                header, worker, buffers,
                (const __fp16 *)buffers->hmx_activation,
                current, group);
            if (next_group < group_count) {
                result = qbh_w4f16_mlp_prepare_group(
                    header, shared, pool,
                    states[next_projection], next_group);
            } else {
                result = qbh_w4f16_start_cross_prefetch(
                    header, shared,
                    &header->projections[QBH_BLOCK_PROJ_DOWN],
                    buffers, cross_prefetch);
            }
            if (qbh_w4f16_mlp_finish_group(
                    header, worker, buffers, pool, current,
                    group, command_start) != 0 || result != 0) {
                qbh_record_projection_failure(
                    header, current->desc,
                    group * group_tiles,
                    54U, result);
                result = -1;
                goto fused_gate_up_failed;
            }
        }
    }
    if (gate.prefetch_active != 0U || up.prefetch_active != 0U) {
        result = -1;
        goto fused_gate_up_failed;
    }
    return 0;

fused_gate_up_failed:
    qbh_w4f16_mlp_drain_prefetch(header, &gate);
    qbh_w4f16_mlp_drain_prefetch(header, &up);
    qbh_w4f16_drain_cross_prefetch(header, cross_prefetch);
    return result;
}

static uint32_t qbh_w4u8_qkvo_batch_tiles(
    const struct qbh_block_header *header,
    const struct qbh_block_projection_desc *desc) {
    if (header == NULL || desc == NULL ||
        header->variant != QBH_BLOCK_W4U8 ||
        header->w4u8_qkvo_pipeline_mode ==
            QBH_BLOCK_W4U8_QKVO_SERIAL) {
        return 0U;
    }
    if (desc == &header->projections[QBH_BLOCK_PROJ_Q] ||
        desc == &header->projections[QBH_BLOCK_PROJ_K] ||
        desc == &header->projections[QBH_BLOCK_PROJ_V]) {
        return header->w4u8_qkvo_pipeline_mode ==
                       QBH_BLOCK_W4U8_QKV_BATCH2
                   ? 2U : 4U;
    }
    if (desc == &header->projections[QBH_BLOCK_PROJ_O] &&
        header->w4u8_qkvo_pipeline_mode >=
            QBH_BLOCK_W4U8_QKVO_BATCH4) {
        return 4U;
    }
    return 0U;
}

static int qbh_run_w4u8_qkvo_pipelined_projection(
    struct qbh_block_header *header, const uint8_t *shared,
    const struct qbh_block_projection_desc *desc,
    struct qbh_block_buffers *buffers,
    struct qbh_block_hmx_worker *worker,
    struct qbh_block_w4f16_pool *w4f16_pool,
    const uint8_t *projection_activation, uint8_t *output,
    uint32_t direct_output, uint32_t batch_tiles) {
    struct qbh_dma_aligned_desc_1d prefetch_descriptors[2]
        __attribute__((aligned(64)));
    uint8_t *compressed_slots[2] = {
        buffers->compressed_weight, buffers->compressed_weight_alt};
    uint8_t *expanded_slots[2] = {
        buffers->expanded_weight, buffers->expanded_weight_alt};
    uint8_t *bias_slots[2] = {
        buffers->scale_or_bias,
        buffers->scale_or_bias +
            QBH_BLOCK_W4U8_QKVO_MAX_BATCH_N_TILES *
                QBH_HMX_BIAS_BYTES};
    uint32_t k_tiles = desc->k / QBH_HMX_INPUT_CHANNELS;
    uint32_t n_tiles = desc->n / QBH_HMX_OUTPUT_CHANNELS;
    uint32_t compressed_tile_bytes =
        k_tiles * QBH_W4_PACKED_TILE_BYTES;
    uint32_t expanded_tile_bytes =
        k_tiles * QBH_HMX_WEIGHT_BYTES;
    uint32_t current_base = 0U;
    uint32_t current_slot = 0U;
    uint32_t current_tiles = batch_tiles;
    uint64_t hmx_start = 0U;
    int hmx_active = 0;
    int result;

    if (batch_tiles == 0U || batch_tiles >
            QBH_BLOCK_W4U8_QKVO_MAX_BATCH_N_TILES ||
        n_tiles == 0U || n_tiles % batch_tiles != 0U) {
        return -1;
    }
    if (desc != &header->projections[QBH_BLOCK_PROJ_O]) {
        header->w4u8_qkv_batch_n_tiles = batch_tiles;
    }

    for (uint32_t batch_base = 0U; batch_base < n_tiles;
         batch_base += batch_tiles) {
        uint32_t slot = (batch_base / batch_tiles) & 1U;
        uint32_t tiles = n_tiles - batch_base;
        uint32_t weight_bytes;
        uint32_t bias_bytes;
        uint64_t dma_start;
        uint64_t wait_start;
        uint64_t expand_start;

        if (tiles > batch_tiles) {
            tiles = batch_tiles;
        }
        weight_bytes = tiles * compressed_tile_bytes;
        bias_bytes = tiles * QBH_HMX_BIAS_BYTES;
        dma_start = HAP_perf_get_qtimer_count();
        result = qbh_dma_start_w4u8_batch_prefetch(
            prefetch_descriptors, compressed_slots[slot],
            shared + desc->weight_offset +
                (size_t)batch_base * compressed_tile_bytes,
            weight_bytes, bias_slots[slot],
            shared + desc->bias_offset +
                (size_t)batch_base * QBH_HMX_BIAS_BYTES,
            bias_bytes);
        if (result != 0) {
            if (hmx_active != 0) {
                (void)qbh_hmx_wait(worker);
            }
            qbh_record_projection_failure(
                header, desc, batch_base, 10U, result);
            return -1;
        }
        if (batch_base != 0U) {
            ++header->w4u8_qkvo_prefetch_count;
        }
        header->weight_ddr_read_bytes += weight_bytes + bias_bytes;
        header->weight_dma_descriptor_count += 2U;
        wait_start = HAP_perf_get_qtimer_count();
        result = qbh_dma_wait_w4u8_batch_prefetch(
            prefetch_descriptors);
        header->w4u8_qkvo_prefetch_wait_ticks +=
            HAP_perf_get_qtimer_count() - wait_start;
        header->weight_dma_ticks +=
            HAP_perf_get_qtimer_count() - dma_start;
        if (result != 0) {
            if (hmx_active != 0) {
                (void)qbh_hmx_wait(worker);
            }
            qbh_record_projection_failure(
                header, desc, batch_base, 11U, result);
            return -1;
        }

        expand_start = HAP_perf_get_qtimer_count();
        for (uint32_t tile = 0U; tile < tiles; ++tile) {
            qbh_unpack_w4_to_s8_hvx(
                compressed_slots[slot] +
                    (size_t)tile * compressed_tile_bytes,
                (int8_t *)(expanded_slots[slot] +
                    (size_t)tile * expanded_tile_bytes),
                k_tiles);
        }
        header->w4u8_qkvo_weight_expand_ticks +=
            HAP_perf_get_qtimer_count() - expand_start;

        if (hmx_active != 0) {
            wait_start = HAP_perf_get_qtimer_count();
            result = qbh_hmx_wait(worker);
            header->projection_hmx_wait_ticks +=
                HAP_perf_get_qtimer_count() - wait_start;
            header->w4u8_qkvo_hmx_lifetime_ticks +=
                HAP_perf_get_qtimer_count() - hmx_start;
            hmx_active = 0;
            if (result != 0) {
                qbh_record_projection_failure(
                    header, desc, current_base, 12U, result);
                return -1;
            }
            if (direct_output != 0U) {
                if (desc != &header->projections[QBH_BLOCK_PROJ_O]) {
                    header->u8_attention_qkv_unpack_skipped +=
                        current_tiles;
                    qbh_hvx_pool_u8_qk_prep_publish(
                        header, desc, w4f16_pool,
                        current_base, current_tiles);
                }
            } else {
                uint64_t unpack_start =
                    HAP_perf_get_qtimer_count();
                for (uint32_t tile = 0U;
                     tile < current_tiles; ++tile) {
                    qbh_unpack_u8_output(
                        buffers->hmx_output +
                            (size_t)tile * QBH_HMX_OUTPUT_BYTES,
                        output, desc->n,
                        (current_base + tile) *
                            QBH_HMX_OUTPUT_CHANNELS);
                }
                header->projection_unpack_ticks +=
                    HAP_perf_get_qtimer_count() - unpack_start;
            }
        }

        current_base = batch_base;
        current_slot = slot;
        current_tiles = tiles;
        hmx_start = HAP_perf_get_qtimer_count();
        qbh_hmx_start(
            worker, QBH_BLOCK_HMX_U8S8, projection_activation,
            expanded_slots[current_slot], bias_slots[current_slot],
            direct_output != 0U
                ? output +
                      (size_t)current_base * QBH_HMX_OUTPUT_BYTES
                : buffers->hmx_output,
            1U, k_tiles, current_tiles);
        hmx_active = 1;
        header->hmx_u8s8_tile_pair_count +=
            k_tiles * current_tiles;
        ++header->hmx_command_count;
        if (desc != &header->projections[QBH_BLOCK_PROJ_O]) {
            ++header->w4u8_qkv_batch_count;
        }
        if (batch_base + batch_tiles < n_tiles) {
            ++header->w4u8_qkvo_overlap_schedule_count;
        }
    }

    if (hmx_active != 0) {
        uint64_t wait_start = HAP_perf_get_qtimer_count();
        result = qbh_hmx_wait(worker);
        header->projection_hmx_wait_ticks +=
            HAP_perf_get_qtimer_count() - wait_start;
        header->w4u8_qkvo_hmx_lifetime_ticks +=
            HAP_perf_get_qtimer_count() - hmx_start;
        if (result != 0) {
            qbh_record_projection_failure(
                header, desc, current_base, 13U, result);
            return -1;
        }
        if (direct_output != 0U) {
            if (desc != &header->projections[QBH_BLOCK_PROJ_O]) {
                header->u8_attention_qkv_unpack_skipped += current_tiles;
                qbh_hvx_pool_u8_qk_prep_publish(
                    header, desc, w4f16_pool,
                    current_base, current_tiles);
            }
        } else {
            uint64_t unpack_start = HAP_perf_get_qtimer_count();
            for (uint32_t tile = 0U; tile < current_tiles; ++tile) {
                qbh_unpack_u8_output(
                    buffers->hmx_output +
                        (size_t)tile * QBH_HMX_OUTPUT_BYTES,
                    output, desc->n,
                    (current_base + tile) *
                        QBH_HMX_OUTPUT_CHANNELS);
            }
            header->projection_unpack_ticks +=
                HAP_perf_get_qtimer_count() - unpack_start;
        }
    }
    return 0;
}

static void qbh_w4u8_qkv_ring_abort(
    struct qbh_w4u8_qkv_ring_state *state) {
    if (state == NULL) {
        return;
    }
    state->abort_status = 1U;
    if (state->pool != NULL) {
        state->pool->attention_qk_stream_abort = 1U;
    }
    asm volatile("barrier" ::: "memory");
    for (uint32_t slot = 0U;
         slot < QBH_BLOCK_W4U8_QKV_RING_SLOTS; ++slot) {
        qurt_sem_up(&state->compressed_free[slot]);
        qurt_sem_up(&state->expanded_free[slot]);
        qurt_sem_up(&state->expanded_ready[slot]);
    }
}

static int qbh_run_w4u8_qkv_ring(
    struct qbh_block_header *header, const uint8_t *shared,
    struct qbh_block_buffers *buffers,
    struct qbh_block_hmx_worker *worker,
    struct qbh_block_w4f16_pool *pool,
    const uint8_t *projection_activation) {
    struct qbh_w4u8_qkv_ring_state state;
    struct qbh_dma_aligned_desc_1d descriptors[2]
        __attribute__((aligned(64)));
    const struct qbh_block_projection_desc *descs[3] = {
        &header->projections[QBH_BLOCK_PROJ_Q],
        &header->projections[QBH_BLOCK_PROJ_K],
        &header->projections[QBH_BLOCK_PROJ_V],
    };
    uint8_t *outputs[3] = {buffers->q, buffers->k, buffers->v};
    uint64_t pipeline_start = HAP_perf_get_qtimer_count();
    uint64_t hmx_start = 0U;
    int hmx_started = 0;
    int pool_started = 0;
    uint64_t local_expand_ticks = 0U;
    int result = -1;

    if (header == NULL || shared == NULL || buffers == NULL ||
        worker == NULL || pool == NULL ||
        projection_activation == NULL ||
        header->variant != QBH_BLOCK_W4U8 ||
        header->w4u8_qkv_ring_expand_workers == 0U ||
        header->w4u8_qkv_ring_expand_workers > 3U ||
        pool->worker_count !=
            QBH_BLOCK_MAX_POOL_HVX_WORKERS ||
        header->attention_hvx_contexts !=
            QBH_BLOCK_MAX_ATTENTION_HVX_CONTEXTS) {
        return -1;
    }

    memset(&state, 0, sizeof(state));
    state.header = header;
    state.shared = shared;
    state.pool = pool;
    state.activation = projection_activation;
    state.generation = 1U;
    state.k_tiles = QBH_BLOCK_HIDDEN / QBH_HMX_INPUT_CHANNELS;
    state.expand_worker_count =
        header->w4u8_qkv_ring_expand_workers;
    state.compressed_slots[0] = buffers->compressed_weight;
    state.compressed_slots[1] = buffers->compressed_weight_alt;
    state.compressed_slots[2] = buffers->middle;
    state.compressed_slots[3] = buffers->down;
    state.expanded_slots[0] = buffers->expanded_weight;
    state.expanded_slots[1] = buffers->expanded_weight_alt;
    state.expanded_slots[2] = (uint8_t *)(
        ((uintptr_t)buffers->gate + QBH_HMX_FP16_TILE_BYTES - 1U) &
        ~((uintptr_t)QBH_HMX_FP16_TILE_BYTES - 1U));
    state.expanded_slots[3] = (uint8_t *)(
        ((uintptr_t)buffers->up + QBH_HMX_FP16_TILE_BYTES - 1U) &
        ~((uintptr_t)QBH_HMX_FP16_TILE_BYTES - 1U));
    for (uint32_t slot = 0U;
         slot < QBH_BLOCK_W4U8_QKV_RING_SLOTS; ++slot) {
        state.bias_slots[slot] = buffers->scale_or_bias +
            (size_t)slot *
                QBH_BLOCK_W4U8_QKV_RING_TILES_PER_BATCH *
                QBH_HMX_BIAS_BYTES;
        qurt_sem_init_val(&state.compressed_free[slot], 1U);
        qurt_sem_init_val(&state.expanded_free[slot], 1U);
        qurt_sem_init_val(&state.expanded_ready[slot], 0U);
    }

    for (uint32_t projection = 0U; projection < 3U; ++projection) {
        const struct qbh_block_projection_desc *desc = descs[projection];
        const uint32_t n_tiles =
            desc->n / QBH_HMX_OUTPUT_CHANNELS;
        if (desc->k != QBH_BLOCK_HIDDEN ||
            n_tiles % QBH_BLOCK_W4U8_QKV_RING_TILES_PER_BATCH != 0U) {
            goto cleanup;
        }
        for (uint32_t first = 0U; first < n_tiles;
             first += QBH_BLOCK_W4U8_QKV_RING_TILES_PER_BATCH) {
            struct qbh_w4u8_qkv_ring_batch *batch;
            if (state.batch_count >=
                QBH_BLOCK_W4U8_QKV_RING_BATCHES) {
                goto cleanup;
            }
            batch = &state.batches[state.batch_count++];
            batch->desc = desc;
            batch->output = outputs[projection];
            batch->first_n_tile = first;
            batch->n_tiles =
                QBH_BLOCK_W4U8_QKV_RING_TILES_PER_BATCH;
        }
    }
    if (state.batch_count != QBH_BLOCK_W4U8_QKV_RING_BATCHES) {
        goto cleanup;
    }

    pool->qkv_ring_state = &state;
    pool->attention_header = header;
    pool->attention_buffers = buffers;
    pool->attention_task_count =
        QBH_BLOCK_HEADS / 2U + QBH_BLOCK_KV_HEADS / 2U;
    pool->next_attention_task = 0U;
    pool->attention_qk_stream_abort = 0U;
    pool->attention_qk_streaming = 1U;
    pool->active_worker_count = pool->worker_count;
    ++pool->attention_qk_generation;
    if (pool->attention_qk_generation == 0U) {
        memset((void *)pool->attention_qk_ready, 0,
               sizeof(pool->attention_qk_ready));
        pool->attention_qk_generation = 1U;
    }
    for (uint32_t index = 0U;
         index < pool->active_worker_count; ++index) {
        struct qbh_block_w4f16_job *job = &pool->jobs[index];
        job->expand_count = 0U;
        job->expand_ticks = 0U;
        job->u8_attention_prepared_group_count = 0U;
        job->u8_qk_quarter_pair_count = 0U;
        job->u8_attention_fused_k_operand_mismatch_count = 0U;
        job->u8_attention_qk_norm_rope_ticks = 0U;
        job->u8_attention_k_pack_ticks = 0U;
        job->attention_qk_norm_task_count = 0U;
        job->command_kind =
            index < state.expand_worker_count
                ? QBH_BLOCK_HVX_POOL_W4U8_QKV_RING
                : QBH_BLOCK_HVX_POOL_U8_QK_PREP;
    }
    worker->kind = QBH_BLOCK_HMX_U8S8_QKV_RING;
    worker->qkv_ring_request = &state;
    worker->command_status = AEE_EFAILED;
    asm volatile("barrier" ::: "memory");
    for (uint32_t index = 0U;
         index < pool->active_worker_count; ++index) {
        qurt_sem_up(&pool->command_ready[index]);
    }
    pool_started = 1;
    hmx_start = HAP_perf_get_qtimer_count();
    qurt_sem_up(&worker->command_ready);
    hmx_started = 1;
    ++header->w4u8_qkv_ring_dispatch_count;
    ++header->w4u8_qkv_ring_hmx_dispatch_count;

    for (uint32_t batch_index = 0U;
         batch_index < state.batch_count; ++batch_index) {
        const struct qbh_w4u8_qkv_ring_batch *batch =
            &state.batches[batch_index];
        const uint32_t slot =
            batch_index % QBH_BLOCK_W4U8_QKV_RING_SLOTS;
        const uint32_t compressed_tile_bytes =
            state.k_tiles * QBH_W4_PACKED_TILE_BYTES;
        const uint32_t weight_bytes =
            batch->n_tiles * compressed_tile_bytes;
        const uint32_t bias_bytes =
            batch->n_tiles * QBH_HMX_BIAS_BYTES;
        uint64_t wait_start = HAP_perf_get_qtimer_count();
        uint64_t dma_start;

        qurt_sem_down(&state.compressed_free[slot]);
        qurt_sem_down(&state.expanded_free[slot]);
        header->w4u8_qkv_ring_producer_slot_wait_ticks +=
            HAP_perf_get_qtimer_count() - wait_start;
        dma_start = HAP_perf_get_qtimer_count();
        if (qbh_dma_start_w4u8_batch_prefetch(
                descriptors, state.compressed_slots[slot],
                shared + batch->desc->weight_offset +
                    (size_t)batch->first_n_tile *
                        compressed_tile_bytes,
                weight_bytes, state.bias_slots[slot],
                shared + batch->desc->bias_offset +
                    (size_t)batch->first_n_tile *
                        QBH_HMX_BIAS_BYTES,
                bias_bytes) != 0) {
            qbh_record_projection_failure(
                header, batch->desc, batch->first_n_tile, 20U, -1);
            qbh_w4u8_qkv_ring_abort(&state);
            goto finish;
        }
        wait_start = HAP_perf_get_qtimer_count();
        if (qbh_dma_wait_w4u8_batch_prefetch(descriptors) != 0) {
            qbh_record_projection_failure(
                header, batch->desc, batch->first_n_tile, 21U, -1);
            qbh_w4u8_qkv_ring_abort(&state);
            goto finish;
        }
        header->w4u8_qkv_ring_dma_wait_ticks +=
            HAP_perf_get_qtimer_count() - wait_start;
        header->w4u8_qkvo_prefetch_wait_ticks +=
            HAP_perf_get_qtimer_count() - wait_start;
        header->weight_dma_ticks +=
            HAP_perf_get_qtimer_count() - dma_start;
        header->weight_ddr_read_bytes += weight_bytes + bias_bytes;
        header->weight_dma_descriptor_count += 2U;
        if (batch->first_n_tile != 0U) {
            ++header->w4u8_qkvo_prefetch_count;
            ++header->w4u8_qkvo_overlap_schedule_count;
        }
        state.dma_ready[batch_index] = state.generation;
        asm volatile("release(%0):at"
                     :
                     : "r"(&state.dma_ready[batch_index])
                     : "memory");
    }

finish:
    if (hmx_started != 0) {
        uint64_t wait_start = HAP_perf_get_qtimer_count();
        const int hmx_result = qbh_hmx_wait(worker);
        header->projection_hmx_wait_ticks +=
            HAP_perf_get_qtimer_count() - wait_start;
        header->w4u8_qkvo_hmx_lifetime_ticks +=
            HAP_perf_get_qtimer_count() - hmx_start;
        worker->qkv_ring_request = NULL;
        if (hmx_result != 0) {
            qbh_w4u8_qkv_ring_abort(&state);
        }
    }
    if (pool_started != 0) {
        uint64_t wait_before = header->attention_qk_norm_pool_wait_ticks;
        if (qbh_hvx_pool_u8_qk_prep_wait_async(header, pool) != 0) {
            qbh_w4u8_qkv_ring_abort(&state);
        }
        header->w4u8_qkv_ring_pool_wait_ticks +=
            header->attention_qk_norm_pool_wait_ticks - wait_before;
    }
    header->w4u8_qkv_ring_slot_count =
        QBH_BLOCK_W4U8_QKV_RING_SLOTS;
    header->w4u8_qkv_ring_expand_worker_count =
        state.expand_worker_count;
    header->w4u8_qkv_ring_prep_worker_count =
        pool->worker_count - state.expand_worker_count;
    header->w4u8_qkv_ring_batch_count += state.hmx_batch_count;
    header->w4u8_qkv_ring_head_publish_count +=
        state.head_publish_count;
    header->w4u8_qkv_ring_hmx_ready_wait_ticks +=
        state.hmx_ready_wait_ticks;
    header->w4u8_qkv_ring_hmx_compute_ticks +=
        state.hmx_compute_ticks;
    for (uint32_t index = 0U;
         index < state.expand_worker_count; ++index) {
        local_expand_ticks += pool->jobs[index].expand_ticks;
        header->w4u8_qkv_ring_expand_task_count +=
            pool->jobs[index].expand_count;
    }
    header->w4u8_qkv_ring_expand_ticks += local_expand_ticks;
    header->w4u8_qkvo_weight_expand_ticks +=
        local_expand_ticks;
    result = state.abort_status == 0U &&
             state.hmx_batch_count == state.batch_count &&
             state.head_publish_count ==
                 QBH_BLOCK_HEADS + QBH_BLOCK_KV_HEADS
                 ? 0 : -1;

cleanup:
    pool->qkv_ring_state = NULL;
    for (uint32_t slot = 0U;
         slot < QBH_BLOCK_W4U8_QKV_RING_SLOTS; ++slot) {
        qurt_sem_destroy(&state.compressed_free[slot]);
        qurt_sem_destroy(&state.expanded_free[slot]);
        qurt_sem_destroy(&state.expanded_ready[slot]);
    }
    header->w4u8_qkv_ring_pipeline_ticks +=
        HAP_perf_get_qtimer_count() - pipeline_start;
    return result;
}

struct qbh_w4f16_qkv_schedule_task {
    const struct qbh_block_projection_desc *desc;
    uint8_t *output;
    uint32_t first_n_tile;
    uint32_t n_tiles;
};

static int qbh_w4f16_qkv_prefix4_task_init(
    struct qbh_block_header *header,
    struct qbh_block_buffers *buffers,
    uint32_t command, uint32_t batch_tiles,
    struct qbh_w4f16_qkv_schedule_task *task) {
    const uint32_t prefix_groups = 4U;
    const uint32_t q_commands_per_group = 8U / batch_tiles;
    const uint32_t kv_commands_per_group = 4U / batch_tiles;
    const uint32_t prefix_q_commands =
        prefix_groups * q_commands_per_group;
    const uint32_t all_k_commands =
        QBH_BLOCK_KV_HEADS * kv_commands_per_group;
    const uint32_t tail_q_commands =
        (QBH_BLOCK_KV_HEADS - prefix_groups) *
        q_commands_per_group;
    uint32_t local;
    uint32_t group;
    uint32_t projection;
    uint32_t projection_local;

    if (header == NULL || buffers == NULL || task == NULL ||
        header->qkv_schedule_mode !=
            QBH_BLOCK_QKV_SCHEDULE_Q_PREFIX4_K_ALL ||
        (batch_tiles != 2U && batch_tiles != 4U)) {
        return -1;
    }
    if (command < prefix_q_commands) {
        projection = QBH_BLOCK_PROJ_Q;
        group = command / q_commands_per_group;
        projection_local = command % q_commands_per_group;
        task->first_n_tile = group * 8U +
            projection_local * batch_tiles;
        task->output = buffers->q;
    } else if (command <
               prefix_q_commands + all_k_commands) {
        local = command - prefix_q_commands;
        projection = QBH_BLOCK_PROJ_K;
        group = local / kv_commands_per_group;
        projection_local = local % kv_commands_per_group;
        task->first_n_tile = group * 4U +
            projection_local * batch_tiles;
        task->output = buffers->k;
    } else if (command < prefix_q_commands +
                              all_k_commands +
                              tail_q_commands) {
        local = command - prefix_q_commands - all_k_commands;
        projection = QBH_BLOCK_PROJ_Q;
        group = prefix_groups +
            local / q_commands_per_group;
        projection_local = local % q_commands_per_group;
        task->first_n_tile = group * 8U +
            projection_local * batch_tiles;
        task->output = buffers->q;
    } else {
        local = command - prefix_q_commands -
            all_k_commands - tail_q_commands;
        if (local >= QBH_BLOCK_KV_HEADS *
                         kv_commands_per_group) {
            return -1;
        }
        projection = QBH_BLOCK_PROJ_V;
        group = local / kv_commands_per_group;
        projection_local = local % kv_commands_per_group;
        task->first_n_tile = group * 4U +
            projection_local * batch_tiles;
        task->output = buffers->v;
    }
    task->desc = &header->projections[projection];
    task->n_tiles = batch_tiles;
    return 0;
}

static __fp16 *qbh_w4f16_qkv_direct_output(
    const struct qbh_w4f16_qkv_schedule_task *task) {
    const uint32_t output_group_elements =
        (QBH_BLOCK_M / QBH_HMX_FP16_ROWS) *
        task->n_tiles * QBH_HMX_FP16_TILE_ELEMENTS;
    return (__fp16 *)task->output +
        (size_t)(task->first_n_tile / task->n_tiles) *
            output_group_elements;
}

static void qbh_w4f16_qkv_trace_command(
    struct qbh_block_header *header,
    const struct qbh_w4f16_qkv_schedule_task *task) {
    uint64_t hash = header->qkv_schedule_command_count == 0U
        ? UINT64_C(1469598103934665603)
        : header->qkv_schedule_trace_hash;
    const uint32_t words[3] = {
        (uint32_t)(task->desc - header->projections),
        task->first_n_tile, task->n_tiles};

    for (uint32_t word = 0U; word < 3U; ++word) {
        for (uint32_t byte = 0U; byte < 4U; ++byte) {
            hash ^= (words[word] >> (byte * 8U)) & 0xffU;
            hash *= UINT64_C(1099511628211);
        }
    }
    header->qkv_schedule_trace_hash = hash;
    ++header->qkv_schedule_command_count;
}

/* EXP-0110: W4F16-only Q-prefix schedule.  It preserves the frozen
 * projection math and supports either row-major or direct Crouton Q/K/V
 * output so projection order and carrier remain independent factors. */
static int qbh_run_w4f16_qkv_prefix4(
    struct qbh_block_header *header, const uint8_t *shared,
    struct qbh_block_buffers *buffers,
    struct qbh_block_hmx_worker *worker,
    struct qbh_block_w4f16_pool *pool,
    const void *activation_tiles,
    struct qbh_block_w4f16_cross_prefetch *cross_prefetch) {
    const uint32_t hmx_batch_tiles =
        QBH_BLOCK_W4F16_HMX_BATCH_N_TILES;
    const uint32_t dma_batch_tiles =
        QBH_BLOCK_W4F16_DMA_BATCH_N_TILES;
    const uint32_t total_commands =
        QBH_BLOCK_KV_HEADS * (16U / hmx_batch_tiles);
    const uint32_t total_batches =
        QBH_BLOCK_KV_HEADS * (16U / dma_batch_tiles);
    const uint32_t k_tiles =
        QBH_BLOCK_HIDDEN / QBH_HMX_FP16_COLS;
    const uint32_t compressed_bytes_per_tile =
        k_tiles * QBH_W4_PACKED_TILE_BYTES;
    const uint32_t direct_qkv =
        qbh_crouton_qkv_enabled(header);
    uint8_t *compressed_slots[2] = {
        buffers->compressed_weight, buffers->compressed_weight_alt};
    uint8_t *expanded_slots[2] = {
        buffers->expanded_weight, buffers->expanded_weight_alt};
    volatile uint32_t ready[QBH_BLOCK_W4F16_MAX_REGIONS];
    struct qbh_dma_aligned_desc_1d prefetch_descriptor
        __attribute__((aligned(64)));
    struct qbh_w4f16_qkv_schedule_task first_dma;
    struct qbh_w4f16_qkv_schedule_task first_hmx;
    uint64_t prefetch_start = 0U;
    int prefetch_active = 0;
    int result;
    uint32_t active_workers;
    uint32_t region_tiles;

    if (header->variant != QBH_BLOCK_W4F16 || pool == NULL ||
        qbh_w4f16_qkv_prefix4_task_init(
            header, buffers, 0U, dma_batch_tiles,
            &first_dma) != 0 ||
        qbh_w4f16_qkv_prefix4_task_init(
            header, buffers, 0U, hmx_batch_tiles,
            &first_hmx) != 0) {
        return -1;
    }
    active_workers = qbh_w4f16_projection_worker_count(
        header, first_hmx.desc, pool);
    region_tiles = qbh_w4f16_projection_region_tiles(
        header, first_hmx.desc);
    qbh_w4f16_note_active_workers(header, active_workers);
    qbh_w4f16_note_effective_region(header, region_tiles);
    memset((void *)ready, 0, sizeof(ready));

    if (qbh_dma_copy(
            header, compressed_slots[0],
            shared + first_dma.desc->weight_offset +
                (size_t)first_dma.first_n_tile *
                    compressed_bytes_per_tile,
            dma_batch_tiles * compressed_bytes_per_tile, 1U) != 0) {
        qbh_record_projection_failure(
            header, first_dma.desc, first_dma.first_n_tile, 64U, -1);
        return -1;
    }
    header->weight_ddr_read_bytes +=
        dma_batch_tiles * compressed_bytes_per_tile;
    ++header->weight_dma_descriptor_count;
    if (total_batches > 1U) {
        struct qbh_w4f16_qkv_schedule_task next_dma;
        if (qbh_w4f16_qkv_prefix4_task_init(
                header, buffers, 1U, dma_batch_tiles,
                &next_dma) != 0) {
            return -1;
        }
        prefetch_start = HAP_perf_get_qtimer_count();
        result = qbh_dma_start_weight_prefetch(
            &prefetch_descriptor, compressed_slots[1],
            shared + next_dma.desc->weight_offset +
                (size_t)next_dma.first_n_tile *
                    compressed_bytes_per_tile,
            dma_batch_tiles * compressed_bytes_per_tile);
        if (result != 0) {
            qbh_record_projection_failure(
                header, next_dma.desc, next_dma.first_n_tile,
                65U, result);
            return -1;
        }
        prefetch_active = 1;
        ++header->w4f16_prefetch_count;
        header->weight_ddr_read_bytes +=
            dma_batch_tiles * compressed_bytes_per_tile;
        ++header->weight_dma_descriptor_count;
    }

    {
        const float *scales = qbh_w4f16_projection_scales(
            header, buffers, first_hmx.desc) +
            (size_t)first_hmx.first_n_tile * 32U;
        uint64_t expand_start = HAP_perf_get_qtimer_count();
        qbh_w4f16_expand_with_main(
            header, pool, compressed_slots[0], scales,
            expanded_slots[0], ready, 1U,
            k_tiles * hmx_batch_tiles, region_tiles,
            active_workers, 0U, 0U);
        header->w4f16_expand_ticks +=
            HAP_perf_get_qtimer_count() - expand_start;
        header->w4f16_first_expand_ticks +=
            HAP_perf_get_qtimer_count() - expand_start;
    }
    if (direct_qkv != 0U) {
        header->crouton_qkv_projection_count += 3U;
    }

    for (uint32_t command = 0U;
         command < total_commands; ++command) {
        struct qbh_w4f16_qkv_schedule_task current;
        const float *current_scales;
        __fp16 *hmx_output = (__fp16 *)buffers->hmx_output;
        uint64_t command_start;
        int cross_prefetch_result = 0;

        if (qbh_w4f16_qkv_prefix4_task_init(
                header, buffers, command, hmx_batch_tiles,
                &current) != 0) {
            return -1;
        }
        current_scales = qbh_w4f16_projection_scales(
            header, buffers, current.desc) +
            (size_t)current.first_n_tile * 32U;
        for (uint32_t tile = 0U;
             tile < hmx_batch_tiles; ++tile) {
            qbh_hmx_fp16_init_channel_scales(
                buffers->scale_or_bias +
                    (size_t)tile * QBH_HMX_FP16_SCALE_BYTES,
                current_scales + (size_t)tile * 32U);
        }
        if (command == 0U) {
            header->w4f16_expand_mismatch_count =
                qbh_audit_unscaled_w4_to_f16_tile(
                    compressed_slots[0], expanded_slots[0],
                    &header->w4f16_expand_first_logical_index,
                    &header->w4f16_expand_expected_half_bits,
                    &header->w4f16_expand_actual_half_bits);
            header->w4f16_expand_mismatch_count +=
                qbh_hmx_fp16_audit_channel_scales(
                    buffers->scale_or_bias, current_scales,
                    &header->w4f16_expand_first_logical_index,
                    &header->w4f16_expand_expected_half_bits,
                    &header->w4f16_expand_actual_half_bits);
            if (header->w4f16_expand_mismatch_count != 0U) {
                qbh_record_projection_failure(
                    header, current.desc,
                    current.first_n_tile, 66U, -1);
                return -1;
            }
        }
        if (direct_qkv != 0U) {
            hmx_output = qbh_w4f16_qkv_direct_output(&current);
        }

        command_start = HAP_perf_get_qtimer_count();
        qbh_hmx_start_fp16_tile_scales(
            worker, activation_tiles,
            expanded_slots[command & 1U],
            buffers->scale_or_bias, hmx_output,
            2U, k_tiles, hmx_batch_tiles);
        ++header->w4f16_streamed_command_count;

        if (command + 1U < total_commands) {
            const uint32_t next_command = command + 1U;
            const uint32_t current_batch = command / 2U;
            const uint32_t next_batch = next_command / 2U;
            struct qbh_w4f16_qkv_schedule_task next;
            const float *next_scales;
            const uint8_t *next_compressed;

            if (qbh_w4f16_qkv_prefix4_task_init(
                    header, buffers, next_command,
                    hmx_batch_tiles, &next) != 0) {
                (void)qbh_hmx_wait(worker);
                return -1;
            }
            if (next_batch != current_batch) {
                uint64_t wait_start = HAP_perf_get_qtimer_count();
                if (prefetch_active == 0) {
                    (void)qbh_hmx_wait(worker);
                    return -1;
                }
                result = qbh_dma_wait_weight_prefetch(
                    &prefetch_descriptor);
                header->w4f16_prefetch_wait_ticks +=
                    HAP_perf_get_qtimer_count() - wait_start;
                header->weight_dma_ticks +=
                    HAP_perf_get_qtimer_count() - prefetch_start;
                prefetch_active = 0;
                if (result != 0) {
                    (void)qbh_hmx_wait(worker);
                    qbh_record_projection_failure(
                        header, next.desc, next.first_n_tile,
                        67U, result);
                    return -1;
                }
                if (next_batch + 1U < total_batches) {
                    struct qbh_w4f16_qkv_schedule_task following;
                    if (qbh_w4f16_qkv_prefix4_task_init(
                            header, buffers, next_batch + 1U,
                            dma_batch_tiles, &following) != 0) {
                        (void)qbh_hmx_wait(worker);
                        return -1;
                    }
                    prefetch_start = HAP_perf_get_qtimer_count();
                    result = qbh_dma_start_weight_prefetch(
                        &prefetch_descriptor,
                        compressed_slots[(next_batch + 1U) & 1U],
                        shared + following.desc->weight_offset +
                            (size_t)following.first_n_tile *
                                compressed_bytes_per_tile,
                        dma_batch_tiles *
                            compressed_bytes_per_tile);
                    if (result != 0) {
                        (void)qbh_hmx_wait(worker);
                        qbh_record_projection_failure(
                            header, following.desc,
                            following.first_n_tile, 68U, result);
                        return -1;
                    }
                    prefetch_active = 1;
                    ++header->w4f16_prefetch_count;
                    header->weight_ddr_read_bytes +=
                        dma_batch_tiles *
                            compressed_bytes_per_tile;
                    ++header->weight_dma_descriptor_count;
                }
            }
            next_compressed =
                compressed_slots[next_batch & 1U] +
                (size_t)(next_command & 1U) *
                    hmx_batch_tiles *
                    compressed_bytes_per_tile;
            next_scales = qbh_w4f16_projection_scales(
                header, buffers, next.desc) +
                (size_t)next.first_n_tile * 32U;
            {
                uint64_t expand_start = HAP_perf_get_qtimer_count();
                qbh_w4f16_expand_with_main(
                    header, pool, next_compressed,
                    next_scales,
                    expanded_slots[next_command & 1U],
                    ready, next_command + 1U,
                    k_tiles * hmx_batch_tiles,
                    region_tiles, active_workers, 0U, 0U);
                header->w4f16_expand_ticks +=
                    HAP_perf_get_qtimer_count() - expand_start;
                header->w4f16_steady_expand_ticks +=
                    HAP_perf_get_qtimer_count() - expand_start;
            }
        } else {
            cross_prefetch_result = qbh_w4f16_start_cross_prefetch(
                header, shared,
                &header->projections[QBH_BLOCK_PROJ_O],
                buffers, cross_prefetch);
        }

        {
            uint64_t wait_start = HAP_perf_get_qtimer_count();
            result = qbh_hmx_wait(worker);
            header->w4f16_hmx_tail_wait_ticks +=
                HAP_perf_get_qtimer_count() - wait_start;
        }
        header->projection_hmx_wait_ticks +=
            HAP_perf_get_qtimer_count() - command_start;
        header->hmx_fp16_tile_pair_count +=
            2U * k_tiles * hmx_batch_tiles;
        ++header->hmx_command_count;
        if (result != 0 || cross_prefetch_result != 0) {
            qbh_w4f16_drain_cross_prefetch(
                header, cross_prefetch);
            qbh_record_projection_failure(
                header, current.desc, current.first_n_tile,
                result != 0 ? 69U : 70U,
                result != 0 ? result : cross_prefetch_result);
            return -1;
        }
        if (direct_qkv != 0U) {
            header->crouton_qkv_unpack_skipped += hmx_batch_tiles;
        } else {
            uint64_t unpack_start = HAP_perf_get_qtimer_count();
            qbh_unpack_fp16_output(
                (const __fp16 *)buffers->hmx_output,
                hmx_batch_tiles, (__fp16 *)current.output,
                current.desc->n,
                current.first_n_tile * QBH_HMX_FP16_COLS);
            header->projection_unpack_ticks +=
                HAP_perf_get_qtimer_count() - unpack_start;
            qbh_capture_row_major_qkv_reference(
                header, current.desc, buffers,
                (const __fp16 *)current.output,
                current.first_n_tile, current.n_tiles);
        }
        qbh_w4f16_qkv_trace_command(header, &current);
        qbh_hvx_pool_qk_norm_rope_publish(
            header, current.desc, pool,
            current.first_n_tile, current.n_tiles);
    }
    if (prefetch_active != 0) {
        qbh_record_projection_failure(
            header, &header->projections[QBH_BLOCK_PROJ_V],
            0U, 71U, -1);
        return -1;
    }
    return 0;
}

static int qbh_run_projection(
    struct qbh_block_header *header, const uint8_t *shared,
    const struct qbh_block_projection_desc *desc,
    struct qbh_block_buffers *buffers,
    struct qbh_block_hmx_worker *worker,
    struct qbh_block_w4f16_pool *w4f16_pool, const void *input,
    void *output, uint32_t activation_tiles_ready,
    const struct qbh_block_projection_desc *next_w4_desc,
    struct qbh_block_w4f16_cross_prefetch *cross_prefetch) {
    uint32_t k_tiles = desc->k / 32U;
    uint32_t n_tiles = desc->n / 32U;
    uint32_t element_bytes =
        header->variant == QBH_BLOCK_W4U8 ? 1U : 2U;
    uint32_t failure_step = 0U;
    uint64_t phase_start = HAP_perf_get_qtimer_count();
    uint8_t *projection_activation =
        (header->mlp_mode == QBH_BLOCK_MLP_STREAMING ||
         header->mlp_mode == QBH_BLOCK_MLP_CROUTON_NATIVE ||
         header->mlp_mode == QBH_BLOCK_MLP_CROUTON_NATIVE_BATCH8) &&
                desc == &header->projections[QBH_BLOCK_PROJ_DOWN]
            ? buffers->q : buffers->hmx_activation;
    volatile uint32_t w4f16_ready[QBH_BLOCK_W4F16_MAX_REGIONS];
    struct qbh_dma_aligned_desc_1d w4f16_prefetch_descriptor
        __attribute__((aligned(64)));
    uint8_t *w4f16_compressed_slots[2] = {
        buffers->compressed_weight, buffers->compressed_weight_alt};
    uint8_t *w4f16_scale_slots[2] = {
        buffers->channel_scale, buffers->channel_scale_alt};
    int w4f16_prefetch_active = 0;
    uint64_t w4f16_prefetch_start = 0U;
    const uint32_t u8_integer_attention =
        qbh_attention_u8_enabled(header->attention_pipeline_mode);
    const uint32_t direct_u8_projection_output =
        u8_integer_attention != 0U &&
        (desc == &header->projections[QBH_BLOCK_PROJ_Q] ||
         desc == &header->projections[QBH_BLOCK_PROJ_K] ||
         desc == &header->projections[QBH_BLOCK_PROJ_V] ||
         (desc == &header->projections[QBH_BLOCK_PROJ_O] &&
          (header->crouton_boundary_mode &
           QBH_BLOCK_CROUTON_BOUNDARY_W4U8_O_OUTPUT) != 0U));

    memset((void *)w4f16_ready, 0, sizeof(w4f16_ready));

    if (activation_tiles_ready != 0U) {
        const uint32_t w4u8_qkv_native_input =
            header->variant == QBH_BLOCK_W4U8 &&
            (header->crouton_boundary_mode &
             QBH_BLOCK_CROUTON_BOUNDARY_W4U8_QKV_INPUT) != 0U &&
            (desc == &header->projections[QBH_BLOCK_PROJ_Q] ||
             desc == &header->projections[QBH_BLOCK_PROJ_K] ||
             desc == &header->projections[QBH_BLOCK_PROJ_V]);
        if ((header->variant == QBH_BLOCK_W4U8 &&
             !(u8_integer_attention != 0U &&
               desc == &header->projections[QBH_BLOCK_PROJ_O]) &&
             w4u8_qkv_native_input == 0U) ||
            ((header->mlp_mode == QBH_BLOCK_MLP_STREAMING ||
              header->mlp_mode == QBH_BLOCK_MLP_CROUTON_NATIVE ||
              header->mlp_mode ==
                  QBH_BLOCK_MLP_CROUTON_NATIVE_BATCH8) &&
             desc == &header->projections[QBH_BLOCK_PROJ_DOWN])) {
            return -1;
        }
        /* Cache-native W4U8 decode writes AV into buffers->q while fixed-M
         * Attention writes AV into buffers->hmx_activation.  Only this
         * dynamic O-projection boundary has a caller-selected carrier; the
         * older FP16 and other prepacked paths retain their established
         * workspace contract. */
        if (header->variant == QBH_BLOCK_W4U8 &&
            u8_integer_attention != 0U &&
            desc == &header->projections[QBH_BLOCK_PROJ_O]) {
            projection_activation = (uint8_t *)(uintptr_t)input;
        }
    } else if (header->variant == QBH_BLOCK_W4U8) {
        qbh_pack_u8_activation((const uint8_t *)input, desc->k,
                               desc->k, projection_activation);
    } else {
        if ((header->mlp_mode == QBH_BLOCK_MLP_STREAMING ||
             header->mlp_mode == QBH_BLOCK_MLP_CROUTON_NATIVE ||
             header->mlp_mode ==
                 QBH_BLOCK_MLP_CROUTON_NATIVE_BATCH8) &&
            desc == &header->projections[QBH_BLOCK_PROJ_DOWN]) {
            ++header->mlp_down_pack_skipped;
        } else {
            qbh_pack_fp16_activation(
                (const __fp16 *)input, desc->k, desc->k,
                (__fp16 *)projection_activation);
        }
        qbh_hmx_fp16_init_unity_scale(buffers->scale_or_bias);
    }
    header->projection_pack_ticks +=
        HAP_perf_get_qtimer_count() - phase_start;
    if (qbh_projection_direct_qkv_crouton(header, desc)) {
        ++header->crouton_qkv_projection_count;
    }
    if (header->numerical_audit_enabled != 0U &&
        desc == &header->projections[QBH_BLOCK_PROJ_DOWN]) {
        header->mlp_down_input_hash = qbh_fnv1a64_bytes(
            projection_activation,
            QBH_BLOCK_M * QBH_BLOCK_INTERMEDIATE * sizeof(uint16_t));
    }
    if (header->variant == QBH_BLOCK_W4F16) {
        return qbh_run_w4f16_projection(
            header, shared, desc, buffers, worker, w4f16_pool,
            projection_activation, output, next_w4_desc,
            cross_prefetch);
    }
    if (header->variant == QBH_BLOCK_F16F16) {
        header->f16f16_weight_batch_n_tiles = 1U;
        if (header->f16f16_projection_mode !=
            QBH_BLOCK_F16F16_PROJECTION_SERIAL) {
            return qbh_run_f16f16_pipelined_projection(
                header, shared, desc, buffers, worker,
                w4f16_pool, projection_activation, output);
        }
    }
    if (header->variant == QBH_BLOCK_W4U8) {
        uint32_t batch_tiles =
            qbh_w4u8_qkvo_batch_tiles(header, desc);
        if (batch_tiles != 0U) {
            return qbh_run_w4u8_qkvo_pipelined_projection(
                header, shared, desc, buffers, worker,
                w4f16_pool, projection_activation,
                (uint8_t *)output, direct_u8_projection_output,
                batch_tiles);
        }
    }
    if (header->variant == QBH_BLOCK_W4F16) {
        uint32_t compressed_bytes =
            k_tiles * QBH_W4_PACKED_TILE_BYTES;
        failure_step = 1U;
        if (qbh_dma_copy(
                header, w4f16_compressed_slots[0],
                shared + desc->weight_offset,
                compressed_bytes, 1U) != 0) {
            qbh_record_projection_failure(
                header, desc, 0U, failure_step, -1);
            return -1;
        }
        failure_step = 2U;
        if (qbh_dma_copy(
                header, w4f16_scale_slots[0],
                shared + desc->scale_offset,
                32U * sizeof(float), 1U) != 0) {
            qbh_record_projection_failure(
                header, desc, 0U, failure_step, -1);
            return -1;
        }
        header->weight_ddr_read_bytes +=
            compressed_bytes + 32U * sizeof(float);
        header->weight_dma_descriptor_count += 2U;
    }

    for (uint32_t n_tile = 0; n_tile < n_tiles; ++n_tile) {
        uint32_t compressed_bytes = k_tiles * QBH_W4_PACKED_TILE_BYTES;
        uint32_t fp16_weight_bytes =
            k_tiles * QBH_HMX_FP16_TILE_BYTES;
        int result;
        if (header->variant == QBH_BLOCK_W4F16) {
            result = 0;
        } else if (header->variant == QBH_BLOCK_F16F16) {
            const uint8_t *source = shared + desc->weight_offset +
                (size_t)n_tile * fp16_weight_bytes;
            failure_step = 1U;
            result = qbh_dma_copy(header, buffers->expanded_weight,
                                  source, fp16_weight_bytes, 1U);
            header->weight_ddr_read_bytes += fp16_weight_bytes;
            ++header->weight_dma_descriptor_count;
        } else {
            const uint8_t *source = shared + desc->weight_offset +
                (size_t)n_tile * compressed_bytes;
            failure_step = 1U;
            result = qbh_dma_copy(header, buffers->compressed_weight,
                                  source, compressed_bytes, 1U);
            header->weight_ddr_read_bytes += compressed_bytes;
            ++header->weight_dma_descriptor_count;
            if (result == 0) {
                failure_step = 3U;
                result = qbh_dma_copy(
                    header, buffers->scale_or_bias,
                    shared + desc->bias_offset +
                        (size_t)n_tile * QBH_HMX_BIAS_BYTES,
                    QBH_HMX_BIAS_BYTES, 1U);
                header->weight_ddr_read_bytes += QBH_HMX_BIAS_BYTES;
                ++header->weight_dma_descriptor_count;
                if (result == 0) {
                    qbh_unpack_w4_to_s8_hvx(
                        buffers->compressed_weight,
                        (int8_t *)buffers->expanded_weight, k_tiles);
                }
            }
        }
        if (result != 0) {
            qbh_record_projection_failure(
                header, desc, n_tile, failure_step, result);
            return -1;
        }
        if (header->variant == QBH_BLOCK_W4F16 &&
            n_tile + 1U < n_tiles) {
            uint32_t next_tile = n_tile + 1U;
            uint32_t next_slot = next_tile & 1U;
            w4f16_prefetch_start = HAP_perf_get_qtimer_count();
            failure_step = 7U;
            result = qbh_dma_start_weight_prefetch(
                &w4f16_prefetch_descriptor,
                w4f16_compressed_slots[next_slot],
                shared + desc->weight_offset +
                    (size_t)next_tile * compressed_bytes,
                compressed_bytes);
            if (result != 0) {
                qbh_record_projection_failure(
                    header, desc, n_tile, failure_step, result);
                return -1;
            }
            w4f16_prefetch_active = 1;
            ++header->w4f16_prefetch_count;
            header->weight_ddr_read_bytes += compressed_bytes;
            ++header->weight_dma_descriptor_count;
        }

        if (header->variant == QBH_BLOCK_W4U8) {
            failure_step = 4U;
            phase_start = HAP_perf_get_qtimer_count();
            result = qbh_hmx_submit(
                worker, QBH_BLOCK_HMX_U8S8,
                projection_activation, buffers->expanded_weight,
                buffers->scale_or_bias,
                direct_u8_projection_output != 0U
                    ? (uint8_t *)output +
                          (size_t)n_tile * QBH_HMX_OUTPUT_BYTES
                    : buffers->hmx_output,
                1U,
                k_tiles, 1U);
            header->projection_hmx_wait_ticks +=
                HAP_perf_get_qtimer_count() - phase_start;
            header->hmx_u8s8_tile_pair_count += k_tiles;
            if (result == 0 && direct_u8_projection_output == 0U) {
                phase_start = HAP_perf_get_qtimer_count();
                qbh_unpack_u8_output(
                    buffers->hmx_output, (uint8_t *)output, desc->n,
                    n_tile * QBH_HMX_OUTPUT_CHANNELS);
                header->projection_unpack_ticks +=
                    HAP_perf_get_qtimer_count() - phase_start;
                qbh_hvx_pool_qk_norm_rope_publish(
                    header, desc, w4f16_pool, n_tile, 1U);
            } else if (result == 0 &&
                       desc != &header->projections[QBH_BLOCK_PROJ_O]) {
                ++header->u8_attention_qkv_unpack_skipped;
                qbh_hvx_pool_u8_qk_prep_publish(
                    header, desc, w4f16_pool, n_tile, 1U);
            }
        } else {
            failure_step = 4U;
            phase_start = HAP_perf_get_qtimer_count();
            if (header->variant == QBH_BLOCK_W4F16) {
                uint32_t region_count =
                    k_tiles / header->w4f16_region_tiles;
                uint32_t generation = n_tile + 1U;
                uint64_t expand_start;
                if (w4f16_pool == NULL) {
                    result = -1;
                    goto projection_command_complete;
                }
                qbh_hmx_fp16_init_channel_scales(
                    buffers->scale_or_bias,
                    (const float *)w4f16_scale_slots[n_tile & 1U]);
                qbh_hmx_start_fp16_streaming(
                    worker, projection_activation,
                    buffers->expanded_weight, buffers->scale_or_bias,
                    buffers->hmx_output, 2U, k_tiles,
                    header->w4f16_region_tiles, w4f16_ready,
                    generation);
                ++header->w4f16_streamed_command_count;
                expand_start = HAP_perf_get_qtimer_count();
                qbh_w4f16_pool_start(
                    w4f16_pool,
                    w4f16_compressed_slots[n_tile & 1U],
                    (const float *)w4f16_scale_slots[n_tile & 1U],
                    buffers->expanded_weight, w4f16_ready,
                    generation, region_count,
                    header->w4f16_region_tiles,
                    w4f16_pool->worker_count, 1U, 1U, UINT32_MAX);
                qbh_w4f16_pool_wait(w4f16_pool);
                header->w4f16_expand_ticks +=
                    HAP_perf_get_qtimer_count() - expand_start;
                if (desc == &header->projections[0] &&
                    n_tile == 0U) {
                    header->w4f16_expand_mismatch_count =
                        qbh_audit_unscaled_w4_to_f16_tile(
                            w4f16_compressed_slots[n_tile & 1U],
                            buffers->expanded_weight,
                            &header->w4f16_expand_first_logical_index,
                            &header->w4f16_expand_expected_half_bits,
                            &header->w4f16_expand_actual_half_bits);
                    if (header->w4f16_expand_mismatch_count == 0U) {
                        header->w4f16_expand_mismatch_count =
                            qbh_hmx_fp16_audit_channel_scales(
                                buffers->scale_or_bias,
                                (const float *)w4f16_scale_slots[0],
                                &header->w4f16_expand_first_logical_index,
                                &header->w4f16_expand_expected_half_bits,
                                &header->w4f16_expand_actual_half_bits);
                    }
                }
                result = qbh_hmx_wait(worker);
            } else {
                result = qbh_hmx_submit(
                    worker, QBH_BLOCK_HMX_FP16,
                    projection_activation, buffers->expanded_weight,
                    buffers->scale_or_bias, buffers->hmx_output, 2U,
                    k_tiles, 1U);
            }
projection_command_complete:
            header->projection_hmx_wait_ticks +=
                HAP_perf_get_qtimer_count() - phase_start;
            header->hmx_fp16_tile_pair_count += 2U * k_tiles;
            if (result == 0) {
                phase_start = HAP_perf_get_qtimer_count();
                qbh_unpack_fp16_output(
                    (const __fp16 *)buffers->hmx_output, 1U,
                    (__fp16 *)output, desc->n,
                    n_tile * QBH_HMX_FP16_COLS);
                header->projection_unpack_ticks +=
                    HAP_perf_get_qtimer_count() - phase_start;
            }
        }
        if (w4f16_prefetch_active != 0) {
            uint64_t wait_start = HAP_perf_get_qtimer_count();
            int prefetch_result;
            failure_step = 8U;
            prefetch_result = qbh_dma_wait_weight_prefetch(
                &w4f16_prefetch_descriptor);
            if (prefetch_result != 0) {
                qbh_record_projection_failure(
                    header, desc, n_tile, failure_step,
                    prefetch_result);
                return -1;
            }
            header->w4f16_prefetch_wait_ticks +=
                HAP_perf_get_qtimer_count() - wait_start;
            header->weight_dma_ticks +=
                HAP_perf_get_qtimer_count() - w4f16_prefetch_start;
            {
                uint32_t next_tile = n_tile + 1U;
                uint32_t next_slot = next_tile & 1U;
                failure_step = 9U;
                if (qbh_dma_copy(
                        header, w4f16_scale_slots[next_slot],
                        shared + desc->scale_offset +
                            (size_t)next_tile * 32U * sizeof(float),
                        32U * sizeof(float), 1U) != 0) {
                    qbh_record_projection_failure(
                        header, desc, n_tile, failure_step, -1);
                    return -1;
                }
                header->weight_ddr_read_bytes += 32U * sizeof(float);
                ++header->weight_dma_descriptor_count;
            }
            w4f16_prefetch_active = 0;
        }
        ++header->hmx_command_count;
        if (result != 0) {
            qbh_record_projection_failure(
                header, desc, n_tile, failure_step, result);
            return -1;
        }
    }
    (void)element_bytes;
    return 0;
}

static uint8_t qbh_quantize(float value,
                            const struct qbh_block_qparam *qparam) {
    float encoded = value / qparam->scale + (float)qparam->zero_point;
    int32_t rounded = (int32_t)roundf(encoded);
    if (rounded < 0) {
        return 0U;
    }
    if (rounded > 255) {
        return UINT8_MAX;
    }
    return (uint8_t)rounded;
}

static float qbh_dequantize(uint8_t value,
                            const struct qbh_block_qparam *qparam) {
    return ((float)value - (float)qparam->zero_point) * qparam->scale;
}

static void qbh_rms_norm_f16(const __fp16 *input,
                             const __fp16 *gamma, __fp16 *output,
                             uint32_t rows, uint32_t width) {
    for (uint32_t row = 0; row < rows; ++row) {
        float sum = 0.0f;
        for (uint32_t channel = 0; channel < width; ++channel) {
            float value = (float)input[(size_t)row * width + channel];
            sum += value * value;
        }
        float inverse = 1.0f / sqrtf(sum / (float)width + 1.0e-6f);
        for (uint32_t channel = 0; channel < width; ++channel) {
            output[(size_t)row * width + channel] = (__fp16)(
                (float)input[(size_t)row * width + channel] * inverse *
                (float)gamma[channel]);
        }
    }
}

static void qbh_rms_norm_u8(const uint8_t *input,
                            const struct qbh_block_qparam *input_qparam,
                            const __fp16 *gamma, uint8_t *output,
                            const struct qbh_block_qparam *output_qparam,
                            uint32_t rows, uint32_t width) {
    for (uint32_t row = 0; row < rows; ++row) {
        float sum = 0.0f;
        for (uint32_t channel = 0; channel < width; ++channel) {
            float value = qbh_dequantize(
                input[(size_t)row * width + channel], input_qparam);
            sum += value * value;
        }
        float inverse = 1.0f / sqrtf(sum / (float)width + 1.0e-6f);
        for (uint32_t channel = 0; channel < width; ++channel) {
            float value = qbh_dequantize(
                input[(size_t)row * width + channel], input_qparam);
            output[(size_t)row * width + channel] = qbh_quantize(
                value * inverse * (float)gamma[channel], output_qparam);
        }
    }
}

static void qbh_qk_norm_rope_f16(
    __fp16 *tensor, uint32_t heads, uint32_t row_stride,
    const __fp16 *gamma, const __fp16 *cosine,
    const __fp16 *sine) {
    for (uint32_t row = 0; row < QBH_BLOCK_M; ++row) {
        for (uint32_t head = 0; head < heads; ++head) {
            __fp16 *values = tensor + (size_t)row * row_stride +
                             head * QBH_BLOCK_HEAD_DIM;
            float sum = 0.0f;
            for (uint32_t channel = 0; channel < QBH_BLOCK_HEAD_DIM;
                 ++channel) {
                float value = (float)values[channel];
                sum += value * value;
            }
            float inverse = 1.0f / sqrtf(
                sum / (float)QBH_BLOCK_HEAD_DIM + 1.0e-6f);
            for (uint32_t channel = 0;
                 channel < QBH_BLOCK_HEAD_DIM / 2U; ++channel) {
                float first = (float)values[channel] * inverse *
                              (float)gamma[channel];
                float second =
                    (float)values[channel + QBH_BLOCK_HEAD_DIM / 2U] *
                    inverse *
                    (float)gamma[channel + QBH_BLOCK_HEAD_DIM / 2U];
                float c0 = (float)cosine[
                    (size_t)row * QBH_BLOCK_HEAD_DIM + channel];
                float s0 = (float)sine[
                    (size_t)row * QBH_BLOCK_HEAD_DIM + channel];
                float c1 = (float)cosine[
                    (size_t)row * QBH_BLOCK_HEAD_DIM + channel +
                    QBH_BLOCK_HEAD_DIM / 2U];
                float s1 = (float)sine[
                    (size_t)row * QBH_BLOCK_HEAD_DIM + channel +
                    QBH_BLOCK_HEAD_DIM / 2U];
                values[channel] = (__fp16)(first * c0 - second * s0);
                values[channel + QBH_BLOCK_HEAD_DIM / 2U] =
                    (__fp16)(second * c1 + first * s1);
            }
        }
    }
}

static void qbh_qk_norm_rope_u8(
    uint8_t *tensor, uint32_t heads, uint32_t row_stride,
    const struct qbh_block_qparam *input_qparam,
    const struct qbh_block_qparam *output_qparam,
    const __fp16 *gamma, const __fp16 *cosine,
    const __fp16 *sine) {
    for (uint32_t row = 0; row < QBH_BLOCK_M; ++row) {
        for (uint32_t head = 0; head < heads; ++head) {
            uint8_t *values = tensor + (size_t)row * row_stride +
                              head * QBH_BLOCK_HEAD_DIM;
            float sum = 0.0f;
            for (uint32_t channel = 0; channel < QBH_BLOCK_HEAD_DIM;
                 ++channel) {
                float value = qbh_dequantize(values[channel], input_qparam);
                sum += value * value;
            }
            float inverse = 1.0f / sqrtf(
                sum / (float)QBH_BLOCK_HEAD_DIM + 1.0e-6f);
            for (uint32_t channel = 0;
                 channel < QBH_BLOCK_HEAD_DIM / 2U; ++channel) {
                float first = qbh_dequantize(values[channel], input_qparam) *
                              inverse * (float)gamma[channel];
                float second = qbh_dequantize(
                                   values[channel +
                                          QBH_BLOCK_HEAD_DIM / 2U],
                                   input_qparam) *
                               inverse *
                               (float)gamma[channel +
                                            QBH_BLOCK_HEAD_DIM / 2U];
                float c0 = (float)cosine[
                    (size_t)row * QBH_BLOCK_HEAD_DIM + channel];
                float s0 = (float)sine[
                    (size_t)row * QBH_BLOCK_HEAD_DIM + channel];
                float c1 = (float)cosine[
                    (size_t)row * QBH_BLOCK_HEAD_DIM + channel +
                    QBH_BLOCK_HEAD_DIM / 2U];
                float s1 = (float)sine[
                    (size_t)row * QBH_BLOCK_HEAD_DIM + channel +
                    QBH_BLOCK_HEAD_DIM / 2U];
                values[channel] = qbh_quantize(
                    first * c0 - second * s0, output_qparam);
                values[channel + QBH_BLOCK_HEAD_DIM / 2U] = qbh_quantize(
                    second * c1 + first * s1, output_qparam);
            }
        }
    }
}

static void qbh_expand_u8_to_f16_in_place(
    uint8_t *buffer, uint32_t elements,
    const struct qbh_block_qparam *qparam) {
    __fp16 *expanded = (__fp16 *)buffer;
    for (uint32_t index = elements; index-- > 0U;) {
        expanded[index] = (__fp16)qbh_dequantize(buffer[index], qparam);
    }
}

static void qbh_pack_fp16_weight_rows_scalar(
    const __fp16 *source, uint32_t source_stride,
    uint32_t source_column, uint32_t k, uint32_t n,
    __fp16 *destination) {
    uint32_t k_tiles = k / QBH_HMX_FP16_ROWS;
    for (uint32_t output = 0; output < n; ++output) {
        uint32_t n_tile = output / QBH_HMX_FP16_COLS;
        uint32_t local_output = output % QBH_HMX_FP16_COLS;
        for (uint32_t input = 0; input < k; ++input) {
            uint32_t k_tile = input / QBH_HMX_FP16_ROWS;
            uint32_t local_input = input % QBH_HMX_FP16_ROWS;
            size_t tile = ((size_t)n_tile * k_tiles + k_tile) *
                          QBH_HMX_FP16_TILE_ELEMENTS;
            destination[tile + qbh_hmx_fp16_tile_offset(
                                   local_input, local_output)] =
                source[(size_t)output * source_stride +
                       source_column + input];
        }
    }
}

static void qbh_pack_fp16_weight_transposed_scalar(
    const __fp16 *source, uint32_t source_stride,
    uint32_t source_column, uint32_t rows, uint32_t columns,
    __fp16 *destination) {
    uint32_t k_tiles = rows / QBH_HMX_FP16_ROWS;
    for (uint32_t output = 0; output < columns; ++output) {
        uint32_t n_tile = output / QBH_HMX_FP16_COLS;
        uint32_t local_output = output % QBH_HMX_FP16_COLS;
        for (uint32_t input = 0; input < rows; ++input) {
            uint32_t k_tile = input / QBH_HMX_FP16_ROWS;
            uint32_t local_input = input % QBH_HMX_FP16_ROWS;
            size_t tile = ((size_t)n_tile * k_tiles + k_tile) *
                          QBH_HMX_FP16_TILE_ELEMENTS;
            destination[tile + qbh_hmx_fp16_tile_offset(
                                   local_input, local_output)] =
                source[(size_t)input * source_stride +
                       source_column + output];
        }
    }
}

/* Word offsets for one FP16 HMX dual-tile.  Each vscatter word contains two
 * adjacent K values and successive words are 128 bytes apart in Crouton. */
static const int32_t qbh_attention_vscatter_offsets[32]
    __attribute__((aligned(128))) = {
        0, 128, 256, 384, 512, 640, 768, 896,
        1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920,
        2048, 2176, 2304, 2432, 2560, 2688, 2816, 2944,
        3072, 3200, 3328, 3456, 3584, 3712, 3840, 3968,
    };

static void qbh_pack_fp16_weight_rows_hvx(
    const __fp16 *source, uint32_t source_stride,
    uint32_t source_column, uint32_t k, uint32_t n,
    __fp16 *destination) {
    uint32_t k_tiles = k / QBH_HMX_FP16_ROWS;
    uint32_t n_tiles = n / QBH_HMX_FP16_COLS;
    HVX_Vector offsets_base = *(const HVX_Vector *)
        qbh_attention_vscatter_offsets;
    HVX_Vector offset_step = Q6_V_vsplat_R(4);

    for (uint32_t n_tile = 0; n_tile < n_tiles; ++n_tile) {
        HVX_Vector offsets = offsets_base;
        for (uint32_t local_output = 0;
             local_output < QBH_HMX_FP16_COLS; ++local_output) {
            const HVX_Vector *input = (const HVX_Vector *)(
                source + (size_t)(n_tile * QBH_HMX_FP16_COLS +
                                  local_output) * source_stride +
                source_column);
            for (uint32_t channel = 0; channel < k; channel += 64U) {
                size_t tile = ((size_t)n_tile * k_tiles +
                               channel / QBH_HMX_FP16_ROWS) *
                              QBH_HMX_FP16_TILE_ELEMENTS;
                Q6_vscatter_RMVwV(
                    (uint32_t)(uintptr_t)(destination + tile),
                    2U * QBH_HMX_FP16_TILE_BYTES - 1U,
                    offsets, *input++);
            }
            offsets = Q6_Vw_vadd_VwVw(offsets, offset_step);
        }
    }
    asm volatile("barrier" ::: "memory");
}

static void qbh_pack_fp16_weight_transposed_hvx(
    const __fp16 *source, uint32_t source_stride,
    uint32_t source_column, uint32_t rows, uint32_t columns,
    __fp16 *destination) {
    uint32_t k_tiles = rows / QBH_HMX_FP16_ROWS;

    for (uint32_t input = 0; input < rows; input += 2U) {
        uint32_t k_tile = input / QBH_HMX_FP16_ROWS;
        uint32_t row_pair =
            (input % QBH_HMX_FP16_ROWS) / 2U;
        const HVX_Vector *source0 = (const HVX_Vector *)(
            source + (size_t)input * source_stride + source_column);
        const HVX_Vector *source1 = (const HVX_Vector *)(
            source + (size_t)(input + 1U) * source_stride +
            source_column);

        for (uint32_t output = 0; output < columns; output += 64U) {
            HVX_VectorPair packed = Q6_W_vshuff_VVR(
                *source1++, *source0++, -2);
            uint32_t n_tile = output / QBH_HMX_FP16_COLS;
            size_t tile0 = ((size_t)n_tile * k_tiles + k_tile) *
                           QBH_HMX_FP16_TILE_ELEMENTS;
            size_t tile1 = ((size_t)(n_tile + 1U) * k_tiles + k_tile) *
                           QBH_HMX_FP16_TILE_ELEMENTS;
            HVX_Vector *output0 =
                (HVX_Vector *)(destination + tile0) + row_pair;
            HVX_Vector *output1 =
                (HVX_Vector *)(destination + tile1) + row_pair;
            *output0 = Q6_V_lo_W(packed);
            *output1 = Q6_V_hi_W(packed);
        }
    }
    asm volatile("barrier" ::: "memory");
}

static __fp16 *qbh_attention_gqa_scratch(
    uint8_t *base, uint32_t context_index) {
    uintptr_t aligned =
        ((uintptr_t)base + QBH_HMX_FP16_TILE_BYTES - 1U) &
        ~((uintptr_t)QBH_HMX_FP16_TILE_BYTES - 1U);
    return (__fp16 *)(aligned +
        (size_t)context_index * QBH_BLOCK_HMX_OUTPUT_MAX_BYTES);
}

static void qbh_attention_copy_v_crouton_weight(
    const __fp16 *source_group_tiles, uint32_t head,
    uint32_t source_group_tiles_per_command,
    __fp16 *destination_weight) {
    static const uint8_t source_tile_for_weight_batch2[8] = {
        0U, 2U, 1U, 3U, 4U, 6U, 5U, 7U};
    static const uint8_t source_tile_for_weight_batch4[8] = {
        0U, 4U, 1U, 5U, 2U, 6U, 3U, 7U};
    const uint8_t *source_tile_for_weight =
        source_group_tiles_per_command == 4U
            ? source_tile_for_weight_batch4
            : source_tile_for_weight_batch2;
    const __fp16 *source = source_group_tiles +
        (size_t)head * 8U * QBH_HMX_FP16_TILE_ELEMENTS;

    for (uint32_t destination_tile = 0U;
         destination_tile < 8U; ++destination_tile) {
        const HVX_Vector *input = (const HVX_Vector *)(source +
            (size_t)source_tile_for_weight[destination_tile] *
                QBH_HMX_FP16_TILE_ELEMENTS);
        HVX_Vector *output = (HVX_Vector *)(destination_weight +
            (size_t)destination_tile * QBH_HMX_FP16_TILE_ELEMENTS);
        for (uint32_t vector = 0U;
             vector < QBH_HMX_FP16_TILE_BYTES / sizeof(HVX_Vector);
             ++vector) {
            output[vector] = input[vector];
        }
    }
    asm volatile("barrier" ::: "memory");
}

static void qbh_attention_copy_av_crouton_to_o_activation(
    const __fp16 *source, uint32_t head,
    __fp16 *destination) {
    const uint32_t destination_tiles_per_row =
        QBH_BLOCK_HIDDEN / QBH_HMX_FP16_COLS;
    const uint32_t head_first_tile =
        head * (QBH_BLOCK_HEAD_DIM / QBH_HMX_FP16_COLS);

    for (uint32_t row_tile = 0U;
         row_tile < QBH_BLOCK_M / QBH_HMX_FP16_ROWS;
         ++row_tile) {
        for (uint32_t column_tile = 0U;
             column_tile < QBH_BLOCK_HEAD_DIM / QBH_HMX_FP16_COLS;
             ++column_tile) {
            const HVX_Vector *input = (const HVX_Vector *)(source +
                (size_t)(row_tile * 4U + column_tile) *
                    QBH_HMX_FP16_TILE_ELEMENTS);
            HVX_Vector *output = (HVX_Vector *)(destination +
                qbh_hmx_fp16_matrix_tile_offset(
                    row_tile, head_first_tile + column_tile,
                    destination_tiles_per_row));
            for (uint32_t vector = 0U;
                 vector < QBH_HMX_FP16_TILE_BYTES /
                              sizeof(HVX_Vector);
                 ++vector) {
                output[vector] = input[vector];
            }
        }
    }
    asm volatile("barrier" ::: "memory");
}

static uint32_t qbh_count_u16_mismatches(
    const uint16_t *left, const uint16_t *right,
    uint32_t elements) {
    uint32_t mismatches = 0U;
    for (uint32_t index = 0U; index < elements; ++index) {
        mismatches += left[index] != right[index];
    }
    return mismatches;
}

static uint32_t qbh_count_u8_mismatches(
    const uint8_t *left, const uint8_t *right,
    uint32_t elements) {
    uint32_t mismatches = 0U;
    for (uint32_t index = 0U; index < elements; ++index) {
        mismatches += left[index] != right[index];
    }
    return mismatches;
}

static void qbh_unpack_fp16_grouped_projection(
    const __fp16 *source, uint32_t n_tiles,
    __fp16 *destination, uint32_t destination_stride,
    uint32_t group_tiles) {
    const uint32_t group_elements =
        (QBH_BLOCK_M / QBH_HMX_FP16_ROWS) *
        group_tiles * QBH_HMX_FP16_TILE_ELEMENTS;
    for (uint32_t first_tile = 0U; first_tile < n_tiles;
         first_tile += group_tiles) {
        qbh_unpack_fp16_output(
            source + (size_t)(first_tile / group_tiles) *
                group_elements,
            group_tiles, destination, destination_stride,
            first_tile * QBH_HMX_FP16_COLS);
    }
}

static void qbh_audit_crouton_qkv_operands(
    struct qbh_block_header *header,
    struct qbh_block_buffers *buffers) {
    const uint32_t head_elements =
        QBH_BLOCK_M * QBH_BLOCK_HEAD_DIM;
    __fp16 *q_reference = (__fp16 *)buffers->gate;
    __fp16 *v_reference = q_reference +
        QBH_BLOCK_M * QBH_BLOCK_HIDDEN;
    __fp16 *k_reference = (__fp16 *)buffers->up;
    __fp16 *reference_tiles = (__fp16 *)buffers->hmx_activation;
    __fp16 *candidate_tiles = reference_tiles + head_elements;
    const uint32_t q_group_tiles = qbh_w4f16_projection_group_tiles(
        header, &header->projections[QBH_BLOCK_PROJ_Q]);
    const uint32_t k_group_tiles = qbh_w4f16_projection_group_tiles(
        header, &header->projections[QBH_BLOCK_PROJ_K]);
    const uint32_t v_group_tiles = qbh_w4f16_projection_group_tiles(
        header, &header->projections[QBH_BLOCK_PROJ_V]);

    qbh_unpack_fp16_grouped_projection(
        (const __fp16 *)buffers->q,
        QBH_BLOCK_HIDDEN / QBH_HMX_FP16_COLS,
        q_reference, QBH_BLOCK_HIDDEN,
        q_group_tiles);
    qbh_unpack_fp16_grouped_projection(
        (const __fp16 *)buffers->k,
        QBH_BLOCK_KV_HIDDEN / QBH_HMX_FP16_COLS,
        k_reference, QBH_BLOCK_KV_HIDDEN,
        k_group_tiles);
    qbh_unpack_fp16_grouped_projection(
        (const __fp16 *)buffers->v,
        QBH_BLOCK_KV_HIDDEN / QBH_HMX_FP16_COLS,
        v_reference, QBH_BLOCK_KV_HIDDEN,
        v_group_tiles);
    qbh_hvx_qk_norm_rope_f16(
        q_reference, QBH_BLOCK_M, QBH_BLOCK_HEADS,
        QBH_BLOCK_HIDDEN, QBH_BLOCK_HEAD_DIM,
        (const __fp16 *)buffers->q_norm_weight,
        (const __fp16 *)buffers->rope_cos,
        (const __fp16 *)buffers->rope_sin, NULL);
    qbh_hvx_qk_norm_rope_f16(
        k_reference, QBH_BLOCK_M, QBH_BLOCK_KV_HEADS,
        QBH_BLOCK_KV_HIDDEN, QBH_BLOCK_HEAD_DIM,
        (const __fp16 *)buffers->k_norm_weight,
        (const __fp16 *)buffers->rope_cos,
        (const __fp16 *)buffers->rope_sin, NULL);

    for (uint32_t head = 0U; head < QBH_BLOCK_HEADS; ++head) {
        qbh_pack_fp16_activation(
            q_reference + head * QBH_BLOCK_HEAD_DIM,
            QBH_BLOCK_HIDDEN, QBH_BLOCK_HEAD_DIM,
            reference_tiles);
        header->crouton_q_operand_mismatch_count +=
            qbh_count_u16_mismatches(
                (const uint16_t *)reference_tiles,
                (const uint16_t *)buffers->attention_projection +
                    (size_t)head * head_elements,
                head_elements);
    }
    for (uint32_t head = 0U; head < QBH_BLOCK_KV_HEADS; ++head) {
        qbh_pack_fp16_weight_rows_hvx(
            k_reference, QBH_BLOCK_KV_HIDDEN,
            head * QBH_BLOCK_HEAD_DIM,
            QBH_BLOCK_HEAD_DIM, QBH_BLOCK_M,
            reference_tiles);
        header->crouton_k_operand_mismatch_count +=
            qbh_count_u16_mismatches(
                (const uint16_t *)reference_tiles,
                (const uint16_t *)buffers->scores +
                    (size_t)head * head_elements,
                head_elements);
        qbh_pack_fp16_weight_transposed_hvx(
            v_reference, QBH_BLOCK_KV_HIDDEN,
            head * QBH_BLOCK_HEAD_DIM, QBH_BLOCK_M,
            QBH_BLOCK_HEAD_DIM, reference_tiles);
        qbh_attention_copy_v_crouton_weight(
            (const __fp16 *)buffers->v, head,
            v_group_tiles,
            candidate_tiles);
        header->crouton_v_operand_mismatch_count +=
            qbh_count_u16_mismatches(
                (const uint16_t *)reference_tiles,
                (const uint16_t *)candidate_tiles,
                head_elements);
    }
    header->qkv_operand_audit_tensor_count += 3U;
}

static void qbh_audit_row_major_qkv_operands(
    struct qbh_block_header *header,
    struct qbh_block_buffers *buffers) {
    __fp16 *q_reference = (__fp16 *)buffers->gate;
    __fp16 *v_reference = q_reference +
        QBH_BLOCK_M * QBH_BLOCK_HIDDEN;
    __fp16 *k_reference = (__fp16 *)buffers->up;

    qbh_hvx_qk_norm_rope_f16(
        q_reference, QBH_BLOCK_M, QBH_BLOCK_HEADS,
        QBH_BLOCK_HIDDEN, QBH_BLOCK_HEAD_DIM,
        (const __fp16 *)buffers->q_norm_weight,
        (const __fp16 *)buffers->rope_cos,
        (const __fp16 *)buffers->rope_sin, NULL);
    qbh_hvx_qk_norm_rope_f16(
        k_reference, QBH_BLOCK_M, QBH_BLOCK_KV_HEADS,
        QBH_BLOCK_KV_HIDDEN, QBH_BLOCK_HEAD_DIM,
        (const __fp16 *)buffers->k_norm_weight,
        (const __fp16 *)buffers->rope_cos,
        (const __fp16 *)buffers->rope_sin, NULL);
    header->crouton_q_operand_mismatch_count +=
        qbh_count_u16_mismatches(
            (const uint16_t *)q_reference,
            (const uint16_t *)buffers->q,
            QBH_BLOCK_M * QBH_BLOCK_HIDDEN);
    header->crouton_k_operand_mismatch_count +=
        qbh_count_u16_mismatches(
            (const uint16_t *)k_reference,
            (const uint16_t *)buffers->k,
            QBH_BLOCK_M * QBH_BLOCK_KV_HIDDEN);
    header->crouton_v_operand_mismatch_count +=
        qbh_count_u16_mismatches(
            (const uint16_t *)v_reference,
            (const uint16_t *)buffers->v,
            QBH_BLOCK_M * QBH_BLOCK_KV_HIDDEN);
    header->qkv_operand_audit_tensor_count += 3U;
}

static int qbh_attention_gqa_submit(
    struct qbh_block_w4f16_pool *pool,
    struct qbh_block_w4f16_job *job,
    const __fp16 *activation, const __fp16 *weight,
    __fp16 *output, uint32_t k_tiles, uint32_t n_tiles) {
    uint64_t queue_start = HAP_perf_get_qtimer_count();
    uint64_t hmx_start;
    int result;

    qurt_mutex_lock(&pool->attention_hmx_mutex);
    job->attention_gqa_queue_wait_ticks +=
        HAP_perf_get_qtimer_count() - queue_start;
    if (pool->attention_gqa_abort != 0U) {
        qurt_mutex_unlock(&pool->attention_hmx_mutex);
        return -1;
    }
    hmx_start = HAP_perf_get_qtimer_count();
    result = qbh_hmx_submit(
        pool->attention_hmx_worker, QBH_BLOCK_HMX_FP16,
        activation, weight,
        pool->attention_buffers->scale_or_bias,
        output, 2U, k_tiles, n_tiles);
    job->attention_gqa_hmx_wait_ticks +=
        HAP_perf_get_qtimer_count() - hmx_start;
    qurt_mutex_unlock(&pool->attention_hmx_mutex);
    if (result != 0) {
        pool->attention_gqa_abort = 1U;
        asm volatile("barrier" ::: "memory");
        return -1;
    }
    return 0;
}

static void qbh_attention_gqa_pool_run_tasks(
    struct qbh_block_w4f16_pool *pool,
    struct qbh_block_w4f16_job *job) {
    struct qbh_block_buffers *buffers = pool->attention_buffers;
    __fp16 *activation = qbh_attention_gqa_scratch(
        buffers->gate, job->worker_index);
    __fp16 *weight = qbh_attention_gqa_scratch(
        buffers->up, job->worker_index);
    __fp16 *output = qbh_attention_gqa_scratch(
        buffers->middle, job->worker_index);
    const size_t head_elements =
        (size_t)QBH_BLOCK_M * QBH_BLOCK_M;
    const uint32_t direct_qkv = qbh_crouton_qkv_enabled(
        pool->attention_header);
    const uint32_t direct_av_o =
        (pool->attention_header->crouton_boundary_mode &
         QBH_BLOCK_CROUTON_BOUNDARY_AV_TO_O) != 0U;

    for (;;) {
        uint32_t kv_head =
            qbh_atomic_fetch_increment(&pool->next_attention_task);
        uint32_t first_q_head;
        uint64_t group_start;

        if (kv_head >= pool->attention_task_count ||
            pool->attention_gqa_abort != 0U) {
            break;
        }
        first_q_head = kv_head *
            (QBH_BLOCK_HEADS / QBH_BLOCK_KV_HEADS);
        group_start = HAP_perf_get_qtimer_count();

        if (pool->attention_header->attention_pipeline_mode !=
            QBH_BLOCK_ATTENTION_PIPELINE_GQA_QKV_OVERLAP) {
            qbh_hvx_qk_norm_rope_f16_head(
                pool->attention_q, QBH_BLOCK_M, QBH_BLOCK_HIDDEN,
                QBH_BLOCK_HEAD_DIM, first_q_head,
                pool->attention_q_gamma, pool->attention_rope_cos,
                pool->attention_rope_sin);
            qbh_hvx_qk_norm_rope_f16_head(
                pool->attention_q, QBH_BLOCK_M, QBH_BLOCK_HIDDEN,
                QBH_BLOCK_HEAD_DIM, first_q_head + 1U,
                pool->attention_q_gamma, pool->attention_rope_cos,
                pool->attention_rope_sin);
            qbh_hvx_qk_norm_rope_f16_head(
                pool->attention_k, QBH_BLOCK_M, QBH_BLOCK_KV_HIDDEN,
                QBH_BLOCK_HEAD_DIM, kv_head,
                pool->attention_k_gamma, pool->attention_rope_cos,
                pool->attention_rope_sin);
        }

        if (direct_qkv != 0U) {
            weight = (__fp16 *)buffers->scores +
                (size_t)kv_head * 2U * head_elements;
        } else {
            qbh_pack_fp16_weight_rows_hvx(
                (const __fp16 *)buffers->k, QBH_BLOCK_KV_HIDDEN,
                kv_head * QBH_BLOCK_HEAD_DIM,
                QBH_BLOCK_HEAD_DIM, QBH_BLOCK_M, weight);
        }
        for (uint32_t local_head = 0U; local_head < 2U; ++local_head) {
            uint32_t head = first_q_head + local_head;
            if (direct_qkv != 0U) {
                activation = (__fp16 *)buffers->attention_projection +
                    (size_t)head * 2U * head_elements;
            } else {
                qbh_pack_fp16_activation(
                    (const __fp16 *)buffers->q +
                        head * QBH_BLOCK_HEAD_DIM,
                    QBH_BLOCK_HIDDEN, QBH_BLOCK_HEAD_DIM, activation);
            }
            if (qbh_attention_gqa_submit(
                    pool, job, activation, weight, output, 4U, 2U) != 0) {
                break;
            }
            qbh_unpack_fp16_output(
                output, 2U,
                (direct_qkv != 0U
                     ? (__fp16 *)buffers->probability
                     : (__fp16 *)buffers->scores) +
                    (size_t)head * head_elements,
                QBH_BLOCK_M, 0U);
        }
        if (pool->attention_gqa_abort != 0U) {
            break;
        }

        if (pool->attention_header->numerical_audit_enabled != 0U) {
            for (uint32_t local_head = 0U;
                 local_head < 2U; ++local_head) {
                uint32_t head = first_q_head + local_head;
                pool->attention_gqa_qk_max_abs[head] = qbh_f16_max_abs(
                    pool->attention_header,
                    (direct_qkv != 0U
                         ? (__fp16 *)buffers->probability
                         : (__fp16 *)buffers->scores) +
                        (size_t)head * head_elements,
                    (uint32_t)head_elements);
            }
        }

        /* EXP-0158: after both Q heads have consumed K, preserve the exact
         * HMX K-weight carrier in this group's dead Q-carrier range.  Every
         * worker owns a distinct GQA group, so this is lock-free and remains
         * entirely in VTCM. */
        if (pool->attention_header->scan_mode ==
                QBH_BLOCK_SCAN_PREFILL &&
            qbh_hmx_native_f16_cache_formats(
                pool->attention_header->kv_cache_k_format,
                pool->attention_header->kv_cache_v_format)) {
            qbh_hvx_copy_aligned_bytes(
                (__fp16 *)buffers->attention_projection +
                    (size_t)first_q_head * 2U * head_elements,
                weight,
                QBH_BLOCK_M * QBH_BLOCK_HEAD_DIM * sizeof(__fp16));
        }

        for (uint32_t local_head = 0U; local_head < 2U; ++local_head) {
            uint32_t head = first_q_head + local_head;
            qbh_hvx_stable_causal_softmax_f16(
                (direct_qkv != 0U
                     ? (__fp16 *)buffers->probability
                     : (__fp16 *)buffers->scores) +
                    (size_t)head * head_elements,
                (direct_qkv != 0U
                     ? (__fp16 *)buffers->scores
                     : (__fp16 *)buffers->probability) +
                    (size_t)head * head_elements,
                1U, QBH_BLOCK_M, QBH_BLOCK_M,
                0.08838834764831845f, NULL);
        }
        if (direct_qkv != 0U) {
            weight = qbh_attention_gqa_scratch(
                buffers->up, job->worker_index);
            qbh_attention_copy_v_crouton_weight(
                (const __fp16 *)buffers->v, kv_head,
                qbh_w4f16_projection_group_tiles(
                    pool->attention_header,
                    &pool->attention_header->projections[
                        QBH_BLOCK_PROJ_V]),
                weight);
        } else {
            qbh_pack_fp16_weight_transposed_hvx(
                (const __fp16 *)buffers->v, QBH_BLOCK_KV_HIDDEN,
                kv_head * QBH_BLOCK_HEAD_DIM, QBH_BLOCK_M,
                QBH_BLOCK_HEAD_DIM, weight);
        }
        for (uint32_t local_head = 0U; local_head < 2U; ++local_head) {
            uint32_t head = first_q_head + local_head;
            qbh_pack_fp16_activation(
                (direct_qkv != 0U
                     ? (const __fp16 *)buffers->scores
                     : (const __fp16 *)buffers->probability) +
                    (size_t)head * head_elements,
                QBH_BLOCK_M, QBH_BLOCK_M, activation);
            if (qbh_attention_gqa_submit(
                    pool, job, activation, weight, output, 2U, 4U) != 0) {
                break;
            }
            if (direct_av_o != 0U) {
                uint64_t copy_start = HAP_perf_get_qtimer_count();
                qbh_attention_copy_av_crouton_to_o_activation(
                    output, head,
                    (__fp16 *)buffers->hmx_activation);
                job->attention_av_o_copy_ticks +=
                    HAP_perf_get_qtimer_count() - copy_start;
            } else {
                qbh_unpack_fp16_output(
                    output, 4U, (__fp16 *)buffers->attention_concat,
                    QBH_BLOCK_HIDDEN, head * QBH_BLOCK_HEAD_DIM);
            }
        }
        if (pool->attention_gqa_abort != 0U) {
            break;
        }
        /* The AV weight is dead after both query heads complete.  Preserve
         * it in the group's dead probability plane for post-Attention DMA to
         * the persistent cache-native V carrier. */
        if (pool->attention_header->scan_mode ==
                QBH_BLOCK_SCAN_PREFILL &&
            qbh_hmx_native_f16_cache_formats(
                pool->attention_header->kv_cache_k_format,
                pool->attention_header->kv_cache_v_format)) {
            qbh_hvx_copy_aligned_bytes(
                (__fp16 *)buffers->probability +
                    (size_t)first_q_head * head_elements,
                weight,
                QBH_BLOCK_M * QBH_BLOCK_HEAD_DIM * sizeof(__fp16));
        }
        job->attention_gqa_ticks +=
            HAP_perf_get_qtimer_count() - group_start;
        ++job->attention_gqa_group_count;
    }
}

static int qbh_hvx_pool_gqa_attention(
    struct qbh_block_header *header,
    struct qbh_block_w4f16_pool *pool,
    struct qbh_block_buffers *buffers,
    struct qbh_block_hmx_worker *worker) {
    struct qbh_block_w4f16_job main_job;
    uint64_t wait_start;

    if (pool == NULL || buffers == NULL || worker == NULL ||
        header->attention_hvx_contexts != 4U ||
        header->attention_hvx_contexts - 1U > pool->worker_count) {
        return -1;
    }
    memset(&main_job, 0, sizeof(main_job));
    main_job.worker_index = header->attention_hvx_contexts - 1U;
    pool->attention_header = header;
    pool->attention_buffers = buffers;
    pool->attention_hmx_worker = worker;
    pool->attention_q = (__fp16 *)buffers->q;
    pool->attention_k = (__fp16 *)buffers->k;
    pool->attention_q_gamma =
        (const __fp16 *)buffers->q_norm_weight;
    pool->attention_k_gamma =
        (const __fp16 *)buffers->k_norm_weight;
    pool->attention_rope_cos = (const __fp16 *)buffers->rope_cos;
    pool->attention_rope_sin = (const __fp16 *)buffers->rope_sin;
    pool->attention_scores = (__fp16 *)buffers->scores;
    pool->attention_probability = (__fp16 *)buffers->probability;
    pool->attention_task_count = QBH_BLOCK_KV_HEADS;
    pool->next_attention_task = 0U;
    pool->attention_gqa_abort = 0U;
    memset(pool->attention_gqa_qk_max_abs, 0,
           sizeof(pool->attention_gqa_qk_max_abs));
    pool->active_worker_count = header->attention_hvx_contexts - 1U;
    for (uint32_t worker_index = 0U;
         worker_index < pool->active_worker_count; ++worker_index) {
        pool->jobs[worker_index].command_kind =
            QBH_BLOCK_HVX_POOL_GQA_ATTENTION;
    }
    asm volatile("barrier" ::: "memory");
    for (uint32_t worker_index = 0U;
         worker_index < pool->active_worker_count; ++worker_index) {
        (void)qurt_sem_up(&pool->command_ready[worker_index]);
    }
    qbh_attention_gqa_pool_run_tasks(pool, &main_job);
    wait_start = HAP_perf_get_qtimer_count();
    qbh_w4f16_pool_wait(pool);
    main_job.attention_gqa_queue_wait_ticks +=
        HAP_perf_get_qtimer_count() - wait_start;
    header->attention_gqa_worker_work_ticks +=
        main_job.attention_gqa_ticks;
    header->attention_gqa_hmx_wait_ticks +=
        main_job.attention_gqa_hmx_wait_ticks;
    header->attention_gqa_queue_wait_ticks +=
        main_job.attention_gqa_queue_wait_ticks;
    header->crouton_av_o_copy_ticks +=
        main_job.attention_av_o_copy_ticks;
    header->attention_gqa_group_count +=
        main_job.attention_gqa_group_count;
    asm volatile("barrier" ::: "memory");
    if (pool->attention_gqa_abort != 0U) {
        return -1;
    }
    if (header->numerical_audit_enabled != 0U) {
        for (uint32_t head = 0U; head < QBH_BLOCK_HEADS; ++head) {
            float value = pool->attention_gqa_qk_max_abs[head];
            if (!isfinite(value)) {
                if (header->numerical_status ==
                    QBH_BLOCK_NUMERICAL_UNCHECKED) {
                    header->numerical_status =
                        QBH_BLOCK_NUMERICAL_ATTENTION_QK;
                }
                header->attention_qk_max_abs = INFINITY;
                break;
            }
            if (value > header->attention_qk_max_abs) {
                header->attention_qk_max_abs = value;
            }
        }
    }
    header->hmx_command_count += 2U * QBH_BLOCK_HEADS;
    header->hmx_fp16_tile_pair_count += 32U * QBH_BLOCK_HEADS;
    return 0;
}

static void qbh_record_f16_nonfinite(struct qbh_block_header *header,
                                     const void *data,
                                     uint32_t elements,
                                     int32_t stage);

static float qbh_f16_max_abs(
    const struct qbh_block_header *header,
    const __fp16 *data, uint32_t elements) {
    if (header->numerical_audit_enabled == 0U) {
        return 0.0f;
    }
    float maximum = 0.0f;
    for (uint32_t index = 0; index < elements; ++index) {
        float value = fabsf((float)data[index]);
        if (!isfinite(value)) {
            return INFINITY;
        }
        if (value > maximum) {
            maximum = value;
        }
    }
    return maximum;
}

static uint64_t qbh_attribution_mark(uint64_t *cursor) {
    uint64_t now = HAP_perf_get_qtimer_count();
    uint64_t elapsed = now - *cursor;
    *cursor = now;
    return elapsed;
}

static uint64_t qbh_attribution_begin(
    const struct qbh_block_header *header) {
    return header->attribution_enabled != 0U &&
                   header->numerical_audit_enabled != 0U
               ? HAP_perf_get_qtimer_count()
               : 0U;
}

static void qbh_attribution_accumulate(
    const struct qbh_block_header *header, uint64_t start,
    uint64_t *accumulator) {
    if (header->attribution_enabled != 0U &&
        header->numerical_audit_enabled != 0U) {
        *accumulator += HAP_perf_get_qtimer_count() - start;
    }
}

static uint64_t qbh_attention_attributed_ticks(
    const struct qbh_block_header *header) {
    return header->attention_setup_ticks +
           header->attention_qk_pack_ticks +
           header->attention_qk_hmx_ticks +
           header->attention_qk_unpack_ticks +
           header->attention_qk_audit_ticks +
           header->attention_softmax_ticks +
           header->attention_softmax_audit_ticks +
           header->attention_av_pack_ticks +
           header->attention_av_hmx_ticks +
           header->attention_av_unpack_ticks +
           header->attention_av_audit_ticks +
           header->attention_gqa_pipeline_ticks +
           header->u8_attention_qk_norm_rope_ticks +
           header->u8_attention_k_pack_ticks +
           header->u8_attention_v_pack_ticks +
           header->u8_attention_qk_hmx_ticks +
           header->u8_attention_qk_requant_ticks +
           header->u8_attention_softmax_ticks +
           header->u8_attention_av_hmx_ticks +
           header->u8_attention_av_requant_ticks +
           header->u8_attention_pipeline_wait_ticks;
}

static int qbh_attention_f16(struct qbh_block_header *header,
                             struct qbh_block_buffers *buffers,
                             struct qbh_block_hmx_worker *worker,
                             struct qbh_block_w4f16_pool *hvx_pool,
                             struct qbh_hvx_check_metrics *softmax_check) {
    const __fp16 *q = (const __fp16 *)buffers->q;
    const __fp16 *k = (const __fp16 *)buffers->k;
    const __fp16 *v = (const __fp16 *)buffers->v;
    __fp16 *scores = (__fp16 *)buffers->scores;
    __fp16 *probability = (__fp16 *)buffers->probability;
    __fp16 *attention = (__fp16 *)buffers->attention_concat;
    uint64_t attribution_cursor = 0U;
    uint32_t packed_qk_kv_head = UINT32_MAX;
    uint32_t packed_av_kv_head = UINT32_MAX;
    if (header->attribution_enabled != 0U) {
        attribution_cursor = HAP_perf_get_qtimer_count();
    }
    qbh_hmx_fp16_init_unity_scale(buffers->scale_or_bias);
    if (header->attribution_enabled != 0U) {
        header->attention_setup_ticks +=
            qbh_attribution_mark(&attribution_cursor);
    }

    if (qbh_attention_gqa_enabled(
            header->attention_pipeline_mode)) {
        uint64_t pipeline_start = HAP_perf_get_qtimer_count();
        if (qbh_hvx_pool_gqa_attention(
                header, hvx_pool, buffers, worker) != 0) {
            return -1;
        }
        qbh_record_f16_nonfinite(
            header, q, QBH_BLOCK_M * QBH_BLOCK_HIDDEN,
            QBH_BLOCK_NUMERICAL_Q_ROPE);
        qbh_record_f16_nonfinite(
            header, k, QBH_BLOCK_M * QBH_BLOCK_KV_HIDDEN,
            QBH_BLOCK_NUMERICAL_K_ROPE);
        qbh_record_f16_nonfinite(
            header, probability, QBH_BLOCK_SCORE_ELEMENTS,
            QBH_BLOCK_NUMERICAL_ATTENTION_SOFTMAX);
        header->attention_probability_max_abs = qbh_f16_max_abs(
            header, probability, QBH_BLOCK_SCORE_ELEMENTS);
        if ((header->crouton_boundary_mode &
             QBH_BLOCK_CROUTON_BOUNDARY_AV_TO_O) != 0U) {
            attention = (__fp16 *)buffers->hmx_activation;
        }
        qbh_record_f16_nonfinite(
            header, attention, QBH_BLOCK_M * QBH_BLOCK_HIDDEN,
            QBH_BLOCK_NUMERICAL_ATTENTION_AV);
        header->attention_av_max_abs = qbh_f16_max_abs(
            header, attention, QBH_BLOCK_M * QBH_BLOCK_HIDDEN);
        header->attention_gqa_pipeline_ticks +=
            HAP_perf_get_qtimer_count() - pipeline_start;
        return 0;
    }

    for (uint32_t head = 0; head < QBH_BLOCK_HEADS; ++head) {
        uint32_t kv_head = head / (QBH_BLOCK_HEADS / QBH_BLOCK_KV_HEADS);
        qbh_pack_fp16_activation(
            q + head * QBH_BLOCK_HEAD_DIM, QBH_BLOCK_HIDDEN,
            QBH_BLOCK_HEAD_DIM, (__fp16 *)buffers->hmx_activation);
        if ((header->attention_pack_mode &
             QBH_BLOCK_ATTENTION_PACK_QK_HVX) != 0U) {
            if (packed_qk_kv_head != kv_head) {
                qbh_pack_fp16_weight_rows_hvx(
                    k, QBH_BLOCK_KV_HIDDEN,
                    kv_head * QBH_BLOCK_HEAD_DIM,
                    QBH_BLOCK_HEAD_DIM, QBH_BLOCK_M,
                    (__fp16 *)buffers->expanded_weight);
                packed_qk_kv_head = kv_head;
            }
        } else {
            qbh_pack_fp16_weight_rows_scalar(
                k, QBH_BLOCK_KV_HIDDEN,
                kv_head * QBH_BLOCK_HEAD_DIM,
                QBH_BLOCK_HEAD_DIM, QBH_BLOCK_M,
                (__fp16 *)buffers->expanded_weight);
        }
        if (header->attribution_enabled != 0U) {
            header->attention_qk_pack_ticks +=
                qbh_attribution_mark(&attribution_cursor);
        }
        if (qbh_hmx_submit(worker, QBH_BLOCK_HMX_FP16,
                           buffers->hmx_activation,
                           buffers->expanded_weight,
                           buffers->scale_or_bias,
                           buffers->hmx_output, 2U, 4U, 2U) != 0) {
            return -1;
        }
        if (header->attribution_enabled != 0U) {
            header->attention_qk_hmx_ticks +=
                qbh_attribution_mark(&attribution_cursor);
        }
        ++header->hmx_command_count;
        header->hmx_fp16_tile_pair_count += 16U;
        qbh_unpack_fp16_output(
            (const __fp16 *)buffers->hmx_output, 2U,
            scores + (size_t)head * QBH_BLOCK_M * QBH_BLOCK_M,
            QBH_BLOCK_M, 0U);
        if (header->attribution_enabled != 0U) {
            header->attention_qk_unpack_ticks +=
                qbh_attribution_mark(&attribution_cursor);
        }
    }
    qbh_record_f16_nonfinite(
        header, scores, QBH_BLOCK_SCORE_ELEMENTS,
        QBH_BLOCK_NUMERICAL_ATTENTION_QK);
    header->attention_qk_max_abs = qbh_f16_max_abs(
        header, scores, QBH_BLOCK_SCORE_ELEMENTS);
    if (header->attribution_enabled != 0U &&
        header->numerical_audit_enabled != 0U) {
        header->attention_qk_audit_ticks +=
            qbh_attribution_mark(&attribution_cursor);
    }

    if (qbh_attention_parallel_softmax_enabled(
            header->attention_pipeline_mode) &&
        softmax_check == NULL) {
        if (qbh_hvx_pool_softmax(
                header, hvx_pool, scores, probability) != 0) {
            return -1;
        }
    } else if ((header->common_ops_mask &
                QBH_BLOCK_COMMON_OP_SOFTMAX) != 0U) {
        qbh_hvx_stable_causal_softmax_f16(
            scores, probability, QBH_BLOCK_HEADS, QBH_BLOCK_M,
            QBH_BLOCK_M, 0.08838834764831845f, softmax_check);
    } else {
      for (uint32_t head = 0; head < QBH_BLOCK_HEADS; ++head) {
        for (uint32_t row = 0; row < QBH_BLOCK_M; ++row) {
            __fp16 *score_row = scores +
                ((size_t)head * QBH_BLOCK_M + row) * QBH_BLOCK_M;
            __fp16 *probability_row = probability +
                ((size_t)head * QBH_BLOCK_M + row) * QBH_BLOCK_M;
            float maximum = -INFINITY;
            float sum = 0.0f;
            for (uint32_t column = 0; column < QBH_BLOCK_M; ++column) {
                float value = column <= row
                                  ? (float)score_row[column] *
                                        0.08838834764831845f
                                  : -INFINITY;
                score_row[column] = column <= row
                                        ? (__fp16)value
                                        : (__fp16)-65504.0f;
                if (value > maximum) {
                    maximum = value;
                }
            }
            for (uint32_t column = 0; column <= row; ++column) {
                float value = expf((float)score_row[column] - maximum);
                probability_row[column] = (__fp16)value;
                sum += value;
            }
            for (uint32_t column = 0; column < QBH_BLOCK_M; ++column) {
                probability_row[column] = column <= row
                    ? (__fp16)((float)probability_row[column] / sum)
                    : (__fp16)0.0f;
            }
        }
      }
    }
    if (header->attribution_enabled != 0U) {
        header->attention_softmax_ticks +=
            qbh_attribution_mark(&attribution_cursor);
    }
    qbh_record_f16_nonfinite(
        header, probability, QBH_BLOCK_SCORE_ELEMENTS,
        QBH_BLOCK_NUMERICAL_ATTENTION_SOFTMAX);
    header->attention_probability_max_abs = qbh_f16_max_abs(
        header, probability, QBH_BLOCK_SCORE_ELEMENTS);
    if (header->attribution_enabled != 0U &&
        header->numerical_audit_enabled != 0U) {
        header->attention_softmax_audit_ticks +=
            qbh_attribution_mark(&attribution_cursor);
    }

    for (uint32_t head = 0; head < QBH_BLOCK_HEADS; ++head) {
        uint32_t kv_head = head / (QBH_BLOCK_HEADS / QBH_BLOCK_KV_HEADS);
        qbh_pack_fp16_activation(
            probability + (size_t)head * QBH_BLOCK_M * QBH_BLOCK_M,
            QBH_BLOCK_M, QBH_BLOCK_M,
            (__fp16 *)buffers->hmx_activation);
        if ((header->attention_pack_mode &
             QBH_BLOCK_ATTENTION_PACK_AV_HVX) != 0U) {
            if (packed_av_kv_head != kv_head) {
                qbh_pack_fp16_weight_transposed_hvx(
                    v, QBH_BLOCK_KV_HIDDEN,
                    kv_head * QBH_BLOCK_HEAD_DIM, QBH_BLOCK_M,
                    QBH_BLOCK_HEAD_DIM,
                    (__fp16 *)buffers->expanded_weight);
                packed_av_kv_head = kv_head;
            }
        } else {
            qbh_pack_fp16_weight_transposed_scalar(
                v, QBH_BLOCK_KV_HIDDEN,
                kv_head * QBH_BLOCK_HEAD_DIM, QBH_BLOCK_M,
                QBH_BLOCK_HEAD_DIM,
                (__fp16 *)buffers->expanded_weight);
        }
        if (header->attribution_enabled != 0U) {
            header->attention_av_pack_ticks +=
                qbh_attribution_mark(&attribution_cursor);
        }
        if (qbh_hmx_submit(worker, QBH_BLOCK_HMX_FP16,
                           buffers->hmx_activation,
                           buffers->expanded_weight,
                           buffers->scale_or_bias,
                           buffers->hmx_output, 2U, 2U, 4U) != 0) {
            return -1;
        }
        if (header->attribution_enabled != 0U) {
            header->attention_av_hmx_ticks +=
                qbh_attribution_mark(&attribution_cursor);
        }
        ++header->hmx_command_count;
        header->hmx_fp16_tile_pair_count += 16U;
        qbh_unpack_fp16_output(
            (const __fp16 *)buffers->hmx_output, 4U, attention,
            QBH_BLOCK_HIDDEN, head * QBH_BLOCK_HEAD_DIM);
        if (header->attribution_enabled != 0U) {
            header->attention_av_unpack_ticks +=
                qbh_attribution_mark(&attribution_cursor);
        }
    }
    qbh_record_f16_nonfinite(
        header, attention, QBH_BLOCK_M * QBH_BLOCK_HIDDEN,
        QBH_BLOCK_NUMERICAL_ATTENTION_AV);
    header->attention_av_max_abs = qbh_f16_max_abs(
        header, attention, QBH_BLOCK_M * QBH_BLOCK_HIDDEN);
    if (header->attribution_enabled != 0U &&
        header->numerical_audit_enabled != 0U) {
        header->attention_av_audit_ticks +=
            qbh_attribution_mark(&attribution_cursor);
    }
    return 0;
}

static void qbh_quantize_f16_buffer(
    const __fp16 *input, uint8_t *output, uint32_t elements,
    const struct qbh_block_qparam *qparam) {
    for (uint32_t index = 0; index < elements; ++index) {
        output[index] = qbh_quantize((float)input[index], qparam);
    }
}

static int qbh_block_attention_config_valid(
    const struct qbh_block_header *header,
    const struct qbh_attention_config *config,
    uint32_t group) {
    return config != NULL &&
           config->abi_version == QBH_ATTENTION_ABI_VERSION &&
           config->group_index == group &&
           (config->fraction_bits == 3U ||
            config->fraction_bits == 4U) &&
           config->division_mode >= QBH_ATTENTION_DIVISION_EXACT &&
           config->division_mode <= QBH_ATTENTION_DIVISION_ENDPOINT &&
           config->q_zero_point ==
               header->qparams[QBH_BLOCK_QP_Q_ROPE].zero_point &&
           config->k_zero_point ==
               header->qparams[QBH_BLOCK_QP_K_ROPE].zero_point &&
           config->v_zero_point ==
               header->qparams[QBH_BLOCK_QP_V].zero_point &&
           config->probability_zero_point == 0 &&
           config->output_zero_point ==
               header->qparams[
                   QBH_BLOCK_QP_ATTENTION_CONCAT].zero_point &&
           config->v_recenter_numerator > 0U &&
           config->v_recenter_denominator > 0U &&
           config->score_shift <= QBH_ATTENTION_MAX_SHIFT &&
           config->score_multiplier > 0U &&
           config->score_multiplier <= QBH_ATTENTION_MAX_MULTIPLIER &&
           config->av_shift <= QBH_ATTENTION_MAX_SHIFT &&
           config->av_multiplier > 0U &&
           config->av_multiplier <= QBH_ATTENTION_MAX_MULTIPLIER;
}

static int qbh_attention_u8_integer(
    struct qbh_block_header *header,
    struct qbh_block_buffers *buffers,
    struct qbh_block_hmx_worker *worker) {
    uint8_t *scratch = buffers->attention_concat;
    int8_t *k_weight = (int8_t *)(
        scratch + QBH_ATTN_U8_K_WEIGHT_OFFSET);
    int8_t *v_weight = (int8_t *)(
        scratch + QBH_ATTN_U8_V_WEIGHT_OFFSET);
    uint32_t *qk_bias = (uint32_t *)(
        scratch + QBH_ATTN_U8_QK_BIAS_OFFSET);
    uint32_t *av_bias = (uint32_t *)(
        scratch + QBH_ATTN_U8_AV_BIAS_OFFSET);
    uint8_t *softmax_scratch =
        scratch + QBH_ATTN_U8_SOFTMAX_SCRATCH_OFFSET;

    if (!qbh_attention_u8_enabled(
            header->attention_pipeline_mode) ||
        header->variant != QBH_BLOCK_W4U8 ||
        (((uintptr_t)buffers->q |
          (uintptr_t)buffers->k |
          (uintptr_t)buffers->v |
          (uintptr_t)buffers->scores |
          (uintptr_t)buffers->probability |
          (uintptr_t)buffers->hmx_activation) &
         (QBH_HMX_OUTPUT_BYTES - 1U)) != 0U ||
        ((uintptr_t)k_weight & 255U) != 0U ||
        ((uintptr_t)v_weight & 255U) != 0U ||
        ((uintptr_t)qk_bias & 255U) != 0U ||
        ((uintptr_t)av_bias & 255U) != 0U ||
        ((uintptr_t)softmax_scratch & 127U) != 0U) {
        return -1;
    }

    header->u8_attention_probability_row_sum_min = UINT32_MAX;
    for (uint32_t group = 0U;
         group < QBH_BLOCK_KV_HEADS; ++group) {
        const struct qbh_attention_config *config =
            &buffers->attention_configs[group];
        const uint32_t first_q_head =
            group * QBH_ATTENTION_Q_HEADS_PER_GROUP;
        uint8_t *q_group = buffers->q +
            (size_t)first_q_head *
                QBH_ATTENTION_HEAD_DIM_TILES *
                QBH_HMX_ACTIVATION_BYTES;
        uint8_t *k_head = buffers->k +
            (size_t)group * QBH_ATTENTION_HEAD_DIM_TILES *
                QBH_HMX_ACTIVATION_BYTES;
        uint8_t *v_head = buffers->v +
            (size_t)group * QBH_ATTENTION_HEAD_DIM_TILES *
                QBH_HMX_ACTIVATION_BYTES;
        uint8_t *score_group = buffers->scores +
            (size_t)first_q_head * QBH_ATTENTION_SCORE_TILES *
                QBH_HMX_OUTPUT_BYTES;
        uint8_t *probability_group = buffers->probability +
            (size_t)first_q_head * QBH_ATTENTION_SCORE_TILES *
                QBH_HMX_ACTIVATION_BYTES;
        uint8_t *output_group = buffers->hmx_activation +
            (size_t)first_q_head *
                QBH_ATTENTION_HEAD_DIM_TILES *
                QBH_HMX_OUTPUT_BYTES;
        struct qbh_attention_u8_telemetry telemetry;
        uint64_t start;

        if (!qbh_block_attention_config_valid(
                header, config, group)) {
            return -1;
        }
        memset(&telemetry, 0, sizeof(telemetry));

        start = HAP_perf_get_qtimer_count();
        for (uint32_t local_head = 0U;
             local_head < QBH_ATTENTION_Q_HEADS_PER_GROUP;
             ++local_head) {
            qbh_hvx_qk_norm_rope_u8_native_head(
                q_group +
                    (size_t)local_head *
                        QBH_ATTENTION_HEAD_DIM_TILES *
                        QBH_HMX_ACTIVATION_BYTES,
                &header->qparams[QBH_BLOCK_QP_Q_PROJECTION],
                &header->qparams[QBH_BLOCK_QP_Q_ROPE],
                (const __fp16 *)buffers->q_norm_weight,
                (const __fp16 *)buffers->rope_cos,
                (const __fp16 *)buffers->rope_sin);
        }
        qbh_hvx_qk_norm_rope_u8_native_head(
            k_head,
            &header->qparams[QBH_BLOCK_QP_K_PROJECTION],
            &header->qparams[QBH_BLOCK_QP_K_ROPE],
            (const __fp16 *)buffers->k_norm_weight,
            (const __fp16 *)buffers->rope_cos,
            (const __fp16 *)buffers->rope_sin);
        header->u8_attention_qk_norm_rope_ticks +=
            HAP_perf_get_qtimer_count() - start;

        start = HAP_perf_get_qtimer_count();
        qbh_attention_u8_pack_k_native(
            k_head, config, k_weight, qk_bias);
        header->u8_attention_k_pack_ticks +=
            HAP_perf_get_qtimer_count() - start;

        start = HAP_perf_get_qtimer_count();
        if (qbh_attention_u8_vdeal_enabled(
                header->attention_pipeline_mode)) {
            qbh_attention_u8_pack_v_native_vgather_vdeal(
                v_head, config, v_weight, av_bias, scratch,
                header->numerical_audit_enabled != 0U
                    ? &telemetry.v_recenter_saturation_count
                    : NULL);
        } else if (qbh_attention_u8_vgather_enabled(
                header->attention_pipeline_mode)) {
            qbh_attention_u8_pack_v_native_vgather(
                v_head, config, v_weight, av_bias, scratch,
                header->numerical_audit_enabled != 0U
                    ? &telemetry.v_recenter_saturation_count
                    : NULL);
        } else {
            qbh_attention_u8_pack_v_native(
                v_head, config, v_weight, av_bias,
                header->numerical_audit_enabled != 0U
                    ? &telemetry.v_recenter_saturation_count
                    : NULL);
        }
        header->u8_attention_v_pack_ticks +=
            HAP_perf_get_qtimer_count() - start;

        start = HAP_perf_get_qtimer_count();
        for (uint32_t local_head = 0U;
             local_head < QBH_ATTENTION_Q_HEADS_PER_GROUP;
             ++local_head) {
            const uint8_t *q_head = q_group +
                (size_t)local_head *
                    QBH_ATTENTION_HEAD_DIM_TILES *
                    QBH_HMX_ACTIVATION_BYTES;
            if (qbh_attention_u8_hmx_batch_enabled(
                    header->attention_pipeline_mode)) {
                if (qbh_hmx_submit(
                        worker, QBH_BLOCK_HMX_U8S8,
                        q_head, k_weight, qk_bias,
                        score_group +
                            (size_t)local_head *
                                QBH_ATTENTION_SCORE_TILES *
                                QBH_HMX_OUTPUT_BYTES,
                        1U, QBH_ATTENTION_HEAD_DIM_TILES,
                        QBH_ATTENTION_SCORE_TILES) != 0) {
                    return -1;
                }
                ++header->hmx_command_count;
                header->hmx_u8s8_tile_pair_count +=
                    QBH_ATTENTION_HEAD_DIM_TILES *
                    QBH_ATTENTION_SCORE_TILES;
                header->u8_attention_qk_execution_count +=
                    QBH_ATTENTION_SCORE_TILES;
            } else {
                for (uint32_t n_tile = 0U;
                     n_tile < QBH_ATTENTION_SCORE_TILES; ++n_tile) {
                    if (qbh_hmx_submit(
                            worker, QBH_BLOCK_HMX_U8S8,
                            q_head,
                            k_weight +
                                (size_t)n_tile *
                                    QBH_ATTENTION_HEAD_DIM_TILES *
                                    QBH_HMX_WEIGHT_BYTES,
                            qk_bias +
                                (size_t)n_tile *
                                    (QBH_HMX_BIAS_BYTES /
                                     sizeof(uint32_t)),
                            score_group +
                                ((size_t)local_head *
                                     QBH_ATTENTION_SCORE_TILES +
                                 n_tile) * QBH_HMX_OUTPUT_BYTES,
                            1U, QBH_ATTENTION_HEAD_DIM_TILES,
                            1U) != 0) {
                        return -1;
                    }
                    ++header->hmx_command_count;
                    header->hmx_u8s8_tile_pair_count +=
                        QBH_ATTENTION_HEAD_DIM_TILES;
                    ++header->u8_attention_qk_execution_count;
                }
            }
        }
        header->u8_attention_qk_hmx_ticks +=
            HAP_perf_get_qtimer_count() - start;

        if (!qbh_attention_u8_fused_qk_requant_enabled(
                header->attention_pipeline_mode)) {
            start = HAP_perf_get_qtimer_count();
            qbh_attention_u8_requant_qk(
                score_group, config,
                header->numerical_audit_enabled != 0U
                    ? &telemetry.score_saturation_count
                    : NULL);
            header->u8_attention_qk_requant_ticks +=
                HAP_perf_get_qtimer_count() - start;
        }

        start = HAP_perf_get_qtimer_count();
        if (qbh_attention_u8_fused_qk_requant_enabled(
                header->attention_pipeline_mode)) {
            if (qbh_attention_u8_lut_templates_enabled(
                    header->attention_pipeline_mode)) {
                qbh_attention_u8_requant_softmax_group_lut_templates(
                    score_group, probability_group,
                    softmax_scratch, config,
                    header->numerical_audit_enabled != 0U
                        ? &telemetry
                        : NULL);
            } else {
                qbh_attention_u8_requant_softmax_group(
                    score_group, probability_group,
                    softmax_scratch, config,
                    header->numerical_audit_enabled != 0U
                        ? &telemetry
                        : NULL);
            }
        } else {
            qbh_attention_u8_softmax_group(
                score_group, probability_group,
                softmax_scratch, config,
                header->numerical_audit_enabled != 0U
                    ? &telemetry
                    : NULL);
        }
        header->u8_attention_softmax_ticks +=
            HAP_perf_get_qtimer_count() - start;

        start = HAP_perf_get_qtimer_count();
        for (uint32_t local_head = 0U;
             local_head < QBH_ATTENTION_Q_HEADS_PER_GROUP;
             ++local_head) {
            const uint8_t *probability = probability_group +
                (size_t)local_head *
                    QBH_ATTENTION_SCORE_TILES *
                    QBH_HMX_ACTIVATION_BYTES;
            if (qbh_attention_u8_hmx_batch_enabled(
                    header->attention_pipeline_mode)) {
                if (qbh_hmx_submit(
                        worker, QBH_BLOCK_HMX_U8S8,
                        probability, v_weight, av_bias,
                        output_group +
                            (size_t)local_head *
                                QBH_ATTENTION_HEAD_DIM_TILES *
                                QBH_HMX_OUTPUT_BYTES,
                        1U, QBH_ATTENTION_SCORE_TILES,
                        QBH_ATTENTION_HEAD_DIM_TILES) != 0) {
                    return -1;
                }
                ++header->hmx_command_count;
                header->hmx_u8s8_tile_pair_count +=
                    QBH_ATTENTION_SCORE_TILES *
                    QBH_ATTENTION_HEAD_DIM_TILES;
                header->u8_attention_av_execution_count +=
                    QBH_ATTENTION_HEAD_DIM_TILES;
            } else {
                for (uint32_t n_tile = 0U;
                     n_tile < QBH_ATTENTION_HEAD_DIM_TILES; ++n_tile) {
                    if (qbh_hmx_submit(
                            worker, QBH_BLOCK_HMX_U8S8,
                            probability,
                            v_weight +
                                (size_t)n_tile *
                                    QBH_ATTENTION_SCORE_TILES *
                                    QBH_HMX_WEIGHT_BYTES,
                            av_bias +
                                (size_t)n_tile *
                                    (QBH_HMX_BIAS_BYTES /
                                     sizeof(uint32_t)),
                            output_group +
                                ((size_t)local_head *
                                     QBH_ATTENTION_HEAD_DIM_TILES +
                                 n_tile) * QBH_HMX_OUTPUT_BYTES,
                            1U, QBH_ATTENTION_SCORE_TILES,
                            1U) != 0) {
                        return -1;
                    }
                    ++header->hmx_command_count;
                    header->hmx_u8s8_tile_pair_count +=
                        QBH_ATTENTION_SCORE_TILES;
                    ++header->u8_attention_av_execution_count;
                }
            }
        }
        header->u8_attention_av_hmx_ticks +=
            HAP_perf_get_qtimer_count() - start;

        start = HAP_perf_get_qtimer_count();
        qbh_attention_u8_requant_av(
            output_group, config);
        header->u8_attention_av_requant_ticks +=
            HAP_perf_get_qtimer_count() - start;

        header->u8_attention_score_saturation_count +=
            telemetry.score_saturation_count;
        header->u8_attention_v_recenter_saturation_count +=
            telemetry.v_recenter_saturation_count;
        header->u8_attention_probability_mask_violation_count +=
            telemetry.probability_mask_violation_count;
        if (telemetry.probability_row_sum_min <
            header->u8_attention_probability_row_sum_min) {
            header->u8_attention_probability_row_sum_min =
                telemetry.probability_row_sum_min;
        }
        if (telemetry.probability_row_sum_max >
            header->u8_attention_probability_row_sum_max) {
            header->u8_attention_probability_row_sum_max =
                telemetry.probability_row_sum_max;
        }
        ++header->u8_attention_group_count;
        header->u8_attention_direct_o_tile_count +=
            QBH_ATTENTION_Q_HEADS_PER_GROUP *
            QBH_ATTENTION_HEAD_DIM_TILES;
    }
    return 0;
}

static int qbh_attention_u8_qk_prep_wait_ready(
    struct qbh_block_w4f16_pool *pool, uint32_t task) {
    while (pool->attention_qk_ready[task] !=
           pool->attention_qk_generation) {
        if (pool->attention_qk_stream_abort != 0U) {
            return -1;
        }
        asm volatile("pause(#8)" : : : "memory");
    }
    asm volatile("barrier" ::: "memory");
    return 0;
}

static void qbh_attention_u8_qk_prep_pool_run_group_tasks(
    struct qbh_block_w4f16_pool *pool,
    struct qbh_block_w4f16_job *job) {
    struct qbh_block_header *header = pool->attention_header;
    struct qbh_block_buffers *buffers = pool->attention_buffers;

    for (;;) {
        const uint32_t group =
            qbh_atomic_fetch_increment(&pool->next_attention_task);
        const struct qbh_attention_config *config;
        const uint32_t first_q_head =
            group * QBH_ATTENTION_Q_HEADS_PER_GROUP;
        uint8_t *q_group;
        uint8_t *k_head;
        uint8_t *scratch;
        int8_t *k_weight;
        uint32_t *qk_bias;
        uint64_t start;

        if (group >= pool->attention_task_count ||
            pool->attention_qk_stream_abort != 0U) {
            break;
        }
        config = &buffers->attention_configs[group];
        if (!qbh_block_attention_config_valid(header, config, group)) {
            pool->attention_qk_stream_abort = 1U;
            asm volatile("barrier" ::: "memory");
            break;
        }
        q_group = buffers->q +
            (size_t)first_q_head *
                QBH_ATTENTION_HEAD_DIM_TILES *
                QBH_HMX_ACTIVATION_BYTES;
        k_head = buffers->k +
            (size_t)group * QBH_ATTENTION_HEAD_DIM_TILES *
                QBH_HMX_ACTIVATION_BYTES;
        scratch = buffers->attention_concat +
            (size_t)group * QBH_ATTN_U8_GROUP_SCRATCH_BYTES;
        k_weight = (int8_t *)(
            scratch + QBH_ATTN_U8_K_WEIGHT_OFFSET);
        qk_bias = (uint32_t *)(
            scratch + QBH_ATTN_U8_QK_BIAS_OFFSET);

        for (uint32_t local_head = 0U;
             local_head < QBH_ATTENTION_Q_HEADS_PER_GROUP;
             ++local_head) {
            if (qbh_attention_u8_qk_prep_wait_ready(
                    pool, first_q_head + local_head) != 0) {
                return;
            }
            start = HAP_perf_get_qtimer_count();
            qbh_hvx_qk_norm_rope_u8_native_head(
                q_group +
                    (size_t)local_head *
                        QBH_ATTENTION_HEAD_DIM_TILES *
                        QBH_HMX_ACTIVATION_BYTES,
                &header->qparams[QBH_BLOCK_QP_Q_PROJECTION],
                &header->qparams[QBH_BLOCK_QP_Q_ROPE],
                (const __fp16 *)buffers->q_norm_weight,
                (const __fp16 *)buffers->rope_cos,
                (const __fp16 *)buffers->rope_sin);
            job->u8_attention_qk_norm_rope_ticks +=
                HAP_perf_get_qtimer_count() - start;
            ++job->attention_qk_norm_task_count;
        }

        if (qbh_attention_u8_qk_prep_wait_ready(
                pool, QBH_BLOCK_HEADS + group) != 0) {
            return;
        }
        start = HAP_perf_get_qtimer_count();
        qbh_hvx_qk_norm_rope_u8_native_k_head(
            k_head,
            &header->qparams[QBH_BLOCK_QP_K_PROJECTION],
            &header->qparams[QBH_BLOCK_QP_K_ROPE],
            (const __fp16 *)buffers->k_norm_weight,
            (const __fp16 *)buffers->rope_cos,
            (const __fp16 *)buffers->rope_sin,
            config, k_weight, qk_bias);
        job->u8_attention_qk_norm_rope_ticks +=
            HAP_perf_get_qtimer_count() - start;
        ++job->attention_qk_norm_task_count;

        if (header->numerical_audit_enabled != 0U) {
            const uint32_t audit_slice_bytes =
                QBH_ATTN_U8_K_WEIGHT_BYTES +
                QBH_ATTN_U8_QK_BIAS_BYTES;
            uint8_t *reference = buffers->attention_projection +
                (size_t)job->worker_index * audit_slice_bytes;
            int8_t *reference_weight = (int8_t *)reference;
            uint32_t *reference_bias = (uint32_t *)(
                reference + QBH_ATTN_U8_K_WEIGHT_BYTES);

            qbh_attention_u8_pack_k_native(
                k_head, config, reference_weight, reference_bias);
            job->u8_attention_fused_k_operand_mismatch_count +=
                qbh_count_u8_mismatches(
                    (const uint8_t *)k_weight,
                    (const uint8_t *)reference_weight,
                    QBH_ATTN_U8_K_WEIGHT_BYTES) +
                qbh_count_u8_mismatches(
                    (const uint8_t *)qk_bias,
                    (const uint8_t *)reference_bias,
                    QBH_ATTN_U8_QK_BIAS_BYTES);
        }
        ++job->u8_attention_prepared_group_count;
    }
}

static void qbh_attention_u8_qk_prep_pool_run_head_tasks(
    struct qbh_block_w4f16_pool *pool,
    struct qbh_block_w4f16_job *job) {
    struct qbh_block_header *header = pool->attention_header;
    struct qbh_block_buffers *buffers = pool->attention_buffers;
    const uint32_t task_count = QBH_BLOCK_HEADS + QBH_BLOCK_KV_HEADS;

    for (;;) {
        const uint32_t task =
            qbh_atomic_fetch_increment(&pool->next_attention_task);
        uint64_t start;

        if (task >= task_count ||
            pool->attention_qk_stream_abort != 0U) {
            break;
        }
        if (qbh_attention_u8_qk_prep_wait_ready(pool, task) != 0) {
            return;
        }

        if (task < QBH_BLOCK_HEADS) {
            uint8_t *q_head = buffers->q +
                (size_t)task * QBH_ATTENTION_HEAD_DIM_TILES *
                    QBH_HMX_ACTIVATION_BYTES;
            start = HAP_perf_get_qtimer_count();
            qbh_hvx_qk_norm_rope_u8_native_head(
                q_head,
                &header->qparams[QBH_BLOCK_QP_Q_PROJECTION],
                &header->qparams[QBH_BLOCK_QP_Q_ROPE],
                (const __fp16 *)buffers->q_norm_weight,
                (const __fp16 *)buffers->rope_cos,
                (const __fp16 *)buffers->rope_sin);
            job->u8_attention_qk_norm_rope_ticks +=
                HAP_perf_get_qtimer_count() - start;
            ++job->attention_qk_norm_task_count;
            continue;
        }

        {
            const uint32_t group = task - QBH_BLOCK_HEADS;
            const struct qbh_attention_config *config =
                &buffers->attention_configs[group];
            uint8_t *k_head = buffers->k +
                (size_t)group * QBH_ATTENTION_HEAD_DIM_TILES *
                    QBH_HMX_ACTIVATION_BYTES;
            uint8_t *scratch = buffers->attention_concat +
                (size_t)group * QBH_ATTN_U8_GROUP_SCRATCH_BYTES;
            int8_t *k_weight = (int8_t *)(
                scratch + QBH_ATTN_U8_K_WEIGHT_OFFSET);
            uint32_t *qk_bias = (uint32_t *)(
                scratch + QBH_ATTN_U8_QK_BIAS_OFFSET);

            if (!qbh_block_attention_config_valid(
                    header, config, group)) {
                pool->attention_qk_stream_abort = 1U;
                asm volatile("barrier" ::: "memory");
                return;
            }
            start = HAP_perf_get_qtimer_count();
            qbh_hvx_qk_norm_rope_u8_native_k_head(
                k_head,
                &header->qparams[QBH_BLOCK_QP_K_PROJECTION],
                &header->qparams[QBH_BLOCK_QP_K_ROPE],
                (const __fp16 *)buffers->k_norm_weight,
                (const __fp16 *)buffers->rope_cos,
                (const __fp16 *)buffers->rope_sin,
                config, k_weight, qk_bias);
            job->u8_attention_qk_norm_rope_ticks +=
                HAP_perf_get_qtimer_count() - start;
            ++job->attention_qk_norm_task_count;

            if (header->numerical_audit_enabled != 0U) {
                const uint32_t audit_slice_bytes =
                    QBH_ATTN_U8_K_WEIGHT_BYTES +
                    QBH_ATTN_U8_QK_BIAS_BYTES;
                uint8_t *reference = buffers->attention_projection +
                    (size_t)job->worker_index * audit_slice_bytes;
                int8_t *reference_weight = (int8_t *)reference;
                uint32_t *reference_bias = (uint32_t *)(
                    reference + QBH_ATTN_U8_K_WEIGHT_BYTES);

                qbh_attention_u8_pack_k_native(
                    k_head, config, reference_weight, reference_bias);
                job->u8_attention_fused_k_operand_mismatch_count +=
                    qbh_count_u8_mismatches(
                        (const uint8_t *)k_weight,
                        (const uint8_t *)reference_weight,
                        QBH_ATTN_U8_K_WEIGHT_BYTES) +
                    qbh_count_u8_mismatches(
                        (const uint8_t *)qk_bias,
                        (const uint8_t *)reference_bias,
                        QBH_ATTN_U8_QK_BIAS_BYTES);
            }
            ++job->u8_attention_prepared_group_count;
        }
    }
}

static void qbh_attention_u8_qk_prep_pool_run_head_pair_tasks(
    struct qbh_block_w4f16_pool *pool,
    struct qbh_block_w4f16_job *job) {
    struct qbh_block_header *header = pool->attention_header;
    struct qbh_block_buffers *buffers = pool->attention_buffers;
    const uint32_t q_pair_count = QBH_BLOCK_HEADS / 2U;
    const uint32_t k_pair_count = QBH_BLOCK_KV_HEADS / 2U;
    const uint32_t task_count = q_pair_count + k_pair_count;

    for (;;) {
        const uint32_t task =
            qbh_atomic_fetch_increment(&pool->next_attention_task);
        uint64_t start;

        if (task >= task_count ||
            pool->attention_qk_stream_abort != 0U) {
            break;
        }

        if (task < q_pair_count) {
            const uint32_t first_head = task * 2U;
            uint8_t *first_q_head;
            uint8_t *second_q_head;

            if (qbh_attention_u8_qk_prep_wait_ready(
                    pool, first_head) != 0 ||
                qbh_attention_u8_qk_prep_wait_ready(
                    pool, first_head + 1U) != 0) {
                return;
            }
            first_q_head = buffers->q +
                (size_t)first_head *
                    QBH_ATTENTION_HEAD_DIM_TILES *
                    QBH_HMX_ACTIVATION_BYTES;
            second_q_head = first_q_head +
                QBH_ATTENTION_HEAD_DIM_TILES *
                    QBH_HMX_ACTIVATION_BYTES;
            start = HAP_perf_get_qtimer_count();
            qbh_hvx_qk_norm_rope_u8_native_head_pair(
                first_q_head, second_q_head,
                &header->qparams[QBH_BLOCK_QP_Q_PROJECTION],
                &header->qparams[QBH_BLOCK_QP_Q_ROPE],
                (const __fp16 *)buffers->q_norm_weight,
                (const __fp16 *)buffers->rope_cos,
                (const __fp16 *)buffers->rope_sin,
                buffers->attention_projection +
                    QBH_QK_PAIR_RSQRT_SCRATCH_OFFSET +
                    (size_t)job->worker_index *
                        QBH_QK_PAIR_RSQRT_SCRATCH_BYTES,
                buffers->attention_projection +
                    QBH_QK_ROPE_SF32_CACHE_OFFSET);
            if (header->w4u8_qk_pair_kernel_mode >=
                QBH_BLOCK_W4U8_QK_PAIR_QUARTER_TILED) {
                ++job->u8_qk_quarter_pair_count;
            }
            job->u8_attention_qk_norm_rope_ticks +=
                HAP_perf_get_qtimer_count() - start;
            job->attention_qk_norm_task_count += 2U;
            continue;
        }

        {
            const uint32_t first_group =
                (task - q_pair_count) * 2U;
            const struct qbh_attention_config *first_config =
                &buffers->attention_configs[first_group];
            const struct qbh_attention_config *second_config =
                first_config + 1U;
            uint8_t *first_k_head;
            uint8_t *second_k_head;
            uint8_t *first_scratch;
            uint8_t *second_scratch;
            int8_t *first_k_weight;
            int8_t *second_k_weight;
            uint32_t *first_qk_bias;
            uint32_t *second_qk_bias;

            if (qbh_attention_u8_qk_prep_wait_ready(
                    pool, QBH_BLOCK_HEADS + first_group) != 0 ||
                qbh_attention_u8_qk_prep_wait_ready(
                    pool, QBH_BLOCK_HEADS + first_group + 1U) != 0) {
                return;
            }
            if (!qbh_block_attention_config_valid(
                    header, first_config, first_group) ||
                !qbh_block_attention_config_valid(
                    header, second_config, first_group + 1U)) {
                pool->attention_qk_stream_abort = 1U;
                asm volatile("barrier" ::: "memory");
                return;
            }

            first_k_head = buffers->k +
                (size_t)first_group *
                    QBH_ATTENTION_HEAD_DIM_TILES *
                    QBH_HMX_ACTIVATION_BYTES;
            second_k_head = first_k_head +
                QBH_ATTENTION_HEAD_DIM_TILES *
                    QBH_HMX_ACTIVATION_BYTES;
            first_scratch = buffers->attention_concat +
                (size_t)first_group *
                    QBH_ATTN_U8_GROUP_SCRATCH_BYTES;
            second_scratch = first_scratch +
                QBH_ATTN_U8_GROUP_SCRATCH_BYTES;
            first_k_weight = (int8_t *)(
                first_scratch + QBH_ATTN_U8_K_WEIGHT_OFFSET);
            second_k_weight = (int8_t *)(
                second_scratch + QBH_ATTN_U8_K_WEIGHT_OFFSET);
            first_qk_bias = (uint32_t *)(
                first_scratch + QBH_ATTN_U8_QK_BIAS_OFFSET);
            second_qk_bias = (uint32_t *)(
                second_scratch + QBH_ATTN_U8_QK_BIAS_OFFSET);

            start = HAP_perf_get_qtimer_count();
            qbh_hvx_qk_norm_rope_u8_native_k_head_pair(
                first_k_head, second_k_head,
                &header->qparams[QBH_BLOCK_QP_K_PROJECTION],
                &header->qparams[QBH_BLOCK_QP_K_ROPE],
                (const __fp16 *)buffers->k_norm_weight,
                (const __fp16 *)buffers->rope_cos,
                (const __fp16 *)buffers->rope_sin,
                first_config, second_config,
                first_k_weight, second_k_weight,
                first_qk_bias, second_qk_bias,
                buffers->attention_projection +
                    QBH_QK_PAIR_RSQRT_SCRATCH_OFFSET +
                    (size_t)job->worker_index *
                        QBH_QK_PAIR_RSQRT_SCRATCH_BYTES,
                buffers->attention_projection +
                    QBH_QK_ROPE_SF32_CACHE_OFFSET);
            if (header->w4u8_qk_pair_kernel_mode >=
                QBH_BLOCK_W4U8_QK_PAIR_QUARTER_TILED) {
                ++job->u8_qk_quarter_pair_count;
            }
            job->u8_attention_qk_norm_rope_ticks +=
                HAP_perf_get_qtimer_count() - start;
            job->attention_qk_norm_task_count += 2U;

            if (header->numerical_audit_enabled != 0U) {
                const uint32_t audit_slice_bytes =
                    QBH_ATTN_U8_K_WEIGHT_BYTES +
                    QBH_ATTN_U8_QK_BIAS_BYTES;
                uint8_t *reference = buffers->attention_projection +
                    (size_t)job->worker_index * audit_slice_bytes;
                int8_t *reference_weight = (int8_t *)reference;
                uint32_t *reference_bias = (uint32_t *)(
                    reference + QBH_ATTN_U8_K_WEIGHT_BYTES);

                for (uint32_t pair = 0U; pair < 2U; ++pair) {
                    uint8_t *k_head = pair == 0U
                        ? first_k_head : second_k_head;
                    const struct qbh_attention_config *config =
                        pair == 0U ? first_config : second_config;
                    int8_t *k_weight = pair == 0U
                        ? first_k_weight : second_k_weight;
                    uint32_t *qk_bias = pair == 0U
                        ? first_qk_bias : second_qk_bias;

                    qbh_attention_u8_pack_k_native(
                        k_head, config,
                        reference_weight, reference_bias);
                    job->u8_attention_fused_k_operand_mismatch_count +=
                        qbh_count_u8_mismatches(
                            (const uint8_t *)k_weight,
                            (const uint8_t *)reference_weight,
                            QBH_ATTN_U8_K_WEIGHT_BYTES) +
                        qbh_count_u8_mismatches(
                            (const uint8_t *)qk_bias,
                            (const uint8_t *)reference_bias,
                            QBH_ATTN_U8_QK_BIAS_BYTES);
                }
            }
            job->u8_attention_prepared_group_count += 2U;
        }
    }
}

static void qbh_attention_u8_qk_prep_pool_run_tasks(
    struct qbh_block_w4f16_pool *pool,
    struct qbh_block_w4f16_job *job) {
    if (pool->attention_header->w4u8_qkvo_pipeline_mode ==
        QBH_BLOCK_W4U8_QKVO_BATCH4_QK_HEAD_PAIRS) {
        qbh_attention_u8_qk_prep_pool_run_head_pair_tasks(pool, job);
    } else if (pool->attention_header->w4u8_qkvo_pipeline_mode ==
               QBH_BLOCK_W4U8_QKVO_BATCH4_QK_HEAD_TASKS) {
        qbh_attention_u8_qk_prep_pool_run_head_tasks(pool, job);
    } else {
        qbh_attention_u8_qk_prep_pool_run_group_tasks(pool, job);
    }
}

static int qbh_hvx_pool_u8_qk_prep_start_async(
    struct qbh_block_header *header,
    struct qbh_block_w4f16_pool *pool,
    struct qbh_block_buffers *buffers) {
    uint32_t worker_count;

    if (header == NULL || pool == NULL || buffers == NULL ||
        !qbh_attention_u8_qkv_overlap_enabled(
            header->attention_pipeline_mode) ||
        header->attention_hvx_contexts < 4U ||
        header->attention_hvx_contexts >
            QBH_BLOCK_MAX_ATTENTION_HVX_CONTEXTS) {
        return -1;
    }
    worker_count = header->attention_hvx_contexts - 1U;
    if (worker_count == 0U || worker_count > pool->worker_count) {
        return -1;
    }
    pool->attention_header = header;
    pool->attention_buffers = buffers;
    if (header->w4u8_qkvo_pipeline_mode ==
        QBH_BLOCK_W4U8_QKVO_BATCH4_QK_HEAD_PAIRS) {
        pool->attention_task_count =
            QBH_BLOCK_HEADS / 2U + QBH_BLOCK_KV_HEADS / 2U;
    } else if (header->w4u8_qkvo_pipeline_mode ==
               QBH_BLOCK_W4U8_QKVO_BATCH4_QK_HEAD_TASKS) {
        pool->attention_task_count =
            QBH_BLOCK_HEADS + QBH_BLOCK_KV_HEADS;
    } else {
        pool->attention_task_count = QBH_BLOCK_KV_HEADS;
    }
    pool->next_attention_task = 0U;
    pool->attention_qk_stream_abort = 0U;
    pool->attention_qk_streaming = 1U;
    pool->active_worker_count = worker_count;
    ++pool->attention_qk_generation;
    if (pool->attention_qk_generation == 0U) {
        memset((void *)pool->attention_qk_ready, 0,
               sizeof(pool->attention_qk_ready));
        pool->attention_qk_generation = 1U;
    }
    for (uint32_t worker = 0U; worker < worker_count; ++worker) {
        struct qbh_block_w4f16_job *job = &pool->jobs[worker];
        job->u8_attention_prepared_group_count = 0U;
        job->u8_qk_quarter_pair_count = 0U;
        job->u8_attention_fused_k_operand_mismatch_count = 0U;
        job->u8_attention_qk_norm_rope_ticks = 0U;
        job->u8_attention_k_pack_ticks = 0U;
        job->attention_qk_norm_task_count = 0U;
        job->command_kind = QBH_BLOCK_HVX_POOL_U8_QK_PREP;
    }
    asm volatile("barrier" ::: "memory");
    for (uint32_t worker = 0U; worker < worker_count; ++worker) {
        (void)qurt_sem_up(&pool->command_ready[worker]);
    }
    return 0;
}

static void qbh_hvx_pool_u8_qk_prep_abort_async(
    struct qbh_block_w4f16_pool *pool) {
    if (pool != NULL) {
        pool->attention_qk_stream_abort = 1U;
        asm volatile("barrier" ::: "memory");
    }
}

static void qbh_hvx_pool_u8_qk_prep_publish(
    const struct qbh_block_header *header,
    const struct qbh_block_projection_desc *desc,
    struct qbh_block_w4f16_pool *pool,
    uint32_t first_n_tile, uint32_t n_tiles) {
    const uint32_t tiles_per_head =
        QBH_BLOCK_HEAD_DIM / QBH_HMX_OUTPUT_CHANNELS;
    uint32_t end_tile;
    uint32_t task;

    if (header == NULL ||
        !qbh_attention_u8_qkv_overlap_enabled(
            header->attention_pipeline_mode) ||
        pool == NULL ||
        (desc != &header->projections[QBH_BLOCK_PROJ_Q] &&
         desc != &header->projections[QBH_BLOCK_PROJ_K])) {
        return;
    }
    end_tile = first_n_tile + n_tiles;
    if (end_tile == 0U || end_tile % tiles_per_head != 0U) {
        return;
    }
    task = end_tile / tiles_per_head - 1U;
    if (desc == &header->projections[QBH_BLOCK_PROJ_K]) {
        task += QBH_BLOCK_HEADS;
    }
    if (task >= QBH_BLOCK_HEADS + QBH_BLOCK_KV_HEADS) {
        return;
    }
    pool->attention_qk_ready[task] =
        pool->attention_qk_generation;
    asm volatile("release(%0):at"
                 :
                 : "r"(&pool->attention_qk_ready[task])
                 : "memory");
}

static int qbh_hvx_pool_u8_qk_prep_wait_async(
    struct qbh_block_header *header,
    struct qbh_block_w4f16_pool *pool) {
    uint64_t wait_start;
    uint32_t completed_groups = 0U;

    if (header == NULL || pool == NULL ||
        pool->active_worker_count == 0U ||
        pool->active_worker_count > pool->worker_count) {
        return -1;
    }
    wait_start = HAP_perf_get_qtimer_count();
    for (uint32_t worker = 0U;
         worker < pool->active_worker_count; ++worker) {
        qurt_sem_down(&pool->command_done[worker]);
    }
    header->attention_qk_norm_pool_wait_ticks +=
        HAP_perf_get_qtimer_count() - wait_start;
    asm volatile("barrier" ::: "memory");
    for (uint32_t worker = 0U;
         worker < pool->active_worker_count; ++worker) {
        struct qbh_block_w4f16_job *job = &pool->jobs[worker];
        completed_groups += job->u8_attention_prepared_group_count;
        header->u8_attention_qk_norm_rope_ticks +=
            job->u8_attention_qk_norm_rope_ticks;
        header->w4u8_qk_quarter_pair_count +=
            job->u8_qk_quarter_pair_count;
        header->attention_qk_norm_task_count +=
            job->attention_qk_norm_task_count;
        job->attention_qk_norm_task_count = 0U;
        header->u8_attention_k_pack_ticks +=
            job->u8_attention_k_pack_ticks;
        header->u8_attention_fused_k_operand_mismatch_count +=
            job->u8_attention_fused_k_operand_mismatch_count;
    }
    pool->attention_qk_streaming = 0U;
    return pool->attention_qk_stream_abort == 0U &&
                   completed_groups == QBH_BLOCK_KV_HEADS
               ? 0
               : -1;
}

static int qbh_attention_u8_pool_submit(
    struct qbh_block_w4f16_pool *pool,
    struct qbh_block_w4f16_job *job,
    const uint8_t *activation, const int8_t *weight,
    const uint32_t *bias, uint8_t *output,
    uint32_t m_tiles, uint32_t k_tiles, uint32_t n_tiles,
    uint64_t *hmx_ticks) {
    const uint64_t queue_start = HAP_perf_get_qtimer_count();
    uint64_t hmx_start;
    int result;

    qurt_mutex_lock(&pool->attention_hmx_mutex);
    job->u8_attention_hmx_queue_wait_ticks +=
        HAP_perf_get_qtimer_count() - queue_start;
    if (pool->attention_gqa_abort != 0U) {
        qurt_mutex_unlock(&pool->attention_hmx_mutex);
        return -1;
    }
    hmx_start = HAP_perf_get_qtimer_count();
    result = qbh_hmx_submit(
        pool->attention_hmx_worker, QBH_BLOCK_HMX_U8S8,
        activation, weight, bias, output,
        m_tiles, k_tiles, n_tiles);
    *hmx_ticks += HAP_perf_get_qtimer_count() - hmx_start;
    qurt_mutex_unlock(&pool->attention_hmx_mutex);
    if (result != 0) {
        pool->attention_gqa_abort = 1U;
        asm volatile("barrier" ::: "memory");
        return -1;
    }
    return 0;
}

struct qbh_attention_u8_group_view {
    const struct qbh_attention_config *config;
    uint8_t *q_group;
    uint8_t *v_head;
    uint8_t *score_group;
    uint8_t *probability_group;
    uint8_t *output_group;
    int8_t *k_weight;
    int8_t *v_weight;
    uint32_t *qk_bias;
    uint32_t *av_bias;
};

static int qbh_attention_u8_group_view_init(
    struct qbh_block_header *header,
    struct qbh_block_buffers *buffers, uint32_t group,
    struct qbh_attention_u8_group_view *view) {
    uint8_t *scratch;
    uint32_t first_q_head;

    if (group >= QBH_BLOCK_KV_HEADS || view == NULL) {
        return -1;
    }
    view->config = &buffers->attention_configs[group];
    if (!qbh_block_attention_config_valid(
            header, view->config, group)) {
        return -1;
    }
    scratch = buffers->attention_concat +
        (size_t)group * QBH_ATTN_U8_GROUP_SCRATCH_BYTES;
    first_q_head = group * QBH_ATTENTION_Q_HEADS_PER_GROUP;
    view->q_group = buffers->q +
        (size_t)first_q_head * QBH_ATTENTION_HEAD_DIM_TILES *
            QBH_HMX_ACTIVATION_BYTES;
    view->v_head = buffers->v +
        (size_t)group * QBH_ATTENTION_HEAD_DIM_TILES *
            QBH_HMX_ACTIVATION_BYTES;
    view->score_group = buffers->scores +
        (size_t)first_q_head * QBH_ATTENTION_SCORE_TILES *
            QBH_HMX_OUTPUT_BYTES;
    view->probability_group = buffers->probability +
        (size_t)first_q_head * QBH_ATTENTION_SCORE_TILES *
            QBH_HMX_ACTIVATION_BYTES;
    view->output_group = buffers->hmx_activation +
        (size_t)first_q_head * QBH_ATTENTION_HEAD_DIM_TILES *
            QBH_HMX_OUTPUT_BYTES;
    view->k_weight = (int8_t *)(
        scratch + QBH_ATTN_U8_K_WEIGHT_OFFSET);
    view->v_weight = (int8_t *)(
        scratch + QBH_ATTN_U8_V_WEIGHT_OFFSET);
    view->qk_bias = (uint32_t *)(
        scratch + QBH_ATTN_U8_QK_BIAS_OFFSET);
    view->av_bias = (uint32_t *)(
        scratch + QBH_ATTN_U8_AV_BIAS_OFFSET);
    return 0;
}

static void qbh_attention_u8_dependency_stream_run_tasks(
    struct qbh_block_w4f16_pool *pool,
    struct qbh_block_w4f16_job *job) {
    struct qbh_block_header *header = pool->attention_header;
    struct qbh_block_buffers *buffers = pool->attention_buffers;
    uint8_t *softmax_scratch = buffers->attention_projection +
        (size_t)job->worker_index * QBH_ATTN_U8_SOFTMAX_SCRATCH_BYTES;
    uint32_t templates_built = 0U;

    job->u8_attention_probability_row_sum_min = UINT32_MAX;
    for (;;) {
        const uint32_t group =
            qbh_atomic_fetch_increment(&pool->next_attention_task);
        struct qbh_attention_u8_group_view view;
        struct qbh_attention_u8_telemetry telemetry;
        struct qbh_attention_u8_telemetry *telemetry_ptr;
        uint8_t *group_scratch;
        uint64_t start;

        if (group >= QBH_BLOCK_KV_HEADS ||
            pool->attention_gqa_abort != 0U) {
            break;
        }
        if (qbh_attention_u8_group_view_init(
                header, buffers, group, &view) != 0) {
            pool->attention_gqa_abort = 1U;
            asm volatile("barrier" ::: "memory");
            return;
        }
        memset(&telemetry, 0, sizeof(telemetry));
        telemetry_ptr = header->numerical_audit_enabled != 0U
            ? &telemetry : NULL;
        group_scratch = buffers->attention_concat +
            (size_t)group * QBH_ATTN_U8_GROUP_SCRATCH_BYTES;
        start = HAP_perf_get_qtimer_count();
        qbh_attention_u8_pack_v_native_vgather_vdeal(
            view.v_head, view.config, view.v_weight, view.av_bias,
            group_scratch,
            telemetry_ptr != NULL
                ? &telemetry.v_recenter_saturation_count : NULL);
        job->u8_attention_v_pack_ticks +=
            HAP_perf_get_qtimer_count() - start;
        if (qbh_attention_u8_pool_submit(
                pool, job, view.q_group, view.k_weight,
                view.qk_bias, view.score_group,
                QBH_ATTENTION_Q_HEADS_PER_GROUP,
                QBH_ATTENTION_HEAD_DIM_TILES,
                QBH_ATTENTION_SCORE_TILES,
                &job->u8_attention_qk_hmx_ticks) != 0) {
            return;
        }
        if (telemetry_ptr != NULL) {
            job->u8_attention_v_recenter_saturation_count +=
                telemetry.v_recenter_saturation_count;
        }
        pool->u8_attention_qk_ready[group] =
            pool->u8_attention_dependency_generation;
        asm volatile("release(%0):at"
                     :
                     : "r"(&pool->u8_attention_qk_ready[group])
                     : "memory");
    }

    for (;;) {
        const uint32_t task = qbh_atomic_fetch_increment(
            &pool->next_attention_softmax_task);
        const uint32_t group = task % QBH_BLOCK_KV_HEADS;
        const uint32_t slice = task / QBH_BLOCK_KV_HEADS;
        const uint32_t first_row =
            slice * QBH_BLOCK_W4U8_SOFTMAX_ROWS_PER_SLICE;
        const uint32_t ready_index =
            group * QBH_BLOCK_W4U8_SOFTMAX_ROW_SLICES + slice;
        struct qbh_attention_u8_group_view view;
        struct qbh_attention_u8_telemetry telemetry;
        struct qbh_attention_u8_telemetry *telemetry_ptr;
        uint64_t start;

        if (task >= QBH_BLOCK_KV_HEADS *
                        QBH_BLOCK_W4U8_SOFTMAX_ROW_SLICES ||
            pool->attention_gqa_abort != 0U) {
            break;
        }
        while (pool->u8_attention_qk_ready[group] !=
               pool->u8_attention_dependency_generation) {
            if (pool->attention_gqa_abort != 0U) {
                return;
            }
            asm volatile("pause(#8)" : : : "memory");
        }
        asm volatile("barrier" ::: "memory");
        if (qbh_attention_u8_group_view_init(
                header, buffers, group, &view) != 0) {
            pool->attention_gqa_abort = 1U;
            asm volatile("barrier" ::: "memory");
            return;
        }
        if (templates_built == 0U) {
            start = HAP_perf_get_qtimer_count();
            qbh_attention_u8_build_sole_lut_template_bank(
                softmax_scratch + QBH_ATTN_U8_SOFTMAX_TEMPLATE_OFFSET);
            asm volatile("barrier" ::: "memory");
            job->u8_attention_softmax_ticks +=
                HAP_perf_get_qtimer_count() - start;
            templates_built = 1U;
        }
        memset(&telemetry, 0, sizeof(telemetry));
        telemetry_ptr = header->numerical_audit_enabled != 0U
            ? &telemetry : NULL;
        start = HAP_perf_get_qtimer_count();
        if (qbh_attention_u8_softmax_shuffle4_enabled(
                header->attention_pipeline_mode)) {
            qbh_attention_u8_requant_softmax_group_rows_prebuilt_templates_shuffle4(
                view.score_group, view.probability_group,
                softmax_scratch,
                buffers->attention_projection +
                    (size_t)QBH_BLOCK_MAX_ATTENTION_HVX_CONTEXTS *
                        QBH_ATTN_U8_SOFTMAX_SCRATCH_BYTES +
                    (size_t)job->worker_index *
                        QBH_ATTN_U8_SOFTMAX_CARRIER_BYTES,
                view.config, telemetry_ptr,
                first_row, QBH_BLOCK_W4U8_SOFTMAX_ROWS_PER_SLICE);
            job->u8_attention_softmax_shuffle4_row_group_count +=
                QBH_BLOCK_W4U8_SOFTMAX_ROWS_PER_SLICE / 4U;
        } else {
            qbh_attention_u8_requant_softmax_group_rows_prebuilt_templates(
                view.score_group, view.probability_group,
                softmax_scratch, view.config, telemetry_ptr,
                first_row, QBH_BLOCK_W4U8_SOFTMAX_ROWS_PER_SLICE);
        }
        job->u8_attention_softmax_ticks +=
            HAP_perf_get_qtimer_count() - start;
        ++job->attention_softmax_task_count;
        if (telemetry_ptr != NULL) {
            job->u8_attention_score_saturation_count +=
                telemetry.score_saturation_count;
            job->u8_attention_probability_mask_violation_count +=
                telemetry.probability_mask_violation_count;
            if (telemetry.probability_row_sum_min <
                job->u8_attention_probability_row_sum_min) {
                job->u8_attention_probability_row_sum_min =
                    telemetry.probability_row_sum_min;
            }
            if (telemetry.probability_row_sum_max >
                job->u8_attention_probability_row_sum_max) {
                job->u8_attention_probability_row_sum_max =
                    telemetry.probability_row_sum_max;
            }
        }
        pool->u8_attention_softmax_ready[ready_index] =
            pool->u8_attention_dependency_generation;
        asm volatile("release(%0):at"
                     :
                     : "r"(&pool->u8_attention_softmax_ready[ready_index])
                     : "memory");
    }

    for (;;) {
        const uint32_t group = qbh_atomic_fetch_increment(
            &pool->next_attention_av_task);
        struct qbh_attention_u8_group_view view;
        uint64_t start;

        if (group >= QBH_BLOCK_KV_HEADS ||
            pool->attention_gqa_abort != 0U) {
            break;
        }
        while (pool->u8_attention_softmax_ready[
                   group * QBH_BLOCK_W4U8_SOFTMAX_ROW_SLICES] !=
                   pool->u8_attention_dependency_generation ||
               pool->u8_attention_softmax_ready[
                   group * QBH_BLOCK_W4U8_SOFTMAX_ROW_SLICES + 1U] !=
                   pool->u8_attention_dependency_generation) {
            if (pool->attention_gqa_abort != 0U) {
                return;
            }
            asm volatile("pause(#8)" : : : "memory");
        }
        asm volatile("barrier" ::: "memory");
        if (qbh_attention_u8_group_view_init(
                header, buffers, group, &view) != 0) {
            pool->attention_gqa_abort = 1U;
            asm volatile("barrier" ::: "memory");
            return;
        }
        if (qbh_attention_u8_pool_submit(
                pool, job, view.probability_group, view.v_weight,
                view.av_bias, view.output_group,
                QBH_ATTENTION_Q_HEADS_PER_GROUP,
                QBH_ATTENTION_SCORE_TILES,
                QBH_ATTENTION_HEAD_DIM_TILES,
                &job->u8_attention_av_hmx_ticks) != 0) {
            return;
        }
        start = HAP_perf_get_qtimer_count();
        qbh_attention_u8_requant_av(view.output_group, view.config);
        job->u8_attention_av_requant_ticks +=
            HAP_perf_get_qtimer_count() - start;
        ++job->u8_attention_group_count;
    }
}

static void qbh_attention_u8_pool_run_tasks(
    struct qbh_block_w4f16_pool *pool,
    struct qbh_block_w4f16_job *job) {
    struct qbh_block_header *header = pool->attention_header;
    struct qbh_block_buffers *buffers = pool->attention_buffers;

    if (qbh_attention_u8_dependency_stream_enabled(
            header->attention_pipeline_mode)) {
        qbh_attention_u8_dependency_stream_run_tasks(pool, job);
        return;
    }

    job->u8_attention_probability_row_sum_min = UINT32_MAX;
    for (;;) {
        const uint32_t group =
            qbh_atomic_fetch_increment(&pool->next_attention_task);
        const struct qbh_attention_config *config;
        uint32_t first_q_head;
        uint8_t *q_group;
        uint8_t *k_head;
        uint8_t *v_head;
        uint8_t *score_group;
        uint8_t *probability_group;
        uint8_t *output_group;
        uint8_t *scratch;
        int8_t *k_weight;
        int8_t *v_weight;
        uint32_t *qk_bias;
        uint32_t *av_bias;
        uint8_t *softmax_scratch;
        struct qbh_attention_u8_telemetry telemetry;
        struct qbh_attention_u8_telemetry *telemetry_ptr;
        uint64_t start;

        if (group >= pool->attention_task_count ||
            pool->attention_gqa_abort != 0U) {
            break;
        }
        config = &buffers->attention_configs[group];
        if (!qbh_block_attention_config_valid(
                header, config, group)) {
            pool->attention_gqa_abort = 1U;
            asm volatile("barrier" ::: "memory");
            break;
        }
        scratch = buffers->attention_concat +
            (size_t)(qbh_attention_u8_qkv_overlap_enabled(
                         header->attention_pipeline_mode)
                         ? group : job->worker_index) *
                QBH_ATTN_U8_GROUP_SCRATCH_BYTES;
        k_weight = (int8_t *)(
            scratch + QBH_ATTN_U8_K_WEIGHT_OFFSET);
        v_weight = (int8_t *)(
            scratch + QBH_ATTN_U8_V_WEIGHT_OFFSET);
        qk_bias = (uint32_t *)(
            scratch + QBH_ATTN_U8_QK_BIAS_OFFSET);
        av_bias = (uint32_t *)(
            scratch + QBH_ATTN_U8_AV_BIAS_OFFSET);
        softmax_scratch =
            scratch + QBH_ATTN_U8_SOFTMAX_SCRATCH_OFFSET;
        first_q_head = group * QBH_ATTENTION_Q_HEADS_PER_GROUP;
        q_group = buffers->q +
            (size_t)first_q_head *
                QBH_ATTENTION_HEAD_DIM_TILES *
                QBH_HMX_ACTIVATION_BYTES;
        k_head = buffers->k +
            (size_t)group * QBH_ATTENTION_HEAD_DIM_TILES *
                QBH_HMX_ACTIVATION_BYTES;
        v_head = buffers->v +
            (size_t)group * QBH_ATTENTION_HEAD_DIM_TILES *
                QBH_HMX_ACTIVATION_BYTES;
        score_group = buffers->scores +
            (size_t)first_q_head * QBH_ATTENTION_SCORE_TILES *
                QBH_HMX_OUTPUT_BYTES;
        probability_group = buffers->probability +
            (size_t)first_q_head * QBH_ATTENTION_SCORE_TILES *
                QBH_HMX_ACTIVATION_BYTES;
        output_group = buffers->hmx_activation +
            (size_t)first_q_head *
                QBH_ATTENTION_HEAD_DIM_TILES *
                QBH_HMX_OUTPUT_BYTES;
        memset(&telemetry, 0, sizeof(telemetry));
        telemetry_ptr = header->numerical_audit_enabled != 0U
            ? &telemetry : NULL;

        if (!qbh_attention_u8_qkv_overlap_enabled(
                header->attention_pipeline_mode)) {
            start = HAP_perf_get_qtimer_count();
            for (uint32_t local_head = 0U;
                 local_head < QBH_ATTENTION_Q_HEADS_PER_GROUP;
                 ++local_head) {
                qbh_hvx_qk_norm_rope_u8_native_head(
                    q_group +
                        (size_t)local_head *
                            QBH_ATTENTION_HEAD_DIM_TILES *
                            QBH_HMX_ACTIVATION_BYTES,
                    &header->qparams[QBH_BLOCK_QP_Q_PROJECTION],
                    &header->qparams[QBH_BLOCK_QP_Q_ROPE],
                    (const __fp16 *)buffers->q_norm_weight,
                    (const __fp16 *)buffers->rope_cos,
                    (const __fp16 *)buffers->rope_sin);
            }
            if (qbh_attention_u8_fused_k_enabled(
                    header->attention_pipeline_mode)) {
                qbh_hvx_qk_norm_rope_u8_native_k_head(
                    k_head,
                    &header->qparams[QBH_BLOCK_QP_K_PROJECTION],
                    &header->qparams[QBH_BLOCK_QP_K_ROPE],
                    (const __fp16 *)buffers->k_norm_weight,
                    (const __fp16 *)buffers->rope_cos,
                    (const __fp16 *)buffers->rope_sin,
                    config, k_weight, qk_bias);
            } else {
                qbh_hvx_qk_norm_rope_u8_native_head(
                    k_head,
                    &header->qparams[QBH_BLOCK_QP_K_PROJECTION],
                    &header->qparams[QBH_BLOCK_QP_K_ROPE],
                    (const __fp16 *)buffers->k_norm_weight,
                    (const __fp16 *)buffers->rope_cos,
                    (const __fp16 *)buffers->rope_sin);
            }
            job->u8_attention_qk_norm_rope_ticks +=
                HAP_perf_get_qtimer_count() - start;

            if (!qbh_attention_u8_fused_k_enabled(
                    header->attention_pipeline_mode)) {
                start = HAP_perf_get_qtimer_count();
                qbh_attention_u8_pack_k_native(
                    k_head, config, k_weight, qk_bias);
                job->u8_attention_k_pack_ticks +=
                    HAP_perf_get_qtimer_count() - start;
            } else if (header->numerical_audit_enabled != 0U) {
                const uint32_t audit_slice_bytes =
                    QBH_ATTN_U8_K_WEIGHT_BYTES +
                    QBH_ATTN_U8_QK_BIAS_BYTES;
                uint8_t *reference = buffers->attention_projection +
                    (size_t)job->worker_index * audit_slice_bytes;
                int8_t *reference_weight = (int8_t *)reference;
                uint32_t *reference_bias = (uint32_t *)(
                    reference + QBH_ATTN_U8_K_WEIGHT_BYTES);

                qbh_attention_u8_pack_k_native(
                    k_head, config, reference_weight, reference_bias);
                job->u8_attention_fused_k_operand_mismatch_count +=
                    qbh_count_u8_mismatches(
                        (const uint8_t *)k_weight,
                        (const uint8_t *)reference_weight,
                        QBH_ATTN_U8_K_WEIGHT_BYTES) +
                    qbh_count_u8_mismatches(
                        (const uint8_t *)qk_bias,
                        (const uint8_t *)reference_bias,
                        QBH_ATTN_U8_QK_BIAS_BYTES);
            }
        }

        start = HAP_perf_get_qtimer_count();
        if (qbh_attention_u8_vdeal_enabled(
                header->attention_pipeline_mode)) {
            qbh_attention_u8_pack_v_native_vgather_vdeal(
                v_head, config, v_weight, av_bias, scratch,
                telemetry_ptr != NULL
                    ? &telemetry.v_recenter_saturation_count
                    : NULL);
        } else if (qbh_attention_u8_vgather_enabled(
                header->attention_pipeline_mode)) {
            qbh_attention_u8_pack_v_native_vgather(
                v_head, config, v_weight, av_bias, scratch,
                telemetry_ptr != NULL
                    ? &telemetry.v_recenter_saturation_count
                    : NULL);
        } else {
            qbh_attention_u8_pack_v_native(
                v_head, config, v_weight, av_bias,
                telemetry_ptr != NULL
                    ? &telemetry.v_recenter_saturation_count
                    : NULL);
        }
        job->u8_attention_v_pack_ticks +=
            HAP_perf_get_qtimer_count() - start;

        if (qbh_attention_u8_gqa_hmx_batch_enabled(
                header->attention_pipeline_mode)) {
            if (qbh_attention_u8_pool_submit(
                    pool, job, q_group, k_weight, qk_bias,
                    score_group, QBH_ATTENTION_Q_HEADS_PER_GROUP,
                    QBH_ATTENTION_HEAD_DIM_TILES,
                    QBH_ATTENTION_SCORE_TILES,
                    &job->u8_attention_qk_hmx_ticks) != 0) {
                break;
            }
        } else {
            for (uint32_t local_head = 0U;
                 local_head < QBH_ATTENTION_Q_HEADS_PER_GROUP;
                 ++local_head) {
                const uint8_t *q_head = q_group +
                    (size_t)local_head *
                        QBH_ATTENTION_HEAD_DIM_TILES *
                        QBH_HMX_ACTIVATION_BYTES;
                if (qbh_attention_u8_hmx_batch_enabled(
                        header->attention_pipeline_mode)) {
                    if (qbh_attention_u8_pool_submit(
                            pool, job, q_head, k_weight, qk_bias,
                            score_group +
                                (size_t)local_head *
                                    QBH_ATTENTION_SCORE_TILES *
                                    QBH_HMX_OUTPUT_BYTES,
                            1U, QBH_ATTENTION_HEAD_DIM_TILES,
                            QBH_ATTENTION_SCORE_TILES,
                            &job->u8_attention_qk_hmx_ticks) != 0) {
                        break;
                    }
                } else {
                    for (uint32_t n_tile = 0U;
                         n_tile < QBH_ATTENTION_SCORE_TILES; ++n_tile) {
                        if (qbh_attention_u8_pool_submit(
                                pool, job, q_head,
                                k_weight +
                                    (size_t)n_tile *
                                        QBH_ATTENTION_HEAD_DIM_TILES *
                                        QBH_HMX_WEIGHT_BYTES,
                                qk_bias +
                                    (size_t)n_tile *
                                        (QBH_HMX_BIAS_BYTES /
                                         sizeof(uint32_t)),
                                score_group +
                                    ((size_t)local_head *
                                         QBH_ATTENTION_SCORE_TILES +
                                     n_tile) * QBH_HMX_OUTPUT_BYTES,
                                1U, QBH_ATTENTION_HEAD_DIM_TILES, 1U,
                                &job->u8_attention_qk_hmx_ticks) != 0) {
                            break;
                        }
                    }
                }
            }
        }
        if (pool->attention_gqa_abort != 0U) {
            break;
        }

        if (!qbh_attention_u8_fused_qk_requant_enabled(
                header->attention_pipeline_mode)) {
            start = HAP_perf_get_qtimer_count();
            qbh_attention_u8_requant_qk(
                score_group, config,
                telemetry_ptr != NULL
                    ? &telemetry.score_saturation_count : NULL);
            job->u8_attention_qk_requant_ticks +=
                HAP_perf_get_qtimer_count() - start;
        }

        start = HAP_perf_get_qtimer_count();
        if (qbh_attention_u8_fused_qk_requant_enabled(
                header->attention_pipeline_mode)) {
            if (qbh_attention_u8_lut_templates_enabled(
                    header->attention_pipeline_mode)) {
                qbh_attention_u8_requant_softmax_group_lut_templates(
                    score_group, probability_group, softmax_scratch,
                    config, telemetry_ptr);
            } else {
                qbh_attention_u8_requant_softmax_group(
                    score_group, probability_group, softmax_scratch,
                    config, telemetry_ptr);
            }
        } else {
            qbh_attention_u8_softmax_group(
                score_group, probability_group, softmax_scratch,
                config, telemetry_ptr);
        }
        job->u8_attention_softmax_ticks +=
            HAP_perf_get_qtimer_count() - start;

        if (qbh_attention_u8_gqa_hmx_batch_enabled(
                header->attention_pipeline_mode)) {
            if (qbh_attention_u8_pool_submit(
                    pool, job, probability_group, v_weight, av_bias,
                    output_group, QBH_ATTENTION_Q_HEADS_PER_GROUP,
                    QBH_ATTENTION_SCORE_TILES,
                    QBH_ATTENTION_HEAD_DIM_TILES,
                    &job->u8_attention_av_hmx_ticks) != 0) {
                break;
            }
        } else {
            for (uint32_t local_head = 0U;
                 local_head < QBH_ATTENTION_Q_HEADS_PER_GROUP;
                 ++local_head) {
                const uint8_t *probability = probability_group +
                    (size_t)local_head * QBH_ATTENTION_SCORE_TILES *
                        QBH_HMX_ACTIVATION_BYTES;
                if (qbh_attention_u8_hmx_batch_enabled(
                        header->attention_pipeline_mode)) {
                    if (qbh_attention_u8_pool_submit(
                            pool, job, probability, v_weight, av_bias,
                            output_group +
                                (size_t)local_head *
                                    QBH_ATTENTION_HEAD_DIM_TILES *
                                    QBH_HMX_OUTPUT_BYTES,
                            1U, QBH_ATTENTION_SCORE_TILES,
                            QBH_ATTENTION_HEAD_DIM_TILES,
                            &job->u8_attention_av_hmx_ticks) != 0) {
                        break;
                    }
                } else {
                    for (uint32_t n_tile = 0U;
                         n_tile < QBH_ATTENTION_HEAD_DIM_TILES; ++n_tile) {
                        if (qbh_attention_u8_pool_submit(
                                pool, job, probability,
                                v_weight +
                                    (size_t)n_tile *
                                        QBH_ATTENTION_SCORE_TILES *
                                        QBH_HMX_WEIGHT_BYTES,
                                av_bias +
                                    (size_t)n_tile *
                                        (QBH_HMX_BIAS_BYTES /
                                         sizeof(uint32_t)),
                                output_group +
                                    ((size_t)local_head *
                                         QBH_ATTENTION_HEAD_DIM_TILES +
                                     n_tile) * QBH_HMX_OUTPUT_BYTES,
                                1U, QBH_ATTENTION_SCORE_TILES, 1U,
                                &job->u8_attention_av_hmx_ticks) != 0) {
                            break;
                        }
                    }
                }
            }
        }
        if (pool->attention_gqa_abort != 0U) {
            break;
        }

        start = HAP_perf_get_qtimer_count();
        qbh_attention_u8_requant_av(output_group, config);
        job->u8_attention_av_requant_ticks +=
            HAP_perf_get_qtimer_count() - start;

        if (telemetry_ptr != NULL) {
            job->u8_attention_score_saturation_count +=
                telemetry.score_saturation_count;
            job->u8_attention_v_recenter_saturation_count +=
                telemetry.v_recenter_saturation_count;
            job->u8_attention_probability_mask_violation_count +=
                telemetry.probability_mask_violation_count;
            if (telemetry.probability_row_sum_min <
                job->u8_attention_probability_row_sum_min) {
                job->u8_attention_probability_row_sum_min =
                    telemetry.probability_row_sum_min;
            }
            if (telemetry.probability_row_sum_max >
                job->u8_attention_probability_row_sum_max) {
                job->u8_attention_probability_row_sum_max =
                    telemetry.probability_row_sum_max;
            }
        }
        ++job->u8_attention_group_count;
    }
}

static void qbh_attention_u8_accumulate_job(
    struct qbh_block_header *header,
    const struct qbh_block_w4f16_job *job) {
    header->attention_softmax_task_count +=
        job->attention_softmax_task_count;
    header->u8_attention_softmax_shuffle4_row_group_count +=
        job->u8_attention_softmax_shuffle4_row_group_count;
    header->u8_attention_qk_norm_rope_ticks +=
        job->u8_attention_qk_norm_rope_ticks;
    header->u8_attention_k_pack_ticks +=
        job->u8_attention_k_pack_ticks;
    header->u8_attention_v_pack_ticks +=
        job->u8_attention_v_pack_ticks;
    header->u8_attention_qk_hmx_ticks +=
        job->u8_attention_qk_hmx_ticks;
    header->u8_attention_qk_requant_ticks +=
        job->u8_attention_qk_requant_ticks;
    header->u8_attention_softmax_ticks +=
        job->u8_attention_softmax_ticks;
    header->u8_attention_av_hmx_ticks +=
        job->u8_attention_av_hmx_ticks;
    header->u8_attention_av_requant_ticks +=
        job->u8_attention_av_requant_ticks;
    header->u8_attention_pipeline_wait_ticks +=
        job->u8_attention_hmx_queue_wait_ticks;
    header->u8_attention_fused_k_operand_mismatch_count +=
        job->u8_attention_fused_k_operand_mismatch_count;
    header->u8_attention_score_saturation_count +=
        job->u8_attention_score_saturation_count;
    header->u8_attention_v_recenter_saturation_count +=
        job->u8_attention_v_recenter_saturation_count;
    header->u8_attention_probability_mask_violation_count +=
        job->u8_attention_probability_mask_violation_count;
    if (job->u8_attention_probability_row_sum_min <
        header->u8_attention_probability_row_sum_min) {
        header->u8_attention_probability_row_sum_min =
            job->u8_attention_probability_row_sum_min;
    }
    if (job->u8_attention_probability_row_sum_max >
        header->u8_attention_probability_row_sum_max) {
        header->u8_attention_probability_row_sum_max =
            job->u8_attention_probability_row_sum_max;
    }
    header->u8_attention_group_count +=
        job->u8_attention_group_count;
}

static int qbh_hvx_pool_u8_attention(
    struct qbh_block_header *header,
    struct qbh_block_w4f16_pool *pool,
    struct qbh_block_buffers *buffers,
    struct qbh_block_hmx_worker *worker) {
    struct qbh_block_w4f16_job main_job;
    uint64_t wait_start;
    uint32_t completed_groups;

    if (pool == NULL || buffers == NULL || worker == NULL ||
        header->attention_hvx_contexts < 4U ||
        header->attention_hvx_contexts >
            QBH_BLOCK_MAX_ATTENTION_HVX_CONTEXTS ||
        header->attention_hvx_contexts - 1U > pool->worker_count) {
        return -1;
    }
    memset(&main_job, 0, sizeof(main_job));
    main_job.worker_index = header->attention_hvx_contexts - 1U;
    pool->attention_header = header;
    pool->attention_buffers = buffers;
    pool->attention_hmx_worker = worker;
    pool->attention_task_count = QBH_BLOCK_KV_HEADS;
    pool->next_attention_task = 0U;
    pool->next_attention_softmax_task = 0U;
    pool->next_attention_av_task = 0U;
    pool->attention_gqa_abort = 0U;
    pool->active_worker_count = header->attention_hvx_contexts - 1U;
    header->u8_attention_probability_row_sum_min = UINT32_MAX;
    if (qbh_attention_u8_dependency_stream_enabled(
            header->attention_pipeline_mode)) {
        ++pool->u8_attention_dependency_generation;
        if (pool->u8_attention_dependency_generation == 0U) {
            memset((void *)pool->u8_attention_qk_ready, 0,
                   sizeof(pool->u8_attention_qk_ready));
            memset((void *)pool->u8_attention_softmax_ready, 0,
                   sizeof(pool->u8_attention_softmax_ready));
            pool->u8_attention_dependency_generation = 1U;
        }
    }
    for (uint32_t worker_index = 0U;
         worker_index < pool->active_worker_count; ++worker_index) {
        struct qbh_block_w4f16_job *job =
            &pool->jobs[worker_index];
        job->attention_softmax_task_count = 0U;
        job->u8_attention_softmax_shuffle4_row_group_count = 0U;
        job->u8_attention_group_count = 0U;
        job->u8_attention_score_saturation_count = 0U;
        job->u8_attention_v_recenter_saturation_count = 0U;
        job->u8_attention_probability_mask_violation_count = 0U;
        job->u8_attention_fused_k_operand_mismatch_count = 0U;
        job->u8_attention_probability_row_sum_min = UINT32_MAX;
        job->u8_attention_probability_row_sum_max = 0U;
        job->u8_attention_qk_norm_rope_ticks = 0U;
        job->u8_attention_k_pack_ticks = 0U;
        job->u8_attention_v_pack_ticks = 0U;
        job->u8_attention_qk_hmx_ticks = 0U;
        job->u8_attention_qk_requant_ticks = 0U;
        job->u8_attention_softmax_ticks = 0U;
        job->u8_attention_av_hmx_ticks = 0U;
        job->u8_attention_av_requant_ticks = 0U;
        job->u8_attention_hmx_queue_wait_ticks = 0U;
        job->command_kind =
            QBH_BLOCK_HVX_POOL_U8_GQA_ATTENTION;
    }
    asm volatile("barrier" ::: "memory");
    for (uint32_t worker_index = 0U;
         worker_index < pool->active_worker_count; ++worker_index) {
        (void)qurt_sem_up(&pool->command_ready[worker_index]);
    }
    qbh_attention_u8_pool_run_tasks(pool, &main_job);
    wait_start = HAP_perf_get_qtimer_count();
    qbh_w4f16_pool_wait(pool);
    main_job.u8_attention_hmx_queue_wait_ticks +=
        HAP_perf_get_qtimer_count() - wait_start;

    qbh_attention_u8_accumulate_job(header, &main_job);
    completed_groups = main_job.u8_attention_group_count;
    for (uint32_t worker_index = 0U;
         worker_index < pool->active_worker_count; ++worker_index) {
        qbh_attention_u8_accumulate_job(
            header, &pool->jobs[worker_index]);
        completed_groups +=
            pool->jobs[worker_index].u8_attention_group_count;
        pool->jobs[worker_index].attention_softmax_task_count = 0U;
    }
    if (pool->attention_gqa_abort != 0U ||
        completed_groups != QBH_BLOCK_KV_HEADS) {
        return -1;
    }
    header->u8_attention_qk_execution_count +=
        completed_groups * QBH_ATTENTION_Q_HEADS_PER_GROUP *
        QBH_ATTENTION_SCORE_TILES;
    header->u8_attention_av_execution_count +=
        completed_groups * QBH_ATTENTION_Q_HEADS_PER_GROUP *
        QBH_ATTENTION_HEAD_DIM_TILES;
    header->hmx_command_count += completed_groups *
        (qbh_attention_u8_gqa_hmx_batch_enabled(
             header->attention_pipeline_mode)
             ? 2U
             : QBH_ATTENTION_Q_HEADS_PER_GROUP *
                   (qbh_attention_u8_hmx_batch_enabled(
                        header->attention_pipeline_mode)
                        ? 2U
                        : (QBH_ATTENTION_SCORE_TILES +
                           QBH_ATTENTION_HEAD_DIM_TILES)));
    header->hmx_u8s8_tile_pair_count +=
        completed_groups * QBH_ATTENTION_Q_HEADS_PER_GROUP *
        (QBH_ATTENTION_SCORE_TILES *
             QBH_ATTENTION_HEAD_DIM_TILES +
         QBH_ATTENTION_HEAD_DIM_TILES *
             QBH_ATTENTION_SCORE_TILES);
    header->u8_attention_direct_o_tile_count +=
        completed_groups * QBH_ATTENTION_Q_HEADS_PER_GROUP *
        QBH_ATTENTION_HEAD_DIM_TILES;
    return 0;
}

static void qbh_residual_add_f16(__fp16 *residual,
                                 const __fp16 *addition,
                                 uint32_t elements) {
    for (uint32_t index = 0; index < elements; ++index) {
        residual[index] = (__fp16)((float)residual[index] +
                                   (float)addition[index]);
    }
}

static void qbh_residual_add_u8(
    const uint8_t *left, const struct qbh_block_qparam *left_qparam,
    const uint8_t *right, const struct qbh_block_qparam *right_qparam,
    uint8_t *output, const struct qbh_block_qparam *output_qparam,
    uint32_t elements) {
    for (uint32_t index = 0; index < elements; ++index) {
        output[index] = qbh_quantize(
            qbh_dequantize(left[index], left_qparam) +
                qbh_dequantize(right[index], right_qparam),
            output_qparam);
    }
}

static void qbh_silu_multiply_f16(const __fp16 *gate,
                                  const __fp16 *up, __fp16 *middle,
                                  uint32_t elements) {
    for (uint32_t index = 0; index < elements; ++index) {
        float gate_value = (float)gate[index];
        middle[index] = (__fp16)(
            gate_value / (1.0f + expf(-gate_value)) * (float)up[index]);
    }
}

static void qbh_silu_multiply_u8(
    const uint8_t *gate, const struct qbh_block_qparam *gate_qparam,
    const uint8_t *up, const struct qbh_block_qparam *up_qparam,
    uint8_t *middle, const struct qbh_block_qparam *middle_qparam,
    uint32_t elements) {
    for (uint32_t index = 0; index < elements; ++index) {
        float gate_value = qbh_dequantize(gate[index], gate_qparam);
        float up_value = qbh_dequantize(up[index], up_qparam);
        middle[index] = qbh_quantize(
            gate_value / (1.0f + expf(-gate_value)) * up_value,
            middle_qparam);
    }
}

static int qbh_stage_metadata(struct qbh_block_header *header,
                              const uint8_t *shared,
                              struct qbh_block_buffers *buffers) {
    const uint32_t rope_chunk_bytes =
        QBH_BLOCK_M * QBH_BLOCK_HEAD_DIM * sizeof(uint16_t);
    const uint32_t offsets[] = {
        header->input_norm_weight_offset,
        header->post_norm_weight_offset,
        header->q_norm_weight_offset,
        header->k_norm_weight_offset,
        header->rope_cos_offset,
        header->rope_sin_offset,
    };
    const uint32_t bytes[] = {
        header->input_norm_weight_bytes,
        header->post_norm_weight_bytes,
        header->q_norm_weight_bytes,
        header->k_norm_weight_bytes,
        rope_chunk_bytes,
        rope_chunk_bytes,
    };
    void *destinations[] = {
        buffers->input_norm_weight,
        buffers->post_norm_weight,
        buffers->q_norm_weight,
        buffers->k_norm_weight,
        buffers->rope_cos,
        buffers->rope_sin,
    };
    uint64_t start = HAP_perf_get_qtimer_count();
    for (uint32_t index = 0; index < 6U; ++index) {
        if (qbh_dma_copy(header, destinations[index],
                         shared + offsets[index], bytes[index], 1U) != 0) {
            return -1;
        }
        header->boundary_ddr_read_bytes += bytes[index];
        ++header->boundary_dma_descriptor_count;
    }
    if (qbh_block_mlp_is_w4u8_streaming(header->mlp_mode)) {
        if (qbh_dma_copy(
                header, buffers->w4u8_silu_lut,
                shared + header->w4u8_silu_lut_offset,
                header->w4u8_silu_lut_bytes, 1U) != 0) {
            return -1;
        }
        header->boundary_ddr_read_bytes +=
            header->w4u8_silu_lut_bytes;
        ++header->boundary_dma_descriptor_count;
        header->w4u8_mlp_lut_vtcm_bytes =
            header->w4u8_silu_lut_bytes;
        header->w4u8_mlp_gather_scratch_vtcm_bytes =
            QBH_BLOCK_W4U8_GATHER_SCRATCH_BYTES;
    }
    if (qbh_attention_u8_enabled(
            header->attention_pipeline_mode)) {
        if (qbh_dma_copy(
                header, buffers->attention_configs,
                shared + header->attention_config_offset,
                header->attention_config_bytes, 1U) != 0) {
            return -1;
        }
        header->boundary_ddr_read_bytes +=
            header->attention_config_bytes;
        ++header->boundary_dma_descriptor_count;
    }
    if (header->variant == QBH_BLOCK_W4F16) {
        uint8_t *destination = buffers->projection_scales;
        for (uint32_t index = 0;
             index < QBH_BLOCK_PROJECTION_COUNT; ++index) {
            const struct qbh_block_projection_desc *desc =
                &header->projections[index];
            if (header->w4f16_pipeline_mode !=
                    QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_GATE4_DMA8_CROSS_PREFETCH ||
                (index != QBH_BLOCK_PROJ_GATE &&
                 index != QBH_BLOCK_PROJ_UP)) {
                if (qbh_dma_copy(
                        header, destination, shared + desc->scale_offset,
                        desc->scale_bytes, 1U) != 0) {
                    return -1;
                }
                header->weight_ddr_read_bytes += desc->scale_bytes;
                ++header->weight_dma_descriptor_count;
            }
            destination += desc->scale_bytes;
        }
        if (header->w4f16_pipeline_mode ==
            QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_GATE4_DMA8_CROSS_PREFETCH) {
            uint32_t cache_source_offset = 0U;
            for (uint32_t index = QBH_BLOCK_PROJ_GATE;
                 index <= QBH_BLOCK_PROJ_UP; ++index) {
                const struct qbh_block_projection_desc *desc =
                    &header->projections[index];
                uint32_t cache_bytes =
                    desc->n / QBH_HMX_FP16_COLS *
                    QBH_HMX_FP16_SCALE_BYTES;
                uint8_t *cache_destination =
                    buffers->gate_up_scale_cache + cache_source_offset;
                if (qbh_dma_copy(
                        header, cache_destination,
                        shared +
                            header->w4f16_gate_up_scale_cache_offset +
                            cache_source_offset,
                        cache_bytes, 1U) != 0) {
                    return -1;
                }
                cache_source_offset += cache_bytes;
                header->weight_ddr_read_bytes += cache_bytes;
                ++header->weight_dma_descriptor_count;
            }
            header->w4f16_gate_up_scale_cache_bytes =
                cache_source_offset;
        }
    }
    header->metadata_stage_ticks += HAP_perf_get_qtimer_count() - start;
    return 0;
}

static int qbh_scan_stage_rope_chunk(
    struct qbh_block_header *header, const uint8_t *shared,
    struct qbh_block_buffers *buffers, uint32_t chunk) {
    const uint32_t bytes =
        QBH_BLOCK_M * QBH_BLOCK_HEAD_DIM * sizeof(uint16_t);
    const uint32_t chunks = qbh_scan_physical_chunks(header);
    uint64_t start;

    if (header->scan_mode == QBH_BLOCK_SCAN_DISABLED ||
        chunks == 1U) {
        return 0;
    }
    if (chunk >= chunks) {
        return -1;
    }
    start = HAP_perf_get_qtimer_count();
    if (qbh_dma_copy(
            header, buffers->rope_cos,
            shared + header->rope_cos_offset + (size_t)chunk * bytes,
            bytes, 1U) != 0 ||
        qbh_dma_copy(
            header, buffers->rope_sin,
            shared + header->rope_sin_offset + (size_t)chunk * bytes,
            bytes, 1U) != 0) {
        return -1;
    }
    header->boundary_ddr_read_bytes += 2U * bytes;
    header->boundary_dma_descriptor_count += 2U;
    header->metadata_stage_ticks +=
        HAP_perf_get_qtimer_count() - start;
    return 0;
}

static int qbh_configure_w4u8_gate_up_layout(
    struct qbh_projection_layout *layout) {
    uint32_t activation_offset = qbh_align_up_u32(
        QBH_BLOCK_M * QBH_BLOCK_INTERMEDIATE,
        QBH_HMX_ACTIVATION_BYTES);
    uint32_t compressed_offset = qbh_align_up_u32(
        activation_offset + layout->activation_bytes,
        QBH_W4_METADATA_ALIGNMENT);
    uint32_t expanded_offset = qbh_align_up_u32(
        compressed_offset + layout->compressed_slot_count *
                                layout->stored_weight_bundle_bytes,
        QBH_W4_METADATA_ALIGNMENT);
    uint32_t output_offset = qbh_align_up_u32(
        expanded_offset + layout->expanded_slot_count *
                              layout->expanded_chunk_slot_bytes,
        QBH_HMX_OUTPUT_BYTES);
    uint32_t plan_bytes = output_offset +
        QBH_BLOCK_W4U8_GATE_UP_PAIR_SLOTS * 2U *
            QBH_HMX_OUTPUT_BYTES;

    layout->vtcm_activation_offset = activation_offset;
    layout->vtcm_compressed_slot0_offset = compressed_offset;
    layout->vtcm_compressed_slot1_offset =
        compressed_offset + layout->w4_bundle_bytes;
    layout->vtcm_chunked_expanded_slots_offset = expanded_offset;
    layout->vtcm_chunked_output_offset = output_offset;
    layout->vtcm_output_offset = output_offset;
    layout->vtcm_chunked_plan_bytes = plan_bytes;
    layout->vtcm_plan_bytes = plan_bytes;
    return plan_bytes <= QBH_EXPECTED_FULL_VTCM_BYTES ? 0 : -1;
}

static int qbh_init_w4u8_gate_up_layout(
    struct qbh_projection_layout *layout, uint32_t ring_slots) {
    if (ring_slots != 8U && ring_slots != 16U) {
        return -1;
    }
    if (qbh_projection_layout_init(
            QBH_PROJECTION_GATE_UP_PAIR,
            QBH_WEIGHT_PACKED_W4_HMX_SCALE,
            QBH_PHYSICAL_PLAN_STREAMING_E7_DMA_CHAIN4, ring_slots,
            QBH_W4_COARSE_CHUNK_TILES, layout) != 0) {
        return -1;
    }
    layout->expanded_slot_count = ring_slots;
    return qbh_configure_w4u8_gate_up_layout(layout);
}

static int qbh_init_w4u8_down_layout(
    struct qbh_projection_layout *layout) {
    return qbh_projection_layout_init(
        QBH_PROJECTION_DOWN,
        QBH_WEIGHT_PACKED_W4_HMX_SCALE,
        QBH_PHYSICAL_PLAN_CHUNKED_DMA_BATCH2, 4U,
        QBH_W4_WIDE_CHUNK_TILES, layout);
}

static void qbh_reset_w4u8_phase_header(
    struct qbh_probe_header *phase, uint32_t workers) {
    memset(phase, 0, sizeof(*phase));
    phase->magic = QBH_PROBE_MAGIC;
    phase->abi_version = QBH_PROBE_ABI_VERSION;
    phase->header_bytes = sizeof(*phase);
    phase->repeat_count = 1U;
    phase->requested_hvx_workers = workers;
    phase->dsp_status = QBH_PROBE_STATUS_DSP_RUNNING;
}

static void qbh_accumulate_w4u8_phase_metrics(
    struct qbh_block_header *header,
    const struct qbh_probe_header *phase,
    const struct qbh_projection_layout *layout,
    uint32_t gate_up_phase) {
    uint64_t command_count =
        gate_up_phase != 0U &&
                qbh_physical_plan_is_streaming(layout->physical_plan)
            ? (layout->n_tiles +
               QBH_BLOCK_W4U8_GATE_UP_HMX_BATCH_N_TILES - 1U) /
                  QBH_BLOCK_W4U8_GATE_UP_HMX_BATCH_N_TILES
            : gate_up_phase == 0U &&
                      layout->chunks_per_output == 2U
                  ? (layout->n_tiles +
                     header->w4u8_down_hmx_batch_outputs - 1U) /
                        header->w4u8_down_hmx_batch_outputs
                  : layout->n_tiles * layout->chunks_per_output;
    header->weight_dma_ticks += phase->weight_stage_ticks;
    header->weight_ddr_read_bytes += layout->stored_weight_bytes;
    header->weight_dma_descriptor_count += phase->dma_descriptor_count;
    header->w4u8_mlp_weight_stage_ticks += phase->weight_stage_ticks;
    header->w4u8_mlp_weight_expand_ticks += phase->weight_expand_ticks;
    header->w4u8_mlp_hmx_compute_ticks += phase->hmx_compute_ticks;
    header->w4u8_mlp_hmx_ready_wait_ticks +=
        phase->hmx_ready_wait_ticks;
    header->w4u8_mlp_producer_slot_wait_ticks +=
        phase->producer_slot_wait_ticks;
    header->w4u8_mlp_expanded_slot_wait_ticks +=
        phase->expanded_slot_wait_ticks;
    header->hmx_u8s8_tile_pair_count += layout->hmx_pairs_per_repeat;
    header->hmx_command_count += command_count;
    if (gate_up_phase != 0U) {
        header->w4u8_mlp_gate_up_pipeline_ticks += phase->pipeline_ticks;
        header->w4u8_mlp_gate_up_hmx_command_count += command_count;
        header->w4u8_mlp_gate_up_hvx_hmx_overlap |=
            phase->hvx_hmx_overlap_observed;
        header->w4u8_mlp_gate_up_hvx_parallel_overlap |=
            phase->hvx_parallel_overlap_observed;
    } else {
        header->w4u8_mlp_down_hmx_batch_n_tiles =
            header->w4u8_down_hmx_batch_outputs;
        header->w4u8_mlp_down_in_command_slot_release_count +=
            phase->hmx_in_command_slot_release_count;
        header->w4u8_mlp_down_producer_progress_command_count +=
            phase->hmx_producer_progress_command_count;
        header->w4u8_mlp_down_pipeline_ticks += phase->pipeline_ticks;
        header->w4u8_mlp_down_hmx_command_count += command_count;
        header->w4u8_mlp_down_hvx_hmx_overlap |=
            phase->hvx_hmx_overlap_observed;
        header->w4u8_mlp_down_hvx_parallel_overlap |=
            phase->hvx_parallel_overlap_observed;
    }
}

static int qbh_copy_w4u8_tail_audit(
    struct qbh_block_header *header, uint8_t *shared,
    uint32_t audit_offset, const void *source, uint32_t bytes) {
    if (header->numerical_audit_enabled == 0U) {
        return 0;
    }
    if (header->u8_attention_audit_output_bytes !=
            QBH_BLOCK_U8_ATTENTION_AUDIT_BYTES ||
        audit_offset > header->u8_attention_audit_output_bytes ||
        bytes > header->u8_attention_audit_output_bytes - audit_offset ||
        qbh_dma_copy(
            header,
            shared + header->u8_attention_audit_output_offset + audit_offset,
            source, bytes, 0U) != 0) {
        return -1;
    }
    header->u8_attention_audit_ddr_write_bytes += bytes;
    return 0;
}

static int qbh_run_w4u8_streaming_mlp(
    struct qbh_block_header *header, uint8_t *shared,
    struct qbh_block_buffers *buffers,
    struct qbh_block_hmx_worker *worker,
    struct qbh_block_w4f16_pool *hvx_pool,
    uint32_t activation_prepacked,
    uint32_t output_native) {
    struct qbh_projection_layout gate_up_layout;
    struct qbh_projection_layout down_layout;
    struct qbh_probe_header gate_up_phase;
    struct qbh_probe_header down_phase;
    struct qbh_w4_hmx_runner runner = {
        .context = worker,
        .max_batch_outputs =
            QBH_BLOCK_W4U8_GATE_UP_HMX_BATCH_N_TILES,
        .max_nonstreaming_batch_outputs =
            header->w4u8_down_hmx_batch_outputs,
        .max_chunks_per_command = 2U,
        .submit = qbh_block_w4_hmx_submit,
    };
    struct qbh_w4_hvx_dispatch_runner hvx_runner = {
        .context = hvx_pool,
        .max_workers = hvx_pool != NULL ? hvx_pool->worker_count : 0U,
        .start = qbh_w4u8_pipeline_pool_start,
        .wait = qbh_w4u8_pipeline_pool_wait,
    };
    struct qbh_block_w4u8_hybrid_hvx_runner hybrid_down_context = {
        .pool = hvx_pool,
        .transient_thread_created = 0U,
    };
    struct qbh_w4_hvx_dispatch_runner hybrid_down_runner = {
        .context = &hybrid_down_context,
        .max_workers = QBH_BLOCK_W4U8_DOWN_HVX_WORKERS,
        .start = qbh_w4u8_hybrid_down_runner_start,
        .wait = qbh_w4u8_hybrid_down_runner_wait,
    };
    uint8_t *mlp_arena = buffers->q;
    uint32_t pair_publish_count = 0U;
    uint32_t pair_consume_count = 0U;
    uint64_t activation_work_ticks = 0U;
    uint64_t unpack_ticks = 0U;
    uint64_t start;
    int result = -1;
    int unlock_status;
    int relock_status;

    if (header->variant != QBH_BLOCK_W4U8 ||
        !qbh_block_mlp_is_w4u8_streaming(header->mlp_mode) ||
        (qbh_block_mlp_uses_persistent_gate_up_hvx(
             header->mlp_mode) &&
         (hvx_pool == NULL ||
          hvx_pool->worker_count <
              QBH_BLOCK_W4U8_GATE_UP_HVX_WORKERS)) ||
        qbh_init_w4u8_gate_up_layout(
            &gate_up_layout, header->w4u8_gate_up_ring_slots) != 0 ||
        qbh_init_w4u8_down_layout(&down_layout) != 0 ||
        header->w4u8_gate_up_bundle_bytes !=
            gate_up_layout.stored_weight_bytes ||
        header->w4u8_down_bundle_bytes !=
            down_layout.stored_weight_bytes) {
        return -1;
    }
    header->w4u8_mlp_vtcm_base_offset =
        (uint32_t)((uintptr_t)mlp_arena -
                   (uintptr_t)header->resource_vtcm_address);
    header->w4u8_mlp_vtcm_plan_bytes = gate_up_layout.vtcm_plan_bytes;
    if (header->w4u8_mlp_vtcm_plan_bytes < down_layout.vtcm_plan_bytes) {
        header->w4u8_mlp_vtcm_plan_bytes = down_layout.vtcm_plan_bytes;
    }
    header->w4u8_mlp_gate_up_hvx_workers =
        QBH_BLOCK_W4U8_GATE_UP_HVX_WORKERS;
    header->w4u8_mlp_down_hvx_workers =
        QBH_BLOCK_W4U8_DOWN_HVX_WORKERS;
    header->w4u8_mlp_gate_up_hmx_batch_n_tiles =
        QBH_BLOCK_W4U8_GATE_UP_HMX_BATCH_N_TILES;
    header->w4u8_mlp_gate_up_expanded_slot_count =
        gate_up_layout.expanded_slot_count;

    if (activation_prepacked != 0U) {
        ++header->w4u8_mlp_input_pack_skipped;
    } else {
        uint64_t pack_ticks;
        start = HAP_perf_get_qtimer_count();
        qbh_pack_u8_activation(
            buffers->normalized, QBH_BLOCK_HIDDEN, QBH_BLOCK_HIDDEN,
            mlp_arena + gate_up_layout.vtcm_activation_offset);
        pack_ticks = HAP_perf_get_qtimer_count() - start;
        header->projection_pack_ticks += pack_ticks;
        header->w4u8_mlp_input_pack_ticks += pack_ticks;
    }

    unlock_status = qurt_hvx_unlock();
    if (unlock_status != AEE_SUCCESS) {
        return -1;
    }

    {
        struct qbh_mlp_gate_up_handoff handoff = {
            .middle_activation = mlp_arena,
            .activation_lut =
                (const uint16_t *)buffers->w4u8_silu_lut,
            .output_multipliers = NULL,
            .activation_gather_scratch =
                buffers->w4u8_gather_scratch,
            .pair_slot_count = QBH_BLOCK_W4U8_GATE_UP_PAIR_SLOTS,
            .pair_publish_count = &pair_publish_count,
            .pair_consume_count = &pair_consume_count,
            .activation_ticks = &activation_work_ticks,
            .stream_fence_mode = header->w4u8_stream_fence_mode,
        };
        qbh_reset_w4u8_phase_header(
            &gate_up_phase, QBH_BLOCK_W4U8_GATE_UP_HVX_WORKERS);
        start = HAP_perf_get_qtimer_count();
        if (qbh_block_mlp_uses_persistent_gate_up_hvx(
                header->mlp_mode)) {
            result = qbh_run_chunked_w4_pipeline_external_hvx(
                &gate_up_phase, &gate_up_layout,
                shared + header->w4u8_gate_up_bundle_offset,
                mlp_arena + gate_up_layout.vtcm_activation_offset,
                mlp_arena, &handoff, &runner, &hvx_runner);
            ++header->w4u8_gate_up_persistent_hvx_dispatch_count;
            header->w4u8_gate_up_persistent_hvx_worker_count +=
                QBH_BLOCK_W4U8_GATE_UP_HVX_WORKERS;
        } else {
            result = qbh_run_chunked_w4_pipeline_external(
                &gate_up_phase, &gate_up_layout,
                shared + header->w4u8_gate_up_bundle_offset,
                mlp_arena + gate_up_layout.vtcm_activation_offset,
                mlp_arena, &handoff, &runner);
            header->w4u8_gate_up_transient_hvx_thread_count +=
                QBH_BLOCK_W4U8_GATE_UP_HVX_WORKERS;
        }
        header->gate_up_ticks += HAP_perf_get_qtimer_count() - start;
        header->w4u8_mlp_activation_work_ticks +=
            activation_work_ticks;
        header->w4u8_mlp_pair_publish_count += pair_publish_count;
        header->w4u8_mlp_pair_consume_count += pair_consume_count;
        if (result != AEE_SUCCESS ||
            pair_publish_count != gate_up_layout.n_tiles / 2U ||
            pair_consume_count != pair_publish_count) {
            result = -1;
            goto relock;
        }
        if (qbh_copy_w4u8_tail_audit(
                header, shared, QBH_BLOCK_U8_TAIL_MIDDLE_OFFSET,
                mlp_arena,
                QBH_BLOCK_M * QBH_BLOCK_INTERMEDIATE) != 0) {
            result = -1;
            goto relock;
        }
        qbh_accumulate_w4u8_phase_metrics(
            header, &gate_up_phase, &gate_up_layout, 1U);
    }

    qbh_reset_w4u8_phase_header(
        &down_phase, QBH_BLOCK_W4U8_DOWN_HVX_WORKERS);
    start = HAP_perf_get_qtimer_count();
    if (header->mlp_mode ==
        QBH_BLOCK_MLP_W4U8_STREAMING_PERSISTENT_MLP_HVX) {
        result = qbh_run_chunked_w4_pipeline_external_hvx(
            &down_phase, &down_layout,
            shared + header->w4u8_down_bundle_offset,
            mlp_arena, mlp_arena, NULL, &runner,
            &hybrid_down_runner);
        ++header->w4u8_down_persistent_hvx_dispatch_count;
        header->w4u8_down_persistent_hvx_worker_count +=
            QBH_BLOCK_W4U8_DOWN_PERSISTENT_HVX_WORKERS;
        ++header->w4u8_down_transient_hvx_thread_count;
    } else {
        result = qbh_run_chunked_w4_pipeline_external(
            &down_phase, &down_layout,
            shared + header->w4u8_down_bundle_offset,
            mlp_arena, mlp_arena, NULL, &runner);
        header->w4u8_down_transient_hvx_thread_count +=
            QBH_BLOCK_W4U8_DOWN_HVX_WORKERS;
    }
    header->down_ticks += HAP_perf_get_qtimer_count() - start;
    if (result != AEE_SUCCESS) {
        result = -1;
        goto relock;
    }
    if (qbh_copy_w4u8_tail_audit(
            header, shared, QBH_BLOCK_U8_TAIL_DOWN_OFFSET,
            mlp_arena + down_layout.vtcm_output_offset,
            QBH_BLOCK_M * QBH_BLOCK_HIDDEN) != 0) {
        result = -1;
        goto relock;
    }
    qbh_accumulate_w4u8_phase_metrics(
        header, &down_phase, &down_layout, 0U);

    if (output_native != 0U) {
        ++header->w4u8_mlp_output_unpack_skipped;
    } else {
        start = HAP_perf_get_qtimer_count();
        for (uint32_t n_tile = 0U; n_tile < down_layout.n_tiles;
             ++n_tile) {
            qbh_unpack_u8_output(
                mlp_arena + down_layout.vtcm_output_offset +
                    (size_t)n_tile * QBH_HMX_OUTPUT_BYTES,
                buffers->down, QBH_BLOCK_HIDDEN,
                n_tile * QBH_HMX_OUTPUT_CHANNELS);
        }
        unpack_ticks = HAP_perf_get_qtimer_count() - start;
        header->projection_unpack_ticks += unpack_ticks;
        header->w4u8_mlp_output_unpack_ticks += unpack_ticks;
    }
    result = 0;

relock:
    relock_status = qurt_hvx_lock(QURT_HVX_MODE_128B);
    if (relock_status != AEE_SUCCESS) {
        return -1;
    }
    return result;
}

static void qbh_record_f16_nonfinite(struct qbh_block_header *header,
                                     const void *data,
                                     uint32_t elements,
                                     int32_t stage) {
    const uint16_t *bits = (const uint16_t *)data;
    if (header->numerical_audit_enabled == 0U ||
        header->numerical_status != QBH_BLOCK_NUMERICAL_UNCHECKED) {
        return;
    }
    for (uint32_t index = 0; index < elements; ++index) {
        if ((bits[index] & UINT16_C(0x7c00)) == UINT16_C(0x7c00)) {
            header->numerical_status = stage;
            return;
        }
    }
}

static int qbh_scan_cache_dma(struct qbh_block_header *header,
                              void *destination, const void *source,
                              uint32_t bytes, uint32_t ddr_to_vtcm) {
    const uint64_t weight_ticks_before = header->weight_dma_ticks;
    const uint64_t start = HAP_perf_get_qtimer_count();
    int result = qbh_dma_copy(
        header, destination, source, bytes, ddr_to_vtcm);
    const uint64_t elapsed = HAP_perf_get_qtimer_count() - start;

    /* Cache is a declared boundary, not projection weight traffic. */
    header->weight_dma_ticks = weight_ticks_before;
    if (result != 0) {
        return result;
    }
    ++header->scan_cache_dma_descriptor_count;
    if (ddr_to_vtcm != 0U) {
        header->scan_cache_ddr_read_bytes += bytes;
        header->scan_cache_stage_ticks += elapsed;
    } else {
        header->scan_cache_ddr_write_bytes += bytes;
        header->scan_cache_append_ticks += elapsed;
    }
    return 0;
}

static int qbh_scan_cache_dma_2d(
    struct qbh_block_header *header, void *destination,
    const void *source, uint32_t row_bytes, uint32_t rows,
    uint32_t source_stride, uint32_t destination_stride) {
    struct qbh_dma_aligned_desc_2d aligned;
    struct qbh_dma_desc_2d *descriptor = &aligned.descriptor;
    const uint64_t start = HAP_perf_get_qtimer_count();

    if (destination == NULL || source == NULL || row_bytes == 0U ||
        rows == 0U || row_bytes > UINT16_MAX || rows > UINT16_MAX ||
        source_stride > UINT16_MAX || destination_stride > UINT16_MAX) {
        return -1;
    }
    if (qbh_dma_wait_idle() != 0) {
        return -2;
    }
    memset(&aligned, 0, sizeof(aligned));
    descriptor->type = QBH_DMA_TYPE_2D;
    descriptor->src_bypass = 1;
    descriptor->dst_bypass = 0;
    descriptor->ordered = 1;
    descriptor->dstate = QBH_DMA_DESC_PENDING;
    descriptor->src = (uint32_t)(uintptr_t)source;
    descriptor->dst = (uint32_t)(uintptr_t)destination;
    descriptor->roi_width = (uint16_t)row_bytes;
    descriptor->roi_height = (uint16_t)rows;
    descriptor->src_stride = (uint16_t)source_stride;
    descriptor->dst_stride = (uint16_t)destination_stride;
    if (qbh_dma_start(descriptor) != 0) {
        return -3;
    }
    if (qbh_dma_wait_idle() != 0) {
        return -4;
    }
    if (descriptor->dstate != QBH_DMA_DESC_COMPLETE) {
        return -5;
    }
    ++header->scan_cache_dma_descriptor_count;
    header->scan_cache_ddr_read_bytes +=
        (uint64_t)row_bytes * rows;
    header->scan_cache_stage_ticks +=
        HAP_perf_get_qtimer_count() - start;
    return 0;
}

static void qbh_hvx_copy_aligned_bytes(
    void *destination, const void *source, uint32_t bytes) {
    HVX_Vector *output = (HVX_Vector *)destination;
    const HVX_Vector *input = (const HVX_Vector *)source;
    const uint32_t vectors = bytes / sizeof(HVX_Vector);

    for (uint32_t index = 0U; index < vectors; ++index) {
        output[index] = input[index];
    }
    asm volatile("barrier" ::: "memory");
}

static void qbh_hvx_zero_aligned_bytes(void *destination,
                                       uint32_t bytes) {
    HVX_Vector *output = (HVX_Vector *)destination;
    const uint32_t vectors = bytes / sizeof(HVX_Vector);
    const HVX_Vector zero = Q6_V_vzero();

    for (uint32_t index = 0U; index < vectors; ++index) {
        output[index] = zero;
    }
    asm volatile("barrier" ::: "memory");
}

static void qbh_hvx_zero_strided_bytes(
    void *destination, uint32_t row_bytes, uint32_t rows,
    uint32_t destination_stride) {
    const HVX_Vector zero = Q6_V_vzero();
    const uint32_t vectors = row_bytes / sizeof(HVX_Vector);

    for (uint32_t row = 0U; row < rows; ++row) {
        HVX_Vector *output = (HVX_Vector *)(
            (uint8_t *)destination + (size_t)row * destination_stride);
        for (uint32_t index = 0U; index < vectors; ++index) {
            output[index] = zero;
        }
    }
    asm volatile("barrier" ::: "memory");
}

static int qbh_scan_append_u8_kv_row_major(
    struct qbh_block_header *header, uint8_t *shared,
    struct qbh_block_buffers *buffers, uint32_t logical_rows,
    uint32_t past_tokens) {
    uint8_t *row_scratch = buffers->attention_projection;
    const uint32_t append_bytes =
        logical_rows * QBH_BLOCK_HEAD_DIM;
    const uint32_t head_stride =
        header->kv_cache_capacity * QBH_BLOCK_HEAD_DIM;

    for (uint32_t head = 0U; head < QBH_BLOCK_KV_HEADS; ++head) {
        const uint8_t *k_head = buffers->k +
            (size_t)head * QBH_ATTENTION_HEAD_DIM_TILES *
                QBH_HMX_ACTIVATION_BYTES;
        const uint8_t *v_head = buffers->v +
            (size_t)head * QBH_ATTENTION_HEAD_DIM_TILES *
                QBH_HMX_ACTIVATION_BYTES;
        uint8_t *k_destination = shared + header->kv_cache_k_offset +
            (size_t)head * head_stride +
            (size_t)past_tokens * QBH_BLOCK_HEAD_DIM;
        uint8_t *v_destination = shared + header->kv_cache_v_offset +
            (size_t)head * head_stride +
            (size_t)past_tokens * QBH_BLOCK_HEAD_DIM;

        qbh_attention_u8_native_head_to_row_major(
            k_head, row_scratch, logical_rows);
        if (qbh_scan_cache_dma(
                header, k_destination, row_scratch,
                append_bytes, 0U) != 0) {
            return -1;
        }
        qbh_attention_u8_native_head_to_row_major(
            v_head, row_scratch, logical_rows);
        if (qbh_scan_cache_dma(
                header, v_destination, row_scratch,
                append_bytes, 0U) != 0) {
            return -1;
        }
    }
    return 0;
}

static int qbh_scan_append_u8_kv_hmx_native(
    struct qbh_block_header *header, uint8_t *shared,
    struct qbh_block_buffers *buffers, uint32_t logical_rows,
    uint32_t past_tokens) {
    const uint32_t padded_capacity =
        QBH_KV_CACHE_HMX_PADDED_CAPACITY(header->kv_cache_capacity);
    const uint32_t weight_bytes =
        QBH_KV_CACHE_HMX_WEIGHT_BYTES_PER_HEAD(
            header->kv_cache_capacity);
    const uint32_t k_bias_bytes =
        QBH_KV_CACHE_HMX_K_BIAS_BYTES_PER_HEAD(
            header->kv_cache_capacity);
    const uint32_t k_head_bytes =
        QBH_KV_CACHE_HMX_K_HEAD_BYTES(header->kv_cache_capacity);
    const uint32_t v_head_bytes =
        QBH_KV_CACHE_HMX_V_HEAD_BYTES(header->kv_cache_capacity);
    uint8_t *scratch = buffers->attention_projection;
    const uint64_t update_start = HAP_perf_get_qtimer_count();

    if (header->kv_cache_padded_capacity != padded_capacity ||
        header->kv_cache_k_bytes !=
            QBH_KV_CACHE_HMX_K_BYTES(header->kv_cache_capacity) ||
        header->kv_cache_v_bytes !=
            QBH_KV_CACHE_HMX_V_BYTES(header->kv_cache_capacity)) {
        return -1;
    }

    if (past_tokens == 0U && logical_rows == QBH_BLOCK_M) {
        const uint32_t native_weight_bytes =
            QBH_ATTENTION_SCORE_TILES *
            QBH_ATTENTION_HEAD_DIM_TILES * QBH_HMX_WEIGHT_BYTES;
        const uint32_t capacity_k_tiles =
            padded_capacity / QBH_HMX_INPUT_CHANNELS;
        int8_t *packed = (int8_t *)scratch;
        int8_t *native_v_weight = (int8_t *)(scratch + weight_bytes);
        uint32_t *bias = (uint32_t *)(
            scratch + 2U * weight_bytes);
        uint8_t *pack_scratch =
            (uint8_t *)bias + k_bias_bytes;

        for (uint32_t head = 0U;
             head < QBH_BLOCK_KV_HEADS; ++head) {
            const struct qbh_attention_config *config =
                &buffers->attention_configs[head];
            const uint8_t *k_head = buffers->k +
                (size_t)head * QBH_ATTENTION_HEAD_DIM_TILES *
                    QBH_HMX_ACTIVATION_BYTES;
            const uint8_t *v_head = buffers->v +
                (size_t)head * QBH_ATTENTION_HEAD_DIM_TILES *
                    QBH_HMX_ACTIVATION_BYTES;
            uint8_t *k_destination =
                shared + header->kv_cache_k_offset +
                (size_t)head * k_head_bytes;
            uint8_t *v_destination =
                shared + header->kv_cache_v_offset +
                (size_t)head * v_head_bytes;

            memset(packed, 0, weight_bytes);
            memset(bias, 0, k_bias_bytes);
            qbh_attention_u8_pack_k_native(
                k_head, config, packed, bias);
            {
                const uint32_t divisor =
                    UINT32_C(1) << config->score_shift;
                const int32_t rounding = config->score_shift == 0U
                                             ? 0
                                             : (int32_t)(divisor / 2U);
                const __fp16 conversion_value =
                    (__fp16)(512.0f / (float)divisor);
                uint16_t conversion_bits;
                memcpy(&conversion_bits, &conversion_value,
                       sizeof(conversion_bits));
                for (uint32_t n_tile = QBH_ATTENTION_SCORE_TILES;
                     n_tile < capacity_k_tiles; ++n_tile) {
                    uint32_t *tile_bias = bias +
                        (size_t)n_tile *
                            (QBH_HMX_BIAS_BYTES / sizeof(uint32_t));
                    for (uint32_t output = 0U;
                         output < QBH_HMX_OUTPUT_CHANNELS; ++output) {
                        tile_bias[output] = conversion_bits;
                        tile_bias[QBH_HMX_OUTPUT_CHANNELS + output] =
                            UINT32_C(128) * divisor +
                            rounding;
                    }
                }
            }
            if (qbh_scan_cache_dma(
                    header, k_destination, packed,
                    weight_bytes, 0U) != 0 ||
                qbh_scan_cache_dma(
                    header, k_destination + weight_bytes, bias,
                    k_bias_bytes, 0U) != 0) {
                return -1;
            }

            memset(packed, 0, weight_bytes);
            qbh_attention_u8_pack_v_native_vgather_vdeal(
                v_head, config, native_v_weight, bias, pack_scratch,
                header->numerical_audit_enabled != 0U
                    ? &header->u8_attention_v_recenter_saturation_count
                    : NULL);
            for (uint32_t output_tile = 0U;
                 output_tile < QBH_ATTENTION_HEAD_DIM_TILES;
                 ++output_tile) {
                memcpy(
                    packed + (size_t)output_tile *
                        capacity_k_tiles * QBH_HMX_WEIGHT_BYTES,
                    native_v_weight + (size_t)output_tile *
                        QBH_ATTENTION_SCORE_TILES *
                            QBH_HMX_WEIGHT_BYTES,
                    native_weight_bytes /
                        QBH_ATTENTION_HEAD_DIM_TILES);
            }
            if (qbh_scan_cache_dma(
                    header, v_destination, packed,
                    weight_bytes, 0U) != 0 ||
                qbh_scan_cache_dma(
                    header, v_destination + weight_bytes, bias,
                    QBH_KV_CACHE_HMX_V_BIAS_BYTES_PER_HEAD,
                    0U) != 0) {
                return -1;
            }
        }
        ++header->u8_cache_native_prefill_build_count;
    } else if (logical_rows == 1U &&
               past_tokens < header->kv_cache_capacity) {
        const uint32_t n_tile =
            past_tokens / QBH_HMX_OUTPUT_CHANNELS;
        const uint32_t output_lane =
            past_tokens % QBH_HMX_OUTPUT_CHANNELS;
        const uint32_t k_tile =
            past_tokens / QBH_HMX_INPUT_CHANNELS;
        const uint32_t input_lane =
            past_tokens % QBH_HMX_INPUT_CHANNELS;
        const uint32_t k_tile_weight_bytes =
            QBH_ATTENTION_HEAD_DIM_TILES * QBH_HMX_WEIGHT_BYTES;
        const uint32_t capacity_k_tiles =
            padded_capacity / QBH_HMX_INPUT_CHANNELS;

        for (uint32_t head = 0U;
             head < QBH_BLOCK_KV_HEADS; ++head) {
            const struct qbh_attention_config *config =
                &buffers->attention_configs[head];
            const uint8_t *k_head = buffers->k +
                (size_t)head * QBH_ATTENTION_HEAD_DIM_TILES *
                    QBH_HMX_ACTIVATION_BYTES;
            const uint8_t *v_head = buffers->v +
                (size_t)head * QBH_ATTENTION_HEAD_DIM_TILES *
                    QBH_HMX_ACTIVATION_BYTES;
            uint8_t *k_cache = shared + header->kv_cache_k_offset +
                (size_t)head * k_head_bytes;
            uint8_t *v_cache = shared + header->kv_cache_v_offset +
                (size_t)head * v_head_bytes;
            int8_t *k_weight = (int8_t *)scratch;
            uint32_t *k_bias = (uint32_t *)(
                scratch + k_tile_weight_bytes);

            if (qbh_scan_cache_dma(
                    header, k_weight,
                    k_cache + (size_t)n_tile *
                        k_tile_weight_bytes,
                    k_tile_weight_bytes, 1U) != 0 ||
                qbh_scan_cache_dma(
                    header, k_bias,
                    k_cache + weight_bytes +
                        (size_t)n_tile * QBH_HMX_BIAS_BYTES,
                    QBH_HMX_BIAS_BYTES, 1U) != 0) {
                return -1;
            }
            qbh_attention_u8_update_k_native_token(
                k_head, 0U, output_lane, config,
                k_weight, k_bias);
            if (qbh_scan_cache_dma(
                    header,
                    k_cache + (size_t)n_tile *
                        k_tile_weight_bytes,
                    k_weight, k_tile_weight_bytes, 0U) != 0 ||
                qbh_scan_cache_dma(
                    header,
                    k_cache + weight_bytes +
                        (size_t)n_tile * QBH_HMX_BIAS_BYTES,
                    k_bias, QBH_HMX_BIAS_BYTES, 0U) != 0) {
                return -1;
            }

            for (uint32_t output_tile = 0U;
                 output_tile < QBH_ATTENTION_HEAD_DIM_TILES;
                 ++output_tile) {
                if (qbh_scan_cache_dma(
                        header,
                        scratch + (size_t)output_tile *
                            QBH_HMX_WEIGHT_BYTES,
                        v_cache +
                            ((size_t)output_tile * capacity_k_tiles +
                             k_tile) * QBH_HMX_WEIGHT_BYTES,
                        QBH_HMX_WEIGHT_BYTES, 1U) != 0) {
                    return -1;
                }
            }
            qbh_attention_u8_update_v_native_token(
                v_head, 0U, input_lane, config,
                (int8_t *)scratch,
                &header->u8_attention_v_recenter_saturation_count);
            for (uint32_t output_tile = 0U;
                 output_tile < QBH_ATTENTION_HEAD_DIM_TILES;
                 ++output_tile) {
                if (qbh_scan_cache_dma(
                        header,
                        v_cache +
                            ((size_t)output_tile * capacity_k_tiles +
                             k_tile) * QBH_HMX_WEIGHT_BYTES,
                        scratch + (size_t)output_tile *
                            QBH_HMX_WEIGHT_BYTES,
                        QBH_HMX_WEIGHT_BYTES, 0U) != 0) {
                    return -1;
                }
            }
        }
        ++header->u8_cache_native_incremental_append_count;
    } else {
        return -1;
    }

    header->u8_cache_native_append_update_ticks +=
        HAP_perf_get_qtimer_count() - update_start;
    return 0;
}

static int qbh_scan_append_u8_kv_hmx_delta(
    struct qbh_block_header *header, uint8_t *shared,
    struct qbh_block_buffers *buffers, uint32_t logical_rows,
    uint32_t past_tokens) {
    const uint32_t weight_bytes =
        QBH_KV_CACHE_HMX_U8_BASE_WEIGHT_BYTES_PER_HEAD;
    const uint32_t k_bias_bytes =
        QBH_KV_CACHE_HMX_U8_K_BASE_BIAS_BYTES_PER_HEAD;
    const uint32_t k_head_bytes =
        QBH_KV_CACHE_HMX_U8_K_DELTA_HEAD_BYTES(
            header->kv_cache_capacity);
    const uint32_t v_head_bytes =
        QBH_KV_CACHE_HMX_U8_V_DELTA_HEAD_BYTES(
            header->kv_cache_capacity);
    const uint32_t delta_row = past_tokens - QBH_BLOCK_M;
    uint8_t *row = buffers->attention_projection;
    const uint64_t update_start = HAP_perf_get_qtimer_count();

    if (!qbh_hmx_native_u8_delta_cache_formats(
            header->kv_cache_k_format,
            header->kv_cache_v_format) ||
        logical_rows != 1U || past_tokens < QBH_BLOCK_M ||
        past_tokens >= header->kv_cache_capacity ||
        header->kv_cache_k_bytes !=
            QBH_KV_CACHE_HMX_U8_K_DELTA_BYTES(
                header->kv_cache_capacity) ||
        header->kv_cache_v_bytes !=
            QBH_KV_CACHE_HMX_U8_V_DELTA_BYTES(
                header->kv_cache_capacity)) {
        return -1;
    }

    for (uint32_t head = 0U;
         head < QBH_BLOCK_KV_HEADS; ++head) {
        const uint8_t *k_head = buffers->k +
            (size_t)head * QBH_ATTENTION_HEAD_DIM_TILES *
                QBH_HMX_ACTIVATION_BYTES;
        const uint8_t *v_head = buffers->v +
            (size_t)head * QBH_ATTENTION_HEAD_DIM_TILES *
                QBH_HMX_ACTIVATION_BYTES;
        uint8_t *k_delta = shared + header->kv_cache_k_offset +
            (size_t)head * k_head_bytes + weight_bytes +
            k_bias_bytes +
            (size_t)delta_row * QBH_BLOCK_HEAD_DIM;
        uint8_t *v_delta = shared + header->kv_cache_v_offset +
            (size_t)head * v_head_bytes + weight_bytes +
            QBH_KV_CACHE_HMX_V_BIAS_BYTES_PER_HEAD +
            (size_t)delta_row * QBH_BLOCK_HEAD_DIM;

        qbh_attention_u8_native_head_to_row_major(k_head, row, 1U);
        if (qbh_scan_cache_dma(
                header, k_delta, row,
                QBH_BLOCK_HEAD_DIM, 0U) != 0) {
            return -1;
        }
        qbh_attention_u8_native_head_to_row_major(v_head, row, 1U);
        if (qbh_scan_cache_dma(
                header, v_delta, row,
                QBH_BLOCK_HEAD_DIM, 0U) != 0) {
            return -1;
        }
    }
    ++header->u8_cache_native_incremental_append_count;
    header->u8_cache_native_append_update_ticks +=
        HAP_perf_get_qtimer_count() - update_start;
    return 0;
}

static int qbh_scan_append_u8_kv_hmx_segmented(
    struct qbh_block_header *header, uint8_t *shared,
    struct qbh_block_buffers *buffers, uint32_t logical_rows,
    uint32_t past_tokens) {
    const uint32_t max_segments =
        QBH_KV_CACHE_HMX_U8_SEGMENT_COUNT(header->kv_cache_capacity);
    const uint32_t sealed_segments_before =
        past_tokens / QBH_KV_CACHE_HMX_U8_SEGMENT_TOKENS;
    const uint32_t active_row =
        past_tokens % QBH_KV_CACHE_HMX_U8_SEGMENT_TOKENS;
    const uint32_t k_head_bytes =
        QBH_KV_CACHE_HMX_U8_K_SEGMENTED_HEAD_BYTES(
            header->kv_cache_capacity);
    const uint32_t v_head_bytes =
        QBH_KV_CACHE_HMX_U8_V_SEGMENTED_HEAD_BYTES(
            header->kv_cache_capacity);
    const uint32_t k_tail_offset =
        max_segments * QBH_KV_CACHE_HMX_U8_SEGMENT_K_BYTES;
    const uint32_t v_tail_offset =
        max_segments * QBH_KV_CACHE_HMX_U8_SEGMENT_V_BYTES +
        QBH_KV_CACHE_HMX_V_BIAS_BYTES_PER_HEAD;
    uint8_t *row = buffers->attention_projection;
    uint8_t *packed = buffers->gate;
    uint8_t *v_lut_scratch = packed +
        QBH_KV_CACHE_HMX_U8_SEGMENT_K_BYTES;
    int32_t lut_v_zero_point = 0;
    uint32_t lut_v_numerator = 0U;
    uint32_t lut_v_denominator = 0U;
    int v_lut_valid = 0;
    const uint64_t update_start = HAP_perf_get_qtimer_count();

    if (!qbh_hmx_native_u8_segmented_cache_formats(
            header->kv_cache_k_format,
            header->kv_cache_v_format) ||
        logical_rows != 1U ||
        sealed_segments_before >= max_segments + 1U ||
        past_tokens >= header->kv_cache_capacity ||
        header->kv_cache_k_bytes !=
            QBH_KV_CACHE_HMX_U8_K_SEGMENTED_BYTES(
                header->kv_cache_capacity) ||
        header->kv_cache_v_bytes !=
            QBH_KV_CACHE_HMX_U8_V_SEGMENTED_BYTES(
                header->kv_cache_capacity)) {
        return -1;
    }

    for (uint32_t head = 0U;
         head < QBH_BLOCK_KV_HEADS; ++head) {
        const uint8_t *k_head = buffers->k +
            (size_t)head * QBH_ATTENTION_HEAD_DIM_TILES *
                QBH_HMX_ACTIVATION_BYTES;
        const uint8_t *v_head = buffers->v +
            (size_t)head * QBH_ATTENTION_HEAD_DIM_TILES *
                QBH_HMX_ACTIVATION_BYTES;
        uint8_t *k_tail = shared + header->kv_cache_k_offset +
            (size_t)head * k_head_bytes + k_tail_offset +
            (size_t)active_row * QBH_BLOCK_HEAD_DIM;
        uint8_t *v_tail = shared + header->kv_cache_v_offset +
            (size_t)head * v_head_bytes + v_tail_offset +
            (size_t)active_row * QBH_BLOCK_HEAD_DIM;

        qbh_attention_u8_native_head_to_row_major(k_head, row, 1U);
        if (qbh_scan_cache_dma(
                header, k_tail, row, QBH_BLOCK_HEAD_DIM, 0U) != 0) {
            return -1;
        }
        qbh_attention_u8_native_head_to_row_major(v_head, row, 1U);
        if (qbh_scan_cache_dma(
                header, v_tail, row, QBH_BLOCK_HEAD_DIM, 0U) != 0) {
            return -1;
        }

        if (active_row + 1U ==
                QBH_KV_CACHE_HMX_U8_SEGMENT_TOKENS) {
            const uint32_t segment = sealed_segments_before;
            const uint32_t block_first =
                (segment /
                 QBH_KV_CACHE_HMX_U8_V_SEGMENT_BLOCK_SEGMENTS) *
                QBH_KV_CACHE_HMX_U8_V_SEGMENT_BLOCK_SEGMENTS;
            const uint32_t block_count =
                max_segments - block_first <
                        QBH_KV_CACHE_HMX_U8_V_SEGMENT_BLOCK_SEGMENTS
                    ? max_segments - block_first
                    : QBH_KV_CACHE_HMX_U8_V_SEGMENT_BLOCK_SEGMENTS;

            if (segment >= max_segments || block_count == 0U ||
                qbh_scan_cache_dma(
                    header, row,
                    shared + header->kv_cache_k_offset +
                        (size_t)head * k_head_bytes + k_tail_offset,
                    QBH_KV_CACHE_HMX_U8_SEGMENT_TAIL_BYTES, 1U) != 0) {
                return -1;
            }
            /* The original EXP-0162 seal path called the scalar full-matrix
             * packer here.  That made one 32-token seal dominate an entire
             * decode window.  The tail is already row-major and aligned, so
             * use the same HVX scatter kernel as the bounded-tail Attention
             * consumer.  It writes all 32 output lanes and their bias words
             * exactly once, preserving the physical segment ABI. */
            qbh_attention_u8_patch_k_delta_rows_hvx(
                row, QBH_KV_CACHE_HMX_U8_SEGMENT_TOKENS,
                &buffers->attention_configs[head],
                (int8_t *)packed,
                (uint32_t *)(packed +
                    QBH_KV_CACHE_HMX_U8_SEGMENT_WEIGHT_BYTES));
            if (qbh_scan_cache_dma(
                    header,
                    shared + header->kv_cache_k_offset +
                        (size_t)head * k_head_bytes +
                        (size_t)segment *
                            QBH_KV_CACHE_HMX_U8_SEGMENT_K_BYTES,
                    packed, QBH_KV_CACHE_HMX_U8_SEGMENT_K_BYTES,
                    0U) != 0 ||
                qbh_scan_cache_dma(
                    header, row,
                    shared + header->kv_cache_v_offset +
                        (size_t)head * v_head_bytes + v_tail_offset,
                    QBH_KV_CACHE_HMX_U8_SEGMENT_TAIL_BYTES, 1U) != 0) {
                return -1;
            }
            if (!v_lut_valid ||
                lut_v_zero_point !=
                    buffers->attention_configs[head].v_zero_point ||
                lut_v_numerator !=
                    buffers->attention_configs[head].v_recenter_numerator ||
                lut_v_denominator !=
                    buffers->attention_configs[head].v_recenter_denominator) {
                qbh_attention_u8_prepare_v_delta_lut(
                    &buffers->attention_configs[head], v_lut_scratch);
                lut_v_zero_point =
                    buffers->attention_configs[head].v_zero_point;
                lut_v_numerator =
                    buffers->attention_configs[head].v_recenter_numerator;
                lut_v_denominator =
                    buffers->attention_configs[head].v_recenter_denominator;
                v_lut_valid = 1;
            }
            qbh_attention_u8_patch_v_delta_rows_hvx(
                row, QBH_KV_CACHE_HMX_U8_SEGMENT_TOKENS,
                &buffers->attention_configs[head],
                (int8_t *)packed, QBH_HMX_WEIGHT_BYTES,
                v_lut_scratch,
                header->numerical_audit_enabled != 0U
                    ? &header->u8_attention_v_recenter_saturation_count
                    : NULL);
            for (uint32_t output_tile = 0U;
                 output_tile < QBH_ATTENTION_HEAD_DIM_TILES;
                 ++output_tile) {
                uint8_t *v_segment_destination =
                    shared + header->kv_cache_v_offset +
                    (size_t)head * v_head_bytes +
                    (size_t)block_first *
                        QBH_KV_CACHE_HMX_U8_SEGMENT_WEIGHT_BYTES +
                    ((size_t)output_tile * block_count +
                     (segment - block_first)) * QBH_HMX_WEIGHT_BYTES;
                if (qbh_scan_cache_dma(
                        header, v_segment_destination,
                        packed +
                            (size_t)output_tile * QBH_HMX_WEIGHT_BYTES,
                        QBH_HMX_WEIGHT_BYTES, 0U) != 0) {
                    return -1;
                }
            }
        }
    }
    ++header->u8_cache_native_incremental_append_count;
    ++header->u8_cache_segment_tail_append_count;
    if (active_row + 1U == QBH_KV_CACHE_HMX_U8_SEGMENT_TOKENS) {
        ++header->u8_cache_segment_seal_count;
        header->u8_cache_segment_sealed_bytes +=
            (uint64_t)QBH_BLOCK_KV_HEADS *
            (QBH_KV_CACHE_HMX_U8_SEGMENT_K_BYTES +
             QBH_KV_CACHE_HMX_U8_SEGMENT_V_BYTES);
    }
    header->u8_cache_native_append_update_ticks +=
        HAP_perf_get_qtimer_count() - update_start;
    return 0;
}

static int qbh_scan_persist_u8_prefill_attention_carriers(
    struct qbh_block_header *header, uint8_t *shared,
    struct qbh_block_buffers *buffers, uint32_t logical_rows,
    uint32_t past_tokens) {
    const uint32_t padded_capacity =
        QBH_KV_CACHE_HMX_PADDED_CAPACITY(header->kv_cache_capacity);
    const uint32_t capacity_k_tiles =
        padded_capacity / QBH_HMX_INPUT_CHANNELS;
    const uint32_t weight_bytes =
        QBH_KV_CACHE_HMX_WEIGHT_BYTES_PER_HEAD(
            header->kv_cache_capacity);
    const uint32_t k_bias_bytes =
        QBH_KV_CACHE_HMX_K_BIAS_BYTES_PER_HEAD(
            header->kv_cache_capacity);
    const int delta_cache = qbh_hmx_native_u8_delta_cache_formats(
        header->kv_cache_k_format, header->kv_cache_v_format);
    const int segmented_cache =
        qbh_hmx_native_u8_segmented_cache_formats(
            header->kv_cache_k_format, header->kv_cache_v_format);
    const uint32_t max_segments =
        QBH_KV_CACHE_HMX_U8_SEGMENT_COUNT(header->kv_cache_capacity);
    const uint32_t stored_weight_bytes = delta_cache
        ? QBH_KV_CACHE_HMX_U8_BASE_WEIGHT_BYTES_PER_HEAD
        : weight_bytes;
    const uint32_t stored_k_bias_bytes = delta_cache
        ? QBH_KV_CACHE_HMX_U8_K_BASE_BIAS_BYTES_PER_HEAD
        : k_bias_bytes;
    const uint32_t k_head_bytes =
        segmented_cache
            ? QBH_KV_CACHE_HMX_U8_K_SEGMENTED_HEAD_BYTES(
                  header->kv_cache_capacity)
        : delta_cache
            ? QBH_KV_CACHE_HMX_U8_K_DELTA_HEAD_BYTES(
                  header->kv_cache_capacity)
            : QBH_KV_CACHE_HMX_K_HEAD_BYTES(
                  header->kv_cache_capacity);
    const uint32_t v_head_bytes =
        segmented_cache
            ? QBH_KV_CACHE_HMX_U8_V_SEGMENTED_HEAD_BYTES(
                  header->kv_cache_capacity)
        : delta_cache
            ? QBH_KV_CACHE_HMX_U8_V_DELTA_HEAD_BYTES(
                  header->kv_cache_capacity)
            : QBH_KV_CACHE_HMX_V_HEAD_BYTES(
                  header->kv_cache_capacity);
    const uint32_t existing_v_output_bytes =
        QBH_ATTENTION_SCORE_TILES * QBH_HMX_WEIGHT_BYTES;
    const uint64_t update_start = HAP_perf_get_qtimer_count();
    uint8_t *staged_weight = buffers->attention_projection;
    uint32_t *staged_bias = (uint32_t *)(
        staged_weight + weight_bytes);

    if (header->w4u8_prefill_cache_mode !=
            QBH_BLOCK_W4U8_PREFILL_CACHE_REUSE_ATTENTION_CARRIERS ||
        !qbh_hmx_native_u8_cache_formats(
            header->kv_cache_k_format,
            header->kv_cache_v_format) ||
        logical_rows != QBH_BLOCK_M || past_tokens != 0U ||
        capacity_k_tiles < QBH_ATTENTION_SCORE_TILES ||
        header->kv_cache_padded_capacity != padded_capacity ||
        header->kv_cache_k_bytes !=
            (segmented_cache
                 ? QBH_KV_CACHE_HMX_U8_K_SEGMENTED_BYTES(
                       header->kv_cache_capacity)
             : delta_cache
                 ? QBH_KV_CACHE_HMX_U8_K_DELTA_BYTES(
                       header->kv_cache_capacity)
                 : QBH_KV_CACHE_HMX_K_BYTES(
                       header->kv_cache_capacity)) ||
        header->kv_cache_v_bytes !=
            (segmented_cache
                 ? QBH_KV_CACHE_HMX_U8_V_SEGMENTED_BYTES(
                       header->kv_cache_capacity)
             : delta_cache
                 ? QBH_KV_CACHE_HMX_U8_V_DELTA_BYTES(
                       header->kv_cache_capacity)
                 : QBH_KV_CACHE_HMX_V_BYTES(
                       header->kv_cache_capacity)) ||
        !qbh_attention_u8_qkv_overlap_enabled(
            header->attention_pipeline_mode)) {
        return -1;
    }

    for (uint32_t head = 0U;
         head < QBH_BLOCK_KV_HEADS; ++head) {
        const struct qbh_attention_config *config =
            &buffers->attention_configs[head];
        const uint8_t *group_scratch =
            buffers->attention_concat +
            (size_t)head * QBH_ATTN_U8_GROUP_SCRATCH_BYTES;
        const int8_t *k_weight = (const int8_t *)(
            group_scratch + QBH_ATTN_U8_K_WEIGHT_OFFSET);
        const int8_t *v_weight = (const int8_t *)(
            group_scratch + QBH_ATTN_U8_V_WEIGHT_OFFSET);
        const uint32_t *qk_bias = (const uint32_t *)(
            group_scratch + QBH_ATTN_U8_QK_BIAS_OFFSET);
        const uint32_t *av_bias = (const uint32_t *)(
            group_scratch + QBH_ATTN_U8_AV_BIAS_OFFSET);
        uint8_t *k_destination =
            shared + header->kv_cache_k_offset +
            (size_t)head * k_head_bytes;
        uint8_t *v_destination =
            shared + header->kv_cache_v_offset +
            (size_t)head * v_head_bytes;

        if (segmented_cache) {
            const uint32_t prefill_segments =
                QBH_BLOCK_M /
                QBH_KV_CACHE_HMX_U8_SEGMENT_TOKENS;
            const uint32_t k_prefix_bytes =
                max_segments * QBH_KV_CACHE_HMX_U8_SEGMENT_K_BYTES;
            const uint32_t v_prefix_bytes =
                max_segments * QBH_KV_CACHE_HMX_U8_SEGMENT_V_BYTES;

            if (max_segments < prefill_segments) {
                return -1;
            }
            qbh_hvx_zero_aligned_bytes(staged_weight, k_prefix_bytes);
            for (uint32_t segment = 0U;
                 segment < prefill_segments; ++segment) {
                uint8_t *segment_destination = staged_weight +
                    (size_t)segment *
                        QBH_KV_CACHE_HMX_U8_SEGMENT_K_BYTES;
                qbh_hvx_copy_aligned_bytes(
                    segment_destination,
                    k_weight +
                        (size_t)segment *
                            QBH_KV_CACHE_HMX_U8_SEGMENT_WEIGHT_BYTES,
                    QBH_KV_CACHE_HMX_U8_SEGMENT_WEIGHT_BYTES);
                qbh_hvx_copy_aligned_bytes(
                    segment_destination +
                        QBH_KV_CACHE_HMX_U8_SEGMENT_WEIGHT_BYTES,
                    (const uint8_t *)qk_bias +
                        (size_t)segment * QBH_HMX_BIAS_BYTES,
                    QBH_HMX_BIAS_BYTES);
            }
            if (qbh_scan_cache_dma(
                    header, k_destination, staged_weight,
                    k_prefix_bytes, 0U) != 0) {
                return -1;
            }

            qbh_hvx_zero_aligned_bytes(
                staged_weight,
                v_prefix_bytes +
                    QBH_KV_CACHE_HMX_V_BIAS_BYTES_PER_HEAD);
            for (uint32_t output_tile = 0U;
                 output_tile < QBH_ATTENTION_HEAD_DIM_TILES;
                 ++output_tile) {
                qbh_hvx_copy_aligned_bytes(
                    staged_weight +
                        (size_t)output_tile * max_segments *
                            QBH_HMX_WEIGHT_BYTES,
                    v_weight +
                        (size_t)output_tile * prefill_segments *
                            QBH_HMX_WEIGHT_BYTES,
                    prefill_segments * QBH_HMX_WEIGHT_BYTES);
            }
            qbh_hvx_copy_aligned_bytes(
                staged_weight + v_prefix_bytes, av_bias,
                QBH_KV_CACHE_HMX_V_BIAS_BYTES_PER_HEAD);
            if (qbh_scan_cache_dma(
                    header, v_destination, staged_weight,
                    v_prefix_bytes +
                        QBH_KV_CACHE_HMX_V_BIAS_BYTES_PER_HEAD,
                    0U) != 0) {
                return -1;
            }
            continue;
        }

        qbh_hvx_zero_aligned_bytes(staged_weight, weight_bytes);
        qbh_hvx_copy_aligned_bytes(
            staged_weight, k_weight, QBH_ATTN_U8_K_WEIGHT_BYTES);
        qbh_hvx_zero_aligned_bytes(staged_bias, k_bias_bytes);
        qbh_hvx_copy_aligned_bytes(
            staged_bias, qk_bias, QBH_ATTN_U8_QK_BIAS_BYTES);
        {
            const uint32_t divisor =
                UINT32_C(1) << config->score_shift;
            const int32_t rounding = config->score_shift == 0U
                                         ? 0
                                         : (int32_t)(divisor / 2U);
            const __fp16 conversion_value =
                (__fp16)(512.0f / (float)divisor);
            uint16_t conversion_bits;
            memcpy(&conversion_bits, &conversion_value,
                   sizeof(conversion_bits));
            for (uint32_t n_tile = QBH_ATTENTION_SCORE_TILES;
                 n_tile < capacity_k_tiles; ++n_tile) {
                uint32_t *tile_bias = staged_bias +
                    (size_t)n_tile *
                        (QBH_HMX_BIAS_BYTES / sizeof(uint32_t));
                for (uint32_t output = 0U;
                     output < QBH_HMX_OUTPUT_CHANNELS; ++output) {
                    tile_bias[output] = conversion_bits;
                    tile_bias[QBH_HMX_OUTPUT_CHANNELS + output] =
                        UINT32_C(128) * divisor + rounding;
                }
            }
        }
        if (qbh_scan_cache_dma(
                header, k_destination, staged_weight,
                stored_weight_bytes, 0U) != 0 ||
            qbh_scan_cache_dma(
                header, k_destination + stored_weight_bytes,
                staged_bias, stored_k_bias_bytes, 0U) != 0) {
            return -1;
        }

        if (delta_cache) {
            qbh_hvx_copy_aligned_bytes(
                staged_weight, v_weight, stored_weight_bytes);
        } else {
            qbh_hvx_zero_aligned_bytes(staged_weight, weight_bytes);
            for (uint32_t output_tile = 0U;
                 output_tile < QBH_ATTENTION_HEAD_DIM_TILES;
                 ++output_tile) {
                qbh_hvx_copy_aligned_bytes(
                    staged_weight + (size_t)output_tile *
                        capacity_k_tiles * QBH_HMX_WEIGHT_BYTES,
                    v_weight + (size_t)output_tile *
                        existing_v_output_bytes,
                    existing_v_output_bytes);
            }
        }
        if (qbh_scan_cache_dma(
                header, v_destination, staged_weight,
                stored_weight_bytes, 0U) != 0 ||
            qbh_scan_cache_dma(
                header, v_destination + stored_weight_bytes,
                av_bias, QBH_KV_CACHE_HMX_V_BIAS_BYTES_PER_HEAD,
                0U) != 0) {
            return -1;
        }
    }

    ++header->u8_cache_native_prefill_reuse_count;
    header->u8_cache_native_prefill_reused_carrier_bytes +=
        (uint64_t)QBH_BLOCK_KV_HEADS *
        (QBH_ATTN_U8_K_WEIGHT_BYTES +
         QBH_ATTN_U8_QK_BIAS_BYTES +
         QBH_ATTN_U8_V_WEIGHT_BYTES +
         QBH_ATTN_U8_AV_BIAS_BYTES);
    header->u8_cache_native_append_update_ticks +=
        HAP_perf_get_qtimer_count() - update_start;
    return 0;
}

static int qbh_scan_append_u8_kv(
    struct qbh_block_header *header, uint8_t *shared,
    struct qbh_block_buffers *buffers, uint32_t logical_rows,
    uint32_t past_tokens) {
    if (qbh_hmx_native_u8_segmented_cache_formats(
            header->kv_cache_k_format,
            header->kv_cache_v_format)) {
        return qbh_scan_append_u8_kv_hmx_segmented(
            header, shared, buffers, logical_rows, past_tokens);
    }
    if (qbh_hmx_native_u8_delta_cache_formats(
            header->kv_cache_k_format,
            header->kv_cache_v_format)) {
        return qbh_scan_append_u8_kv_hmx_delta(
            header, shared, buffers, logical_rows, past_tokens);
    }
    if (qbh_hmx_native_u8_cache_formats(
            header->kv_cache_k_format,
            header->kv_cache_v_format)) {
        return qbh_scan_append_u8_kv_hmx_native(
            header, shared, buffers, logical_rows, past_tokens);
    }
    return qbh_scan_append_u8_kv_row_major(
        header, shared, buffers, logical_rows, past_tokens);
}

static void qbh_scan_unpack_fp16_weight_rows(
    const __fp16 *weight_tiles, __fp16 *rows,
    uint32_t valid_rows) {
    const uint32_t k_tiles =
        QBH_BLOCK_HEAD_DIM / QBH_HMX_FP16_COLS;

    for (uint32_t row = 0U; row < valid_rows; ++row) {
        const uint32_t n_tile = row / QBH_HMX_FP16_COLS;
        const uint32_t local_output = row % QBH_HMX_FP16_COLS;
        for (uint32_t channel = 0U;
             channel < QBH_BLOCK_HEAD_DIM; ++channel) {
            const uint32_t k_tile = channel / QBH_HMX_FP16_ROWS;
            const uint32_t local_input = channel % QBH_HMX_FP16_ROWS;
            const size_t tile =
                ((size_t)n_tile * k_tiles + k_tile) *
                QBH_HMX_FP16_TILE_ELEMENTS;
            rows[(size_t)row * QBH_BLOCK_HEAD_DIM + channel] =
                weight_tiles[tile + qbh_hmx_fp16_tile_offset(
                    local_input, local_output)];
        }
    }
}

static void qbh_scan_gather_fp16_head(
    const __fp16 *source, uint32_t source_stride,
    uint32_t head, __fp16 *rows, uint32_t valid_rows) {
    for (uint32_t row = 0U; row < valid_rows; ++row) {
        memcpy(
            rows + (size_t)row * QBH_BLOCK_HEAD_DIM,
            source + (size_t)row * source_stride +
                (size_t)head * QBH_BLOCK_HEAD_DIM,
            QBH_BLOCK_HEAD_DIM * sizeof(__fp16));
    }
}

static int qbh_scan_append_f16_kv_row_major(
    struct qbh_block_header *header, uint8_t *shared,
    struct qbh_block_buffers *buffers, uint32_t logical_rows,
    uint32_t past_tokens) {
    __fp16 *row_scratch = (__fp16 *)buffers->up;
    __fp16 *v_rows = (__fp16 *)buffers->gate;
    const uint32_t append_bytes =
        logical_rows * QBH_BLOCK_HEAD_DIM * sizeof(__fp16);
    const uint32_t head_stride =
        header->kv_cache_capacity * QBH_BLOCK_HEAD_DIM;
    const uint32_t head_elements =
        QBH_BLOCK_M * QBH_BLOCK_HEAD_DIM;
    const uint32_t direct_qkv = qbh_crouton_qkv_enabled(header);

    /* Direct W4F16 QKV keeps V in grouped HMX-output order.  The native
     * Attention path can consume that carrier after a tile permutation, but
     * a persistent head-major KV cache requires logical rows.  Unpack the
     * complete V projection once into dead post-QKV scratch, then gather each
     * head; treating the grouped carrier as one ordinary HMX output silently
     * interleaves the batch2/batch4 command groups. */
    if (direct_qkv != 0U) {
        qbh_unpack_fp16_grouped_projection(
            (const __fp16 *)buffers->v,
            QBH_BLOCK_KV_HIDDEN / QBH_HMX_FP16_COLS,
            v_rows, QBH_BLOCK_KV_HIDDEN,
            qbh_w4f16_projection_group_tiles(
                header, &header->projections[QBH_BLOCK_PROJ_V]));
    }

    for (uint32_t head = 0U; head < QBH_BLOCK_KV_HEADS; ++head) {
        __fp16 *k_destination = (__fp16 *)(
            shared + header->kv_cache_k_offset) +
            (size_t)head * head_stride +
            (size_t)past_tokens * QBH_BLOCK_HEAD_DIM;
        __fp16 *v_destination = (__fp16 *)(
            shared + header->kv_cache_v_offset) +
            (size_t)head * head_stride +
            (size_t)past_tokens * QBH_BLOCK_HEAD_DIM;

        if (direct_qkv != 0U) {
            qbh_scan_unpack_fp16_weight_rows(
                (const __fp16 *)buffers->scores +
                    (size_t)head * head_elements,
                row_scratch, logical_rows);
        } else {
            qbh_scan_gather_fp16_head(
                (const __fp16 *)buffers->k,
                QBH_BLOCK_KV_HIDDEN, head,
                row_scratch, logical_rows);
        }
        if (qbh_scan_cache_dma(
                header, k_destination, row_scratch,
                append_bytes, 0U) != 0) {
            return -1;
        }

        if (direct_qkv != 0U) {
            qbh_scan_gather_fp16_head(
                v_rows, QBH_BLOCK_KV_HIDDEN, head,
                row_scratch, logical_rows);
        } else {
            qbh_scan_gather_fp16_head(
                (const __fp16 *)buffers->v,
                QBH_BLOCK_KV_HIDDEN, head,
                row_scratch, logical_rows);
        }
        if (qbh_scan_cache_dma(
                header, v_destination, row_scratch,
                append_bytes, 0U) != 0) {
            return -1;
        }
    }
    return 0;
}

static int qbh_scan_append_f16_kv_hmx_native(
    struct qbh_block_header *header, uint8_t *shared,
    struct qbh_block_buffers *buffers, uint32_t logical_rows,
    uint32_t past_tokens) {
    const uint32_t weight_bytes =
        QBH_KV_CACHE_HMX_F16_WEIGHT_BYTES_PER_HEAD(
            header->kv_cache_capacity);
    const uint32_t head_stride =
        QBH_KV_CACHE_HMX_F16_K_HEAD_BYTES(
            header->kv_cache_capacity);
    const uint32_t row_bytes =
        QBH_BLOCK_HEAD_DIM * sizeof(uint16_t);
    const uint32_t direct_qkv = qbh_crouton_qkv_enabled(header);
    __fp16 *row = (__fp16 *)buffers->up;
    __fp16 *v_rows = (__fp16 *)buffers->gate;
    const uint64_t update_start = HAP_perf_get_qtimer_count();

    if (logical_rows != 1U ||
        past_tokens < QBH_BLOCK_M ||
        past_tokens >= header->kv_cache_capacity) {
        return -1;
    }
    if (direct_qkv != 0U) {
        qbh_unpack_fp16_grouped_projection(
            (const __fp16 *)buffers->v,
            QBH_BLOCK_KV_HIDDEN / QBH_HMX_FP16_COLS,
            v_rows, QBH_BLOCK_KV_HIDDEN,
            qbh_w4f16_projection_group_tiles(
                header, &header->projections[QBH_BLOCK_PROJ_V]));
    }
    for (uint32_t head = 0U; head < QBH_BLOCK_KV_HEADS; ++head) {
        uint8_t *k_delta = shared + header->kv_cache_k_offset +
            (size_t)head * head_stride + weight_bytes +
            (size_t)(past_tokens - QBH_BLOCK_M) * row_bytes;
        uint8_t *v_delta = shared + header->kv_cache_v_offset +
            (size_t)head * head_stride + weight_bytes +
            (size_t)(past_tokens - QBH_BLOCK_M) * row_bytes;
        if (direct_qkv != 0U) {
            qbh_scan_unpack_fp16_weight_rows(
                (const __fp16 *)buffers->scores +
                    (size_t)head * QBH_BLOCK_M *
                        QBH_BLOCK_HEAD_DIM,
                row, logical_rows);
        } else {
            qbh_scan_gather_fp16_head(
                (const __fp16 *)buffers->k,
                QBH_BLOCK_KV_HIDDEN, head,
                row, logical_rows);
        }
        if (qbh_scan_cache_dma(
                header, k_delta, row, row_bytes, 0U) != 0) {
            return -1;
        }
        if (direct_qkv != 0U) {
            qbh_scan_gather_fp16_head(
                v_rows, QBH_BLOCK_KV_HIDDEN, head,
                row, logical_rows);
        } else {
            qbh_scan_gather_fp16_head(
                (const __fp16 *)buffers->v,
                QBH_BLOCK_KV_HIDDEN, head,
                row, logical_rows);
        }
        if (qbh_scan_cache_dma(
                header, v_delta, row, row_bytes, 0U) != 0) {
            return -1;
        }
    }
    ++header->f16_cache_native_incremental_append_count;
    header->f16_cache_native_append_update_ticks +=
        HAP_perf_get_qtimer_count() - update_start;
    return 0;
}

static int qbh_scan_append_f16_kv(
    struct qbh_block_header *header, uint8_t *shared,
    struct qbh_block_buffers *buffers, uint32_t logical_rows,
    uint32_t past_tokens) {
    if (qbh_hmx_native_f16_cache_formats(
            header->kv_cache_k_format,
            header->kv_cache_v_format)) {
        return qbh_scan_append_f16_kv_hmx_native(
            header, shared, buffers, logical_rows, past_tokens);
    }
    return qbh_scan_append_f16_kv_row_major(
        header, shared, buffers, logical_rows, past_tokens);
}

static int qbh_scan_persist_f16_prefill_attention_carriers(
    struct qbh_block_header *header, uint8_t *shared,
    struct qbh_block_buffers *buffers, uint32_t logical_rows,
    uint32_t past_tokens) {
    const uint32_t head_stride =
        QBH_KV_CACHE_HMX_F16_K_HEAD_BYTES(
            header->kv_cache_capacity);
    const uint32_t staged_bytes = head_stride;
    const uint32_t existing_bytes =
        logical_rows * QBH_BLOCK_HEAD_DIM * sizeof(__fp16);
    __fp16 *staging = (__fp16 *)buffers->gate;
    const uint64_t update_start = HAP_perf_get_qtimer_count();

    if (past_tokens != 0U || logical_rows != QBH_BLOCK_M) {
        return -1;
    }
    for (uint32_t head = 0U; head < QBH_BLOCK_KV_HEADS; ++head) {
        const uint32_t first_q_head =
            head * QBH_ATTENTION_Q_HEADS_PER_GROUP;
        const __fp16 *k_carrier =
            (const __fp16 *)buffers->attention_projection +
            (size_t)first_q_head * 2U * QBH_BLOCK_M * QBH_BLOCK_M;
        const __fp16 *v_carrier =
            (const __fp16 *)buffers->probability +
            (size_t)first_q_head * QBH_BLOCK_M * QBH_BLOCK_M;
        uint8_t *k_destination = shared + header->kv_cache_k_offset +
            (size_t)head * head_stride;
        uint8_t *v_destination = shared + header->kv_cache_v_offset +
            (size_t)head * head_stride;

        qbh_hvx_zero_aligned_bytes(staging, staged_bytes);
        qbh_hvx_copy_aligned_bytes(staging, k_carrier, existing_bytes);
        if (qbh_scan_cache_dma(
                header, k_destination, staging,
                staged_bytes, 0U) != 0) {
            return -1;
        }

        qbh_hvx_zero_aligned_bytes(staging, staged_bytes);
        qbh_hvx_copy_aligned_bytes(staging, v_carrier, existing_bytes);
        if (qbh_scan_cache_dma(
                header, v_destination, staging,
                staged_bytes, 0U) != 0) {
            return -1;
        }
    }
    ++header->f16_cache_native_prefill_reuse_count;
    header->f16_cache_native_prefill_reused_carrier_bytes +=
        (uint64_t)QBH_BLOCK_KV_HEADS * 2U * existing_bytes;
    header->f16_cache_native_append_update_ticks +=
        HAP_perf_get_qtimer_count() - update_start;
    return 0;
}

static void qbh_scan_softmax_f16(
    const __fp16 *scores, __fp16 *probability,
    uint32_t logical_rows, uint32_t past_tokens,
    uint32_t padded_tokens) {
    const uint32_t head_plane = QBH_BLOCK_M * padded_tokens;
    const float scale = 0.08838834764831845f;

    memset(
        probability, 0,
        QBH_ATTENTION_Q_HEADS_PER_GROUP * head_plane *
            sizeof(__fp16));
    for (uint32_t head = 0U;
         head < QBH_ATTENTION_Q_HEADS_PER_GROUP; ++head) {
        for (uint32_t row = 0U; row < logical_rows; ++row) {
            const uint32_t valid = past_tokens + row + 1U;
            const __fp16 *score_row = scores +
                (size_t)head * head_plane +
                (size_t)row * padded_tokens;
            __fp16 *probability_row = probability +
                (size_t)head * head_plane +
                (size_t)row * padded_tokens;
            float maximum = -INFINITY;
            float sum = 0.0f;

            for (uint32_t token = 0U; token < valid; ++token) {
                const float value = (float)score_row[token] * scale;
                if (value > maximum) {
                    maximum = value;
                }
            }
            for (uint32_t token = 0U; token < valid; ++token) {
                const float value =
                    expf((float)score_row[token] * scale - maximum);
                probability_row[token] = (__fp16)value;
                sum += value;
            }
            for (uint32_t token = 0U; token < valid; ++token) {
                probability_row[token] = (__fp16)(
                    (float)probability_row[token] / sum);
            }
        }
    }
}

static int qbh_scan_f16_attention(
    struct qbh_block_header *header, uint8_t *shared,
    struct qbh_block_buffers *buffers,
    struct qbh_block_hmx_worker *worker,
    uint32_t logical_rows, uint32_t past_tokens) {
    const uint32_t valid_tokens = past_tokens + logical_rows;
    const uint32_t padded_tokens = qbh_align_up(
        valid_tokens, QBH_HMX_FP16_COLS);
    const uint32_t kv_tiles =
        padded_tokens / QBH_HMX_FP16_COLS;
    const uint32_t plane_bytes =
        padded_tokens * QBH_BLOCK_HEAD_DIM * sizeof(__fp16);
    const uint32_t required_bytes = 3U * plane_bytes;
    const uint32_t overlay_capacity =
        (uint32_t)(buffers->compressed_weight - buffers->scores);
    __fp16 *plane_a = (__fp16 *)buffers->scores;
    __fp16 *weight = (__fp16 *)(buffers->scores + plane_bytes);
    __fp16 *plane_c = (__fp16 *)(buffers->scores + 2U * plane_bytes);
    const uint32_t head_elements =
        QBH_BLOCK_M * QBH_BLOCK_HEAD_DIM;
    const uint32_t score_head_elements =
        QBH_BLOCK_M * padded_tokens;
    const uint32_t direct_qkv = qbh_crouton_qkv_enabled(header);
    const uint32_t native_cache = qbh_hmx_native_f16_cache_formats(
        header->kv_cache_k_format, header->kv_cache_v_format);
    const uint32_t native_base_bytes =
        QBH_KV_CACHE_HMX_F16_WEIGHT_BYTES_PER_HEAD(
            header->kv_cache_capacity);
    const uint32_t cache_head_stride_bytes = native_cache
        ? QBH_KV_CACHE_HMX_F16_K_HEAD_BYTES(
              header->kv_cache_capacity)
        : header->kv_cache_capacity * QBH_BLOCK_HEAD_DIM *
              sizeof(__fp16);
    const uint64_t dynamic_start = HAP_perf_get_qtimer_count();

    __fp16 *q_group_scratch = (__fp16 *)buffers->k;
    __fp16 *delta_rows = (__fp16 *)buffers->gate;
    __fp16 *delta_weight = (__fp16 *)buffers->up;
    const uint32_t decode_rows = native_cache != 0U
        ? valid_tokens - QBH_BLOCK_M : 0U;
    const uint32_t delta_bytes =
        decode_rows * QBH_BLOCK_HEAD_DIM * sizeof(uint16_t);

    header->scan_attention_overlay_capacity_bytes = overlay_capacity;
    header->scan_attention_overlay_required_bytes = required_bytes;
    header->scan_total_kv_length = valid_tokens;
    header->scan_padded_kv_length = padded_tokens;
    if (required_bytes > overlay_capacity ||
        kv_tiles > QBH_BLOCK_SCAN_MAX_KV_TILES) {
        return -1;
    }
    qbh_hmx_fp16_init_unity_scale(buffers->scale_or_bias);

    /*
     * The three long-context planes intentionally overlay the post-QKV VTCM
     * lifetime through compressed_weight.  At L=4096 that range includes
     * attention_projection and hmx_activation, so neither may remain the
     * live Q/AV carrier.  Preserve logical Q before touching the planes and
     * use the now-dead row-major K buffer as one bounded two-head HMX carrier.
     */
    if (direct_qkv != 0U) {
        for (uint32_t head = 0U; head < QBH_BLOCK_HEADS; ++head) {
            qbh_unpack_fp16_output(
                (const __fp16 *)buffers->attention_projection +
                    (size_t)head * head_elements,
                QBH_BLOCK_HEAD_DIM / QBH_HMX_FP16_COLS,
                (__fp16 *)buffers->q, QBH_BLOCK_HIDDEN,
                head * QBH_BLOCK_HEAD_DIM);
        }
    }

    for (uint32_t group = 0U; group < QBH_BLOCK_KV_HEADS; ++group) {
        const uint32_t first_q_head =
            group * QBH_ATTENTION_Q_HEADS_PER_GROUP;
        const uint8_t *cache_k =
            shared + header->kv_cache_k_offset +
            (size_t)group * cache_head_stride_bytes;
        const uint8_t *cache_v =
            shared + header->kv_cache_v_offset +
            (size_t)group * cache_head_stride_bytes;
        __fp16 *q_group = q_group_scratch;
        uint64_t start;

        if (native_cache != 0U) {
            if (qbh_scan_cache_dma(
                    header, weight, cache_k,
                    native_base_bytes, 1U) != 0) {
                return -1;
            }
            qbh_hvx_zero_aligned_bytes(
                delta_rows,
                QBH_HMX_FP16_COLS * QBH_BLOCK_HEAD_DIM *
                    sizeof(__fp16));
            if (qbh_scan_cache_dma(
                    header, delta_rows,
                    cache_k + native_base_bytes,
                    delta_bytes, 1U) != 0) {
                return -1;
            }
            start = HAP_perf_get_qtimer_count();
            qbh_pack_fp16_weight_rows_hvx(
                delta_rows, QBH_BLOCK_HEAD_DIM, 0U,
                QBH_BLOCK_HEAD_DIM, QBH_HMX_FP16_COLS,
                delta_weight);
            qbh_hvx_copy_aligned_bytes(
                weight +
                    (size_t)(QBH_BLOCK_M / QBH_HMX_FP16_COLS) *
                        QBH_ATTENTION_HEAD_DIM_TILES *
                        QBH_HMX_FP16_TILE_ELEMENTS,
                delta_weight,
                QBH_HMX_FP16_COLS * QBH_BLOCK_HEAD_DIM *
                    sizeof(__fp16));
            header->f16_cache_native_append_update_ticks +=
                HAP_perf_get_qtimer_count() - start;
        } else {
            memset(plane_a, 0, plane_bytes);
            if (qbh_scan_cache_dma(
                    header, plane_a, cache_k,
                    valid_tokens * QBH_BLOCK_HEAD_DIM * sizeof(__fp16),
                    1U) != 0) {
                return -1;
            }
            start = HAP_perf_get_qtimer_count();
            qbh_pack_fp16_weight_rows_hvx(
                plane_a, QBH_BLOCK_HEAD_DIM, 0U,
                QBH_BLOCK_HEAD_DIM, padded_tokens, weight);
            header->attention_qk_pack_ticks +=
                HAP_perf_get_qtimer_count() - start;
            ++header->f16_cache_full_prefix_pack_count;
        }

        for (uint32_t local_head = 0U;
             local_head < QBH_ATTENTION_Q_HEADS_PER_GROUP;
             ++local_head) {
            qbh_pack_fp16_activation(
                (const __fp16 *)buffers->q +
                    (size_t)(first_q_head + local_head) *
                        QBH_BLOCK_HEAD_DIM,
                QBH_BLOCK_HIDDEN, QBH_BLOCK_HEAD_DIM,
                q_group + (size_t)local_head * head_elements);
        }
        start = HAP_perf_get_qtimer_count();
        if (qbh_hmx_submit(
                worker, QBH_BLOCK_HMX_FP16,
                q_group, weight, buffers->scale_or_bias, plane_c,
                2U * QBH_ATTENTION_Q_HEADS_PER_GROUP,
                QBH_BLOCK_HEAD_DIM / QBH_HMX_FP16_COLS,
                kv_tiles) != 0) {
            return -1;
        }
        header->attention_qk_hmx_ticks +=
            HAP_perf_get_qtimer_count() - start;
        start = HAP_perf_get_qtimer_count();
        for (uint32_t local_head = 0U;
             local_head < QBH_ATTENTION_Q_HEADS_PER_GROUP;
             ++local_head) {
            qbh_unpack_fp16_output(
                plane_c + (size_t)local_head *
                    2U * kv_tiles * QBH_HMX_FP16_TILE_ELEMENTS,
                kv_tiles,
                plane_a + (size_t)local_head * score_head_elements,
                padded_tokens, 0U);
        }
        header->attention_qk_unpack_ticks +=
            HAP_perf_get_qtimer_count() - start;

        start = HAP_perf_get_qtimer_count();
        qbh_scan_softmax_f16(
            plane_a, plane_c, logical_rows,
            past_tokens, padded_tokens);
        header->attention_softmax_ticks +=
            HAP_perf_get_qtimer_count() - start;

        start = HAP_perf_get_qtimer_count();
        for (uint32_t local_head = 0U;
             local_head < QBH_ATTENTION_Q_HEADS_PER_GROUP;
             ++local_head) {
            qbh_pack_fp16_activation(
                plane_c + (size_t)local_head * score_head_elements,
                padded_tokens, padded_tokens,
                plane_a + (size_t)local_head *
                    2U * kv_tiles * QBH_HMX_FP16_TILE_ELEMENTS);
        }
        header->attention_av_pack_ticks +=
            HAP_perf_get_qtimer_count() - start;

        if (native_cache != 0U) {
            if (qbh_scan_cache_dma(
                    header, delta_weight, cache_v,
                    native_base_bytes, 1U) != 0) {
                return -1;
            }
            start = HAP_perf_get_qtimer_count();
            for (uint32_t output_tile = 0U;
                 output_tile < QBH_ATTENTION_HEAD_DIM_TILES;
                 ++output_tile) {
                qbh_hvx_copy_aligned_bytes(
                    weight +
                        (size_t)output_tile * kv_tiles *
                            QBH_HMX_FP16_TILE_ELEMENTS,
                    delta_weight +
                        (size_t)output_tile *
                            (QBH_BLOCK_M / QBH_HMX_FP16_ROWS) *
                            QBH_HMX_FP16_TILE_ELEMENTS,
                    (QBH_BLOCK_M / QBH_HMX_FP16_ROWS) *
                        QBH_HMX_FP16_TILE_BYTES);
            }
            header->f16_cache_native_append_update_ticks +=
                HAP_perf_get_qtimer_count() - start;
            qbh_hvx_zero_aligned_bytes(
                delta_rows,
                QBH_HMX_FP16_COLS * QBH_BLOCK_HEAD_DIM *
                    sizeof(__fp16));
            if (qbh_scan_cache_dma(
                    header, delta_rows,
                    cache_v + native_base_bytes,
                    delta_bytes, 1U) != 0) {
                return -1;
            }
            start = HAP_perf_get_qtimer_count();
            qbh_pack_fp16_weight_transposed_hvx(
                delta_rows, QBH_BLOCK_HEAD_DIM, 0U,
                QBH_HMX_FP16_COLS, QBH_BLOCK_HEAD_DIM,
                delta_weight);
            for (uint32_t output_tile = 0U;
                 output_tile < QBH_ATTENTION_HEAD_DIM_TILES;
                 ++output_tile) {
                qbh_hvx_copy_aligned_bytes(
                    weight +
                        ((size_t)output_tile * kv_tiles +
                         QBH_BLOCK_M / QBH_HMX_FP16_ROWS) *
                            QBH_HMX_FP16_TILE_ELEMENTS,
                    delta_weight +
                        (size_t)output_tile *
                            QBH_HMX_FP16_TILE_ELEMENTS,
                    QBH_HMX_FP16_TILE_BYTES);
            }
            header->f16_cache_native_append_update_ticks +=
                HAP_perf_get_qtimer_count() - start;
        } else {
            memset(plane_c, 0, plane_bytes);
            if (qbh_scan_cache_dma(
                    header, plane_c, cache_v,
                    valid_tokens * QBH_BLOCK_HEAD_DIM * sizeof(__fp16),
                    1U) != 0) {
                return -1;
            }
            start = HAP_perf_get_qtimer_count();
            qbh_pack_fp16_weight_transposed_hvx(
                plane_c, QBH_BLOCK_HEAD_DIM, 0U,
                padded_tokens, QBH_BLOCK_HEAD_DIM, weight);
            header->attention_av_pack_ticks +=
                HAP_perf_get_qtimer_count() - start;
            ++header->f16_cache_full_prefix_pack_count;
        }
        start = HAP_perf_get_qtimer_count();
        if (qbh_hmx_submit(
                worker, QBH_BLOCK_HMX_FP16,
                plane_a, weight, buffers->scale_or_bias, q_group,
                2U * QBH_ATTENTION_Q_HEADS_PER_GROUP,
                kv_tiles,
                QBH_BLOCK_HEAD_DIM / QBH_HMX_FP16_COLS) != 0) {
            return -1;
        }
        header->attention_av_hmx_ticks +=
            HAP_perf_get_qtimer_count() - start;
        start = HAP_perf_get_qtimer_count();
        for (uint32_t local_head = 0U;
             local_head < QBH_ATTENTION_Q_HEADS_PER_GROUP;
             ++local_head) {
            qbh_unpack_fp16_output(
                q_group + (size_t)local_head *
                    2U * (QBH_BLOCK_HEAD_DIM / QBH_HMX_FP16_COLS) *
                        QBH_HMX_FP16_TILE_ELEMENTS,
                QBH_BLOCK_HEAD_DIM / QBH_HMX_FP16_COLS,
                (__fp16 *)buffers->q, QBH_BLOCK_HIDDEN,
                (first_q_head + local_head) * QBH_BLOCK_HEAD_DIM);
        }
        header->attention_av_unpack_ticks +=
            HAP_perf_get_qtimer_count() - start;
    }

    memcpy(
        buffers->attention_concat, buffers->q,
        QBH_BLOCK_M * QBH_BLOCK_HIDDEN * sizeof(__fp16));
    header->hmx_command_count += 2U * QBH_BLOCK_KV_HEADS;
    header->hmx_fp16_tile_pair_count +=
        32U * QBH_BLOCK_KV_HEADS * kv_tiles;
    header->attention_gqa_group_count += QBH_BLOCK_KV_HEADS;
    header->scan_dynamic_attention_ticks +=
        HAP_perf_get_qtimer_count() - dynamic_start;
    return 0;
}

static int qbh_scan_audit_f16_q(
    struct qbh_block_header *header, uint8_t *shared,
    struct qbh_block_buffers *buffers) {
    __fp16 *scratch = (__fp16 *)buffers->gate;
    const uint32_t bytes =
        QBH_BLOCK_M * QBH_BLOCK_HIDDEN * sizeof(__fp16);
    const __fp16 *source;

    if (header->scan_attention_audit_output_bytes !=
            QBH_BLOCK_SCAN_F16_AUDIT_BYTES ||
        !qbh_range_valid(
            header->scan_attention_audit_output_offset,
            header->scan_attention_audit_output_bytes,
            header->shared_bytes)) {
        return -1;
    }
    if (qbh_crouton_qkv_enabled(header) != 0U) {
        const uint32_t head_elements =
            QBH_BLOCK_M * QBH_BLOCK_HEAD_DIM;
        for (uint32_t head = 0U; head < QBH_BLOCK_HEADS; ++head) {
            qbh_unpack_fp16_output(
                (const __fp16 *)buffers->attention_projection +
                    (size_t)head * head_elements,
                QBH_BLOCK_HEAD_DIM / QBH_HMX_FP16_COLS,
                scratch, QBH_BLOCK_HIDDEN,
                head * QBH_BLOCK_HEAD_DIM);
        }
        source = scratch;
    } else {
        source = (const __fp16 *)buffers->q;
    }
    if (qbh_dma_copy(
            header,
            shared + header->scan_attention_audit_output_offset,
            source, bytes, 0U) != 0) {
        return -1;
    }
    header->u8_attention_audit_ddr_write_bytes += bytes;
    return 0;
}

static int qbh_scan_audit_f16_attention(
    struct qbh_block_header *header, uint8_t *shared,
    struct qbh_block_buffers *buffers) {
    const uint32_t bytes =
        QBH_BLOCK_M * QBH_BLOCK_HIDDEN * sizeof(__fp16);
    if (qbh_dma_copy(
            header,
            shared + header->scan_attention_audit_output_offset + bytes,
            buffers->attention_concat, bytes, 0U) != 0) {
        return -1;
    }
    header->u8_attention_audit_ddr_write_bytes += bytes;
    return 0;
}

static int qbh_scan_audit_f16_o_projection(
    struct qbh_block_header *header, uint8_t *shared,
    struct qbh_block_buffers *buffers) {
    const uint32_t bytes =
        QBH_BLOCK_M * QBH_BLOCK_HIDDEN * sizeof(__fp16);
    if (qbh_dma_copy(
            header,
            shared + header->scan_attention_audit_output_offset +
                2U * bytes,
            buffers->attention_projection, bytes, 0U) != 0) {
        return -1;
    }
    header->u8_attention_audit_ddr_write_bytes += bytes;
    return 0;
}

struct qbh_scan_u8_delta_attention_slot {
    uint8_t *probability;
    int8_t *weight;
    uint8_t *scores;
    uint32_t *qk_bias;
    uint32_t *av_bias;
    struct qbh_attention_u8_telemetry telemetry;
};

static void qbh_scan_u8_delta_slot_init(
    struct qbh_scan_u8_delta_attention_slot *slot,
    uint8_t *base, uint32_t plane_bytes, uint32_t qk_bias_bytes) {
    slot->probability = base;
    slot->weight = (int8_t *)(base + plane_bytes);
    slot->scores = base + 2U * plane_bytes;
    slot->qk_bias = (uint32_t *)(base + 3U * plane_bytes);
    slot->av_bias = (uint32_t *)(
        (uint8_t *)slot->qk_bias + qk_bias_bytes);
    memset(&slot->telemetry, 0, sizeof(slot->telemetry));
}

static int qbh_scan_prepare_u8_delta_k(
    struct qbh_block_header *header,
    struct qbh_scan_u8_delta_attention_slot *slot,
    const uint8_t *cache_k, uint8_t *delta_rows,
    uint32_t decode_rows, uint32_t delta_bytes,
    uint32_t plane_bytes,
    const struct qbh_attention_config *config) {
    uint32_t *tail_bias = slot->qk_bias +
        (size_t)(QBH_BLOCK_M / QBH_HMX_OUTPUT_CHANNELS) *
            (QBH_HMX_BIAS_BYTES / sizeof(uint32_t));
    const uint32_t tail_tiles =
        (decode_rows + QBH_HMX_OUTPUT_CHANNELS - 1U) /
        QBH_HMX_OUTPUT_CHANNELS;
    const uint32_t divisor = UINT32_C(1) << config->score_shift;
    const int32_t rounding = config->score_shift == 0U
        ? 0 : (int32_t)(divisor / 2U);
    uint64_t start;

    memset(&slot->telemetry, 0, sizeof(slot->telemetry));
    qbh_hvx_zero_aligned_bytes(
        slot->weight +
            QBH_KV_CACHE_HMX_U8_BASE_WEIGHT_BYTES_PER_HEAD,
        plane_bytes -
            QBH_KV_CACHE_HMX_U8_BASE_WEIGHT_BYTES_PER_HEAD);
    if (qbh_scan_cache_dma(
            header, slot->weight, cache_k,
            QBH_KV_CACHE_HMX_U8_BASE_WEIGHT_BYTES_PER_HEAD,
            1U) != 0 ||
        qbh_scan_cache_dma(
            header, slot->qk_bias,
            cache_k +
                QBH_KV_CACHE_HMX_U8_BASE_WEIGHT_BYTES_PER_HEAD,
            QBH_KV_CACHE_HMX_U8_K_BASE_BIAS_BYTES_PER_HEAD,
            1U) != 0) {
        return -1;
    }
    for (uint32_t tile = 0U; tile < tail_tiles; ++tile) {
        uint32_t *tile_bias = tail_bias +
            (size_t)tile *
                (QBH_HMX_BIAS_BYTES / sizeof(uint32_t));
        for (uint32_t output = 0U;
             output < QBH_HMX_OUTPUT_CHANNELS; ++output) {
            tile_bias[output] = slot->qk_bias[0];
            tile_bias[QBH_HMX_OUTPUT_CHANNELS + output] =
                UINT32_C(128) * divisor + rounding;
        }
    }
    if (qbh_scan_cache_dma(
            header, delta_rows,
            cache_k +
                QBH_KV_CACHE_HMX_U8_BASE_WEIGHT_BYTES_PER_HEAD +
                QBH_KV_CACHE_HMX_U8_K_BASE_BIAS_BYTES_PER_HEAD,
            delta_bytes, 1U) != 0) {
        return -1;
    }
    start = HAP_perf_get_qtimer_count();
    for (uint32_t first = 0U; first < decode_rows;
         first += QBH_HMX_OUTPUT_CHANNELS) {
        const uint32_t count =
            decode_rows - first < QBH_HMX_OUTPUT_CHANNELS
                ? decode_rows - first
                : QBH_HMX_OUTPUT_CHANNELS;
        const uint32_t tile = first / QBH_HMX_OUTPUT_CHANNELS;
        qbh_attention_u8_patch_k_delta_rows_hvx(
            delta_rows + (size_t)first * QBH_BLOCK_HEAD_DIM,
            count, config,
            slot->weight +
                (size_t)(QBH_BLOCK_M /
                         QBH_HMX_OUTPUT_CHANNELS + tile) *
                    QBH_ATTENTION_HEAD_DIM_TILES *
                    QBH_HMX_WEIGHT_BYTES,
            slot->qk_bias +
                (size_t)(QBH_BLOCK_M /
                         QBH_HMX_OUTPUT_CHANNELS + tile) *
                    (QBH_HMX_BIAS_BYTES / sizeof(uint32_t)));
    }
    header->u8_attention_k_pack_ticks +=
        HAP_perf_get_qtimer_count() - start;
    return 0;
}

static int qbh_scan_prepare_u8_delta_v(
    struct qbh_block_header *header,
    struct qbh_scan_u8_delta_attention_slot *slot,
    const uint8_t *cache_v, uint8_t *delta_rows,
    uint8_t *delta_lut, uint32_t decode_rows,
    uint32_t delta_bytes, uint32_t kv_tiles,
    int32_t *delta_lut_v_zero_point,
    uint32_t *delta_lut_numerator,
    uint32_t *delta_lut_denominator,
    const struct qbh_attention_config *config) {
    const uint32_t base_tile_bytes =
        QBH_ATTENTION_SCORE_TILES * QBH_HMX_WEIGHT_BYTES;
    const uint32_t destination_tile_stride =
        kv_tiles * QBH_HMX_WEIGHT_BYTES;
    const uint32_t tail_tile_bytes =
        destination_tile_stride - base_tile_bytes;
    uint64_t start;

    if (config->v_zero_point != *delta_lut_v_zero_point ||
        config->v_recenter_numerator != *delta_lut_numerator ||
        config->v_recenter_denominator != *delta_lut_denominator) {
        qbh_attention_u8_prepare_v_delta_lut(config, delta_lut);
        *delta_lut_v_zero_point = config->v_zero_point;
        *delta_lut_numerator = config->v_recenter_numerator;
        *delta_lut_denominator = config->v_recenter_denominator;
    }
    qbh_hvx_zero_strided_bytes(
        slot->weight + base_tile_bytes, tail_tile_bytes,
        QBH_ATTENTION_HEAD_DIM_TILES, destination_tile_stride);
    if (qbh_scan_cache_dma_2d(
            header, slot->weight, cache_v,
            base_tile_bytes, QBH_ATTENTION_HEAD_DIM_TILES,
            base_tile_bytes, destination_tile_stride) != 0 ||
        qbh_scan_cache_dma(
            header, slot->av_bias,
            cache_v +
                QBH_KV_CACHE_HMX_U8_BASE_WEIGHT_BYTES_PER_HEAD,
            QBH_KV_CACHE_HMX_V_BIAS_BYTES_PER_HEAD,
            1U) != 0 ||
        qbh_scan_cache_dma(
            header, delta_rows,
            cache_v +
                QBH_KV_CACHE_HMX_U8_BASE_WEIGHT_BYTES_PER_HEAD +
                QBH_KV_CACHE_HMX_V_BIAS_BYTES_PER_HEAD,
            delta_bytes, 1U) != 0) {
        return -1;
    }
    start = HAP_perf_get_qtimer_count();
    for (uint32_t first = 0U; first < decode_rows;
         first += QBH_HMX_INPUT_CHANNELS) {
        const uint32_t count =
            decode_rows - first < QBH_HMX_INPUT_CHANNELS
                ? decode_rows - first
                : QBH_HMX_INPUT_CHANNELS;
        const uint32_t tile = first / QBH_HMX_INPUT_CHANNELS;
        qbh_attention_u8_patch_v_delta_rows_hvx(
            delta_rows + (size_t)first * QBH_BLOCK_HEAD_DIM,
            count, config,
            slot->weight +
                (size_t)(QBH_BLOCK_M /
                         QBH_HMX_INPUT_CHANNELS + tile) *
                    QBH_HMX_WEIGHT_BYTES,
            destination_tile_stride, delta_lut,
            header->numerical_audit_enabled != 0U
                ? &slot->telemetry.v_recenter_saturation_count
                : NULL);
    }
    header->u8_attention_v_pack_ticks +=
        HAP_perf_get_qtimer_count() - start;
    return 0;
}

static void qbh_scan_accumulate_u8_attention_telemetry(
    struct qbh_block_header *header,
    const struct qbh_attention_u8_telemetry *telemetry) {
    header->u8_attention_score_saturation_count +=
        telemetry->score_saturation_count;
    header->u8_attention_v_recenter_saturation_count +=
        telemetry->v_recenter_saturation_count;
    header->u8_attention_probability_mask_violation_count +=
        telemetry->probability_mask_violation_count;
    if (header->u8_attention_probability_row_sum_min == 0U ||
        telemetry->probability_row_sum_min <
            header->u8_attention_probability_row_sum_min) {
        header->u8_attention_probability_row_sum_min =
            telemetry->probability_row_sum_min;
    }
    if (telemetry->probability_row_sum_max >
        header->u8_attention_probability_row_sum_max) {
        header->u8_attention_probability_row_sum_max =
            telemetry->probability_row_sum_max;
    }
}

static int qbh_scan_u8_attention_delta_pipeline(
    struct qbh_block_header *header, uint8_t *shared,
    struct qbh_block_buffers *buffers,
    struct qbh_block_hmx_worker *worker,
    uint32_t logical_rows, uint32_t past_tokens,
    uint32_t valid_tokens, uint32_t padded_tokens,
    uint32_t kv_tiles, uint32_t plane_bytes,
    uint32_t qk_bias_bytes, uint32_t slot_stride,
    uint32_t k_cache_head_stride,
    uint32_t v_cache_head_stride,
    uint32_t decode_rows, uint32_t delta_bytes,
    uint64_t dynamic_start) {
    struct qbh_scan_u8_delta_attention_slot slots[2];
    uint8_t *delta_rows = buffers->gate;
    int32_t delta_lut_v_zero_point =
        buffers->attention_configs[0].v_zero_point;
    uint32_t delta_lut_numerator =
        buffers->attention_configs[0].v_recenter_numerator;
    uint32_t delta_lut_denominator =
        buffers->attention_configs[0].v_recenter_denominator;
    uint64_t active_qk_compute_before;

    qbh_scan_u8_delta_slot_init(
        &slots[0], buffers->scores, plane_bytes, qk_bias_bytes);
    qbh_scan_u8_delta_slot_init(
        &slots[1], buffers->scores + slot_stride,
        plane_bytes, qk_bias_bytes);

    {
        const uint8_t *cache_k =
            shared + header->kv_cache_k_offset;
        uint8_t *q_group = buffers->q;
        if (qbh_scan_prepare_u8_delta_k(
                header, &slots[0], cache_k, delta_rows,
                decode_rows, delta_bytes, plane_bytes,
                &buffers->attention_configs[0]) != 0) {
            return -1;
        }
        active_qk_compute_before = worker->compute_ticks;
        qbh_hmx_start(
            worker, QBH_BLOCK_HMX_U8S8,
            q_group, slots[0].weight, slots[0].qk_bias,
            slots[0].scores,
            QBH_ATTENTION_Q_HEADS_PER_GROUP,
            QBH_ATTENTION_HEAD_DIM_TILES, kv_tiles);
    }

    for (uint32_t first = 0U;
         first < QBH_BLOCK_KV_HEADS; first += 2U) {
        const uint32_t second = first + 1U;
        const uint32_t next = first + 2U;
        struct qbh_scan_u8_delta_attention_slot *first_slot =
            &slots[first & 1U];
        struct qbh_scan_u8_delta_attention_slot *second_slot =
            &slots[second & 1U];
        const struct qbh_attention_config *first_config =
            &buffers->attention_configs[first];
        const struct qbh_attention_config *second_config =
            &buffers->attention_configs[second];
        uint8_t *first_q_group = buffers->q +
            (size_t)first * QBH_ATTENTION_Q_HEADS_PER_GROUP *
                QBH_ATTENTION_HEAD_DIM_TILES *
                QBH_HMX_ACTIVATION_BYTES;
        uint8_t *second_q_group = buffers->q +
            (size_t)second * QBH_ATTENTION_Q_HEADS_PER_GROUP *
                QBH_ATTENTION_HEAD_DIM_TILES *
                QBH_HMX_ACTIVATION_BYTES;
        const uint8_t *second_cache_k =
            shared + header->kv_cache_k_offset +
            (size_t)second * k_cache_head_stride;
        const uint8_t *first_cache_v =
            shared + header->kv_cache_v_offset +
            (size_t)first * v_cache_head_stride;
        const uint8_t *second_cache_v =
            shared + header->kv_cache_v_offset +
            (size_t)second * v_cache_head_stride;
        uint64_t wait_start;
        uint64_t compute_before;
        uint64_t start;

        /* QK[first] is active while K[second] is reconstructed. */
        if (qbh_scan_prepare_u8_delta_k(
                header, second_slot, second_cache_k, delta_rows,
                decode_rows, delta_bytes, plane_bytes,
                second_config) != 0) {
            return -1;
        }
        wait_start = HAP_perf_get_qtimer_count();
        if (qbh_hmx_wait(worker) != 0) {
            return -1;
        }
        header->u8_attention_pipeline_wait_ticks +=
            HAP_perf_get_qtimer_count() - wait_start;
        header->u8_attention_qk_hmx_ticks +=
            worker->compute_ticks - active_qk_compute_before;

        active_qk_compute_before = worker->compute_ticks;
        qbh_hmx_start(
            worker, QBH_BLOCK_HMX_U8S8,
            second_q_group, second_slot->weight,
            second_slot->qk_bias, second_slot->scores,
            QBH_ATTENTION_Q_HEADS_PER_GROUP,
            QBH_ATTENTION_HEAD_DIM_TILES, kv_tiles);

        start = HAP_perf_get_qtimer_count();
        qbh_attention_u8_requant_softmax_dynamic(
            first_slot->scores, first_slot->probability,
            logical_rows, past_tokens, valid_tokens,
            padded_tokens, first_config, &first_slot->telemetry);
        header->u8_attention_softmax_ticks +=
            HAP_perf_get_qtimer_count() - start;
        if (qbh_scan_prepare_u8_delta_v(
                header, first_slot, first_cache_v, delta_rows,
                buffers->up, decode_rows, delta_bytes, kv_tiles,
                &delta_lut_v_zero_point, &delta_lut_numerator,
                &delta_lut_denominator, first_config) != 0) {
            return -1;
        }

        wait_start = HAP_perf_get_qtimer_count();
        if (qbh_hmx_wait(worker) != 0) {
            return -1;
        }
        header->u8_attention_pipeline_wait_ticks +=
            HAP_perf_get_qtimer_count() - wait_start;
        header->u8_attention_qk_hmx_ticks +=
            worker->compute_ticks - active_qk_compute_before;

        compute_before = worker->compute_ticks;
        qbh_hmx_start(
            worker, QBH_BLOCK_HMX_U8S8,
            first_slot->probability, first_slot->weight,
            first_slot->av_bias, first_q_group,
            QBH_ATTENTION_Q_HEADS_PER_GROUP,
            kv_tiles, QBH_ATTENTION_HEAD_DIM_TILES);

        start = HAP_perf_get_qtimer_count();
        qbh_attention_u8_requant_softmax_dynamic(
            second_slot->scores, second_slot->probability,
            logical_rows, past_tokens, valid_tokens,
            padded_tokens, second_config, &second_slot->telemetry);
        header->u8_attention_softmax_ticks +=
            HAP_perf_get_qtimer_count() - start;
        if (qbh_scan_prepare_u8_delta_v(
                header, second_slot, second_cache_v, delta_rows,
                buffers->up, decode_rows, delta_bytes, kv_tiles,
                &delta_lut_v_zero_point, &delta_lut_numerator,
                &delta_lut_denominator, second_config) != 0) {
            return -1;
        }

        wait_start = HAP_perf_get_qtimer_count();
        if (qbh_hmx_wait(worker) != 0) {
            return -1;
        }
        header->u8_attention_pipeline_wait_ticks +=
            HAP_perf_get_qtimer_count() - wait_start;
        header->u8_attention_av_hmx_ticks +=
            worker->compute_ticks - compute_before;

        compute_before = worker->compute_ticks;
        qbh_hmx_start(
            worker, QBH_BLOCK_HMX_U8S8,
            second_slot->probability, second_slot->weight,
            second_slot->av_bias, second_q_group,
            QBH_ATTENTION_Q_HEADS_PER_GROUP,
            kv_tiles, QBH_ATTENTION_HEAD_DIM_TILES);

        start = HAP_perf_get_qtimer_count();
        qbh_attention_u8_requant_av(first_q_group, first_config);
        header->u8_attention_av_requant_ticks +=
            HAP_perf_get_qtimer_count() - start;
        qbh_scan_accumulate_u8_attention_telemetry(
            header, &first_slot->telemetry);

        if (next < QBH_BLOCK_KV_HEADS) {
            const uint8_t *next_cache_k =
                shared + header->kv_cache_k_offset +
                (size_t)next * k_cache_head_stride;
            if (qbh_scan_prepare_u8_delta_k(
                    header, first_slot, next_cache_k, delta_rows,
                    decode_rows, delta_bytes, plane_bytes,
                    &buffers->attention_configs[next]) != 0) {
                return -1;
            }
        }

        wait_start = HAP_perf_get_qtimer_count();
        if (qbh_hmx_wait(worker) != 0) {
            return -1;
        }
        header->u8_attention_pipeline_wait_ticks +=
            HAP_perf_get_qtimer_count() - wait_start;
        header->u8_attention_av_hmx_ticks +=
            worker->compute_ticks - compute_before;

        if (next < QBH_BLOCK_KV_HEADS) {
            uint8_t *next_q_group = buffers->q +
                (size_t)next * QBH_ATTENTION_Q_HEADS_PER_GROUP *
                    QBH_ATTENTION_HEAD_DIM_TILES *
                    QBH_HMX_ACTIVATION_BYTES;
            active_qk_compute_before = worker->compute_ticks;
            qbh_hmx_start(
                worker, QBH_BLOCK_HMX_U8S8,
                next_q_group, first_slot->weight,
                first_slot->qk_bias, first_slot->scores,
                QBH_ATTENTION_Q_HEADS_PER_GROUP,
                QBH_ATTENTION_HEAD_DIM_TILES, kv_tiles);
        }

        start = HAP_perf_get_qtimer_count();
        qbh_attention_u8_requant_av(second_q_group, second_config);
        header->u8_attention_av_requant_ticks +=
            HAP_perf_get_qtimer_count() - start;
        qbh_scan_accumulate_u8_attention_telemetry(
            header, &second_slot->telemetry);
    }

    header->u8_attention_group_count += QBH_BLOCK_KV_HEADS;
    header->u8_attention_qk_execution_count +=
        QBH_BLOCK_KV_HEADS * QBH_ATTENTION_Q_HEADS_PER_GROUP * kv_tiles;
    header->u8_attention_av_execution_count +=
        QBH_BLOCK_KV_HEADS * QBH_ATTENTION_Q_HEADS_PER_GROUP *
        QBH_ATTENTION_HEAD_DIM_TILES;
    header->hmx_command_count += 2U * QBH_BLOCK_KV_HEADS;
    header->hmx_u8s8_tile_pair_count +=
        2U * QBH_BLOCK_KV_HEADS * QBH_ATTENTION_Q_HEADS_PER_GROUP *
        QBH_ATTENTION_HEAD_DIM_TILES * kv_tiles;
    header->u8_attention_direct_o_tile_count +=
        QBH_BLOCK_HEADS * QBH_ATTENTION_HEAD_DIM_TILES;
    header->scan_dynamic_attention_ticks +=
        HAP_perf_get_qtimer_count() - dynamic_start;
    return 0;
}

static int qbh_scan_prepare_u8_segmented_short_k(
    struct qbh_block_header *header,
    struct qbh_scan_u8_delta_attention_slot *slot,
    const uint8_t *cache_k, const uint8_t *tail_k_rows,
    uint8_t *row_scratch,
    uint32_t sealed_segments, uint32_t tail_rows,
    const struct qbh_attention_config *config) {
    const uint32_t divisor = UINT32_C(1) << config->score_shift;
    const int32_t rounding = config->score_shift == 0U
        ? 0 : (int32_t)(divisor / 2U);
    int8_t *tail_weight = slot->weight +
        (size_t)sealed_segments *
            QBH_KV_CACHE_HMX_U8_SEGMENT_WEIGHT_BYTES;
    uint32_t *tail_bias = slot->qk_bias +
        (size_t)sealed_segments *
            (QBH_HMX_BIAS_BYTES / sizeof(uint32_t));
    uint64_t start;

    memset(&slot->telemetry, 0, sizeof(slot->telemetry));
    if (qbh_scan_cache_dma_2d(
            header, slot->weight, cache_k,
            QBH_KV_CACHE_HMX_U8_SEGMENT_WEIGHT_BYTES,
            sealed_segments,
            QBH_KV_CACHE_HMX_U8_SEGMENT_K_BYTES,
            QBH_KV_CACHE_HMX_U8_SEGMENT_WEIGHT_BYTES) != 0 ||
        qbh_scan_cache_dma_2d(
            header, slot->qk_bias,
            cache_k + QBH_KV_CACHE_HMX_U8_SEGMENT_WEIGHT_BYTES,
            QBH_HMX_BIAS_BYTES, sealed_segments,
            QBH_KV_CACHE_HMX_U8_SEGMENT_K_BYTES,
            QBH_HMX_BIAS_BYTES) != 0) {
        return -1;
    }
    if (tail_rows == 0U) {
        return 0;
    }
    qbh_hvx_zero_aligned_bytes(
        tail_weight, QBH_KV_CACHE_HMX_U8_SEGMENT_WEIGHT_BYTES);
    for (uint32_t output = 0U;
         output < QBH_HMX_OUTPUT_CHANNELS; ++output) {
        tail_bias[output] = slot->qk_bias[0];
        tail_bias[QBH_HMX_OUTPUT_CHANNELS + output] =
            UINT32_C(128) * divisor + rounding;
    }
    if (tail_k_rows == NULL ||
        qbh_scan_cache_dma(
            header, row_scratch, tail_k_rows,
            tail_rows * QBH_BLOCK_HEAD_DIM, 1U) != 0) {
        return -1;
    }
    start = HAP_perf_get_qtimer_count();
    qbh_attention_u8_patch_k_delta_rows_hvx(
        row_scratch, tail_rows, config, tail_weight, tail_bias);
    header->u8_attention_k_pack_ticks +=
        HAP_perf_get_qtimer_count() - start;
    return 0;
}

static int qbh_scan_prepare_u8_segmented_short_v(
    struct qbh_block_header *header,
    struct qbh_scan_u8_delta_attention_slot *slot,
    const uint8_t *cache_v, const uint8_t *tail_v_rows,
    uint8_t *row_scratch,
    uint8_t *pack_scratch, uint32_t max_segments,
    uint32_t sealed_segments,
    uint32_t tail_rows, uint32_t segment_count,
    int32_t *lut_v_zero_point,
    uint32_t *lut_numerator,
    uint32_t *lut_denominator,
    const struct qbh_attention_config *config) {
    const uint32_t destination_stride =
        segment_count * QBH_HMX_WEIGHT_BYTES;
    const uint32_t source_plane_bytes =
        max_segments * QBH_HMX_WEIGHT_BYTES;
    uint64_t start;

    if (qbh_scan_cache_dma_2d(
            header, slot->weight, cache_v,
            sealed_segments * QBH_HMX_WEIGHT_BYTES,
            QBH_ATTENTION_HEAD_DIM_TILES,
            source_plane_bytes, destination_stride) != 0) {
        return -1;
    }
    if (qbh_scan_cache_dma(
            header, slot->av_bias,
            cache_v + (size_t)max_segments *
                QBH_KV_CACHE_HMX_U8_SEGMENT_V_BYTES,
            QBH_KV_CACHE_HMX_V_BIAS_BYTES_PER_HEAD, 1U) != 0 ||
        (tail_rows != 0U && tail_v_rows == NULL)) {
        return -1;
    }
    if (tail_rows == 0U) {
        return 0;
    }
    if (config->v_zero_point != *lut_v_zero_point ||
        config->v_recenter_numerator != *lut_numerator ||
        config->v_recenter_denominator != *lut_denominator) {
        qbh_attention_u8_prepare_v_delta_lut(config, pack_scratch);
        *lut_v_zero_point = config->v_zero_point;
        *lut_numerator = config->v_recenter_numerator;
        *lut_denominator = config->v_recenter_denominator;
    }
    qbh_hvx_zero_strided_bytes(
        slot->weight +
            (size_t)sealed_segments * QBH_HMX_WEIGHT_BYTES,
        QBH_HMX_WEIGHT_BYTES,
        QBH_ATTENTION_HEAD_DIM_TILES, destination_stride);
    start = HAP_perf_get_qtimer_count();
    if (qbh_scan_cache_dma(
            header, row_scratch, tail_v_rows,
            tail_rows * QBH_BLOCK_HEAD_DIM, 1U) != 0) {
        return -1;
    }
    qbh_attention_u8_patch_v_delta_rows_hvx(
        row_scratch, tail_rows, config,
        slot->weight +
            (size_t)sealed_segments * QBH_HMX_WEIGHT_BYTES,
        destination_stride, pack_scratch,
        header->numerical_audit_enabled != 0U
            ? &slot->telemetry.v_recenter_saturation_count
            : NULL);
    header->u8_attention_v_pack_ticks +=
        HAP_perf_get_qtimer_count() - start;
    return 0;
}

static int qbh_scan_u8_attention_segmented_short_pipeline(
    struct qbh_block_header *header, uint8_t *shared,
    struct qbh_block_buffers *buffers,
    struct qbh_block_hmx_worker *worker,
    uint32_t logical_rows, uint32_t past_tokens,
    uint32_t valid_tokens, uint32_t padded_tokens,
    uint32_t segment_count, uint32_t plane_bytes,
    uint32_t qk_bias_bytes, uint32_t slot_stride,
    uint32_t k_cache_head_stride,
    uint32_t v_cache_head_stride,
    uint32_t max_segments, uint32_t sealed_segments,
    uint32_t tail_rows,
    uint64_t dynamic_start) {
    struct qbh_scan_u8_delta_attention_slot slots[2];
    uint8_t *row_scratch = buffers->gate;
    uint8_t *pack_scratch = buffers->up;
    int32_t lut_v_zero_point =
        buffers->attention_configs[0].v_zero_point;
    uint32_t lut_numerator =
        buffers->attention_configs[0].v_recenter_numerator;
    uint32_t lut_denominator =
        buffers->attention_configs[0].v_recenter_denominator;
    uint64_t active_qk_compute_before;

    qbh_scan_u8_delta_slot_init(
        &slots[0], buffers->scores, plane_bytes, qk_bias_bytes);
    qbh_scan_u8_delta_slot_init(
        &slots[1], buffers->scores + slot_stride,
        plane_bytes, qk_bias_bytes);
    qbh_attention_u8_prepare_v_delta_lut(
        &buffers->attention_configs[0], pack_scratch);

    if (qbh_scan_prepare_u8_segmented_short_k(
            header, &slots[0],
            shared + header->kv_cache_k_offset,
            shared + header->kv_cache_k_offset +
                (size_t)max_segments *
                    QBH_KV_CACHE_HMX_U8_SEGMENT_K_BYTES,
            row_scratch, sealed_segments, tail_rows,
            &buffers->attention_configs[0]) != 0) {
        return -1;
    }
    active_qk_compute_before = worker->compute_ticks;
    qbh_hmx_start(
        worker, QBH_BLOCK_HMX_U8S8,
        buffers->q, slots[0].weight, slots[0].qk_bias,
        slots[0].scores, QBH_ATTENTION_Q_HEADS_PER_GROUP,
        QBH_ATTENTION_HEAD_DIM_TILES, segment_count);

    for (uint32_t first = 0U;
         first < QBH_BLOCK_KV_HEADS; first += 2U) {
        const uint32_t second = first + 1U;
        const uint32_t next = first + 2U;
        struct qbh_scan_u8_delta_attention_slot *first_slot =
            &slots[first & 1U];
        struct qbh_scan_u8_delta_attention_slot *second_slot =
            &slots[second & 1U];
        const struct qbh_attention_config *first_config =
            &buffers->attention_configs[first];
        const struct qbh_attention_config *second_config =
            &buffers->attention_configs[second];
        uint8_t *first_q_group = buffers->q +
            (size_t)first * QBH_ATTENTION_Q_HEADS_PER_GROUP *
                QBH_ATTENTION_HEAD_DIM_TILES *
                QBH_HMX_ACTIVATION_BYTES;
        uint8_t *second_q_group = buffers->q +
            (size_t)second * QBH_ATTENTION_Q_HEADS_PER_GROUP *
                QBH_ATTENTION_HEAD_DIM_TILES *
                QBH_HMX_ACTIVATION_BYTES;
        uint64_t wait_start;
        uint64_t compute_before;
        uint64_t start;

        if (qbh_scan_prepare_u8_segmented_short_k(
                header, second_slot,
                shared + header->kv_cache_k_offset +
                    (size_t)second * k_cache_head_stride,
                shared + header->kv_cache_k_offset +
                    (size_t)second * k_cache_head_stride +
                    (size_t)max_segments *
                        QBH_KV_CACHE_HMX_U8_SEGMENT_K_BYTES,
                row_scratch, sealed_segments, tail_rows,
                second_config) != 0) {
            return -1;
        }
        wait_start = HAP_perf_get_qtimer_count();
        if (qbh_hmx_wait(worker) != 0) {
            return -1;
        }
        header->u8_attention_pipeline_wait_ticks +=
            HAP_perf_get_qtimer_count() - wait_start;
        header->u8_attention_qk_hmx_ticks +=
            worker->compute_ticks - active_qk_compute_before;

        active_qk_compute_before = worker->compute_ticks;
        qbh_hmx_start(
            worker, QBH_BLOCK_HMX_U8S8,
            second_q_group, second_slot->weight,
            second_slot->qk_bias, second_slot->scores,
            QBH_ATTENTION_Q_HEADS_PER_GROUP,
            QBH_ATTENTION_HEAD_DIM_TILES, segment_count);

        start = HAP_perf_get_qtimer_count();
        qbh_attention_u8_requant_softmax_dynamic(
            first_slot->scores, first_slot->probability,
            logical_rows, past_tokens, valid_tokens,
            padded_tokens, first_config, &first_slot->telemetry);
        header->u8_attention_softmax_ticks +=
            HAP_perf_get_qtimer_count() - start;
        if (qbh_scan_prepare_u8_segmented_short_v(
                header, first_slot,
                shared + header->kv_cache_v_offset +
                    (size_t)first * v_cache_head_stride,
                shared + header->kv_cache_v_offset +
                    (size_t)first * v_cache_head_stride +
                    (size_t)max_segments *
                        QBH_KV_CACHE_HMX_U8_SEGMENT_V_BYTES +
                    QBH_KV_CACHE_HMX_V_BIAS_BYTES_PER_HEAD,
                row_scratch, pack_scratch,
                max_segments, sealed_segments, tail_rows,
                segment_count,
                &lut_v_zero_point, &lut_numerator,
                &lut_denominator,
                first_config) != 0) {
            return -1;
        }

        wait_start = HAP_perf_get_qtimer_count();
        if (qbh_hmx_wait(worker) != 0) {
            return -1;
        }
        header->u8_attention_pipeline_wait_ticks +=
            HAP_perf_get_qtimer_count() - wait_start;
        header->u8_attention_qk_hmx_ticks +=
            worker->compute_ticks - active_qk_compute_before;

        compute_before = worker->compute_ticks;
        qbh_hmx_start(
            worker, QBH_BLOCK_HMX_U8S8,
            first_slot->probability, first_slot->weight,
            first_slot->av_bias, first_q_group,
            QBH_ATTENTION_Q_HEADS_PER_GROUP,
            segment_count, QBH_ATTENTION_HEAD_DIM_TILES);

        start = HAP_perf_get_qtimer_count();
        qbh_attention_u8_requant_softmax_dynamic(
            second_slot->scores, second_slot->probability,
            logical_rows, past_tokens, valid_tokens,
            padded_tokens, second_config, &second_slot->telemetry);
        header->u8_attention_softmax_ticks +=
            HAP_perf_get_qtimer_count() - start;
        if (qbh_scan_prepare_u8_segmented_short_v(
                header, second_slot,
                shared + header->kv_cache_v_offset +
                    (size_t)second * v_cache_head_stride,
                shared + header->kv_cache_v_offset +
                    (size_t)second * v_cache_head_stride +
                    (size_t)max_segments *
                        QBH_KV_CACHE_HMX_U8_SEGMENT_V_BYTES +
                    QBH_KV_CACHE_HMX_V_BIAS_BYTES_PER_HEAD,
                row_scratch, pack_scratch,
                max_segments, sealed_segments, tail_rows,
                segment_count,
                &lut_v_zero_point, &lut_numerator,
                &lut_denominator,
                second_config) != 0) {
            return -1;
        }

        wait_start = HAP_perf_get_qtimer_count();
        if (qbh_hmx_wait(worker) != 0) {
            return -1;
        }
        header->u8_attention_pipeline_wait_ticks +=
            HAP_perf_get_qtimer_count() - wait_start;
        header->u8_attention_av_hmx_ticks +=
            worker->compute_ticks - compute_before;

        compute_before = worker->compute_ticks;
        qbh_hmx_start(
            worker, QBH_BLOCK_HMX_U8S8,
            second_slot->probability, second_slot->weight,
            second_slot->av_bias, second_q_group,
            QBH_ATTENTION_Q_HEADS_PER_GROUP,
            segment_count, QBH_ATTENTION_HEAD_DIM_TILES);

        start = HAP_perf_get_qtimer_count();
        qbh_attention_u8_requant_av(first_q_group, first_config);
        header->u8_attention_av_requant_ticks +=
            HAP_perf_get_qtimer_count() - start;
        qbh_scan_accumulate_u8_attention_telemetry(
            header, &first_slot->telemetry);

        if (next < QBH_BLOCK_KV_HEADS &&
            qbh_scan_prepare_u8_segmented_short_k(
                header, first_slot,
                shared + header->kv_cache_k_offset +
                    (size_t)next * k_cache_head_stride,
                shared + header->kv_cache_k_offset +
                    (size_t)next * k_cache_head_stride +
                    (size_t)max_segments *
                        QBH_KV_CACHE_HMX_U8_SEGMENT_K_BYTES,
                row_scratch, sealed_segments, tail_rows,
                &buffers->attention_configs[next]) != 0) {
            return -1;
        }

        wait_start = HAP_perf_get_qtimer_count();
        if (qbh_hmx_wait(worker) != 0) {
            return -1;
        }
        header->u8_attention_pipeline_wait_ticks +=
            HAP_perf_get_qtimer_count() - wait_start;
        header->u8_attention_av_hmx_ticks +=
            worker->compute_ticks - compute_before;

        if (next < QBH_BLOCK_KV_HEADS) {
            active_qk_compute_before = worker->compute_ticks;
            qbh_hmx_start(
                worker, QBH_BLOCK_HMX_U8S8,
                buffers->q +
                    (size_t)next *
                        QBH_ATTENTION_Q_HEADS_PER_GROUP *
                        QBH_ATTENTION_HEAD_DIM_TILES *
                        QBH_HMX_ACTIVATION_BYTES,
                first_slot->weight, first_slot->qk_bias,
                first_slot->scores,
                QBH_ATTENTION_Q_HEADS_PER_GROUP,
                QBH_ATTENTION_HEAD_DIM_TILES, segment_count);
        }

        start = HAP_perf_get_qtimer_count();
        qbh_attention_u8_requant_av(second_q_group, second_config);
        header->u8_attention_av_requant_ticks +=
            HAP_perf_get_qtimer_count() - start;
        qbh_scan_accumulate_u8_attention_telemetry(
            header, &second_slot->telemetry);
    }

    header->u8_attention_group_count += QBH_BLOCK_KV_HEADS;
    header->u8_attention_qk_execution_count +=
        QBH_BLOCK_KV_HEADS * QBH_ATTENTION_Q_HEADS_PER_GROUP *
        segment_count;
    header->u8_attention_av_execution_count +=
        QBH_BLOCK_KV_HEADS * QBH_ATTENTION_Q_HEADS_PER_GROUP *
        QBH_ATTENTION_HEAD_DIM_TILES;
    header->hmx_command_count += 2U * QBH_BLOCK_KV_HEADS;
    header->hmx_u8s8_tile_pair_count +=
        2U * QBH_BLOCK_KV_HEADS *
        QBH_ATTENTION_Q_HEADS_PER_GROUP *
        QBH_ATTENTION_HEAD_DIM_TILES * segment_count;
    header->u8_attention_direct_o_tile_count +=
        QBH_BLOCK_HEADS * QBH_ATTENTION_HEAD_DIM_TILES;
    header->scan_dynamic_attention_ticks +=
        HAP_perf_get_qtimer_count() - dynamic_start;
    return 0;
}

static uint8_t qbh_scan_clip_u8(int64_t value) {
    if (value < 0) {
        return 0U;
    }
    if (value > UINT8_MAX) {
        return UINT8_MAX;
    }
    return (uint8_t)value;
}

static void qbh_scan_segment_qk_hvx_head(
    const uint8_t *q_head, const int8_t *k_weight,
    const uint32_t *qk_bias,
    const struct qbh_attention_config *config,
    uint8_t *score_tile) {
    const HVX_Vector zero = Q6_V_vzero();
    HVX_Vector accumulator = Q6_V_vzero();

    for (uint32_t k_tile = 0U;
         k_tile < QBH_ATTENTION_HEAD_DIM_TILES; ++k_tile) {
        const uint8_t *q_tile = q_head +
            (size_t)k_tile * QBH_HMX_ACTIVATION_BYTES;
        const int8_t *weight_tile = k_weight +
            (size_t)k_tile * QBH_HMX_WEIGHT_BYTES;
        for (uint32_t input_group = 0U;
             input_group < QBH_HMX_INPUT_CHANNELS / 4U;
             ++input_group) {
            uint32_t query_word;
            const HVX_Vector weights = *(const HVX_Vector *)(
                weight_tile +
                (size_t)input_group * QBH_BLOCK_ALIGNMENT);
            memcpy(&query_word,
                   q_tile + input_group * 4U,
                   sizeof(query_word));
            accumulator = Q6_Vw_vrmpyacc_VwVubVb(
                accumulator, Q6_V_vsplat_R(query_word),
                weights);
        }
    }
    accumulator = Q6_Vw_vadd_VwVw(
        accumulator,
        *(const HVX_Vector *)(qk_bias +
            QBH_HMX_OUTPUT_CHANNELS));
    accumulator = Q6_Vw_vasr_VwR(
        accumulator, config->score_shift);
    *(HVX_Vector *)score_tile = Q6_Vub_vsat_VhVh(
        Q6_Vh_vsat_VwVw(accumulator, zero), zero);
}

static void qbh_scan_segment_record_histogram_head(
    uint32_t *histogram, uint8_t *active_scores,
    uint32_t *active_count, const uint8_t *scores,
    uint32_t valid_rows) {
    for (uint32_t token = 0U; token < valid_rows; ++token) {
        const uint8_t score = scores[token];
        if (histogram[score] == 0U) {
            active_scores[(*active_count)++] = score;
        }
        ++histogram[score];
    }
}

static void qbh_scan_segment_accumulate_v_direct_head(
    int32_t *accumulators, const uint8_t *probability_map,
    const uint8_t *scores, const int8_t *v_weight,
    uint32_t valid_rows) {
    HVX_Vector sums[QBH_ATTENTION_HEAD_DIM_TILES]
        __attribute__((aligned(QBH_BLOCK_ALIGNMENT)));

    for (uint32_t output_tile = 0U;
         output_tile < QBH_ATTENTION_HEAD_DIM_TILES;
         ++output_tile) {
        sums[output_tile] = *(const HVX_Vector *)(
            accumulators +
            (size_t)output_tile * QBH_HMX_OUTPUT_CHANNELS);
    }

    for (uint32_t input_group = 0U;
         input_group * 4U < valid_rows; ++input_group) {
        const uint32_t segment_first = input_group * 4U;
        uint32_t probability_word = 0U;

        for (uint32_t lane = 0U; lane < 4U; ++lane) {
            const uint32_t segment_token = segment_first + lane;
            if (segment_token >= valid_rows) {
                break;
            }
            probability_word |=
                (uint32_t)probability_map[scores[segment_token]] <<
                (lane * 8U);
        }
        for (uint32_t output_tile = 0U;
             output_tile < QBH_ATTENTION_HEAD_DIM_TILES; ++output_tile) {
            const HVX_Vector weights = *(const HVX_Vector *)(
                v_weight +
                (size_t)output_tile * QBH_HMX_WEIGHT_BYTES +
                (size_t)input_group * QBH_BLOCK_ALIGNMENT);
            sums[output_tile] = Q6_Vw_vrmpyacc_VwVubVb(
                sums[output_tile],
                Q6_V_vsplat_R(probability_word), weights);
        }
    }
    for (uint32_t output_tile = 0U;
         output_tile < QBH_ATTENTION_HEAD_DIM_TILES;
         ++output_tile) {
        *(HVX_Vector *)(accumulators +
            (size_t)output_tile * QBH_HMX_OUTPUT_CHANNELS) =
            sums[output_tile];
    }
}

static void qbh_scan_segment_prepare_probability_maps(
    const uint32_t *histogram,
    const uint8_t *active_scores,
    const uint32_t active_counts[2],
    const struct qbh_attention_config *config,
    struct qbh_attention_u8_telemetry *telemetry,
    uint8_t *probability_maps, uint32_t valid_tokens);
static void qbh_scan_segment_finalize_av(
    uint8_t *output_group, const int32_t *accumulators,
    const struct qbh_attention_config *config);

static void qbh_attention_u8_segmented_pool_run_tasks(
    struct qbh_block_w4f16_pool *pool,
    struct qbh_block_w4f16_job *job) {
    for (;;) {
        const uint32_t task = qbh_atomic_fetch_increment(
            &pool->next_attention_task);
        const uint32_t group =
            task / QBH_ATTENTION_Q_HEADS_PER_GROUP;
        const uint32_t local_head =
            task % QBH_ATTENTION_Q_HEADS_PER_GROUP;
        uint8_t *group_base;
        uint8_t *histogram_base;
        uint8_t *active_score_base;
        uint8_t *probability_map_base;
        uint8_t *accumulator_base;
        uint8_t *ring;
        uint8_t *q_head;

        if (task >= QBH_BLOCK_HEADS ||
            pool->attention_gqa_abort != 0U) {
            break;
        }
        group_base = pool->segmented_attention_scratch +
            (size_t)group * pool->segmented_attention_group_stride;
        histogram_base = group_base;
        active_score_base = histogram_base +
            QBH_BLOCK_U8_SEGMENT_HISTOGRAM_BYTES;
        probability_map_base = active_score_base +
            QBH_BLOCK_U8_SEGMENT_ACTIVE_SCORE_BYTES;
        accumulator_base = probability_map_base +
            QBH_ATTENTION_Q_HEADS_PER_GROUP * 256U;
        ring = group_base + pool->segmented_attention_ring_offset;
        q_head = pool->attention_buffers->q +
            (size_t)task *
                QBH_ATTENTION_HEAD_DIM_TILES *
                QBH_HMX_ACTIVATION_BYTES;

        for (uint32_t local_segment = 0U;
             local_segment < pool->segmented_attention_batch_count;
             ++local_segment) {
            uint8_t *slot_base = ring +
                (size_t)local_segment *
                    QBH_BLOCK_U8_SEGMENT_SLOT_BYTES;
            const uint64_t qk_start = HAP_perf_get_qtimer_count();

            qbh_scan_segment_qk_hvx_head(
                q_head,
                (const int8_t *)(slot_base +
                    QBH_BLOCK_U8_SEGMENT_K_WEIGHT_OFFSET),
                (const uint32_t *)(slot_base +
                    QBH_BLOCK_U8_SEGMENT_K_BIAS_OFFSET),
                &pool->attention_buffers->attention_configs[group],
                slot_base + QBH_BLOCK_U8_SEGMENT_SCORE_OFFSET +
                    (size_t)local_head * QBH_HMX_OUTPUT_BYTES);
            job->u8_attention_qk_requant_ticks +=
                HAP_perf_get_qtimer_count() - qk_start;
            if (pool->segmented_attention_phase == 1U) {
                qbh_scan_segment_record_histogram_head(
                    (uint32_t *)histogram_base +
                        (size_t)local_head * 256U,
                    active_score_base + (size_t)local_head * 256U,
                    &pool->segmented_attention_active_counts
                        [group][local_head],
                    slot_base + QBH_BLOCK_U8_SEGMENT_SCORE_OFFSET +
                        (size_t)local_head * QBH_HMX_OUTPUT_BYTES,
                    pool->segmented_attention_rows[local_segment]);
            } else {
                const uint64_t av_start =
                    HAP_perf_get_qtimer_count();
                qbh_scan_segment_accumulate_v_direct_head(
                    (int32_t *)accumulator_base +
                        (size_t)local_head * QBH_BLOCK_HEAD_DIM,
                    probability_map_base +
                        (size_t)local_head * 256U,
                    slot_base + QBH_BLOCK_U8_SEGMENT_SCORE_OFFSET +
                        (size_t)local_head * QBH_HMX_OUTPUT_BYTES,
                    (const int8_t *)(slot_base +
                        QBH_BLOCK_U8_SEGMENT_V_WEIGHT_OFFSET),
                    pool->segmented_attention_rows[local_segment]);
                job->u8_attention_av_requant_ticks +=
                    HAP_perf_get_qtimer_count() - av_start;
            }
        }
    }
}

static int qbh_hvx_pool_u8_segmented_attention_start_async(
    struct qbh_block_w4f16_pool *pool) {
    const uint32_t worker_count =
        pool->attention_header->attention_hvx_contexts - 2U;

    if (worker_count == 0U || worker_count > pool->worker_count) {
        return -1;
    }
    pool->next_attention_task = 0U;
    pool->active_worker_count = worker_count;
    for (uint32_t worker = 0U; worker < worker_count; ++worker) {
        if (pool->jobs[worker].command_kind !=
            QBH_BLOCK_HVX_POOL_NONE) {
            return -1;
        }
        pool->jobs[worker].command_kind =
            QBH_BLOCK_HVX_POOL_U8_SEGMENTED_ATTENTION;
    }
    asm volatile("barrier" ::: "memory");
    for (uint32_t worker = 0U; worker < worker_count; ++worker) {
        (void)qurt_sem_up(&pool->command_ready[worker]);
    }
    return 0;
}

static int qbh_hvx_pool_u8_segmented_attention_wait_async(
    struct qbh_block_w4f16_pool *pool) {
    qbh_w4f16_pool_wait(pool);
    return pool->attention_gqa_abort == 0U ? 0 : -1;
}

static void qbh_scan_segment_prepare_probability_maps(
    const uint32_t *histogram,
    const uint8_t *active_scores,
    const uint32_t active_counts[2],
    const struct qbh_attention_config *config,
    struct qbh_attention_u8_telemetry *telemetry,
    uint8_t *probability_maps, uint32_t valid_tokens) {
    uint32_t row_sum_min = UINT32_MAX;
    uint32_t row_sum_max = 0U;

    for (uint32_t head = 0U;
         head < QBH_ATTENTION_Q_HEADS_PER_GROUP; ++head) {
        const uint32_t *head_histogram =
            histogram + (size_t)head * 256U;
        uint8_t *probability_by_raw =
            probability_maps + (size_t)head * 256U;
        uint32_t probability_sum = 0U;
        uint32_t saturation_count = 0U;

        qbh_attention_u8_probability_map_from_active_histogram(
            head_histogram,
            active_scores + (size_t)head * 256U,
            active_counts[head], valid_tokens, config,
            probability_by_raw, &probability_sum,
            &saturation_count);
        telemetry->score_saturation_count += saturation_count;
        if (probability_sum < row_sum_min) {
            row_sum_min = probability_sum;
        }
        if (probability_sum > row_sum_max) {
            row_sum_max = probability_sum;
        }
    }
    telemetry->probability_row_sum_min = row_sum_min;
    telemetry->probability_row_sum_max = row_sum_max;
}

static void qbh_scan_segment_finalize_av(
    uint8_t *output_group, const int32_t *accumulators,
    const struct qbh_attention_config *config) {
    const int64_t divisor = INT64_C(1) << config->av_shift;
    const int64_t rounding = config->av_shift == 0U
        ? 0 : divisor / 2;
    const int32_t hmx_output_zero_point =
        config->av_multiplier == 1U
            ? config->output_zero_point
            : (int32_t)QBH_ATTENTION_HMX_CENTER;

    qbh_hvx_zero_aligned_bytes(
        output_group,
        QBH_ATTENTION_Q_HEADS_PER_GROUP *
            QBH_ATTENTION_HEAD_DIM_TILES *
            QBH_HMX_OUTPUT_BYTES);
    for (uint32_t head = 0U;
         head < QBH_ATTENTION_Q_HEADS_PER_GROUP; ++head) {
        for (uint32_t channel = 0U;
             channel < QBH_BLOCK_HEAD_DIM; ++channel) {
            int64_t scaled;
            uint8_t output;
            const int32_t dot = accumulators[
                (size_t)head * QBH_BLOCK_HEAD_DIM + channel];
            scaled = ((int64_t)dot +
                      (int64_t)hmx_output_zero_point * divisor +
                      rounding) >> config->av_shift;
            output = qbh_scan_clip_u8(scaled);
            if (config->av_multiplier != 1U) {
                output = qbh_scan_clip_u8(
                    ((int32_t)output -
                     (int32_t)QBH_ATTENTION_HMX_CENTER) *
                        (int32_t)config->av_multiplier +
                    config->output_zero_point);
            }
            output_group[
                (size_t)head * QBH_ATTENTION_HEAD_DIM_TILES *
                    QBH_HMX_OUTPUT_BYTES +
                (size_t)(channel / QBH_HMX_OUTPUT_CHANNELS) *
                    QBH_HMX_OUTPUT_BYTES +
                channel % QBH_HMX_OUTPUT_CHANNELS] = output;
        }
    }
    asm volatile("barrier" ::: "memory");
}

static int qbh_scan_stage_u8_segment(
    struct qbh_block_header *header, uint8_t *slot_base,
    const uint8_t *cache_k, const uint8_t *cache_v,
    const uint8_t *tail_k_head, const uint8_t *tail_v_head,
    uint32_t segment, uint32_t sealed_segments,
    uint32_t tail_rows, uint8_t *row_scratch,
    uint8_t *bias_scratch,
    const struct qbh_attention_config *config,
    struct qbh_attention_u8_telemetry *telemetry,
    uint32_t stage_k, uint32_t stage_v) {
    int8_t *k_weight = (int8_t *)(slot_base +
        QBH_BLOCK_U8_SEGMENT_K_WEIGHT_OFFSET);
    uint32_t *qk_bias = (uint32_t *)(slot_base +
        QBH_BLOCK_U8_SEGMENT_K_BIAS_OFFSET);
    int8_t *v_weight = (int8_t *)(slot_base +
        QBH_BLOCK_U8_SEGMENT_V_WEIGHT_OFFSET);
    const uint32_t max_segments =
        QBH_KV_CACHE_HMX_U8_SEGMENT_COUNT(header->kv_cache_capacity);

    if (segment < sealed_segments) {
        if (stage_k != 0U && qbh_scan_cache_dma(
                header, k_weight,
                cache_k + (size_t)segment *
                    QBH_KV_CACHE_HMX_U8_SEGMENT_K_BYTES,
                QBH_KV_CACHE_HMX_U8_SEGMENT_K_BYTES, 1U) != 0) {
            return -1;
        }
        if (stage_v != 0U) {
            const uint32_t block_first =
                (segment /
                 QBH_KV_CACHE_HMX_U8_V_SEGMENT_BLOCK_SEGMENTS) *
                QBH_KV_CACHE_HMX_U8_V_SEGMENT_BLOCK_SEGMENTS;
            const uint32_t block_count =
                max_segments - block_first <
                        QBH_KV_CACHE_HMX_U8_V_SEGMENT_BLOCK_SEGMENTS
                    ? max_segments - block_first
                    : QBH_KV_CACHE_HMX_U8_V_SEGMENT_BLOCK_SEGMENTS;
            const uint32_t source_plane_stride =
                block_count * QBH_HMX_WEIGHT_BYTES;
            const uint8_t *source = cache_v +
                (size_t)block_first *
                    QBH_KV_CACHE_HMX_U8_SEGMENT_WEIGHT_BYTES +
                (size_t)(segment - block_first) *
                    QBH_HMX_WEIGHT_BYTES;
            if (qbh_scan_cache_dma_2d(
                    header, v_weight, source,
                    QBH_HMX_WEIGHT_BYTES,
                    QBH_ATTENTION_HEAD_DIM_TILES,
                    source_plane_stride, QBH_HMX_WEIGHT_BYTES) != 0) {
                return -1;
            }
        }
        return 0;
    }

    if (segment != sealed_segments || tail_rows == 0U ||
        tail_rows > QBH_KV_CACHE_HMX_U8_SEGMENT_TOKENS) {
        return -1;
    }
    if (stage_k != 0U) {
        if (tail_k_head == NULL) {
            return -1;
        }
        if (qbh_scan_cache_dma(
                header, row_scratch, tail_k_head,
                tail_rows * QBH_BLOCK_HEAD_DIM, 1U) != 0) {
            return -1;
        }
        qbh_attention_u8_pack_k_row_major(
            row_scratch, tail_rows,
            QBH_KV_CACHE_HMX_U8_SEGMENT_TOKENS,
            config, k_weight, qk_bias);
    }
    if (stage_v != 0U) {
        if (tail_v_head == NULL) {
            return -1;
        }
        if (qbh_scan_cache_dma(
                header, row_scratch, tail_v_head,
                tail_rows * QBH_BLOCK_HEAD_DIM, 1U) != 0) {
            return -1;
        }
        qbh_attention_u8_pack_v_row_major(
            row_scratch, tail_rows,
            QBH_KV_CACHE_HMX_U8_SEGMENT_TOKENS,
            config, v_weight, (uint32_t *)bias_scratch,
            header->numerical_audit_enabled != 0U
                ? &telemetry->v_recenter_saturation_count : NULL);
    }
    return 0;
}

static int qbh_scan_stage_u8_segment_batch(
    struct qbh_block_header *header, uint8_t *shared,
    struct qbh_block_buffers *buffers,
    struct qbh_attention_u8_telemetry *telemetry,
    uint32_t k_head_stride, uint32_t v_head_stride,
    uint32_t group_stride, uint32_t bank_ring_offset,
    uint32_t sealed_segments, uint32_t tail_rows,
    uint8_t *row_scratch, uint8_t *bias_scratch,
    uint32_t batch_first, uint32_t batch_count,
    uint32_t stage_k, uint32_t stage_v) {
    const uint32_t max_segments =
        QBH_KV_CACHE_HMX_U8_SEGMENT_COUNT(header->kv_cache_capacity);
    for (uint32_t group = 0U;
         group < QBH_BLOCK_KV_HEADS; ++group) {
        const uint8_t *cache_k =
            shared + header->kv_cache_k_offset +
            (size_t)group * k_head_stride;
        const uint8_t *cache_v =
            shared + header->kv_cache_v_offset +
            (size_t)group * v_head_stride;
        const uint8_t *tail_k_head = cache_k +
            (size_t)max_segments *
                QBH_KV_CACHE_HMX_U8_SEGMENT_K_BYTES;
        const uint8_t *tail_v_head = cache_v +
            (size_t)max_segments *
                QBH_KV_CACHE_HMX_U8_SEGMENT_V_BYTES +
            QBH_KV_CACHE_HMX_V_BIAS_BYTES_PER_HEAD;
        uint8_t *ring = buffers->scores +
            (size_t)group * group_stride + bank_ring_offset;

        for (uint32_t local_segment = 0U;
             local_segment < batch_count; ++local_segment) {
            if (qbh_scan_stage_u8_segment(
                    header,
                    ring + (size_t)local_segment *
                    QBH_BLOCK_U8_SEGMENT_SLOT_BYTES,
                    cache_k, cache_v,
                    tail_k_head, tail_v_head,
                    batch_first + local_segment,
                    sealed_segments, tail_rows,
                    row_scratch, bias_scratch,
                    &buffers->attention_configs[group],
                    &telemetry[group], stage_k, stage_v) != 0) {
                return -1;
            }
        }
    }
    return 0;
}

static int qbh_scan_run_u8_segmented_async_phase(
    struct qbh_block_header *header, uint8_t *shared,
    struct qbh_block_buffers *buffers,
    struct qbh_block_w4f16_pool *pool,
    struct qbh_attention_u8_telemetry *telemetry,
    uint32_t k_head_stride, uint32_t v_head_stride,
    uint32_t group_stride, uint32_t base_ring_offset,
    uint32_t sealed_segments, uint32_t tail_rows,
    uint32_t segment_count, uint8_t *row_scratch,
    uint8_t *bias_scratch, uint32_t phase,
    uint32_t stage_k, uint32_t stage_v) {
    uint32_t batch_first = 0U;
    uint32_t batch_count =
        segment_count < QBH_BLOCK_U8_SEGMENT_ASYNC_BANK_SLOTS
            ? segment_count
            : QBH_BLOCK_U8_SEGMENT_ASYNC_BANK_SLOTS;
    uint32_t bank = 0U;

    if (qbh_scan_stage_u8_segment_batch(
            header, shared, buffers, telemetry,
            k_head_stride, v_head_stride, group_stride,
            base_ring_offset, sealed_segments, tail_rows,
            row_scratch, bias_scratch, batch_first, batch_count,
            stage_k, stage_v) != 0) {
        return -1;
    }

    for (;;) {
        const uint32_t next_first = batch_first + batch_count;
        const uint32_t next_count =
            next_first < segment_count
                ? (segment_count - next_first <
                           QBH_BLOCK_U8_SEGMENT_ASYNC_BANK_SLOTS
                       ? segment_count - next_first
                       : QBH_BLOCK_U8_SEGMENT_ASYNC_BANK_SLOTS)
                : 0U;
        const uint32_t next_bank = bank ^ 1U;

        pool->segmented_attention_phase = phase;
        pool->segmented_attention_ring_offset =
            base_ring_offset +
            bank * QBH_BLOCK_U8_SEGMENT_ASYNC_BANK_SLOTS *
                QBH_BLOCK_U8_SEGMENT_SLOT_BYTES;
        pool->segmented_attention_batch_count = batch_count;
        for (uint32_t local_segment = 0U;
             local_segment < batch_count; ++local_segment) {
            const uint32_t segment = batch_first + local_segment;
            pool->segmented_attention_rows[local_segment] =
                segment < sealed_segments
                    ? QBH_KV_CACHE_HMX_U8_SEGMENT_TOKENS
                    : tail_rows;
        }
        if (qbh_hvx_pool_u8_segmented_attention_start_async(pool) != 0) {
            return -1;
        }

        if (next_count != 0U &&
            qbh_scan_stage_u8_segment_batch(
                header, shared, buffers, telemetry,
                k_head_stride, v_head_stride, group_stride,
                base_ring_offset +
                    next_bank *
                        QBH_BLOCK_U8_SEGMENT_ASYNC_BANK_SLOTS *
                        QBH_BLOCK_U8_SEGMENT_SLOT_BYTES,
                sealed_segments, tail_rows,
                row_scratch, bias_scratch, next_first, next_count,
                stage_k, stage_v) != 0) {
            pool->attention_gqa_abort = 1U;
            (void)qbh_hvx_pool_u8_segmented_attention_wait_async(pool);
            return -1;
        }
        if (qbh_hvx_pool_u8_segmented_attention_wait_async(pool) != 0) {
            return -1;
        }
        if (next_count == 0U) {
            break;
        }
        batch_first = next_first;
        batch_count = next_count;
        bank = next_bank;
    }
    pool->segmented_attention_ring_offset = base_ring_offset;
    return 0;
}

static int qbh_scan_u8_attention_segmented(
    struct qbh_block_header *header, uint8_t *shared,
    struct qbh_block_buffers *buffers,
    struct qbh_block_hmx_worker *worker,
    struct qbh_block_w4f16_pool *pool,
    uint32_t logical_rows, uint32_t past_tokens) {
    const uint32_t valid_tokens = past_tokens + logical_rows;
    const uint32_t max_segments =
        QBH_KV_CACHE_HMX_U8_SEGMENT_COUNT(header->kv_cache_capacity);
    const uint32_t sealed_segments =
        valid_tokens / QBH_KV_CACHE_HMX_U8_SEGMENT_TOKENS;
    const uint32_t tail_rows =
        valid_tokens % QBH_KV_CACHE_HMX_U8_SEGMENT_TOKENS;
    const uint32_t segment_count =
        sealed_segments + (tail_rows != 0U ? 1U : 0U);
    const uint32_t short_padded_tokens =
        segment_count * QBH_KV_CACHE_HMX_U8_SEGMENT_TOKENS;
    const uint32_t short_plane_bytes =
        short_padded_tokens * QBH_BLOCK_HEAD_DIM;
    const uint32_t short_qk_bias_bytes =
        segment_count * QBH_HMX_BIAS_BYTES;
    const uint32_t short_slot_stride = qbh_align_up(
        3U * short_plane_bytes + short_qk_bias_bytes +
            QBH_ATTENTION_HEAD_DIM_TILES * QBH_HMX_BIAS_BYTES,
        QBH_HMX_FP16_TILE_BYTES);
    const uint32_t k_head_stride =
        QBH_KV_CACHE_HMX_U8_K_SEGMENTED_HEAD_BYTES(
            header->kv_cache_capacity);
    const uint32_t v_head_stride =
        QBH_KV_CACHE_HMX_U8_V_SEGMENTED_HEAD_BYTES(
            header->kv_cache_capacity);
    const uint32_t overlay_capacity =
        (uint32_t)(buffers->compressed_weight - buffers->scores);
    const uint32_t fused_short =
        segment_count <= QBH_BLOCK_U8_SEGMENT_RING_SLOTS &&
        2U * short_slot_stride <= overlay_capacity;
    const uint32_t probability_map_bytes =
        QBH_ATTENTION_Q_HEADS_PER_GROUP * 256U;
    const uint32_t histogram_offset = 0U;
    const uint32_t active_score_offset =
        histogram_offset +
        QBH_BLOCK_U8_SEGMENT_HISTOGRAM_BYTES;
    const uint32_t probability_map_offset =
        active_score_offset +
        QBH_BLOCK_U8_SEGMENT_ACTIVE_SCORE_BYTES;
    const uint32_t accumulator_offset = qbh_align_up(
        probability_map_offset + probability_map_bytes,
        QBH_BLOCK_ALIGNMENT);
    const uint32_t ring_offset = qbh_align_up(
        accumulator_offset +
            QBH_BLOCK_U8_SEGMENT_ACCUMULATOR_BYTES,
        QBH_BLOCK_ALIGNMENT);
    const uint32_t group_stride = qbh_align_up(
        ring_offset +
        QBH_BLOCK_U8_SEGMENT_RING_SLOTS *
            QBH_BLOCK_U8_SEGMENT_SLOT_BYTES,
        QBH_BLOCK_ALIGNMENT);
    const uint32_t required_bytes = fused_short != 0U
        ? 2U * short_slot_stride
        : QBH_BLOCK_KV_HEADS * group_stride;
    uint8_t *row_scratch = buffers->gate;
    uint8_t *bias_scratch = buffers->up;
    struct qbh_attention_u8_telemetry
        telemetry[QBH_BLOCK_KV_HEADS];
    struct qbh_block_w4f16_job main_job;
    const uint64_t dynamic_start = HAP_perf_get_qtimer_count();

    header->scan_attention_overlay_capacity_bytes = overlay_capacity;
    header->scan_attention_overlay_required_bytes = required_bytes;
    header->scan_total_kv_length = valid_tokens;
    header->scan_padded_kv_length =
        segment_count * QBH_KV_CACHE_HMX_U8_SEGMENT_TOKENS;
    if (logical_rows != 1U || valid_tokens > header->kv_cache_capacity ||
        sealed_segments > max_segments ||
        required_bytes > overlay_capacity ||
        segment_count > QBH_BLOCK_SCAN_MAX_KV_TILES ||
        pool == NULL || worker == NULL ||
        header->attention_hvx_contexts < 4U ||
        header->attention_hvx_contexts >
            QBH_BLOCK_MAX_ATTENTION_HVX_CONTEXTS ||
        header->attention_hvx_contexts - 1U > pool->worker_count ||
        header->kv_cache_k_bytes !=
            QBH_KV_CACHE_HMX_U8_K_SEGMENTED_BYTES(
                header->kv_cache_capacity) ||
        header->kv_cache_v_bytes !=
            QBH_KV_CACHE_HMX_U8_V_SEGMENTED_BYTES(
                header->kv_cache_capacity)) {
        return -1;
    }
    if (header->numerical_audit_enabled != 0U) {
        uint8_t *audit = shared + header->u8_attention_audit_output_offset;
        if (qbh_dma_copy(
                header, audit, buffers->q,
                QBH_BLOCK_U8_ATTENTION_Q_BYTES, 0U) != 0) {
            return -1;
        }
        if (qbh_dma_copy(
                header, audit + QBH_BLOCK_U8_ATTENTION_Q_BYTES,
                buffers->k, QBH_BLOCK_U8_ATTENTION_KV_BYTES, 0U) != 0) {
            return -1;
        }
        if (qbh_dma_copy(
                header,
                audit + QBH_BLOCK_U8_ATTENTION_Q_BYTES +
                    QBH_BLOCK_U8_ATTENTION_KV_BYTES,
                buffers->v, QBH_BLOCK_U8_ATTENTION_KV_BYTES, 0U) != 0) {
            return -1;
        }
        header->u8_attention_audit_ddr_write_bytes +=
            QBH_BLOCK_U8_ATTENTION_Q_BYTES +
            2U * QBH_BLOCK_U8_ATTENTION_KV_BYTES;
    }

    if (fused_short != 0U) {
        return qbh_scan_u8_attention_segmented_short_pipeline(
            header, shared, buffers, worker,
            logical_rows, past_tokens, valid_tokens,
            short_padded_tokens, segment_count,
            short_plane_bytes, short_qk_bias_bytes,
            short_slot_stride, k_head_stride, v_head_stride,
            max_segments, sealed_segments, tail_rows,
            dynamic_start);
    }

    memset(&main_job, 0, sizeof(main_job));
    main_job.worker_index = header->attention_hvx_contexts - 1U;
    pool->attention_header = header;
    pool->attention_buffers = buffers;
    pool->attention_gqa_abort = 0U;
    pool->segmented_attention_scratch = buffers->scores;
    pool->segmented_attention_group_stride = group_stride;
    pool->segmented_attention_ring_offset = ring_offset;
    pool->segmented_attention_valid_tokens = valid_tokens;
    pool->segmented_attention_telemetry = telemetry;
    for (uint32_t worker_index = 0U;
         worker_index < header->attention_hvx_contexts - 1U;
         ++worker_index) {
        pool->jobs[worker_index].u8_attention_qk_requant_ticks = 0U;
        pool->jobs[worker_index].u8_attention_softmax_ticks = 0U;
        pool->jobs[worker_index].u8_attention_av_requant_ticks = 0U;
    }
    for (uint32_t group = 0U;
         group < QBH_BLOCK_KV_HEADS; ++group) {
        uint8_t *group_base = buffers->scores +
            (size_t)group * group_stride;
        memset(&telemetry[group], 0, sizeof(telemetry[group]));
        memset(pool->segmented_attention_active_counts[group], 0,
               sizeof(pool->segmented_attention_active_counts[group]));
        memset(pool->segmented_attention_probability_sums[group], 0,
               sizeof(pool->segmented_attention_probability_sums[group]));
        memset(pool->segmented_attention_saturation_counts[group], 0,
               sizeof(pool->segmented_attention_saturation_counts[group]));
        pool->segmented_short_qk_ready[group] = 0U;
        pool->segmented_short_probability_head_count[group] = 0U;
        pool->segmented_short_probability_ready[group] = 0U;
        qbh_hvx_zero_aligned_bytes(
            group_base + histogram_offset,
            QBH_BLOCK_U8_SEGMENT_HISTOGRAM_BYTES);
    }

    if (fused_short != 0U) {
        pool->segmented_attention_phase = 3U;
        pool->segmented_attention_batch_count = segment_count;
        for (uint32_t segment = 0U;
             segment < segment_count; ++segment) {
            pool->segmented_attention_rows[segment] =
                segment < sealed_segments
                    ? QBH_KV_CACHE_HMX_U8_SEGMENT_TOKENS
                    : tail_rows;
        }
        for (uint32_t group = 0U;
             group < QBH_BLOCK_KV_HEADS; ++group) {
            const struct qbh_attention_config *config =
                &buffers->attention_configs[group];
            const uint8_t *cache_k =
                shared + header->kv_cache_k_offset +
                (size_t)group * k_head_stride;
            const uint8_t *cache_v =
                shared + header->kv_cache_v_offset +
                (size_t)group * v_head_stride;
            uint8_t *ring = buffers->scores +
                (size_t)group * group_stride + ring_offset;
            int8_t *short_k = (int8_t *)(ring +
                QBH_BLOCK_U8_SHORT_K_WEIGHT_OFFSET);
            uint32_t *short_bias = (uint32_t *)(ring +
                QBH_BLOCK_U8_SHORT_K_BIAS_OFFSET);
            int8_t *short_v = (int8_t *)(ring +
                QBH_BLOCK_U8_SHORT_V_WEIGHT_OFFSET);
            uint32_t *short_av_bias = (uint32_t *)(
                buffers->scores + (size_t)group * group_stride +
                accumulator_offset);

            for (uint32_t segment = 0U;
                 segment < segment_count; ++segment) {
                int8_t *k_destination = short_k +
                    (size_t)segment *
                        QBH_KV_CACHE_HMX_U8_SEGMENT_WEIGHT_BYTES;
                uint32_t *bias_destination = short_bias +
                    (size_t)segment *
                        (QBH_HMX_BIAS_BYTES / sizeof(uint32_t));
                if (segment < sealed_segments) {
                    const uint8_t *k_source = cache_k +
                        (size_t)segment *
                            QBH_KV_CACHE_HMX_U8_SEGMENT_K_BYTES;
                    if (qbh_scan_cache_dma(
                            header, k_destination, k_source,
                            QBH_KV_CACHE_HMX_U8_SEGMENT_WEIGHT_BYTES,
                            1U) != 0 ||
                        qbh_scan_cache_dma(
                            header, bias_destination,
                            k_source +
                                QBH_KV_CACHE_HMX_U8_SEGMENT_WEIGHT_BYTES,
                            QBH_HMX_BIAS_BYTES, 1U) != 0) {
                        return -1;
                    }
                } else {
                    if (segment != sealed_segments || tail_rows == 0U ||
                        qbh_scan_cache_dma(
                            header, row_scratch,
                            cache_k +
                                (size_t)sealed_segments *
                                    QBH_KV_CACHE_HMX_U8_SEGMENT_K_BYTES,
                            tail_rows * QBH_BLOCK_HEAD_DIM, 1U) != 0) {
                        return -1;
                    }
                    qbh_attention_u8_pack_k_row_major(
                        row_scratch, tail_rows,
                        QBH_KV_CACHE_HMX_U8_SEGMENT_TOKENS,
                        config, k_destination, bias_destination);
                }
            }
            for (uint32_t output_tile = 0U;
                 output_tile < QBH_ATTENTION_HEAD_DIM_TILES;
                 ++output_tile) {
                if (qbh_scan_cache_dma_2d(
                        header,
                        short_v + (size_t)output_tile *
                            segment_count * QBH_HMX_WEIGHT_BYTES,
                        cache_v + (size_t)output_tile *
                            QBH_HMX_WEIGHT_BYTES,
                        QBH_HMX_WEIGHT_BYTES, sealed_segments,
                        QBH_KV_CACHE_HMX_U8_SEGMENT_WEIGHT_BYTES,
                        QBH_HMX_WEIGHT_BYTES) != 0) {
                    return -1;
                }
            }
            if (qbh_scan_cache_dma(
                    header, row_scratch,
                    cache_v +
                        (size_t)sealed_segments *
                            QBH_KV_CACHE_HMX_U8_SEGMENT_V_BYTES +
                        QBH_KV_CACHE_HMX_V_BIAS_BYTES_PER_HEAD,
                    tail_rows * QBH_BLOCK_HEAD_DIM, 1U) != 0) {
                return -1;
            }
            qbh_attention_u8_pack_v_row_major(
                row_scratch, tail_rows,
                QBH_KV_CACHE_HMX_U8_SEGMENT_TOKENS,
                config, (int8_t *)bias_scratch,
                (uint32_t *)(bias_scratch +
                    QBH_KV_CACHE_HMX_U8_SEGMENT_WEIGHT_BYTES),
                header->numerical_audit_enabled != 0U
                    ? &telemetry[group].v_recenter_saturation_count
                    : NULL);
            for (uint32_t output_tile = 0U;
                 output_tile < QBH_ATTENTION_HEAD_DIM_TILES;
                 ++output_tile) {
                qbh_hvx_copy_aligned_bytes(
                    short_v +
                        ((size_t)output_tile * segment_count +
                         sealed_segments) * QBH_HMX_WEIGHT_BYTES,
                    bias_scratch +
                        (size_t)output_tile * QBH_HMX_WEIGHT_BYTES,
                    QBH_HMX_WEIGHT_BYTES);
            }
            if (qbh_scan_cache_dma(
                    header, short_av_bias,
                    cache_v +
                        (size_t)sealed_segments *
                            QBH_KV_CACHE_HMX_U8_SEGMENT_V_BYTES,
                    QBH_KV_CACHE_HMX_V_BIAS_BYTES_PER_HEAD,
                    1U) != 0) {
                return -1;
            }
        }
        pool->segmented_attention_phase = 3U;
        if (qbh_hvx_pool_u8_segmented_attention_start_async(pool) != 0) {
            return -1;
        }
        {
            uint64_t compute_before = worker->compute_ticks;
            uint8_t *ring = buffers->scores + ring_offset;

            qbh_hmx_start(
                worker, QBH_BLOCK_HMX_U8S8,
                buffers->q,
                ring + QBH_BLOCK_U8_SHORT_K_WEIGHT_OFFSET,
                ring + QBH_BLOCK_U8_SHORT_K_BIAS_OFFSET,
                ring + QBH_BLOCK_U8_SHORT_SCORE_OFFSET,
                QBH_ATTENTION_Q_HEADS_PER_GROUP,
                QBH_ATTENTION_HEAD_DIM_TILES, segment_count);
            for (uint32_t group = 0U;
                 group < QBH_BLOCK_KV_HEADS; ++group) {
                const uint64_t wait_start =
                    HAP_perf_get_qtimer_count();
                if (qbh_hmx_wait(worker) != 0) {
                    pool->attention_gqa_abort = 1U;
                    (void)qbh_hvx_pool_u8_segmented_attention_wait_async(
                        pool);
                    return -1;
                }
                header->u8_attention_pipeline_wait_ticks +=
                    HAP_perf_get_qtimer_count() - wait_start;
                header->u8_attention_qk_hmx_ticks +=
                    worker->compute_ticks - compute_before;
                pool->segmented_short_qk_ready[group] = 1U;
                asm volatile("release(%0):at"
                             :
                             : "r"(&pool->segmented_short_qk_ready[group])
                             : "memory");
                if (group + 1U < QBH_BLOCK_KV_HEADS) {
                    ring = buffers->scores +
                        (size_t)(group + 1U) * group_stride +
                        ring_offset;
                    compute_before = worker->compute_ticks;
                    qbh_hmx_start(
                        worker, QBH_BLOCK_HMX_U8S8,
                        buffers->q +
                            (size_t)(group + 1U) *
                                QBH_ATTENTION_Q_HEADS_PER_GROUP *
                                QBH_ATTENTION_HEAD_DIM_TILES *
                                QBH_HMX_ACTIVATION_BYTES,
                        ring + QBH_BLOCK_U8_SHORT_K_WEIGHT_OFFSET,
                        ring + QBH_BLOCK_U8_SHORT_K_BIAS_OFFSET,
                        ring + QBH_BLOCK_U8_SHORT_SCORE_OFFSET,
                        QBH_ATTENTION_Q_HEADS_PER_GROUP,
                        QBH_ATTENTION_HEAD_DIM_TILES, segment_count);
                }
            }
        }
        {
            uint32_t active_group = 0U;
            uint64_t compute_before;
            uint64_t ready_wait_start = HAP_perf_get_qtimer_count();
            uint8_t *ring;

            while (pool->segmented_short_probability_ready[0] == 0U) {
                if (pool->attention_gqa_abort != 0U ||
                    HAP_perf_get_qtimer_count() - ready_wait_start >
                        QBH_BLOCK_DMA_DESCRIPTOR_TIMEOUT_TICKS) {
                    pool->attention_gqa_abort = 1U;
                    (void)qbh_hvx_pool_u8_segmented_attention_wait_async(
                        pool);
                    return -1;
                }
                asm volatile("pause(#8)" : : : "memory");
            }
            asm volatile("barrier" ::: "memory");
            ring = buffers->scores + ring_offset;
            compute_before = worker->compute_ticks;
            qbh_hmx_start(
                worker, QBH_BLOCK_HMX_U8S8,
                ring + QBH_BLOCK_U8_SHORT_K_WEIGHT_OFFSET,
                ring + QBH_BLOCK_U8_SHORT_V_WEIGHT_OFFSET,
                buffers->scores + accumulator_offset,
                buffers->q,
                QBH_ATTENTION_Q_HEADS_PER_GROUP,
                segment_count, QBH_ATTENTION_HEAD_DIM_TILES);

            for (;;) {
                uint64_t wait_start = HAP_perf_get_qtimer_count();
                uint8_t *q_group = buffers->q +
                    (size_t)active_group *
                        QBH_ATTENTION_Q_HEADS_PER_GROUP *
                        QBH_ATTENTION_HEAD_DIM_TILES *
                        QBH_HMX_ACTIVATION_BYTES;
                if (qbh_hmx_wait(worker) != 0) {
                    pool->attention_gqa_abort = 1U;
                    (void)qbh_hvx_pool_u8_segmented_attention_wait_async(
                        pool);
                    return -1;
                }
                header->u8_attention_pipeline_wait_ticks +=
                    HAP_perf_get_qtimer_count() - wait_start;
                header->u8_attention_av_hmx_ticks +=
                    worker->compute_ticks - compute_before;
                if (active_group + 1U < QBH_BLOCK_KV_HEADS) {
                    const uint32_t next_group = active_group + 1U;
                    ready_wait_start = HAP_perf_get_qtimer_count();
                    while (pool->segmented_short_probability_ready[
                               next_group] == 0U) {
                        if (pool->attention_gqa_abort != 0U ||
                            HAP_perf_get_qtimer_count() -
                                    ready_wait_start >
                                QBH_BLOCK_DMA_DESCRIPTOR_TIMEOUT_TICKS) {
                            pool->attention_gqa_abort = 1U;
                            (void)
                                qbh_hvx_pool_u8_segmented_attention_wait_async(
                                    pool);
                            return -1;
                        }
                        asm volatile("pause(#8)" : : : "memory");
                    }
                    asm volatile("barrier" ::: "memory");
                    ring = buffers->scores +
                        (size_t)next_group * group_stride + ring_offset;
                    compute_before = worker->compute_ticks;
                    qbh_hmx_start(
                        worker, QBH_BLOCK_HMX_U8S8,
                        ring + QBH_BLOCK_U8_SHORT_K_WEIGHT_OFFSET,
                        ring + QBH_BLOCK_U8_SHORT_V_WEIGHT_OFFSET,
                        buffers->scores +
                            (size_t)next_group * group_stride +
                            accumulator_offset,
                        buffers->q +
                            (size_t)next_group *
                                QBH_ATTENTION_Q_HEADS_PER_GROUP *
                                QBH_ATTENTION_HEAD_DIM_TILES *
                                QBH_HMX_ACTIVATION_BYTES,
                        QBH_ATTENTION_Q_HEADS_PER_GROUP,
                        segment_count, QBH_ATTENTION_HEAD_DIM_TILES);
                }
                {
                    const uint64_t start =
                        HAP_perf_get_qtimer_count();
                    qbh_attention_u8_requant_av(
                        q_group,
                        &buffers->attention_configs[active_group]);
                    header->u8_attention_av_requant_ticks +=
                        HAP_perf_get_qtimer_count() - start;
                }
                if (++active_group >= QBH_BLOCK_KV_HEADS) {
                    break;
                }
            }
        }
        if (qbh_hvx_pool_u8_segmented_attention_wait_async(pool) != 0) {
            return -1;
        }
        header->hmx_command_count += 2U * QBH_BLOCK_KV_HEADS;
        header->hmx_u8s8_tile_pair_count +=
            2U * QBH_BLOCK_KV_HEADS *
            QBH_ATTENTION_Q_HEADS_PER_GROUP *
            QBH_ATTENTION_HEAD_DIM_TILES * segment_count;
    } else {
    if (qbh_scan_run_u8_segmented_async_phase(
            header, shared, buffers, pool, telemetry,
            k_head_stride, v_head_stride, group_stride,
            ring_offset, sealed_segments, tail_rows,
            segment_count, row_scratch, bias_scratch,
            1U, 1U, 0U) != 0) {
        return -1;
    }

    for (uint32_t group = 0U;
         group < QBH_BLOCK_KV_HEADS; ++group) {
        uint8_t *group_base = buffers->scores +
            (size_t)group * group_stride;
        const uint64_t start = HAP_perf_get_qtimer_count();
        qbh_scan_segment_prepare_probability_maps(
            (const uint32_t *)(group_base + histogram_offset),
            group_base + active_score_offset,
            pool->segmented_attention_active_counts[group],
            &buffers->attention_configs[group],
            &telemetry[group],
            group_base + probability_map_offset,
            valid_tokens);
        header->u8_attention_softmax_ticks +=
            HAP_perf_get_qtimer_count() - start;
        qbh_hvx_zero_aligned_bytes(
            group_base + accumulator_offset,
            QBH_BLOCK_U8_SEGMENT_ACCUMULATOR_BYTES);
    }

    if (qbh_scan_run_u8_segmented_async_phase(
            header, shared, buffers, pool, telemetry,
            k_head_stride, v_head_stride, group_stride,
            ring_offset, sealed_segments, tail_rows,
            segment_count, row_scratch, bias_scratch,
            2U, 1U, 1U) != 0) {
        return -1;
    }
    }

    header->u8_attention_qk_requant_ticks +=
        main_job.u8_attention_qk_requant_ticks;
    header->u8_attention_softmax_ticks +=
        main_job.u8_attention_softmax_ticks;
    header->u8_attention_av_requant_ticks +=
        main_job.u8_attention_av_requant_ticks;
    for (uint32_t worker_index = 0U;
         worker_index < header->attention_hvx_contexts - 1U;
         ++worker_index) {
        header->u8_attention_qk_requant_ticks +=
            pool->jobs[worker_index].u8_attention_qk_requant_ticks;
        header->u8_attention_softmax_ticks +=
            pool->jobs[worker_index].u8_attention_softmax_ticks;
        header->u8_attention_av_requant_ticks +=
            pool->jobs[worker_index].u8_attention_av_requant_ticks;
    }
    for (uint32_t group = 0U;
         group < QBH_BLOCK_KV_HEADS; ++group) {
        if (fused_short != 0U) {
            const uint32_t sum0 =
                pool->segmented_attention_probability_sums[group][0];
            const uint32_t sum1 =
                pool->segmented_attention_probability_sums[group][1];
            telemetry[group].score_saturation_count +=
                pool->segmented_attention_saturation_counts[group][0] +
                pool->segmented_attention_saturation_counts[group][1];
            telemetry[group].probability_row_sum_min =
                sum0 < sum1 ? sum0 : sum1;
            telemetry[group].probability_row_sum_max =
                sum0 > sum1 ? sum0 : sum1;
        } else {
            uint8_t *group_base = buffers->scores +
                (size_t)group * group_stride;
            uint8_t *q_group = buffers->q +
                (size_t)group * QBH_ATTENTION_Q_HEADS_PER_GROUP *
                    QBH_ATTENTION_HEAD_DIM_TILES *
                    QBH_HMX_ACTIVATION_BYTES;
            const uint64_t start = HAP_perf_get_qtimer_count();
            qbh_scan_segment_finalize_av(
                q_group,
                (const int32_t *)(group_base + accumulator_offset),
                &buffers->attention_configs[group]);
            header->u8_attention_av_requant_ticks +=
                HAP_perf_get_qtimer_count() - start;
        }
        qbh_scan_accumulate_u8_attention_telemetry(
            header, &telemetry[group]);
    }

    header->u8_attention_group_count += QBH_BLOCK_KV_HEADS;
    header->u8_attention_qk_execution_count +=
        (fused_short != 0U ? 1U : 2U) * QBH_BLOCK_KV_HEADS *
        QBH_ATTENTION_Q_HEADS_PER_GROUP *
        segment_count;
    header->u8_attention_av_execution_count +=
        QBH_BLOCK_KV_HEADS * QBH_ATTENTION_Q_HEADS_PER_GROUP *
        QBH_ATTENTION_HEAD_DIM_TILES;
    header->u8_attention_direct_o_tile_count +=
        QBH_BLOCK_HEADS * QBH_ATTENTION_HEAD_DIM_TILES;
    header->scan_dynamic_attention_ticks +=
        HAP_perf_get_qtimer_count() - dynamic_start;
    return 0;
}

static int qbh_scan_u8_attention(
    struct qbh_block_header *header, uint8_t *shared,
    struct qbh_block_buffers *buffers,
    struct qbh_block_hmx_worker *worker,
    struct qbh_block_w4f16_pool *pool,
    uint32_t logical_rows, uint32_t past_tokens) {
    const uint32_t valid_tokens = past_tokens + logical_rows;
    const int hmx_segmented_cache =
        qbh_hmx_native_u8_segmented_cache_formats(
            header->kv_cache_k_format,
            header->kv_cache_v_format);
    const int hmx_native_cache = qbh_hmx_native_u8_cache_formats(
        header->kv_cache_k_format, header->kv_cache_v_format);
    const int hmx_delta_cache =
        qbh_hmx_native_u8_delta_cache_formats(
            header->kv_cache_k_format,
            header->kv_cache_v_format);
    const int direct_delta_reconstruction =
        hmx_delta_cache &&
        header->w4u8_delta_reconstruction_mode >=
            QBH_BLOCK_W4U8_DELTA_RECONSTRUCTION_DIRECT;
    const int delta_reconstruction_pipeline =
        hmx_delta_cache &&
        header->w4u8_delta_reconstruction_mode ==
            QBH_BLOCK_W4U8_DELTA_RECONSTRUCTION_PIPELINE;
    const uint32_t padded_tokens = hmx_native_cache
        ? header->kv_cache_padded_capacity
        : qbh_align_up(valid_tokens, QBH_HMX_INPUT_CHANNELS);
    const uint32_t kv_tiles =
        padded_tokens / QBH_HMX_INPUT_CHANNELS;
    const uint32_t plane_bytes =
        padded_tokens * QBH_BLOCK_HEAD_DIM;
    const uint32_t qk_bias_bytes =
        kv_tiles * QBH_HMX_BIAS_BYTES;
    const uint32_t single_slot_bytes =
        3U * plane_bytes + qk_bias_bytes +
        QBH_ATTENTION_HEAD_DIM_TILES * QBH_HMX_BIAS_BYTES;
    const uint32_t slot_stride = qbh_align_up(
        single_slot_bytes, QBH_HMX_FP16_TILE_BYTES);
    const uint32_t required_bytes = delta_reconstruction_pipeline
        ? 2U * slot_stride : single_slot_bytes;
    const uint32_t overlay_capacity =
        (uint32_t)(buffers->compressed_weight - buffers->scores);
    uint8_t *plane_a = buffers->scores;
    int8_t *weight = (int8_t *)(plane_a + plane_bytes);
    uint8_t *plane_c = plane_a + 2U * plane_bytes;
    uint32_t *qk_bias = (uint32_t *)(plane_a + 3U * plane_bytes);
    uint32_t *av_bias = (uint32_t *)((uint8_t *)qk_bias + qk_bias_bytes);
    const uint32_t k_cache_head_stride = hmx_native_cache
        ? (hmx_delta_cache
               ? QBH_KV_CACHE_HMX_U8_K_DELTA_HEAD_BYTES(
                     header->kv_cache_capacity)
               : QBH_KV_CACHE_HMX_K_HEAD_BYTES(
                     header->kv_cache_capacity))
        : header->kv_cache_capacity * QBH_BLOCK_HEAD_DIM;
    const uint32_t v_cache_head_stride = hmx_native_cache
        ? (hmx_delta_cache
               ? QBH_KV_CACHE_HMX_U8_V_DELTA_HEAD_BYTES(
                     header->kv_cache_capacity)
               : QBH_KV_CACHE_HMX_V_HEAD_BYTES(
                     header->kv_cache_capacity))
        : header->kv_cache_capacity * QBH_BLOCK_HEAD_DIM;
    const uint32_t decode_rows = hmx_delta_cache
        ? valid_tokens - QBH_BLOCK_M : 0U;
    const uint32_t delta_bytes =
        decode_rows * QBH_BLOCK_HEAD_DIM;
    uint8_t *delta_rows = buffers->gate;
    int32_t delta_lut_v_zero_point = 0;
    uint32_t delta_lut_numerator = 0U;
    uint32_t delta_lut_denominator = 0U;
    const uint64_t dynamic_start = HAP_perf_get_qtimer_count();

    if (hmx_segmented_cache) {
        return qbh_scan_u8_attention_segmented(
            header, shared, buffers, worker, pool,
            logical_rows, past_tokens);
    }

    header->scan_attention_overlay_capacity_bytes = overlay_capacity;
    header->scan_attention_overlay_required_bytes = required_bytes;
    header->scan_total_kv_length = valid_tokens;
    header->scan_padded_kv_length = padded_tokens;
    if (required_bytes > overlay_capacity ||
        padded_tokens >
            QBH_BLOCK_SCAN_MAX_KV_TILES * QBH_HMX_INPUT_CHANNELS) {
        return -1;
    }
    if (header->numerical_audit_enabled != 0U) {
        uint8_t *audit = shared + header->u8_attention_audit_output_offset;
        if (qbh_dma_copy(
                header, audit, buffers->q,
                QBH_BLOCK_U8_ATTENTION_Q_BYTES, 0U) != 0) {
            return -1;
        }
        header->u8_attention_audit_ddr_write_bytes +=
            QBH_BLOCK_U8_ATTENTION_Q_BYTES;
    }
    if (hmx_delta_cache) {
        const struct qbh_attention_config *config =
            &buffers->attention_configs[0];
        qbh_attention_u8_prepare_v_delta_lut(config, buffers->up);
        delta_lut_v_zero_point = config->v_zero_point;
        delta_lut_numerator = config->v_recenter_numerator;
        delta_lut_denominator = config->v_recenter_denominator;
    }
    if (delta_reconstruction_pipeline) {
        return qbh_scan_u8_attention_delta_pipeline(
            header, shared, buffers, worker,
            logical_rows, past_tokens, valid_tokens,
            padded_tokens, kv_tiles, plane_bytes,
            qk_bias_bytes, slot_stride,
            k_cache_head_stride, v_cache_head_stride,
            decode_rows, delta_bytes, dynamic_start);
    }

    for (uint32_t group = 0U; group < QBH_BLOCK_KV_HEADS; ++group) {
        const struct qbh_attention_config *config =
            &buffers->attention_configs[group];
        const uint32_t first_q_head =
            group * QBH_ATTENTION_Q_HEADS_PER_GROUP;
        uint8_t *q_group = buffers->q +
            (size_t)first_q_head * QBH_ATTENTION_HEAD_DIM_TILES *
                QBH_HMX_ACTIVATION_BYTES;
        const uint8_t *cache_k =
            shared + header->kv_cache_k_offset +
            (size_t)group * k_cache_head_stride;
        const uint8_t *cache_v =
            shared + header->kv_cache_v_offset +
            (size_t)group * v_cache_head_stride;
        struct qbh_attention_u8_telemetry telemetry;
        uint64_t start;

        memset(&telemetry, 0, sizeof(telemetry));
        if (hmx_delta_cache &&
            (config->v_zero_point != delta_lut_v_zero_point ||
             config->v_recenter_numerator != delta_lut_numerator ||
             config->v_recenter_denominator != delta_lut_denominator)) {
            qbh_attention_u8_prepare_v_delta_lut(config, buffers->up);
            delta_lut_v_zero_point = config->v_zero_point;
            delta_lut_numerator = config->v_recenter_numerator;
            delta_lut_denominator = config->v_recenter_denominator;
        }
        if (hmx_native_cache) {
            if (hmx_delta_cache) {
                uint32_t *tail_bias = qk_bias +
                    (size_t)(QBH_BLOCK_M /
                             QBH_HMX_OUTPUT_CHANNELS) *
                        (QBH_HMX_BIAS_BYTES / sizeof(uint32_t));
                const uint32_t tail_tiles =
                    (decode_rows + QBH_HMX_OUTPUT_CHANNELS - 1U) /
                    QBH_HMX_OUTPUT_CHANNELS;
                const uint32_t divisor =
                    UINT32_C(1) << config->score_shift;
                const int32_t rounding = config->score_shift == 0U
                    ? 0 : (int32_t)(divisor / 2U);

                if (direct_delta_reconstruction) {
                    qbh_hvx_zero_aligned_bytes(
                        weight +
                            QBH_KV_CACHE_HMX_U8_BASE_WEIGHT_BYTES_PER_HEAD,
                        plane_bytes -
                            QBH_KV_CACHE_HMX_U8_BASE_WEIGHT_BYTES_PER_HEAD);
                } else {
                    qbh_hvx_zero_aligned_bytes(weight, plane_bytes);
                }
                if (qbh_scan_cache_dma(
                        header, weight, cache_k,
                        QBH_KV_CACHE_HMX_U8_BASE_WEIGHT_BYTES_PER_HEAD,
                        1U) != 0 ||
                    qbh_scan_cache_dma(
                        header, qk_bias,
                        cache_k +
                            QBH_KV_CACHE_HMX_U8_BASE_WEIGHT_BYTES_PER_HEAD,
                        QBH_KV_CACHE_HMX_U8_K_BASE_BIAS_BYTES_PER_HEAD,
                        1U) != 0) {
                    return -1;
                }
                for (uint32_t tile = 0U; tile < tail_tiles; ++tile) {
                    uint32_t *tile_bias = tail_bias +
                        (size_t)tile *
                            (QBH_HMX_BIAS_BYTES / sizeof(uint32_t));
                    for (uint32_t output = 0U;
                         output < QBH_HMX_OUTPUT_CHANNELS; ++output) {
                        tile_bias[output] = qk_bias[0];
                        tile_bias[QBH_HMX_OUTPUT_CHANNELS + output] =
                            UINT32_C(128) * divisor + rounding;
                    }
                }
                if (qbh_scan_cache_dma(
                        header, delta_rows,
                        cache_k +
                            QBH_KV_CACHE_HMX_U8_BASE_WEIGHT_BYTES_PER_HEAD +
                            QBH_KV_CACHE_HMX_U8_K_BASE_BIAS_BYTES_PER_HEAD,
                        delta_bytes, 1U) != 0) {
                    return -1;
                }
                start = HAP_perf_get_qtimer_count();
                for (uint32_t first = 0U; first < decode_rows;
                     first += QBH_HMX_OUTPUT_CHANNELS) {
                    const uint32_t count =
                        decode_rows - first <
                                QBH_HMX_OUTPUT_CHANNELS
                            ? decode_rows - first
                            : QBH_HMX_OUTPUT_CHANNELS;
                    const uint32_t tile =
                        first / QBH_HMX_OUTPUT_CHANNELS;
                    qbh_attention_u8_patch_k_delta_rows_hvx(
                        delta_rows +
                            (size_t)first * QBH_BLOCK_HEAD_DIM,
                        count, config,
                        weight +
                            (size_t)(QBH_BLOCK_M /
                                     QBH_HMX_OUTPUT_CHANNELS + tile) *
                                QBH_ATTENTION_HEAD_DIM_TILES *
                                QBH_HMX_WEIGHT_BYTES,
                        qk_bias +
                            (size_t)(QBH_BLOCK_M /
                                     QBH_HMX_OUTPUT_CHANNELS + tile) *
                                (QBH_HMX_BIAS_BYTES /
                                 sizeof(uint32_t)));
                }
                header->u8_attention_k_pack_ticks +=
                    HAP_perf_get_qtimer_count() - start;
            } else if (qbh_scan_cache_dma(
                           header, weight, cache_k,
                           plane_bytes, 1U) != 0 ||
                       qbh_scan_cache_dma(
                           header, qk_bias,
                           cache_k + plane_bytes,
                           qk_bias_bytes, 1U) != 0) {
                return -1;
            }
        } else {
            memset(plane_a, 0, plane_bytes);
            if (qbh_scan_cache_dma(
                    header, plane_a, cache_k,
                    valid_tokens * QBH_BLOCK_HEAD_DIM, 1U) != 0) {
                return -1;
            }
            start = HAP_perf_get_qtimer_count();
            qbh_attention_u8_pack_k_row_major(
                plane_a, valid_tokens, padded_tokens,
                config, weight, qk_bias);
            header->u8_attention_k_pack_ticks +=
                HAP_perf_get_qtimer_count() - start;
            ++header->u8_cache_full_prefix_pack_count;
        }
        start = HAP_perf_get_qtimer_count();
        if (qbh_hmx_submit(
                worker, QBH_BLOCK_HMX_U8S8,
                q_group, weight, qk_bias, plane_c,
                QBH_ATTENTION_Q_HEADS_PER_GROUP,
                QBH_ATTENTION_HEAD_DIM_TILES, kv_tiles) != 0) {
            return -1;
        }
        header->u8_attention_qk_hmx_ticks +=
            HAP_perf_get_qtimer_count() - start;
        start = HAP_perf_get_qtimer_count();
        qbh_attention_u8_requant_softmax_dynamic(
            plane_c, plane_a, logical_rows, past_tokens,
            valid_tokens, padded_tokens, config, &telemetry);
        header->u8_attention_softmax_ticks +=
            HAP_perf_get_qtimer_count() - start;

        /* Scores are dead after Softmax, so plane C may stage row-major V. */
        if (hmx_native_cache) {
            if (hmx_delta_cache) {
                uint8_t *base_weight = buffers->attention_projection;
                const uint32_t base_tile_bytes =
                    QBH_ATTENTION_SCORE_TILES *
                    QBH_HMX_WEIGHT_BYTES;
                const uint32_t destination_tile_stride =
                    kv_tiles * QBH_HMX_WEIGHT_BYTES;
                const uint32_t tail_tile_bytes =
                    destination_tile_stride - base_tile_bytes;

                if (direct_delta_reconstruction) {
                    qbh_hvx_zero_strided_bytes(
                        weight + base_tile_bytes,
                        tail_tile_bytes,
                        QBH_ATTENTION_HEAD_DIM_TILES,
                        destination_tile_stride);
                } else {
                    qbh_hvx_zero_aligned_bytes(weight, plane_bytes);
                }
                if ((direct_delta_reconstruction
                         ? qbh_scan_cache_dma_2d(
                               header, weight, cache_v,
                               base_tile_bytes,
                               QBH_ATTENTION_HEAD_DIM_TILES,
                               base_tile_bytes,
                               destination_tile_stride)
                         : qbh_scan_cache_dma(
                               header, base_weight, cache_v,
                               QBH_KV_CACHE_HMX_U8_BASE_WEIGHT_BYTES_PER_HEAD,
                               1U)) != 0 ||
                    qbh_scan_cache_dma(
                        header, av_bias,
                        cache_v +
                            QBH_KV_CACHE_HMX_U8_BASE_WEIGHT_BYTES_PER_HEAD,
                        QBH_KV_CACHE_HMX_V_BIAS_BYTES_PER_HEAD,
                        1U) != 0) {
                    return -1;
                }
                if (!direct_delta_reconstruction) {
                    for (uint32_t output_tile = 0U;
                         output_tile < QBH_ATTENTION_HEAD_DIM_TILES;
                         ++output_tile) {
                        qbh_hvx_copy_aligned_bytes(
                            weight +
                                (size_t)output_tile * kv_tiles *
                                    QBH_HMX_WEIGHT_BYTES,
                            base_weight +
                                (size_t)output_tile *
                                    QBH_ATTENTION_SCORE_TILES *
                                    QBH_HMX_WEIGHT_BYTES,
                            base_tile_bytes);
                    }
                }
                if (qbh_scan_cache_dma(
                        header, delta_rows,
                        cache_v +
                            QBH_KV_CACHE_HMX_U8_BASE_WEIGHT_BYTES_PER_HEAD +
                            QBH_KV_CACHE_HMX_V_BIAS_BYTES_PER_HEAD,
                        delta_bytes, 1U) != 0) {
                    return -1;
                }
                start = HAP_perf_get_qtimer_count();
                for (uint32_t first = 0U; first < decode_rows;
                     first += QBH_HMX_INPUT_CHANNELS) {
                    const uint32_t count =
                        decode_rows - first <
                                QBH_HMX_INPUT_CHANNELS
                            ? decode_rows - first
                            : QBH_HMX_INPUT_CHANNELS;
                    const uint32_t tile =
                        first / QBH_HMX_INPUT_CHANNELS;
                    qbh_attention_u8_patch_v_delta_rows_hvx(
                        delta_rows +
                            (size_t)first * QBH_BLOCK_HEAD_DIM,
                        count, config,
                        weight +
                            (size_t)(QBH_BLOCK_M /
                                     QBH_HMX_INPUT_CHANNELS + tile) *
                                QBH_HMX_WEIGHT_BYTES,
                        kv_tiles * QBH_HMX_WEIGHT_BYTES,
                        buffers->up,
                        header->numerical_audit_enabled != 0U
                            ? &telemetry.v_recenter_saturation_count
                            : NULL);
                }
                header->u8_attention_v_pack_ticks +=
                    HAP_perf_get_qtimer_count() - start;
            } else if (qbh_scan_cache_dma(
                           header, weight, cache_v,
                           plane_bytes, 1U) != 0 ||
                       qbh_scan_cache_dma(
                           header, av_bias,
                           cache_v + plane_bytes,
                           QBH_KV_CACHE_HMX_V_BIAS_BYTES_PER_HEAD,
                           1U) != 0) {
                return -1;
            }
        } else {
            memset(plane_c, 0, plane_bytes);
            if (qbh_scan_cache_dma(
                    header, plane_c, cache_v,
                    valid_tokens * QBH_BLOCK_HEAD_DIM, 1U) != 0) {
                return -1;
            }
            start = HAP_perf_get_qtimer_count();
            qbh_attention_u8_pack_v_row_major(
                plane_c, valid_tokens, padded_tokens,
                config, weight, av_bias,
                &telemetry.v_recenter_saturation_count);
            header->u8_attention_v_pack_ticks +=
                HAP_perf_get_qtimer_count() - start;
            ++header->u8_cache_full_prefix_pack_count;
        }
        start = HAP_perf_get_qtimer_count();
        if (qbh_hmx_submit(
                worker, QBH_BLOCK_HMX_U8S8,
                plane_a, weight, av_bias, q_group,
                QBH_ATTENTION_Q_HEADS_PER_GROUP,
                kv_tiles, QBH_ATTENTION_HEAD_DIM_TILES) != 0) {
            return -1;
        }
        qbh_attention_u8_requant_av(q_group, config);
        header->u8_attention_av_hmx_ticks +=
            HAP_perf_get_qtimer_count() - start;
        header->u8_attention_score_saturation_count +=
            telemetry.score_saturation_count;
        header->u8_attention_v_recenter_saturation_count +=
            telemetry.v_recenter_saturation_count;
        header->u8_attention_probability_mask_violation_count +=
            telemetry.probability_mask_violation_count;
        if (header->u8_attention_probability_row_sum_min == 0U ||
            telemetry.probability_row_sum_min <
                header->u8_attention_probability_row_sum_min) {
            header->u8_attention_probability_row_sum_min =
                telemetry.probability_row_sum_min;
        }
        if (telemetry.probability_row_sum_max >
            header->u8_attention_probability_row_sum_max) {
            header->u8_attention_probability_row_sum_max =
                telemetry.probability_row_sum_max;
        }
    }
    header->u8_attention_group_count += QBH_BLOCK_KV_HEADS;
    header->u8_attention_qk_execution_count +=
        QBH_BLOCK_KV_HEADS * QBH_ATTENTION_Q_HEADS_PER_GROUP * kv_tiles;
    header->u8_attention_av_execution_count +=
        QBH_BLOCK_KV_HEADS * QBH_ATTENTION_Q_HEADS_PER_GROUP *
        QBH_ATTENTION_HEAD_DIM_TILES;
    header->hmx_command_count += 2U * QBH_BLOCK_KV_HEADS;
    header->hmx_u8s8_tile_pair_count +=
        2U * QBH_BLOCK_KV_HEADS * QBH_ATTENTION_Q_HEADS_PER_GROUP *
        QBH_ATTENTION_HEAD_DIM_TILES * kv_tiles;
    header->u8_attention_direct_o_tile_count +=
        QBH_BLOCK_HEADS * QBH_ATTENTION_HEAD_DIM_TILES;
    header->scan_dynamic_attention_ticks +=
        HAP_perf_get_qtimer_count() - dynamic_start;
    return 0;
}

static int qbh_run_one_block(struct qbh_block_header *header,
                             uint8_t *shared,
                             struct qbh_block_buffers *buffers,
                             struct qbh_block_hmx_worker *worker,
                             struct qbh_block_w4f16_pool *w4f16_pool,
                             uint32_t input_offset,
                             uint32_t logical_rows,
                             uint32_t past_tokens,
                             uint32_t input_resident) {
    uint32_t hidden_elements = QBH_BLOCK_M * QBH_BLOCK_HIDDEN;
    uint32_t intermediate_elements =
        QBH_BLOCK_M * QBH_BLOCK_INTERMEDIATE;
    struct qbh_hvx_check_metrics rms_check_metrics;
    struct qbh_hvx_check_metrics rope_check_metrics;
    struct qbh_hvx_check_metrics softmax_check_metrics;
    struct qbh_hvx_check_metrics silu_check_metrics;
    struct qbh_block_w4f16_cross_prefetch cross_prefetch;
    struct qbh_hvx_check_metrics *rms_check =
        header->numerical_audit_enabled != 0U &&
        header->common_ops_mask == QBH_BLOCK_COMMON_OP_RMS_NORM
            ? &rms_check_metrics : NULL;
    struct qbh_hvx_check_metrics *rope_check =
        header->numerical_audit_enabled != 0U &&
        header->common_ops_mask == QBH_BLOCK_COMMON_OP_ROPE
            ? &rope_check_metrics : NULL;
    struct qbh_hvx_check_metrics *softmax_check =
        header->numerical_audit_enabled != 0U &&
        header->common_ops_mask == QBH_BLOCK_COMMON_OP_SOFTMAX
            ? &softmax_check_metrics : NULL;
    struct qbh_hvx_check_metrics *silu_check =
        header->numerical_audit_enabled != 0U &&
        header->common_ops_mask == QBH_BLOCK_COMMON_OP_SILU
            ? &silu_check_metrics : NULL;
    uint64_t start;
    uint64_t audit_start;
    uint64_t attention_attributed_before = 0U;
    uint64_t gate_up_weight_dma_before = 0U;
    uint64_t gate_up_expand_before = 0U;
    uint64_t gate_up_expand_work_before = 0U;
    uint64_t gate_up_expand_pool_wait_before = 0U;
    uint64_t gate_up_prefetch_wait_before = 0U;
    uint64_t gate_up_hmx_wait_before = 0U;
    uint64_t gate_up_hmx_tail_wait_before = 0U;
    uint64_t gate_up_unpack_before = 0U;
    uint64_t gate_up_stream_work_before = 0U;
    uint64_t gate_up_stream_ready_wait_before = 0U;
    uint64_t gate_up_stream_join_wait_before = 0U;
    uint64_t gate_up_hmx_command_before = 0U;
    uint32_t qkv_overlap_enabled =
        header->attention_pipeline_mode ==
            QBH_BLOCK_ATTENTION_PIPELINE_GQA_QKV_OVERLAP;
    uint32_t u8_integer_attention_enabled =
        qbh_attention_u8_enabled(
            header->attention_pipeline_mode);
    uint32_t scan_enabled =
        header->scan_mode != QBH_BLOCK_SCAN_DISABLED;
    uint32_t scan_dynamic_attention =
        scan_enabled != 0U && past_tokens != 0U;
    uint32_t reuse_prefill_attention_carriers =
        scan_enabled != 0U && past_tokens == 0U &&
        logical_rows == QBH_BLOCK_M &&
        header->variant == QBH_BLOCK_W4U8 &&
        u8_integer_attention_enabled != 0U &&
        header->w4u8_prefill_cache_mode ==
            QBH_BLOCK_W4U8_PREFILL_CACHE_REUSE_ATTENTION_CARRIERS &&
        qbh_hmx_native_u8_cache_formats(
            header->kv_cache_k_format,
            header->kv_cache_v_format);
    uint32_t reuse_f16_prefill_attention_carriers =
        scan_enabled != 0U && past_tokens == 0U &&
        logical_rows == QBH_BLOCK_M &&
        header->variant != QBH_BLOCK_W4U8 &&
        qbh_attention_gqa_enabled(header->attention_pipeline_mode) &&
        qbh_hmx_native_f16_cache_formats(
            header->kv_cache_k_format,
            header->kv_cache_v_format);
    uint32_t u8_qkv_overlap_enabled =
        qbh_attention_u8_qkv_overlap_enabled(
            header->attention_pipeline_mode);
    uint32_t crouton_qkv_enabled =
        qbh_crouton_qkv_enabled(header);
    uint32_t crouton_av_o_enabled =
        u8_integer_attention_enabled != 0U ||
        (header->variant != QBH_BLOCK_W4U8 &&
         (header->crouton_boundary_mode &
          QBH_BLOCK_CROUTON_BOUNDARY_AV_TO_O) != 0U);
    uint32_t crouton_input_norm_enabled =
        header->variant != QBH_BLOCK_W4U8 &&
        (header->crouton_boundary_mode &
         QBH_BLOCK_CROUTON_BOUNDARY_INPUT_NORM) != 0U;
    uint32_t crouton_post_norm_enabled =
        header->variant != QBH_BLOCK_W4U8 &&
        (header->crouton_boundary_mode &
         QBH_BLOCK_CROUTON_BOUNDARY_POST_NORM) != 0U;
    uint32_t qkv_overlap_first_worker =
        header->variant == QBH_BLOCK_W4F16 ? 2U : 0U;
    uint32_t qkv_overlap_worker_count =
        header->variant == QBH_BLOCK_W4F16 ? 1U : 3U;
    uint32_t w4u8_mlp_native_input_enabled =
        header->variant == QBH_BLOCK_W4U8 &&
        qbh_block_mlp_is_w4u8_streaming(header->mlp_mode) &&
        (header->crouton_boundary_mode &
         QBH_BLOCK_CROUTON_BOUNDARY_W4U8_MLP_INPUT) != 0U;
    uint32_t w4u8_mlp_native_output_enabled =
        header->variant == QBH_BLOCK_W4U8 &&
        qbh_block_mlp_is_w4u8_streaming(header->mlp_mode) &&
        (header->crouton_boundary_mode &
         QBH_BLOCK_CROUTON_BOUNDARY_W4U8_MLP_OUTPUT) != 0U;
    uint32_t w4u8_qkv_native_input_enabled =
        header->variant == QBH_BLOCK_W4U8 &&
        (header->crouton_boundary_mode &
         QBH_BLOCK_CROUTON_BOUNDARY_W4U8_QKV_INPUT) != 0U;
    uint32_t w4u8_qkv_ring_enabled =
        w4u8_qkv_native_input_enabled != 0U &&
        header->w4u8_qkv_ring_expand_workers != 0U;
    uint32_t w4u8_o_native_output_enabled =
        header->variant == QBH_BLOCK_W4U8 &&
        (header->crouton_boundary_mode &
         QBH_BLOCK_CROUTON_BOUNDARY_W4U8_O_OUTPUT) != 0U;
    struct qbh_projection_layout w4u8_gate_up_layout;
    struct qbh_projection_layout w4u8_down_layout;
    uint8_t *w4u8_mlp_native_activation = NULL;
    const uint8_t *w4u8_mlp_native_output = NULL;
    int post_attention_norm_fused = 0;

    qbh_hvx_check_reset(&rms_check_metrics);
    qbh_hvx_check_reset(&rope_check_metrics);
    qbh_hvx_check_reset(&softmax_check_metrics);
    qbh_hvx_check_reset(&silu_check_metrics);
    memset(&cross_prefetch, 0, sizeof(cross_prefetch));
    if (w4u8_mlp_native_input_enabled != 0U) {
        if (qbh_init_w4u8_gate_up_layout(
                &w4u8_gate_up_layout,
                header->w4u8_gate_up_ring_slots) != 0) {
            return QBH_BLOCK_STATUS_MLP_STREAM_FAILED;
        }
        w4u8_mlp_native_activation = buffers->q +
            w4u8_gate_up_layout.vtcm_activation_offset;
    }
    if (w4u8_mlp_native_output_enabled != 0U) {
        if (qbh_init_w4u8_down_layout(&w4u8_down_layout) != 0) {
            return QBH_BLOCK_STATUS_MLP_STREAM_FAILED;
        }
        w4u8_mlp_native_output = buffers->q +
            w4u8_down_layout.vtcm_output_offset;
    }

    start = HAP_perf_get_qtimer_count();
    if (input_resident == 0U) {
        const uint32_t input_bytes =
            QBH_BLOCK_M * QBH_BLOCK_HIDDEN *
            (header->variant == QBH_BLOCK_W4U8 ? 1U : 2U);
        if (qbh_dma_copy(header, buffers->residual,
                         shared + input_offset, input_bytes, 1U) != 0) {
            header->input_dma_status = -1;
            return QBH_BLOCK_STATUS_INPUT_DMA_FAILED;
        }
        header->boundary_ddr_read_bytes += input_bytes;
        ++header->boundary_dma_descriptor_count;
    }
    header->input_stage_ticks += HAP_perf_get_qtimer_count() - start;

    start = HAP_perf_get_qtimer_count();
    if (header->variant == QBH_BLOCK_W4U8) {
        if ((header->common_ops_mask &
             QBH_BLOCK_COMMON_OP_RMS_NORM) != 0U) {
            if (w4u8_qkv_native_input_enabled != 0U &&
                header->u8_norm_reduction_mode ==
                    QBH_BLOCK_U8_NORM_REDUCTION_HVX_TREE_QK_BATCHED_RSQRT_SHARED_ROPE_PARALLEL_INPUT) {
                if (qbh_hvx_pool_u8_input_norm(
                        header, w4f16_pool, buffers->residual,
                        &header->qparams[QBH_BLOCK_QP_BLOCK_INPUT],
                        (const __fp16 *)buffers->input_norm_weight,
                        buffers->hmx_activation,
                        &header->qparams[QBH_BLOCK_QP_INPUT_NORM]) != 0) {
                    return QBH_BLOCK_STATUS_INPUT_NORM_FAILED;
                }
            } else if (w4u8_qkv_native_input_enabled != 0U) {
                qbh_hvx_rms_norm_u8_native_activation(
                    buffers->residual,
                    &header->qparams[QBH_BLOCK_QP_BLOCK_INPUT],
                    (const __fp16 *)buffers->input_norm_weight,
                    buffers->hmx_activation,
                    &header->qparams[QBH_BLOCK_QP_INPUT_NORM],
                    QBH_BLOCK_M, QBH_BLOCK_HIDDEN);
            } else {
                qbh_hvx_rms_norm_u8(
                    buffers->residual,
                    &header->qparams[QBH_BLOCK_QP_BLOCK_INPUT],
                    (const __fp16 *)buffers->input_norm_weight,
                    buffers->normalized,
                    &header->qparams[QBH_BLOCK_QP_INPUT_NORM],
                    QBH_BLOCK_M, QBH_BLOCK_HIDDEN);
            }
        } else {
            qbh_rms_norm_u8(
                buffers->residual,
                &header->qparams[QBH_BLOCK_QP_BLOCK_INPUT],
                (const __fp16 *)buffers->input_norm_weight,
                buffers->normalized,
                &header->qparams[QBH_BLOCK_QP_INPUT_NORM],
                QBH_BLOCK_M, QBH_BLOCK_HIDDEN);
        }
    } else {
        if ((header->fp16_common_schedule_mode &
             QBH_BLOCK_FP16_COMMON_SCHEDULE_INPUT_NORM_POOL) != 0U &&
            rms_check == NULL) {
            const uint64_t norm_start = HAP_perf_get_qtimer_count();
            if (qbh_hvx_pool_fp16_input_norm(
                    header, w4f16_pool,
                    (const __fp16 *)buffers->residual,
                    (const __fp16 *)buffers->input_norm_weight,
                    crouton_input_norm_enabled != 0U
                        ? (__fp16 *)buffers->hmx_activation
                        : (__fp16 *)buffers->normalized,
                    crouton_input_norm_enabled) != 0) {
                return QBH_BLOCK_STATUS_INPUT_NORM_FAILED;
            }
            if (crouton_input_norm_enabled != 0U) {
                header->crouton_norm_store_ticks +=
                    HAP_perf_get_qtimer_count() - norm_start;
                ++header->crouton_norm_projection_count;
                qbh_hmx_fp16_init_unity_scale(buffers->scale_or_bias);
            }
        } else if (crouton_input_norm_enabled != 0U) {
            uint64_t norm_start = HAP_perf_get_qtimer_count();
            qbh_hvx_rms_norm_f16_crouton(
                (const __fp16 *)buffers->residual,
                (const __fp16 *)buffers->input_norm_weight,
                (__fp16 *)buffers->hmx_activation,
                QBH_BLOCK_M, QBH_BLOCK_HIDDEN);
            header->crouton_norm_store_ticks +=
                HAP_perf_get_qtimer_count() - norm_start;
            ++header->crouton_norm_projection_count;
            qbh_hmx_fp16_init_unity_scale(buffers->scale_or_bias);
        } else if ((header->common_ops_mask &
                    QBH_BLOCK_COMMON_OP_RMS_NORM) != 0U) {
            qbh_hvx_rms_norm_f16(
                (const __fp16 *)buffers->residual,
                (const __fp16 *)buffers->input_norm_weight,
                (__fp16 *)buffers->normalized,
                QBH_BLOCK_M, QBH_BLOCK_HIDDEN, rms_check);
        } else {
            qbh_rms_norm_f16(
                (const __fp16 *)buffers->residual,
                (const __fp16 *)buffers->input_norm_weight,
                (__fp16 *)buffers->normalized,
                QBH_BLOCK_M, QBH_BLOCK_HIDDEN);
        }
        audit_start = qbh_attribution_begin(header);
        qbh_record_f16_nonfinite(
            header,
            crouton_input_norm_enabled != 0U
                ? buffers->hmx_activation : buffers->normalized,
            hidden_elements,
            QBH_BLOCK_NUMERICAL_INPUT_NORM);
        qbh_attribution_accumulate(
            header, audit_start, &header->input_norm_audit_ticks);
    }
    if (header->numerical_audit_enabled != 0U &&
        header->variant == QBH_BLOCK_W4U8 &&
        w4u8_qkv_native_input_enabled != 0U) {
        header->u8_input_norm_actual_hash = qbh_fnv1a64_bytes(
            buffers->hmx_activation,
            QBH_BLOCK_M * QBH_BLOCK_HIDDEN);
    }
    if (header->w4u8_boundary_audit_enabled != 0U) {
        const uint32_t boundary_bytes =
            QBH_BLOCK_M * QBH_BLOCK_HIDDEN;
        if (qbh_dma_copy(
                header,
                shared + header->w4u8_boundary_audit_output_offset,
                buffers->hmx_activation, boundary_bytes, 0U) != 0) {
            return QBH_BLOCK_STATUS_INPUT_NORM_FAILED;
        }
        header->w4u8_boundary_audit_ddr_write_bytes += boundary_bytes;
    }
    header->input_norm_ticks += HAP_perf_get_qtimer_count() - start;

    start = HAP_perf_get_qtimer_count();
    if (header->variant == QBH_BLOCK_W4U8 &&
        header->u8_norm_reduction_mode >=
            QBH_BLOCK_U8_NORM_REDUCTION_HVX_TREE_QK_BATCHED_RSQRT_SHARED_ROPE) {
        qbh_hvx_qk_rope_preconvert_sf32(
            (const __fp16 *)buffers->rope_cos,
            (const __fp16 *)buffers->rope_sin,
            buffers->attention_projection +
                QBH_QK_ROPE_SF32_CACHE_OFFSET);
    }
    if (qkv_overlap_enabled != 0U &&
        qbh_hvx_pool_qk_norm_rope_start_async(
            header, w4f16_pool, (__fp16 *)buffers->q,
            (__fp16 *)buffers->k,
            (__fp16 *)buffers->attention_projection,
            (__fp16 *)buffers->scores,
            crouton_qkv_enabled,
            (const __fp16 *)buffers->q_norm_weight,
            (const __fp16 *)buffers->k_norm_weight,
            (const __fp16 *)buffers->rope_cos,
            (const __fp16 *)buffers->rope_sin,
            0U, QBH_BLOCK_HEADS + QBH_BLOCK_KV_HEADS,
            qkv_overlap_first_worker,
            qkv_overlap_worker_count) != 0) {
        return QBH_BLOCK_STATUS_QK_NORM_ROPE_FAILED;
    }
    if (u8_qkv_overlap_enabled != 0U &&
        w4u8_qkv_ring_enabled == 0U &&
        qbh_hvx_pool_u8_qk_prep_start_async(
            header, w4f16_pool, buffers) != 0) {
        return QBH_BLOCK_STATUS_QK_NORM_ROPE_FAILED;
    }
    if (header->qkv_schedule_mode ==
            QBH_BLOCK_QKV_SCHEDULE_Q_PREFIX4_K_ALL) {
        if (qbh_run_w4f16_qkv_prefix4(
                header, shared, buffers, worker, w4f16_pool,
                buffers->hmx_activation, &cross_prefetch) != 0) {
            if (qkv_overlap_enabled != 0U) {
                qbh_hvx_pool_qk_norm_rope_abort_async(w4f16_pool);
                (void)qbh_hvx_pool_qk_norm_rope_wait_async(
                    header, w4f16_pool, qkv_overlap_first_worker,
                    qkv_overlap_worker_count);
            }
            return QBH_BLOCK_STATUS_QKV_FAILED;
        }
    } else if (w4u8_qkv_ring_enabled != 0U) {
        if (qbh_run_w4u8_qkv_ring(
                header, shared, buffers, worker, w4f16_pool,
                buffers->hmx_activation) != 0) {
            return QBH_BLOCK_STATUS_QKV_FAILED;
        }
    } else {
    if (qbh_run_projection(
            header, shared, &header->projections[QBH_BLOCK_PROJ_Q],
            buffers, worker, w4f16_pool,
            crouton_input_norm_enabled != 0U
                ? buffers->hmx_activation : buffers->normalized,
            buffers->q,
            crouton_input_norm_enabled |
                w4u8_qkv_native_input_enabled,
            &header->projections[QBH_BLOCK_PROJ_K],
            &cross_prefetch) != 0) {
        if (qkv_overlap_enabled != 0U) {
            qbh_hvx_pool_qk_norm_rope_abort_async(w4f16_pool);
            (void)qbh_hvx_pool_qk_norm_rope_wait_async(
                header, w4f16_pool, qkv_overlap_first_worker,
                qkv_overlap_worker_count);
        }
        if (u8_qkv_overlap_enabled != 0U) {
            qbh_hvx_pool_u8_qk_prep_abort_async(w4f16_pool);
            (void)qbh_hvx_pool_u8_qk_prep_wait_async(
                header, w4f16_pool);
        }
        return QBH_BLOCK_STATUS_QKV_FAILED;
    }
    if (qbh_run_projection(
            header, shared, &header->projections[QBH_BLOCK_PROJ_K],
            buffers, worker, w4f16_pool, buffers->normalized,
            buffers->k,
            qkv_overlap_enabled | w4u8_qkv_native_input_enabled,
            &header->projections[QBH_BLOCK_PROJ_V],
            &cross_prefetch) != 0) {
        if (qkv_overlap_enabled != 0U) {
            qbh_hvx_pool_qk_norm_rope_abort_async(w4f16_pool);
            (void)qbh_hvx_pool_qk_norm_rope_wait_async(
                header, w4f16_pool, qkv_overlap_first_worker,
                qkv_overlap_worker_count);
        }
        if (u8_qkv_overlap_enabled != 0U) {
            qbh_hvx_pool_u8_qk_prep_abort_async(w4f16_pool);
            (void)qbh_hvx_pool_u8_qk_prep_wait_async(
                header, w4f16_pool);
        }
        return QBH_BLOCK_STATUS_QKV_FAILED;
    }
    if (qbh_run_projection(
            header, shared, &header->projections[QBH_BLOCK_PROJ_V],
            buffers, worker, w4f16_pool, buffers->normalized,
            buffers->v,
            qkv_overlap_enabled | w4u8_qkv_native_input_enabled,
            &header->projections[QBH_BLOCK_PROJ_O],
            &cross_prefetch) != 0) {
        if (u8_qkv_overlap_enabled != 0U) {
            qbh_hvx_pool_u8_qk_prep_abort_async(w4f16_pool);
            (void)qbh_hvx_pool_u8_qk_prep_wait_async(
                header, w4f16_pool);
        }
        return QBH_BLOCK_STATUS_QKV_FAILED;
    }
    }
    if (u8_qkv_overlap_enabled != 0U &&
        w4u8_qkv_ring_enabled == 0U &&
        qbh_hvx_pool_u8_qk_prep_wait_async(
            header, w4f16_pool) != 0) {
        return QBH_BLOCK_STATUS_QK_NORM_ROPE_FAILED;
    }
    if (qkv_overlap_enabled != 0U) {
        if (qbh_hvx_pool_qk_norm_rope_wait_async(
                header, w4f16_pool, qkv_overlap_first_worker,
                qkv_overlap_worker_count) != 0) {
            return QBH_BLOCK_STATUS_QK_NORM_ROPE_FAILED;
        }
        if (header->numerical_audit_enabled != 0U) {
            if (crouton_qkv_enabled != 0U) {
                qbh_audit_crouton_qkv_operands(header, buffers);
            } else if (header->variant == QBH_BLOCK_W4F16) {
                qbh_audit_row_major_qkv_operands(header, buffers);
            }
        }
        audit_start = qbh_attribution_begin(header);
        qbh_record_f16_nonfinite(
            header,
            crouton_qkv_enabled != 0U
                ? buffers->attention_projection : buffers->q,
            QBH_BLOCK_M * QBH_BLOCK_HIDDEN,
            QBH_BLOCK_NUMERICAL_Q_ROPE);
        qbh_record_f16_nonfinite(
            header,
            crouton_qkv_enabled != 0U ? buffers->scores : buffers->k,
            QBH_BLOCK_M * QBH_BLOCK_KV_HIDDEN,
            QBH_BLOCK_NUMERICAL_K_ROPE);
        qbh_record_f16_nonfinite(
            header, buffers->v,
            QBH_BLOCK_M * QBH_BLOCK_KV_HIDDEN,
            QBH_BLOCK_NUMERICAL_V);
        qbh_attribution_accumulate(
            header, audit_start, &header->qkv_audit_ticks);
        if (crouton_qkv_enabled != 0U) {
            header->crouton_qk_operand_count +=
                QBH_BLOCK_HEADS + QBH_BLOCK_KV_HEADS;
        }
    } else if (header->variant != QBH_BLOCK_W4U8) {
        audit_start = qbh_attribution_begin(header);
        qbh_record_f16_nonfinite(
            header, buffers->q, QBH_BLOCK_M * QBH_BLOCK_HIDDEN,
            QBH_BLOCK_NUMERICAL_Q);
        qbh_record_f16_nonfinite(
            header, buffers->k, QBH_BLOCK_M * QBH_BLOCK_KV_HIDDEN,
            QBH_BLOCK_NUMERICAL_K);
        qbh_record_f16_nonfinite(
            header, buffers->v, QBH_BLOCK_M * QBH_BLOCK_KV_HIDDEN,
            QBH_BLOCK_NUMERICAL_V);
        qbh_attribution_accumulate(
            header, audit_start, &header->qkv_audit_ticks);
    }
    header->qkv_projection_ticks += HAP_perf_get_qtimer_count() - start;

    start = HAP_perf_get_qtimer_count();
    if (u8_integer_attention_enabled != 0U) {
        /* Native Q/K projection tiles are normalized and rotated inside
         * the per-GQA integer Attention pipeline. */
    } else if (header->variant != QBH_BLOCK_W4U8 &&
        qbh_attention_gqa_enabled(
            header->attention_pipeline_mode)) {
        /* GQA performs Q/K normalization and RoPE inside each group. The
         * QKV-overlap mode has already completed the same arithmetic from
         * head-readiness events emitted by the Q/K projections. */
    } else if (header->variant == QBH_BLOCK_W4U8) {
        if ((header->common_ops_mask & QBH_BLOCK_COMMON_OP_ROPE) != 0U) {
            qbh_hvx_qk_norm_rope_u8(
                buffers->q, QBH_BLOCK_M, QBH_BLOCK_HEADS,
                QBH_BLOCK_HIDDEN, QBH_BLOCK_HEAD_DIM,
                &header->qparams[QBH_BLOCK_QP_Q_PROJECTION],
                &header->qparams[QBH_BLOCK_QP_Q_ROPE],
                (const __fp16 *)buffers->q_norm_weight,
                (const __fp16 *)buffers->rope_cos,
                (const __fp16 *)buffers->rope_sin);
            qbh_hvx_qk_norm_rope_u8(
                buffers->k, QBH_BLOCK_M, QBH_BLOCK_KV_HEADS,
                QBH_BLOCK_KV_HIDDEN, QBH_BLOCK_HEAD_DIM,
                &header->qparams[QBH_BLOCK_QP_K_PROJECTION],
                &header->qparams[QBH_BLOCK_QP_K_ROPE],
                (const __fp16 *)buffers->k_norm_weight,
                (const __fp16 *)buffers->rope_cos,
                (const __fp16 *)buffers->rope_sin);
            qbh_hvx_expand_u8_to_f16_in_place(
                buffers->q, QBH_BLOCK_M * QBH_BLOCK_HIDDEN,
                &header->qparams[QBH_BLOCK_QP_Q_ROPE]);
            qbh_hvx_expand_u8_to_f16_in_place(
                buffers->k, QBH_BLOCK_M * QBH_BLOCK_KV_HIDDEN,
                &header->qparams[QBH_BLOCK_QP_K_ROPE]);
            qbh_hvx_expand_u8_to_f16_in_place(
                buffers->v, QBH_BLOCK_M * QBH_BLOCK_KV_HIDDEN,
                &header->qparams[QBH_BLOCK_QP_V]);
        } else {
            qbh_qk_norm_rope_u8(
                buffers->q, QBH_BLOCK_HEADS, QBH_BLOCK_HIDDEN,
                &header->qparams[QBH_BLOCK_QP_Q_PROJECTION],
                &header->qparams[QBH_BLOCK_QP_Q_ROPE],
                (const __fp16 *)buffers->q_norm_weight,
                (const __fp16 *)buffers->rope_cos,
                (const __fp16 *)buffers->rope_sin);
            qbh_qk_norm_rope_u8(
                buffers->k, QBH_BLOCK_KV_HEADS, QBH_BLOCK_KV_HIDDEN,
                &header->qparams[QBH_BLOCK_QP_K_PROJECTION],
                &header->qparams[QBH_BLOCK_QP_K_ROPE],
                (const __fp16 *)buffers->k_norm_weight,
                (const __fp16 *)buffers->rope_cos,
                (const __fp16 *)buffers->rope_sin);
            qbh_expand_u8_to_f16_in_place(
                buffers->q, QBH_BLOCK_M * QBH_BLOCK_HIDDEN,
                &header->qparams[QBH_BLOCK_QP_Q_ROPE]);
            qbh_expand_u8_to_f16_in_place(
                buffers->k, QBH_BLOCK_M * QBH_BLOCK_KV_HIDDEN,
                &header->qparams[QBH_BLOCK_QP_K_ROPE]);
            qbh_expand_u8_to_f16_in_place(
                buffers->v, QBH_BLOCK_M * QBH_BLOCK_KV_HIDDEN,
                &header->qparams[QBH_BLOCK_QP_V]);
        }
    } else {
        if (qbh_attention_parallel_qk_norm_enabled(
                header->attention_pipeline_mode) &&
            rope_check == NULL) {
            if (qbh_hvx_pool_qk_norm_rope(
                    header, w4f16_pool,
                    (__fp16 *)buffers->q, (__fp16 *)buffers->k,
                    (const __fp16 *)buffers->q_norm_weight,
                    (const __fp16 *)buffers->k_norm_weight,
                    (const __fp16 *)buffers->rope_cos,
                    (const __fp16 *)buffers->rope_sin) != 0) {
                return QBH_BLOCK_STATUS_QK_NORM_ROPE_FAILED;
            }
        } else if ((header->common_ops_mask &
                    QBH_BLOCK_COMMON_OP_ROPE) != 0U) {
            qbh_hvx_qk_norm_rope_f16(
                (__fp16 *)buffers->q, QBH_BLOCK_M, QBH_BLOCK_HEADS,
                QBH_BLOCK_HIDDEN, QBH_BLOCK_HEAD_DIM,
                (const __fp16 *)buffers->q_norm_weight,
                (const __fp16 *)buffers->rope_cos,
                (const __fp16 *)buffers->rope_sin, rope_check);
            qbh_hvx_qk_norm_rope_f16(
                (__fp16 *)buffers->k, QBH_BLOCK_M,
                QBH_BLOCK_KV_HEADS, QBH_BLOCK_KV_HIDDEN,
                QBH_BLOCK_HEAD_DIM,
                (const __fp16 *)buffers->k_norm_weight,
                (const __fp16 *)buffers->rope_cos,
                (const __fp16 *)buffers->rope_sin, rope_check);
        } else {
            qbh_qk_norm_rope_f16(
                (__fp16 *)buffers->q, QBH_BLOCK_HEADS,
                QBH_BLOCK_HIDDEN,
                (const __fp16 *)buffers->q_norm_weight,
                (const __fp16 *)buffers->rope_cos,
                (const __fp16 *)buffers->rope_sin);
            qbh_qk_norm_rope_f16(
                (__fp16 *)buffers->k, QBH_BLOCK_KV_HEADS,
                QBH_BLOCK_KV_HIDDEN,
                (const __fp16 *)buffers->k_norm_weight,
                (const __fp16 *)buffers->rope_cos,
                (const __fp16 *)buffers->rope_sin);
        }
    }
    if (!qbh_attention_gqa_enabled(
            header->attention_pipeline_mode) &&
        u8_integer_attention_enabled == 0U) {
        audit_start = qbh_attribution_begin(header);
        qbh_record_f16_nonfinite(
            header, buffers->q, QBH_BLOCK_M * QBH_BLOCK_HIDDEN,
            QBH_BLOCK_NUMERICAL_Q_ROPE);
        qbh_record_f16_nonfinite(
            header, buffers->k, QBH_BLOCK_M * QBH_BLOCK_KV_HIDDEN,
            QBH_BLOCK_NUMERICAL_K_ROPE);
        qbh_attribution_accumulate(
            header, audit_start, &header->qk_norm_rope_audit_ticks);
    }
    header->qk_norm_rope_ticks += HAP_perf_get_qtimer_count() - start;

    if (scan_enabled != 0U &&
        header->variant != QBH_BLOCK_W4U8 &&
        header->numerical_audit_enabled != 0U &&
        qbh_scan_audit_f16_q(
            header, shared, buffers) != 0) {
        return QBH_BLOCK_STATUS_ATTENTION_FAILED;
    }

    if (scan_enabled != 0U &&
        reuse_prefill_attention_carriers == 0U &&
        reuse_f16_prefill_attention_carriers == 0U) {
        const uint64_t cache_append_start =
            HAP_perf_get_qtimer_count();
        const uint64_t cache_append_dma_before =
            header->scan_cache_append_ticks;
        int cache_status =
            header->variant == QBH_BLOCK_W4U8
                ? (u8_integer_attention_enabled != 0U
                       ? qbh_scan_append_u8_kv(
                             header, shared, buffers, logical_rows,
                             past_tokens)
                       : -1)
                : qbh_scan_append_f16_kv(
                      header, shared, buffers, logical_rows,
                      past_tokens);
        const uint64_t cache_append_elapsed =
            HAP_perf_get_qtimer_count() - cache_append_start;
        const uint64_t cache_append_dma =
            header->scan_cache_append_ticks -
            cache_append_dma_before;
        if (cache_append_elapsed >= cache_append_dma) {
            header->scan_cache_pack_ticks +=
                cache_append_elapsed - cache_append_dma;
        }
        if (cache_status != 0) {
            return QBH_BLOCK_STATUS_ATTENTION_FAILED;
        }
    }

    if (header->attribution_enabled != 0U) {
        attention_attributed_before =
            qbh_attention_attributed_ticks(header);
    }
    start = HAP_perf_get_qtimer_count();
    if ((u8_integer_attention_enabled != 0U
             ? (scan_dynamic_attention != 0U
                    ? qbh_scan_u8_attention(
                          header, shared, buffers, worker,
                          w4f16_pool,
                          logical_rows, past_tokens)
                    : (w4f16_pool != NULL
                           ? qbh_hvx_pool_u8_attention(
                                 header, w4f16_pool, buffers, worker)
                           : qbh_attention_u8_integer(
                                 header, buffers, worker)))
             : (scan_dynamic_attention != 0U
                    ? qbh_scan_f16_attention(
                          header, shared, buffers, worker,
                          logical_rows, past_tokens)
                    : qbh_attention_f16(
                          header, buffers, worker, w4f16_pool,
                          softmax_check))) != 0) {
        return QBH_BLOCK_STATUS_ATTENTION_FAILED;
    }
    if (scan_enabled != 0U &&
        header->variant != QBH_BLOCK_W4U8 &&
        header->numerical_audit_enabled != 0U &&
        qbh_scan_audit_f16_attention(
            header, shared, buffers) != 0) {
        return QBH_BLOCK_STATUS_ATTENTION_FAILED;
    }
    if (u8_integer_attention_enabled != 0U &&
        scan_dynamic_attention == 0U &&
        header->numerical_audit_enabled != 0U) {
        if (header->numerical_status ==
                QBH_BLOCK_NUMERICAL_UNCHECKED &&
            header->u8_attention_fused_k_operand_mismatch_count != 0U) {
            header->numerical_status =
                QBH_BLOCK_NUMERICAL_ATTENTION_QK;
        }
        header->u8_attention_actual_score_hash = qbh_fnv1a64_bytes(
            buffers->scores, QBH_BLOCK_SCORE_ELEMENTS);
        header->u8_attention_actual_probability_hash = qbh_fnv1a64_bytes(
            buffers->probability, QBH_BLOCK_SCORE_ELEMENTS);
        header->u8_attention_actual_av_hash = qbh_fnv1a64_bytes(
            buffers->hmx_activation,
            QBH_BLOCK_M * QBH_BLOCK_HIDDEN);
        if (header->numerical_status ==
                QBH_BLOCK_NUMERICAL_UNCHECKED &&
            header->u8_attention_expected_score_hash != 0U &&
            header->u8_attention_actual_score_hash !=
                header->u8_attention_expected_score_hash) {
            header->numerical_status =
                QBH_BLOCK_NUMERICAL_ATTENTION_QK;
        }
        if (header->numerical_status ==
                QBH_BLOCK_NUMERICAL_UNCHECKED &&
            header->u8_attention_expected_probability_hash != 0U &&
            header->u8_attention_actual_probability_hash !=
                header->u8_attention_expected_probability_hash) {
            header->numerical_status =
                QBH_BLOCK_NUMERICAL_ATTENTION_SOFTMAX;
        }
        if (header->numerical_status ==
                QBH_BLOCK_NUMERICAL_UNCHECKED &&
            header->u8_attention_expected_av_hash != 0U &&
            header->u8_attention_actual_av_hash !=
                header->u8_attention_expected_av_hash) {
            header->numerical_status =
                QBH_BLOCK_NUMERICAL_ATTENTION_AV;
        }
        {
            uint8_t *audit = (uint8_t *)(uintptr_t)(
                shared + header->u8_attention_audit_output_offset);
            uint32_t offset = 0U;
#define QBH_COPY_U8_ATTN_AUDIT(source_, bytes_)                         \
            do {                                                        \
                if (qbh_dma_copy(                                       \
                        header, audit + offset, (source_),               \
                        (bytes_), 0U) != 0) {                            \
                    return QBH_BLOCK_STATUS_ATTENTION_FAILED;           \
                }                                                       \
                offset += (bytes_);                                     \
            } while (0)
            QBH_COPY_U8_ATTN_AUDIT(
                buffers->q, QBH_BLOCK_U8_ATTENTION_Q_BYTES);
            QBH_COPY_U8_ATTN_AUDIT(
                buffers->k, QBH_BLOCK_U8_ATTENTION_KV_BYTES);
            QBH_COPY_U8_ATTN_AUDIT(
                buffers->v, QBH_BLOCK_U8_ATTENTION_KV_BYTES);
            QBH_COPY_U8_ATTN_AUDIT(
                buffers->scores, QBH_BLOCK_U8_ATTENTION_SCORE_BYTES);
            QBH_COPY_U8_ATTN_AUDIT(
                buffers->probability,
                QBH_BLOCK_U8_ATTENTION_SCORE_BYTES);
            QBH_COPY_U8_ATTN_AUDIT(
                buffers->hmx_activation,
                QBH_BLOCK_U8_ATTENTION_AV_BYTES);
#undef QBH_COPY_U8_ATTN_AUDIT
            header->u8_attention_audit_ddr_write_bytes += offset;
        }
    }
    if (u8_integer_attention_enabled != 0U &&
        scan_dynamic_attention != 0U &&
        header->numerical_audit_enabled != 0U) {
        uint8_t *audit = shared +
            header->u8_attention_audit_output_offset;
        const uint32_t av_offset =
            QBH_BLOCK_U8_ATTENTION_Q_BYTES +
            2U * QBH_BLOCK_U8_ATTENTION_KV_BYTES +
            2U * QBH_BLOCK_U8_ATTENTION_SCORE_BYTES;
        if (qbh_dma_copy(
                header, audit + av_offset, buffers->q,
                QBH_BLOCK_U8_ATTENTION_AV_BYTES, 0U) != 0) {
            return QBH_BLOCK_STATUS_ATTENTION_FAILED;
        }
        header->u8_attention_audit_ddr_write_bytes +=
            QBH_BLOCK_U8_ATTENTION_AV_BYTES;
    }
    if (crouton_qkv_enabled != 0U) {
        header->crouton_av_weight_count += QBH_BLOCK_KV_HEADS;
    }
    if (crouton_av_o_enabled != 0U) {
        header->crouton_av_o_head_count += QBH_BLOCK_HEADS;
        header->crouton_av_unpack_skipped += QBH_BLOCK_HEADS;
    }
    if (header->variant == QBH_BLOCK_W4U8 &&
        u8_integer_attention_enabled == 0U) {
        if ((header->common_ops_mask &
             QBH_BLOCK_COMMON_OP_SOFTMAX) != 0U) {
            qbh_hvx_quantize_f16_to_u8(
                (const __fp16 *)buffers->attention_concat,
                buffers->attention_concat, hidden_elements,
                &header->qparams[QBH_BLOCK_QP_ATTENTION_CONCAT]);
        } else {
            qbh_quantize_f16_buffer(
                (const __fp16 *)buffers->attention_concat,
                buffers->attention_concat, hidden_elements,
                &header->qparams[QBH_BLOCK_QP_ATTENTION_CONCAT]);
        }
    }
    {
        uint64_t attention_end = HAP_perf_get_qtimer_count();
        uint64_t attention_elapsed = attention_end - start;
        header->attention_ticks += attention_elapsed;
        if (header->attribution_enabled != 0U) {
            uint64_t attributed_delta =
                qbh_attention_attributed_ticks(header) -
                attention_attributed_before;
            if (attention_elapsed > attributed_delta) {
                header->attention_unattributed_ticks +=
                    attention_elapsed - attributed_delta;
            }
        }
    }

    if (reuse_prefill_attention_carriers != 0U) {
        const uint64_t cache_append_start =
            HAP_perf_get_qtimer_count();
        const uint64_t cache_append_dma_before =
            header->scan_cache_append_ticks;
        int cache_status =
            qbh_scan_persist_u8_prefill_attention_carriers(
                header, shared, buffers, logical_rows, past_tokens);
        const uint64_t cache_append_elapsed =
            HAP_perf_get_qtimer_count() - cache_append_start;
        const uint64_t cache_append_dma =
            header->scan_cache_append_ticks -
            cache_append_dma_before;
        if (cache_append_elapsed >= cache_append_dma) {
            header->scan_cache_pack_ticks +=
                cache_append_elapsed - cache_append_dma;
        }
        if (cache_status != 0) {
            return QBH_BLOCK_STATUS_ATTENTION_FAILED;
        }
    } else if (reuse_f16_prefill_attention_carriers != 0U) {
        const uint64_t cache_append_start =
            HAP_perf_get_qtimer_count();
        const uint64_t cache_append_dma_before =
            header->scan_cache_append_ticks;
        int cache_status =
            qbh_scan_persist_f16_prefill_attention_carriers(
                header, shared, buffers, logical_rows, past_tokens);
        const uint64_t cache_append_elapsed =
            HAP_perf_get_qtimer_count() - cache_append_start;
        const uint64_t cache_append_dma =
            header->scan_cache_append_ticks -
            cache_append_dma_before;
        if (cache_append_elapsed >= cache_append_dma) {
            header->scan_cache_pack_ticks +=
                cache_append_elapsed - cache_append_dma;
        }
        if (cache_status != 0) {
            return QBH_BLOCK_STATUS_ATTENTION_FAILED;
        }
    }

    start = HAP_perf_get_qtimer_count();
    if (scan_dynamic_attention != 0U &&
        header->variant != QBH_BLOCK_W4U8) {
        qbh_pack_fp16_activation(
            (const __fp16 *)buffers->q,
            QBH_BLOCK_HIDDEN, QBH_BLOCK_HIDDEN,
            (__fp16 *)buffers->hmx_activation);
        qbh_hmx_fp16_init_unity_scale(buffers->scale_or_bias);
    }
    if (qbh_run_projection(
            header, shared, &header->projections[QBH_BLOCK_PROJ_O],
            buffers, worker, w4f16_pool,
            scan_dynamic_attention != 0U
                ? (header->variant == QBH_BLOCK_W4U8
                       ? buffers->q
                       : buffers->hmx_activation)
                : (crouton_av_o_enabled != 0U
                       ? buffers->hmx_activation
                       : buffers->attention_concat),
            buffers->attention_projection,
            scan_dynamic_attention != 0U
                ? 1U : crouton_av_o_enabled,
            &header->projections[QBH_BLOCK_PROJ_GATE],
            &cross_prefetch) != 0) {
        return QBH_BLOCK_STATUS_O_PROJECTION_FAILED;
    }
    if (header->variant == QBH_BLOCK_W4U8 &&
        qbh_copy_w4u8_tail_audit(
            header, shared, QBH_BLOCK_U8_TAIL_O_OFFSET,
            buffers->attention_projection,
            QBH_BLOCK_M * QBH_BLOCK_HIDDEN) != 0) {
        return QBH_BLOCK_STATUS_O_PROJECTION_FAILED;
    }
    if (scan_enabled != 0U &&
        header->variant != QBH_BLOCK_W4U8 &&
        header->numerical_audit_enabled != 0U &&
        qbh_scan_audit_f16_o_projection(
            header, shared, buffers) != 0) {
        return QBH_BLOCK_STATUS_O_PROJECTION_FAILED;
    }
    if (header->variant != QBH_BLOCK_W4U8) {
        audit_start = qbh_attribution_begin(header);
        qbh_record_f16_nonfinite(
            header, buffers->attention_projection, hidden_elements,
            QBH_BLOCK_NUMERICAL_O);
        qbh_attribution_accumulate(
            header, audit_start, &header->o_projection_audit_ticks);
    }
    header->o_projection_ticks += HAP_perf_get_qtimer_count() - start;

    start = HAP_perf_get_qtimer_count();
    if (header->variant == QBH_BLOCK_W4U8) {
        if (header->residual_mode ==
                QBH_BLOCK_RESIDUAL_HVX_FUSED_POST_NORM ||
            header->residual_mode ==
                QBH_BLOCK_RESIDUAL_HVX_FUSED_POST_NORM_POOL4 ||
            header->residual_mode ==
                QBH_BLOCK_RESIDUAL_HVX_FUSED_POST_NORM_POOL6 ||
            header->residual_mode ==
                QBH_BLOCK_RESIDUAL_HVX_FUSED_POST_NORM_POOL6_SHUFFLE4) {
            if (w4u8_mlp_native_input_enabled != 0U) {
                if (w4u8_o_native_output_enabled != 0U) {
                    if (header->residual_mode ==
                            QBH_BLOCK_RESIDUAL_HVX_FUSED_POST_NORM_POOL4 ||
                        header->residual_mode ==
                            QBH_BLOCK_RESIDUAL_HVX_FUSED_POST_NORM_POOL6 ||
                        header->residual_mode ==
                            QBH_BLOCK_RESIDUAL_HVX_FUSED_POST_NORM_POOL6_SHUFFLE4) {
                        if (qbh_hvx_pool_u8_native_residual(
                                header, w4f16_pool,
                                buffers->residual,
                                &header->qparams[QBH_BLOCK_QP_BLOCK_INPUT],
                                buffers->attention_projection,
                                &header->qparams[QBH_BLOCK_QP_ATTENTION_PROJECTION],
                                &header->qparams[QBH_BLOCK_QP_POST_ATTENTION_RESIDUAL],
                                (const __fp16 *)buffers->post_norm_weight,
                                w4u8_mlp_native_activation,
                                &header->qparams[QBH_BLOCK_QP_POST_ATTENTION_NORM],
                                QBH_BLOCK_U8_RESIDUAL_POST_NORM) != 0) {
                            return QBH_BLOCK_STATUS_RESIDUAL_POOL_FAILED;
                        }
                    } else {
                        qbh_hvx_residual_rms_norm_u8_native_io(
                            buffers->residual,
                            &header->qparams[QBH_BLOCK_QP_BLOCK_INPUT],
                            buffers->attention_projection,
                            &header->qparams[QBH_BLOCK_QP_ATTENTION_PROJECTION],
                            &header->qparams[QBH_BLOCK_QP_POST_ATTENTION_RESIDUAL],
                            (const __fp16 *)buffers->post_norm_weight,
                            w4u8_mlp_native_activation,
                            &header->qparams[QBH_BLOCK_QP_POST_ATTENTION_NORM],
                            QBH_BLOCK_M, QBH_BLOCK_HIDDEN);
                    }
                } else {
                    qbh_hvx_residual_rms_norm_u8_native_activation(
                        buffers->residual,
                        &header->qparams[QBH_BLOCK_QP_BLOCK_INPUT],
                        buffers->attention_projection,
                        &header->qparams[QBH_BLOCK_QP_ATTENTION_PROJECTION],
                        &header->qparams[QBH_BLOCK_QP_POST_ATTENTION_RESIDUAL],
                        (const __fp16 *)buffers->post_norm_weight,
                        w4u8_mlp_native_activation,
                        &header->qparams[QBH_BLOCK_QP_POST_ATTENTION_NORM],
                        QBH_BLOCK_M, QBH_BLOCK_HIDDEN);
                }
            } else {
                qbh_hvx_residual_rms_norm_u8(
                    buffers->residual,
                    &header->qparams[QBH_BLOCK_QP_BLOCK_INPUT],
                    buffers->attention_projection,
                    &header->qparams[QBH_BLOCK_QP_ATTENTION_PROJECTION],
                    &header->qparams[QBH_BLOCK_QP_POST_ATTENTION_RESIDUAL],
                    (const __fp16 *)buffers->post_norm_weight,
                    buffers->normalized,
                    &header->qparams[QBH_BLOCK_QP_POST_ATTENTION_NORM],
                    QBH_BLOCK_M, QBH_BLOCK_HIDDEN);
            }
            post_attention_norm_fused = 1;
        } else if (header->residual_mode == QBH_BLOCK_RESIDUAL_HVX) {
            qbh_hvx_residual_add_u8(
                buffers->residual,
                &header->qparams[QBH_BLOCK_QP_BLOCK_INPUT],
                buffers->attention_projection,
                &header->qparams[QBH_BLOCK_QP_ATTENTION_PROJECTION],
                buffers->residual,
                &header->qparams[QBH_BLOCK_QP_POST_ATTENTION_RESIDUAL],
                hidden_elements);
        } else {
            qbh_residual_add_u8(
                buffers->residual,
                &header->qparams[QBH_BLOCK_QP_BLOCK_INPUT],
                buffers->attention_projection,
                &header->qparams[QBH_BLOCK_QP_ATTENTION_PROJECTION],
                buffers->residual,
                &header->qparams[QBH_BLOCK_QP_POST_ATTENTION_RESIDUAL],
                hidden_elements);
        }
    } else {
        if (header->residual_mode ==
                QBH_BLOCK_RESIDUAL_HVX_FUSED_POST_NORM ||
            header->residual_mode ==
                QBH_BLOCK_RESIDUAL_HVX_FUSED_POST_NORM_POOL4 ||
            header->residual_mode ==
                QBH_BLOCK_RESIDUAL_HVX_FUSED_POST_NORM_POOL6 ||
            header->residual_mode ==
                QBH_BLOCK_RESIDUAL_HVX_FUSED_POST_NORM_POOL6_SHUFFLE4) {
            uint64_t norm_start = HAP_perf_get_qtimer_count();
            if ((header->fp16_common_schedule_mode &
                 QBH_BLOCK_FP16_COMMON_SCHEDULE_POST_RESIDUAL_NORM_POOL) !=
                    0U) {
                if (qbh_hvx_pool_fp16_post_residual_norm(
                        header, w4f16_pool,
                        (__fp16 *)buffers->residual,
                        (const __fp16 *)buffers->attention_projection,
                        (const __fp16 *)buffers->post_norm_weight,
                        crouton_post_norm_enabled != 0U
                            ? (__fp16 *)buffers->hmx_activation
                            : (__fp16 *)buffers->normalized,
                        crouton_post_norm_enabled) != 0) {
                    return QBH_BLOCK_STATUS_RESIDUAL_POOL_FAILED;
                }
                if (crouton_post_norm_enabled != 0U) {
                    ++header->crouton_norm_projection_count;
                    qbh_hmx_fp16_init_unity_scale(
                        buffers->scale_or_bias);
                }
            } else if (crouton_post_norm_enabled != 0U) {
                qbh_hvx_residual_rms_norm_f16_crouton(
                    (__fp16 *)buffers->residual,
                    (const __fp16 *)buffers->attention_projection,
                    (const __fp16 *)buffers->post_norm_weight,
                    (__fp16 *)buffers->hmx_activation,
                    QBH_BLOCK_M, QBH_BLOCK_HIDDEN);
                ++header->crouton_norm_projection_count;
                qbh_hmx_fp16_init_unity_scale(
                    buffers->scale_or_bias);
            } else {
                qbh_hvx_residual_rms_norm_f16(
                    (__fp16 *)buffers->residual,
                    (const __fp16 *)buffers->attention_projection,
                    (const __fp16 *)buffers->post_norm_weight,
                    (__fp16 *)buffers->normalized,
                    QBH_BLOCK_M, QBH_BLOCK_HIDDEN);
            }
            if (crouton_post_norm_enabled != 0U) {
                header->crouton_norm_store_ticks +=
                    HAP_perf_get_qtimer_count() - norm_start;
            }
            post_attention_norm_fused = 1;
        } else if (header->residual_mode == QBH_BLOCK_RESIDUAL_HVX) {
            qbh_hvx_residual_add_f16(
                (__fp16 *)buffers->residual,
                (const __fp16 *)buffers->attention_projection,
                hidden_elements);
        } else {
            qbh_residual_add_f16(
                (__fp16 *)buffers->residual,
                (const __fp16 *)buffers->attention_projection,
                hidden_elements);
        }
        audit_start = qbh_attribution_begin(header);
        qbh_record_f16_nonfinite(
            header, buffers->residual, hidden_elements,
            QBH_BLOCK_NUMERICAL_POST_RESIDUAL);
        qbh_attribution_accumulate(
            header, audit_start,
            &header->post_attention_residual_audit_ticks);
    }
    if (header->variant == QBH_BLOCK_W4U8 &&
        qbh_copy_w4u8_tail_audit(
            header, shared, QBH_BLOCK_U8_TAIL_POST_RESIDUAL_OFFSET,
            buffers->residual,
            QBH_BLOCK_M * QBH_BLOCK_HIDDEN) != 0) {
        return QBH_BLOCK_STATUS_RESIDUAL_POOL_FAILED;
    }
    header->post_attention_residual_ticks +=
        HAP_perf_get_qtimer_count() - start;

    start = HAP_perf_get_qtimer_count();
    if (header->variant == QBH_BLOCK_W4U8) {
        if (post_attention_norm_fused != 0) {
            /* Materialized by the U8 residual/RMSNorm two-pass fusion. */
        } else if ((header->common_ops_mask &
                    QBH_BLOCK_COMMON_OP_RMS_NORM) != 0U) {
            if (w4u8_mlp_native_input_enabled != 0U) {
                qbh_hvx_rms_norm_u8_native_activation(
                    buffers->residual,
                    &header->qparams[QBH_BLOCK_QP_POST_ATTENTION_RESIDUAL],
                    (const __fp16 *)buffers->post_norm_weight,
                    w4u8_mlp_native_activation,
                    &header->qparams[QBH_BLOCK_QP_POST_ATTENTION_NORM],
                    QBH_BLOCK_M, QBH_BLOCK_HIDDEN);
            } else {
                qbh_hvx_rms_norm_u8(
                    buffers->residual,
                    &header->qparams[QBH_BLOCK_QP_POST_ATTENTION_RESIDUAL],
                    (const __fp16 *)buffers->post_norm_weight,
                    buffers->normalized,
                    &header->qparams[QBH_BLOCK_QP_POST_ATTENTION_NORM],
                    QBH_BLOCK_M, QBH_BLOCK_HIDDEN);
            }
        } else {
            qbh_rms_norm_u8(
                buffers->residual,
                &header->qparams[QBH_BLOCK_QP_POST_ATTENTION_RESIDUAL],
                (const __fp16 *)buffers->post_norm_weight,
                buffers->normalized,
                &header->qparams[QBH_BLOCK_QP_POST_ATTENTION_NORM],
                QBH_BLOCK_M, QBH_BLOCK_HIDDEN);
        }
    } else {
        if (post_attention_norm_fused != 0) {
            /* Materialized by the fused residual/RMSNorm first pass. */
        } else if (crouton_post_norm_enabled != 0U) {
            uint64_t norm_start = HAP_perf_get_qtimer_count();
            qbh_hvx_rms_norm_f16_crouton(
                (const __fp16 *)buffers->residual,
                (const __fp16 *)buffers->post_norm_weight,
                (__fp16 *)buffers->hmx_activation,
                QBH_BLOCK_M, QBH_BLOCK_HIDDEN);
            header->crouton_norm_store_ticks +=
                HAP_perf_get_qtimer_count() - norm_start;
            ++header->crouton_norm_projection_count;
            qbh_hmx_fp16_init_unity_scale(buffers->scale_or_bias);
        } else if ((header->common_ops_mask &
                    QBH_BLOCK_COMMON_OP_RMS_NORM) != 0U) {
            qbh_hvx_rms_norm_f16(
                (const __fp16 *)buffers->residual,
                (const __fp16 *)buffers->post_norm_weight,
                (__fp16 *)buffers->normalized,
                QBH_BLOCK_M, QBH_BLOCK_HIDDEN, rms_check);
        } else {
            qbh_rms_norm_f16(
                (const __fp16 *)buffers->residual,
                (const __fp16 *)buffers->post_norm_weight,
                (__fp16 *)buffers->normalized,
                QBH_BLOCK_M, QBH_BLOCK_HIDDEN);
        }
        audit_start = qbh_attribution_begin(header);
        qbh_record_f16_nonfinite(
            header,
            crouton_post_norm_enabled != 0U
                ? buffers->hmx_activation : buffers->normalized,
            hidden_elements,
            QBH_BLOCK_NUMERICAL_POST_NORM);
        qbh_attribution_accumulate(
            header, audit_start,
            &header->post_attention_norm_audit_ticks);
    }
    if (header->variant == QBH_BLOCK_W4U8 &&
        w4u8_mlp_native_activation != NULL &&
        qbh_copy_w4u8_tail_audit(
            header, shared, QBH_BLOCK_U8_TAIL_POST_NORM_OFFSET,
            w4u8_mlp_native_activation,
            QBH_BLOCK_M * QBH_BLOCK_HIDDEN) != 0) {
        return QBH_BLOCK_STATUS_RESIDUAL_POOL_FAILED;
    }
    header->post_attention_norm_ticks +=
        HAP_perf_get_qtimer_count() - start;

    if (qbh_block_mlp_is_w4u8_streaming(header->mlp_mode)) {
        if (qbh_run_w4u8_streaming_mlp(
                header, shared, buffers, worker, w4f16_pool,
                w4u8_mlp_native_input_enabled,
                w4u8_mlp_native_output_enabled) != 0) {
            return QBH_BLOCK_STATUS_MLP_STREAM_FAILED;
        }
        goto w4u8_mlp_complete;
    }

    if (header->variant == QBH_BLOCK_W4F16) {
        gate_up_weight_dma_before = header->weight_dma_ticks;
        gate_up_expand_before = header->w4f16_expand_ticks;
        gate_up_expand_work_before = header->w4f16_expand_work_ticks;
        gate_up_expand_pool_wait_before =
            header->w4f16_expand_pool_wait_ticks;
        gate_up_prefetch_wait_before =
            header->w4f16_prefetch_wait_ticks;
        gate_up_hmx_wait_before = header->projection_hmx_wait_ticks;
        gate_up_hmx_tail_wait_before =
            header->w4f16_hmx_tail_wait_ticks;
        gate_up_unpack_before = header->projection_unpack_ticks;
        gate_up_stream_work_before =
            header->mlp_stream_worker_work_ticks +
            header->mlp_stream_main_work_ticks;
        gate_up_stream_ready_wait_before =
            header->mlp_stream_ready_wait_ticks;
        gate_up_stream_join_wait_before =
            header->mlp_stream_join_wait_ticks;
        gate_up_hmx_command_before = header->hmx_command_count;
    }
    start = HAP_perf_get_qtimer_count();
    if (header->variant == QBH_BLOCK_F16F16 &&
        header->f16f16_projection_mode ==
            QBH_BLOCK_F16F16_PROJECTION_GATE8_INTERLEAVED &&
        header->mlp_mode == QBH_BLOCK_MLP_CROUTON_NATIVE_BATCH8) {
        if (crouton_post_norm_enabled == 0U) {
            uint64_t pack_start = HAP_perf_get_qtimer_count();
            qbh_pack_fp16_activation(
                (const __fp16 *)buffers->normalized,
                QBH_BLOCK_HIDDEN, QBH_BLOCK_HIDDEN,
                (__fp16 *)buffers->hmx_activation);
            header->projection_pack_ticks +=
                HAP_perf_get_qtimer_count() - pack_start;
        }
        if (qbh_mlp_stream_pipeline_start(
                header, w4f16_pool,
                (const __fp16 *)buffers->gate,
                (const __fp16 *)buffers->up,
                (__fp16 *)buffers->middle,
                (__fp16 *)buffers->q) != 0) {
            return QBH_BLOCK_STATUS_MLP_STREAM_FAILED;
        }
        if (qbh_run_f16f16_interleaved_gate_up(
                header, shared, buffers, worker, w4f16_pool,
                buffers->hmx_activation) != 0) {
            (void)qbh_mlp_stream_pipeline_wait(
                header, w4f16_pool, 1U);
            return QBH_BLOCK_STATUS_GATE_UP_FAILED;
        }
    } else if (header->variant == QBH_BLOCK_W4F16 &&
        (header->mlp_mode == QBH_BLOCK_MLP_STREAMING ||
         header->mlp_mode == QBH_BLOCK_MLP_CROUTON_NATIVE ||
         header->mlp_mode == QBH_BLOCK_MLP_CROUTON_NATIVE_BATCH8)) {
        if (qbh_mlp_stream_pipeline_start(
                header, w4f16_pool,
                (const __fp16 *)buffers->gate,
                (const __fp16 *)buffers->up,
                (__fp16 *)buffers->middle,
                (__fp16 *)buffers->q) != 0) {
            return QBH_BLOCK_STATUS_MLP_STREAM_FAILED;
        }
        if (qbh_run_w4f16_interleaved_gate_up(
                header, shared, buffers, worker, w4f16_pool,
                &cross_prefetch) != 0) {
            (void)qbh_mlp_stream_pipeline_wait(
                header, w4f16_pool, 1U);
            return QBH_BLOCK_STATUS_GATE_UP_FAILED;
        }
    } else {
        if (qbh_run_projection(
                header, shared,
                &header->projections[QBH_BLOCK_PROJ_GATE], buffers,
                worker, w4f16_pool,
                crouton_post_norm_enabled != 0U
                    ? buffers->hmx_activation : buffers->normalized,
                buffers->gate, crouton_post_norm_enabled,
                &header->projections[QBH_BLOCK_PROJ_UP],
                &cross_prefetch) != 0) {
            return QBH_BLOCK_STATUS_GATE_UP_FAILED;
        }
        if (qbh_mlp_stream_pipeline_start(
                header, w4f16_pool,
                (const __fp16 *)buffers->gate,
                (const __fp16 *)buffers->up,
                (__fp16 *)buffers->middle,
                (__fp16 *)buffers->q) != 0) {
            return QBH_BLOCK_STATUS_MLP_STREAM_FAILED;
        }
        if (qbh_run_projection(
                header, shared,
                &header->projections[QBH_BLOCK_PROJ_UP], buffers,
                worker, w4f16_pool,
                crouton_post_norm_enabled != 0U
                    ? buffers->hmx_activation : buffers->normalized,
                buffers->up, crouton_post_norm_enabled,
                &header->projections[QBH_BLOCK_PROJ_DOWN],
                &cross_prefetch) != 0) {
            (void)qbh_mlp_stream_pipeline_wait(
                header, w4f16_pool, 1U);
            return QBH_BLOCK_STATUS_GATE_UP_FAILED;
        }
    }
    if (qbh_mlp_stream_pipeline_wait(
            header, w4f16_pool, 0U) != 0) {
        return QBH_BLOCK_STATUS_MLP_STREAM_FAILED;
    }
    if (header->variant != QBH_BLOCK_W4U8 &&
        header->mlp_mode != QBH_BLOCK_MLP_CROUTON_NATIVE &&
        header->mlp_mode != QBH_BLOCK_MLP_CROUTON_NATIVE_BATCH8) {
        audit_start = qbh_attribution_begin(header);
        qbh_record_f16_nonfinite(
            header, buffers->gate, intermediate_elements,
            QBH_BLOCK_NUMERICAL_GATE);
        qbh_record_f16_nonfinite(
            header, buffers->up, intermediate_elements,
            QBH_BLOCK_NUMERICAL_UP);
        qbh_attribution_accumulate(
            header, audit_start, &header->gate_up_audit_ticks);
    }
    header->gate_up_ticks += HAP_perf_get_qtimer_count() - start;
    if (header->variant == QBH_BLOCK_W4F16) {
        header->w4f16_gate_up_weight_dma_ticks +=
            header->weight_dma_ticks - gate_up_weight_dma_before;
        header->w4f16_gate_up_expand_ticks +=
            header->w4f16_expand_ticks - gate_up_expand_before;
        header->w4f16_gate_up_expand_work_ticks +=
            header->w4f16_expand_work_ticks -
            gate_up_expand_work_before;
        header->w4f16_gate_up_expand_pool_wait_ticks +=
            header->w4f16_expand_pool_wait_ticks -
            gate_up_expand_pool_wait_before;
        header->w4f16_gate_up_prefetch_wait_ticks +=
            header->w4f16_prefetch_wait_ticks -
            gate_up_prefetch_wait_before;
        header->w4f16_gate_up_hmx_wait_ticks +=
            header->projection_hmx_wait_ticks -
            gate_up_hmx_wait_before;
        header->w4f16_gate_up_hmx_tail_wait_ticks +=
            header->w4f16_hmx_tail_wait_ticks -
            gate_up_hmx_tail_wait_before;
        header->w4f16_gate_up_unpack_ticks +=
            header->projection_unpack_ticks - gate_up_unpack_before;
        header->w4f16_gate_up_stream_work_ticks +=
            header->mlp_stream_worker_work_ticks +
                header->mlp_stream_main_work_ticks -
            gate_up_stream_work_before;
        header->w4f16_gate_up_stream_ready_wait_ticks +=
            header->mlp_stream_ready_wait_ticks -
            gate_up_stream_ready_wait_before;
        header->w4f16_gate_up_stream_join_wait_ticks +=
            header->mlp_stream_join_wait_ticks -
            gate_up_stream_join_wait_before;
        header->w4f16_gate_up_hmx_command_count +=
            header->hmx_command_count - gate_up_hmx_command_before;
    }

    start = HAP_perf_get_qtimer_count();
    if (header->variant == QBH_BLOCK_W4U8) {
        qbh_silu_multiply_u8(
            buffers->gate, &header->qparams[QBH_BLOCK_QP_GATE],
            buffers->up, &header->qparams[QBH_BLOCK_QP_UP],
            buffers->middle, &header->qparams[QBH_BLOCK_QP_MIDDLE],
            intermediate_elements);
    } else {
        if (header->mlp_mode == QBH_BLOCK_MLP_STREAMING) {
            qbh_hvx_silu_multiply_f16_audit(
                (const __fp16 *)buffers->gate,
                (const __fp16 *)buffers->up,
                (const __fp16 *)buffers->middle,
                intermediate_elements, silu_check);
        } else if (header->mlp_mode ==
                       QBH_BLOCK_MLP_CROUTON_NATIVE ||
                   header->mlp_mode ==
                       QBH_BLOCK_MLP_CROUTON_NATIVE_BATCH8) {
            /* The streaming worker already wrote the Crouton Down carrier. */
        } else if ((header->common_ops_mask &
                    QBH_BLOCK_COMMON_OP_SILU) != 0U) {
            if (qbh_hvx_pool_silu(
                    header, w4f16_pool,
                    (const __fp16 *)buffers->gate,
                    (const __fp16 *)buffers->up,
                    (__fp16 *)buffers->middle, intermediate_elements,
                    silu_check) != 0) {
                return QBH_BLOCK_STATUS_ACTIVATION_FAILED;
            }
        } else {
            qbh_silu_multiply_f16(
                (const __fp16 *)buffers->gate,
                (const __fp16 *)buffers->up,
                (__fp16 *)buffers->middle, intermediate_elements);
        }
        if (header->mlp_mode != QBH_BLOCK_MLP_CROUTON_NATIVE &&
            header->mlp_mode != QBH_BLOCK_MLP_CROUTON_NATIVE_BATCH8) {
            audit_start = qbh_attribution_begin(header);
            qbh_record_f16_nonfinite(
                header, buffers->middle, intermediate_elements,
                QBH_BLOCK_NUMERICAL_MIDDLE);
            qbh_attribution_accumulate(
                header, audit_start, &header->activation_audit_ticks);
        }
    }
    header->activation_ticks += HAP_perf_get_qtimer_count() - start;

    start = HAP_perf_get_qtimer_count();
    if (qbh_run_projection(
            header, shared,
            &header->projections[QBH_BLOCK_PROJ_DOWN], buffers,
            worker, w4f16_pool, buffers->middle, buffers->down,
            0U, NULL, &cross_prefetch) != 0) {
        return QBH_BLOCK_STATUS_DOWN_FAILED;
    }
    if (header->variant != QBH_BLOCK_W4U8) {
        audit_start = qbh_attribution_begin(header);
        qbh_record_f16_nonfinite(
            header, buffers->down, hidden_elements,
            QBH_BLOCK_NUMERICAL_DOWN);
        qbh_attribution_accumulate(
            header, audit_start, &header->down_audit_ticks);
    }
    header->down_ticks += HAP_perf_get_qtimer_count() - start;

w4u8_mlp_complete:
    start = HAP_perf_get_qtimer_count();
    if (header->variant == QBH_BLOCK_W4U8) {
        if (w4u8_mlp_native_output_enabled != 0U) {
            if (header->residual_mode ==
                    QBH_BLOCK_RESIDUAL_HVX_FUSED_POST_NORM_POOL4 ||
                header->residual_mode ==
                    QBH_BLOCK_RESIDUAL_HVX_FUSED_POST_NORM_POOL6 ||
                header->residual_mode ==
                    QBH_BLOCK_RESIDUAL_HVX_FUSED_POST_NORM_POOL6_SHUFFLE4) {
                if (qbh_hvx_pool_u8_native_residual(
                        header, w4f16_pool,
                        buffers->residual,
                        &header->qparams[QBH_BLOCK_QP_POST_ATTENTION_RESIDUAL],
                        w4u8_mlp_native_output,
                        &header->qparams[QBH_BLOCK_QP_DOWN],
                        &header->qparams[QBH_BLOCK_QP_BLOCK_OUTPUT],
                        NULL, NULL, NULL,
                        QBH_BLOCK_U8_RESIDUAL_FINAL) != 0) {
                    return QBH_BLOCK_STATUS_RESIDUAL_POOL_FAILED;
                }
            } else {
                qbh_hvx_residual_add_u8_native_output(
                    buffers->residual,
                    &header->qparams[QBH_BLOCK_QP_POST_ATTENTION_RESIDUAL],
                    w4u8_mlp_native_output,
                    &header->qparams[QBH_BLOCK_QP_DOWN],
                    &header->qparams[QBH_BLOCK_QP_BLOCK_OUTPUT],
                    QBH_BLOCK_M, QBH_BLOCK_HIDDEN);
            }
        } else if (header->residual_mode != QBH_BLOCK_RESIDUAL_SCALAR) {
            qbh_hvx_residual_add_u8(
                buffers->residual,
                &header->qparams[QBH_BLOCK_QP_POST_ATTENTION_RESIDUAL],
                buffers->down, &header->qparams[QBH_BLOCK_QP_DOWN],
                buffers->residual,
                &header->qparams[QBH_BLOCK_QP_BLOCK_OUTPUT],
                hidden_elements);
        } else {
            qbh_residual_add_u8(
                buffers->residual,
                &header->qparams[QBH_BLOCK_QP_POST_ATTENTION_RESIDUAL],
                buffers->down, &header->qparams[QBH_BLOCK_QP_DOWN],
                buffers->residual,
                &header->qparams[QBH_BLOCK_QP_BLOCK_OUTPUT],
                hidden_elements);
        }
    } else {
        if (header->residual_mode != QBH_BLOCK_RESIDUAL_SCALAR) {
            qbh_hvx_residual_add_f16(
                (__fp16 *)buffers->residual,
                (const __fp16 *)buffers->down,
                hidden_elements);
        } else {
            qbh_residual_add_f16(
                (__fp16 *)buffers->residual,
                (const __fp16 *)buffers->down,
                hidden_elements);
        }
        audit_start = qbh_attribution_begin(header);
        qbh_record_f16_nonfinite(
            header, buffers->residual, hidden_elements,
            QBH_BLOCK_NUMERICAL_OUTPUT);
        qbh_attribution_accumulate(
            header, audit_start, &header->final_residual_audit_ticks);
    }
    if (header->variant == QBH_BLOCK_W4U8 &&
        qbh_copy_w4u8_tail_audit(
            header, shared, QBH_BLOCK_U8_TAIL_FINAL_OFFSET,
            buffers->residual,
            QBH_BLOCK_M * QBH_BLOCK_HIDDEN) != 0) {
        return QBH_BLOCK_STATUS_RESIDUAL_POOL_FAILED;
    }
    if (header->numerical_status == QBH_BLOCK_NUMERICAL_UNCHECKED) {
        header->numerical_status = QBH_BLOCK_NUMERICAL_OK;
    }
    header->common_op_rms_max_abs = rms_check_metrics.max_abs;
    header->common_op_rms_cosine =
        qbh_hvx_check_cosine(&rms_check_metrics);
    header->common_op_rope_max_abs = rope_check_metrics.max_abs;
    header->common_op_rope_cosine =
        qbh_hvx_check_cosine(&rope_check_metrics);
    header->common_op_softmax_max_abs = softmax_check_metrics.max_abs;
    header->common_op_softmax_cosine =
        qbh_hvx_check_cosine(&softmax_check_metrics);
    header->common_op_silu_max_abs = silu_check_metrics.max_abs;
    header->common_op_silu_cosine =
        qbh_hvx_check_cosine(&silu_check_metrics);
    header->common_op_nonfinite_count =
        rms_check_metrics.nonfinite_count +
        rope_check_metrics.nonfinite_count +
        softmax_check_metrics.nonfinite_count +
        silu_check_metrics.nonfinite_count;
    header->common_op_softmax_mask_violation_count =
        softmax_check_metrics.mask_violation_count;
    header->final_residual_ticks += HAP_perf_get_qtimer_count() - start;
    return QBH_BLOCK_STATUS_OK;
}

AEEResult qbh_run_block_rpc(int32_t shared_fd, uint32_t shared_bytes,
                            uint8_t *vtcm, uint32_t vtcm_bytes,
                            uint32_t hmx_context_id,
                            uint32_t prepared_session_run_index) {
    struct qbh_block_header *header = NULL;
    struct qbh_block_buffers buffers;
    struct qbh_block_hmx_worker worker;
    struct qbh_block_w4f16_pool w4f16_pool;
    qurt_thread_attr_t attributes;
    qurt_thread_t thread;
    uint8_t *shared = NULL;
    int cache_status;
    int result;
    int thread_created = 0;
    int thread_joined = 0;
    int hvx_pool_created = 0;
    int main_hvx_locked = 0;
    int thread_exit_status = AEE_EFAILED;
    uint64_t invocation_start = 0U;
    uint64_t teardown_start = 0U;

    if (vtcm == NULL || vtcm_bytes != QBH_EXPECTED_FULL_VTCM_BYTES ||
        hmx_context_id == 0U ||
        shared_bytes < sizeof(struct qbh_block_header)) {
        return AEE_EBADPARM;
    }
    result = HAP_mmap_get(shared_fd, (void **)&shared, NULL);
    if (result != AEE_SUCCESS || shared == NULL) {
        return result != AEE_SUCCESS ? result : AEE_EFAILED;
    }
    header = (struct qbh_block_header *)shared;
    cache_status = qurt_mem_cache_clean(
        (qurt_addr_t)header, (qurt_size_t)sizeof(*header),
        QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
    if (cache_status == 0 &&
        header->replay_mode == QBH_BLOCK_REPLAY_CONTINUOUS &&
        header->replay_session_bytes ==
            sizeof(struct qbh_decode_session_state) &&
        header->replay_session_offset >= sizeof(*header) &&
        header->replay_session_offset <= shared_bytes &&
        header->replay_session_bytes <=
            shared_bytes - header->replay_session_offset) {
        cache_status = qurt_mem_cache_clean(
            (qurt_addr_t)(shared + header->replay_session_offset),
            (qurt_size_t)header->replay_session_bytes,
            QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
    }
    if (cache_status != 0 || !qbh_header_valid(header, shared_bytes)) {
        if (cache_status == 0) {
            header->dsp_status = QBH_BLOCK_STATUS_BAD_HEADER;
        }
        result = cache_status == 0 ? AEE_EBADPARM : AEE_EFAILED;
        goto publish;
    }

    memset((uint8_t *)header + offsetof(struct qbh_block_header, dsp_status),
           0, sizeof(*header) -
                  offsetof(struct qbh_block_header, dsp_status));
    header->dsp_status = QBH_BLOCK_STATUS_DSP_RUNNING;
    header->cache_status = cache_status;
    header->prepared_session_run_index = prepared_session_run_index;
    header->resource_vtcm_address = (uint32_t)(uintptr_t)vtcm;
    header->resource_hmx_context_id = hmx_context_id;
    header->vtcm_requested_bytes = QBH_EXPECTED_FULL_VTCM_BYTES;
    header->vtcm_acquired_bytes = vtcm_bytes;
    qbh_hvx_u8_set_norm_reduction_mode(
        header->u8_norm_reduction_mode);
    qbh_hvx_u8_set_qk_pair_kernel_mode(
        header->w4u8_qk_pair_kernel_mode);
    header->w4u8_qk_pair_kernel_mode_observed =
        header->w4u8_qk_pair_kernel_mode;

    if (header->full_stack_stage_mode ==
        QBH_BLOCK_FULL_STACK_MAP_GATE) {
        uint64_t all_layers_hash = UINT64_C(1469598103934665603);
        const uint32_t middle = QBH_VERTICAL_SLICE_LAYER_COUNT / 2U;
        for (uint32_t slice_index = 0U;
             slice_index < QBH_VERTICAL_SLICE_LAYER_COUNT;
             ++slice_index) {
            uint64_t layer_hash = qbh_map_gate_layer_hash(
                shared, &header->slice_layers[slice_index]);
            all_layers_hash ^= layer_hash;
            all_layers_hash *= UINT64_C(1099511628211);
            if (slice_index == 0U) {
                header->full_stack_map_gate_first_layer_hash = layer_hash;
            }
            if (slice_index == middle) {
                header->full_stack_map_gate_middle_layer_hash = layer_hash;
            }
            if (slice_index + 1U == QBH_VERTICAL_SLICE_LAYER_COUNT) {
                header->full_stack_map_gate_last_layer_hash = layer_hash;
            }
            ++header->full_stack_map_gate_layer_count;
        }
        header->full_stack_map_gate_hash = all_layers_hash;
        header->dsp_status = QBH_BLOCK_STATUS_OK;
        result = AEE_SUCCESS;
        goto publish;
    }

    if (qbh_plan_buffers(vtcm, vtcm_bytes, header->variant,
                         header->f16f16_projection_mode,
                         header->w4f16_pipeline_mode,
                         header->attention_pipeline_mode,
                         header->mlp_mode, header->generation_mode,
                         &buffers,
                         &header->vtcm_peak_plan_bytes) != 0) {
        header->dsp_status = QBH_BLOCK_STATUS_ARENA_FAILED;
        result = AEE_ENOMEMORY;
        goto publish;
    }

    if (header->attribution_enabled != 0U) {
        invocation_start = HAP_perf_get_qtimer_count();
    }

    memset(&worker, 0, sizeof(worker));
    worker.hmx_context_id = hmx_context_id;
    qurt_sem_init_val(&worker.command_ready, 0U);
    qurt_sem_init_val(&worker.command_done, 0U);
    qurt_sem_init_val(&worker.worker_started, 0U);
    qurt_thread_attr_init(&attributes);
    qurt_thread_attr_set_name(&attributes, "qbh-block-hmx");
    qurt_thread_attr_set_stack_addr(&attributes, qbh_block_hmx_stack);
    qurt_thread_attr_set_stack_size(&attributes,
                                    QBH_BLOCK_HMX_STACK_BYTES);
    qurt_thread_attr_set_priority(
        &attributes, qurt_thread_get_priority(qurt_thread_get_id()));
    qurt_thread_attr_set_detachstate(&attributes,
                                     QURT_THREAD_ATTR_CREATE_JOINABLE);
    result = qurt_thread_create(&thread, &attributes,
                                qbh_hmx_worker_main, &worker);
    if (result != QURT_EOK) {
        header->dsp_status = QBH_BLOCK_STATUS_HMX_WORKER_FAILED;
        goto destroy_semaphores;
    }
    thread_created = 1;
    qurt_sem_down(&worker.worker_started);
    header->hmx_lock_status = worker.lock_status;
    if (worker.lock_status != AEE_SUCCESS) {
        header->dsp_status = QBH_BLOCK_STATUS_HMX_WORKER_FAILED;
        result = AEE_EFAILED;
        goto stop_worker;
    }
    if (qurt_hvx_lock(QURT_HVX_MODE_128B) != AEE_SUCCESS) {
        header->dsp_status = QBH_BLOCK_STATUS_HMX_WORKER_FAILED;
        result = AEE_EFAILED;
        goto stop_worker;
    }
    main_hvx_locked = 1;
    if (header->variant == QBH_BLOCK_W4F16 ||
        (header->variant == QBH_BLOCK_W4U8 &&
         qbh_attention_u8_enabled(
             header->attention_pipeline_mode) &&
         header->attention_hvx_contexts > 1U) ||
        (header->variant == QBH_BLOCK_F16F16 &&
         ((header->mlp_mode != QBH_BLOCK_MLP_CONTROL &&
           header->mlp_hvx_contexts > 1U) ||
          (header->attention_pipeline_mode !=
               QBH_BLOCK_ATTENTION_PIPELINE_CONTROL &&
           header->attention_hvx_contexts > 1U)))) {
        uint32_t pool_worker_count =
            header->variant == QBH_BLOCK_W4F16
                ? header->w4f16_requested_hvx_workers
                : (header->variant == QBH_BLOCK_W4U8
                       ? header->attention_hvx_contexts - 1U
                       : header->mlp_hvx_contexts - 1U);
        if (header->variant == QBH_BLOCK_F16F16 &&
            header->attention_pipeline_mode !=
                QBH_BLOCK_ATTENTION_PIPELINE_CONTROL &&
            header->attention_hvx_contexts - 1U >
                pool_worker_count) {
            pool_worker_count =
                header->attention_hvx_contexts - 1U;
        }
        if (qbh_w4f16_pool_create(
                &w4f16_pool, pool_worker_count) != 0) {
            header->dsp_status = header->variant == QBH_BLOCK_W4F16
                ? QBH_BLOCK_STATUS_W4F16_PIPELINE_FAILED
                : (header->attention_pipeline_mode !=
                       QBH_BLOCK_ATTENTION_PIPELINE_CONTROL
                       ? QBH_BLOCK_STATUS_ATTENTION_POOL_FAILED
                       : QBH_BLOCK_STATUS_MLP_POOL_FAILED);
            if (header->variant == QBH_BLOCK_W4F16) {
                header->w4f16_pool_status = -1;
            }
            header->mlp_pool_status = -1;
            header->attention_pool_status = -1;
            result = AEE_EFAILED;
            goto stop_worker;
        }
        hvx_pool_created = 1;
        if (header->variant == QBH_BLOCK_W4F16) {
            header->w4f16_hvx_workers_created =
                w4f16_pool.created_workers;
            header->w4f16_hvx_workers_locked =
                w4f16_pool.created_workers;
        }
        if (header->mlp_mode != QBH_BLOCK_MLP_CONTROL) {
            header->mlp_hvx_workers_created =
                header->mlp_hvx_contexts - 1U;
            header->mlp_hvx_workers_locked =
                header->mlp_hvx_contexts - 1U;
        }
        if (header->attention_pipeline_mode !=
            QBH_BLOCK_ATTENTION_PIPELINE_CONTROL) {
            header->attention_hvx_workers_created =
                header->attention_hvx_contexts - 1U;
            header->attention_hvx_workers_locked =
                header->attention_hvx_contexts - 1U;
        }
    }

    header->qtimer_start = HAP_perf_get_qtimer_count();
    if (header->attribution_enabled != 0U) {
        header->runtime_setup_ticks =
            header->qtimer_start - invocation_start;
    }
    header->scan_logical_m_observed = header->logical_m;
    header->scan_physical_chunk_count = qbh_scan_physical_chunks(header);
    header->scan_total_kv_length =
        header->initial_kv_length + header->logical_m;
    header->scan_padded_kv_length = qbh_align_up(
        header->scan_total_kv_length, QBH_HMX_INPUT_CHANNELS);
    header->scan_useful_query_rows = header->logical_m;
    header->scan_physical_query_rows =
        header->scan_physical_chunk_count * QBH_BLOCK_M;
    if (qbh_slice_enabled(header)) {
        struct qbh_decode_session_state *state =
            (struct qbh_decode_session_state *)(
                shared + header->replay_session_offset);
        const uint32_t generation_enabled =
            qbh_generation_enabled(header->generation_mode);
        const uint32_t generation_step = state->completed_step_count;
        const uint32_t physical_tensor_bytes =
            QBH_BLOCK_M * QBH_BLOCK_HIDDEN *
            (header->variant == QBH_BLOCK_W4U8 ? 1U : 2U);
        if (header->repeat_count != 1U ||
            header->scan_physical_chunk_count != 1U) {
            header->dsp_status = QBH_BLOCK_STATUS_BAD_HEADER;
            result = AEE_EBADPARM;
            goto stop_worker;
        }
        if (generation_enabled != 0U &&
            qbh_stage_generation_embedding(
                header, shared, &buffers) != 0) {
            header->dsp_status = QBH_BLOCK_STATUS_EMBEDDING_FAILED;
            result = AEE_EFAILED;
            goto stop_worker;
        }
        for (uint32_t slice_index = 0U;
             slice_index < QBH_VERTICAL_SLICE_LAYER_COUNT;
             ++slice_index) {
            const uint32_t layer_index =
                QBH_VERTICAL_SLICE_FIRST_LAYER + slice_index;
            struct qbh_decode_layer_state *layer =
                &state->layers[layer_index];
            struct qbh_block_slice_layer_profile *profile =
                &header->slice_profiles[slice_index];
            uint64_t layer_start = HAP_perf_get_qtimer_count();
            uint64_t metadata_before = header->metadata_stage_ticks;
            uint64_t input_before = header->input_stage_ticks;
            uint64_t input_norm_before = header->input_norm_ticks;
            uint64_t qkv_before = header->qkv_projection_ticks;
            uint64_t qk_norm_before = header->qk_norm_rope_ticks;
            uint64_t attention_before = header->attention_ticks;
            uint64_t o_before = header->o_projection_ticks;
            uint64_t post_residual_before =
                header->post_attention_residual_ticks;
            uint64_t post_norm_before =
                header->post_attention_norm_ticks;
            uint64_t gate_up_before = header->gate_up_ticks;
            uint64_t activation_before = header->activation_ticks;
            uint64_t down_before = header->down_ticks;
            uint64_t final_before = header->final_residual_ticks;
            uint64_t weight_bytes_before = header->weight_ddr_read_bytes;
            uint64_t cache_read_before = header->scan_cache_ddr_read_bytes;
            uint64_t cache_write_before =
                header->scan_cache_ddr_write_bytes;
            uint64_t cache_pack_before =
                header->scan_cache_pack_ticks;
            uint64_t cache_append_dma_before =
                header->scan_cache_append_ticks;
            uint64_t block_orchestration_ticks = 0U;
            uint64_t layer_bookkeeping_ticks;
            int block_status;

            memset(profile, 0, sizeof(*profile));
            profile->layer_index = layer_index;
            profile->cache_valid_before = layer->valid_length;
            qbh_bind_slice_layer(header, slice_index);
            header->initial_kv_length = layer->valid_length;
            layer_bookkeeping_ticks =
                HAP_perf_get_qtimer_count() - layer_start;
            if (qbh_stage_metadata(header, shared, &buffers) != 0 ||
                qbh_scan_stage_rope_chunk(
                    header, shared, &buffers, 0U) != 0) {
                profile->status = QBH_BLOCK_STATUS_METADATA_DMA_FAILED;
                header->dsp_status =
                    QBH_BLOCK_STATUS_METADATA_DMA_FAILED;
                result = AEE_EFAILED;
                goto stop_worker;
            }
            {
                const uint64_t block_start =
                    HAP_perf_get_qtimer_count();
                uint64_t block_named_ticks;
                uint64_t block_end;
                block_status = qbh_run_one_block(
                    header, shared, &buffers, &worker,
                    hvx_pool_created != 0 ? &w4f16_pool : NULL,
                    header->input_offset, header->logical_m,
                    layer->valid_length,
                    generation_enabled != 0U || slice_index != 0U);
                block_end = HAP_perf_get_qtimer_count();
                block_named_ticks =
                    (header->input_stage_ticks - input_before) +
                    (header->input_norm_ticks - input_norm_before) +
                    (header->qkv_projection_ticks - qkv_before) +
                    (header->qk_norm_rope_ticks - qk_norm_before) +
                    (header->attention_ticks - attention_before) +
                    (header->o_projection_ticks - o_before) +
                    (header->post_attention_residual_ticks -
                        post_residual_before) +
                    (header->post_attention_norm_ticks - post_norm_before) +
                    (header->gate_up_ticks - gate_up_before) +
                    (header->activation_ticks - activation_before) +
                    (header->down_ticks - down_before) +
                    (header->final_residual_ticks - final_before) +
                    (header->scan_cache_pack_ticks - cache_pack_before) +
                    (header->scan_cache_append_ticks -
                        cache_append_dma_before);
                if (block_end - block_start > block_named_ticks) {
                    block_orchestration_ticks =
                        block_end - block_start - block_named_ticks;
                    header->block_orchestration_ticks +=
                        block_orchestration_ticks;
                }
            }
            profile->status = block_status;
            if (block_status != QBH_BLOCK_STATUS_OK) {
                header->dsp_status = block_status;
                result = AEE_EFAILED;
                goto stop_worker;
            }
            if (header->full_stack_stage_mode ==
                QBH_BLOCK_FULL_STACK_HIDDEN_CAPTURE) {
                const uint64_t capture_start =
                    HAP_perf_get_qtimer_count();
                uint8_t *capture_destination = shared +
                    header->full_stack_hidden_capture_offset +
                    slice_index *
                        header->full_stack_hidden_capture_layer_bytes;
                if (qbh_dma_copy(
                        header, capture_destination, buffers.residual,
                        physical_tensor_bytes, 0U) != 0) {
                    header->output_dma_status = -2;
                    header->dsp_status =
                        QBH_BLOCK_STATUS_HIDDEN_CAPTURE_DMA_FAILED;
                    result = AEE_EFAILED;
                    goto stop_worker;
                }
                ++header->full_stack_hidden_capture_layer_count;
                ++header->full_stack_hidden_capture_dma_descriptor_count;
                header->full_stack_hidden_capture_ddr_write_bytes +=
                    physical_tensor_bytes;
                header->full_stack_hidden_capture_ticks +=
                    HAP_perf_get_qtimer_count() - capture_start;
            }
            {
                const uint64_t bookkeeping_start =
                    HAP_perf_get_qtimer_count();
                uint64_t layer_named_ticks;
                uint64_t layer_end;
                layer->valid_length += header->logical_m;
                layer->append_count += header->logical_m;
                ++header->block_invocation_count;
                profile->cache_valid_after = layer->valid_length;
                profile->hidden_ddr_read_bytes =
                    generation_enabled == 0U && slice_index == 0U
                        ? physical_tensor_bytes : 0U;
                profile->metadata_stage_ticks =
                    header->metadata_stage_ticks - metadata_before;
                profile->input_stage_ticks =
                    header->input_stage_ticks - input_before;
                profile->input_norm_ticks =
                    header->input_norm_ticks - input_norm_before;
                profile->qkv_projection_ticks =
                    header->qkv_projection_ticks - qkv_before;
                profile->qk_norm_rope_ticks =
                    header->qk_norm_rope_ticks - qk_norm_before;
                profile->attention_ticks =
                    header->attention_ticks - attention_before;
                profile->o_projection_ticks =
                    header->o_projection_ticks - o_before;
                profile->post_attention_residual_ticks =
                    header->post_attention_residual_ticks -
                    post_residual_before;
                profile->post_attention_norm_ticks =
                    header->post_attention_norm_ticks - post_norm_before;
                profile->gate_up_ticks =
                    header->gate_up_ticks - gate_up_before;
                profile->activation_ticks =
                    header->activation_ticks - activation_before;
                profile->down_ticks = header->down_ticks - down_before;
                profile->final_residual_ticks =
                    header->final_residual_ticks - final_before;
                profile->cache_append_pack_ticks =
                    header->scan_cache_pack_ticks - cache_pack_before;
                profile->cache_append_dma_ticks =
                    header->scan_cache_append_ticks -
                    cache_append_dma_before;
                profile->block_orchestration_ticks =
                    block_orchestration_ticks;
                profile->weight_ddr_read_bytes =
                    header->weight_ddr_read_bytes - weight_bytes_before;
                profile->cache_ddr_read_bytes =
                    header->scan_cache_ddr_read_bytes - cache_read_before;
                profile->cache_ddr_write_bytes =
                    header->scan_cache_ddr_write_bytes - cache_write_before;
                layer_end = HAP_perf_get_qtimer_count();
                layer_bookkeeping_ticks +=
                    layer_end - bookkeeping_start;
                profile->layer_bookkeeping_ticks =
                    layer_bookkeeping_ticks;
                header->layer_bookkeeping_ticks +=
                    layer_bookkeeping_ticks;
                profile->layer_ticks = layer_end - layer_start;
                layer_named_ticks =
                    profile->metadata_stage_ticks +
                    profile->input_stage_ticks +
                    profile->input_norm_ticks +
                    profile->qkv_projection_ticks +
                    profile->qk_norm_rope_ticks +
                    profile->attention_ticks +
                    profile->o_projection_ticks +
                    profile->post_attention_residual_ticks +
                    profile->post_attention_norm_ticks +
                    profile->gate_up_ticks +
                    profile->activation_ticks +
                    profile->down_ticks +
                    profile->final_residual_ticks +
                    profile->cache_append_pack_ticks +
                    profile->cache_append_dma_ticks +
                    profile->block_orchestration_ticks +
                    profile->layer_bookkeeping_ticks;
                if (profile->layer_ticks > layer_named_ticks) {
                    const uint64_t bookkeeping_gap =
                        profile->layer_ticks - layer_named_ticks;
                    profile->layer_bookkeeping_ticks += bookkeeping_gap;
                    header->layer_bookkeeping_ticks += bookkeeping_gap;
                }
            }
        }
        if (generation_enabled != 0U) {
            const int generation_status =
                qbh_generation_w4u8_enabled(header->generation_mode)
                    ? qbh_run_generation_head_w4u8(
                          header, shared, &buffers, &worker,
                          hvx_pool_created != 0 ? &w4f16_pool : NULL,
                          header->logical_m, generation_step)
                    : qbh_run_generation_head_w4f16(
                          header, shared, &buffers, &worker,
                          hvx_pool_created != 0 ? &w4f16_pool : NULL,
                          header->logical_m, generation_step);
            if (generation_status != 0) {
                header->dsp_status = generation_status == -2
                    ? QBH_BLOCK_STATUS_FINAL_NORM_FAILED
                    : QBH_BLOCK_STATUS_LM_HEAD_FAILED;
                result = AEE_EFAILED;
                goto stop_worker;
            }
        } else {
            uint64_t output_start = HAP_perf_get_qtimer_count();
            if (qbh_dma_copy(
                    header, shared + header->output_offset,
                    buffers.residual, physical_tensor_bytes, 0U) != 0) {
                header->output_dma_status = -1;
                header->dsp_status = QBH_BLOCK_STATUS_OUTPUT_DMA_FAILED;
                result = AEE_EFAILED;
                goto stop_worker;
            }
            header->boundary_ddr_write_bytes += physical_tensor_bytes;
            ++header->boundary_dma_descriptor_count;
            header->output_stage_ticks +=
                HAP_perf_get_qtimer_count() - output_start;
            header->slice_profiles[
                QBH_VERTICAL_SLICE_LAYER_COUNT - 1U]
                .hidden_ddr_write_bytes = physical_tensor_bytes;
        }
        state->next_position += header->logical_m;
        ++state->completed_step_count;
    } else {
        if (qbh_stage_metadata(header, shared, &buffers) != 0) {
            header->dsp_status = QBH_BLOCK_STATUS_METADATA_DMA_FAILED;
            result = AEE_EFAILED;
            goto stop_worker;
        }
        for (uint32_t repeat = 0; repeat < header->repeat_count; ++repeat) {
            for (uint32_t chunk = 0U;
                 chunk < header->scan_physical_chunk_count; ++chunk) {
                const uint32_t first_row = chunk * QBH_BLOCK_M;
                const uint32_t remaining = header->logical_m - first_row;
                const uint32_t logical_rows =
                    remaining < QBH_BLOCK_M ? remaining : QBH_BLOCK_M;
                const uint32_t element_bytes =
                    header->variant == QBH_BLOCK_W4U8 ? 1U : 2U;
                const uint32_t physical_tensor_bytes =
                    QBH_BLOCK_M * QBH_BLOCK_HIDDEN * element_bytes;
                const uint32_t input_offset =
                    header->input_offset + chunk * physical_tensor_bytes;
                int block_status;

                if (qbh_scan_stage_rope_chunk(
                        header, shared, &buffers, chunk) != 0) {
                    header->dsp_status =
                        QBH_BLOCK_STATUS_METADATA_DMA_FAILED;
                    result = AEE_EFAILED;
                    goto stop_worker;
                }
                block_status = qbh_run_one_block(
                    header, shared, &buffers, &worker,
                    hvx_pool_created != 0 ? &w4f16_pool : NULL,
                    input_offset, logical_rows,
                    header->initial_kv_length + first_row, 0U);
                if (block_status != QBH_BLOCK_STATUS_OK) {
                    header->dsp_status = block_status;
                    result = AEE_EFAILED;
                    goto stop_worker;
                }
                if (header->scan_mode != QBH_BLOCK_SCAN_DISABLED &&
                    repeat + 1U == header->repeat_count) {
                    uint64_t output_start = HAP_perf_get_qtimer_count();
                    if (qbh_dma_copy(
                            header,
                            shared + header->output_offset +
                                chunk * physical_tensor_bytes,
                            buffers.residual, physical_tensor_bytes,
                            0U) != 0) {
                        header->output_dma_status = -1;
                        header->dsp_status =
                            QBH_BLOCK_STATUS_OUTPUT_DMA_FAILED;
                        result = AEE_EFAILED;
                        goto stop_worker;
                    }
                    header->boundary_ddr_write_bytes +=
                        physical_tensor_bytes;
                    ++header->boundary_dma_descriptor_count;
                    header->output_stage_ticks +=
                        HAP_perf_get_qtimer_count() - output_start;
                }
            }
            ++header->block_invocation_count;
        }
    }
    if (!qbh_slice_enabled(header) &&
        header->scan_mode == QBH_BLOCK_SCAN_DISABLED) {
        uint64_t output_start = HAP_perf_get_qtimer_count();
        if (qbh_dma_copy(header, shared + header->output_offset,
                         buffers.residual, header->output_bytes, 0U) != 0) {
            header->output_dma_status = -1;
            header->dsp_status = QBH_BLOCK_STATUS_OUTPUT_DMA_FAILED;
            result = AEE_EFAILED;
            goto stop_worker;
        }
        header->boundary_ddr_write_bytes += header->output_bytes;
        ++header->boundary_dma_descriptor_count;
        {
            uint64_t output_end = HAP_perf_get_qtimer_count();
            header->output_stage_ticks += output_end - output_start;
            if (header->attribution_enabled != 0U) {
                teardown_start = output_end;
            }
        }
    } else if (header->attribution_enabled != 0U) {
        teardown_start = HAP_perf_get_qtimer_count();
    }
    header->dsp_status = QBH_BLOCK_STATUS_OK;
    if (header->replay_mode == QBH_BLOCK_REPLAY_CONTINUOUS &&
        !qbh_slice_enabled(header)) {
        struct qbh_decode_session_state *state =
            (struct qbh_decode_session_state *)(
                shared + header->replay_session_offset);
        struct qbh_decode_layer_state *layer =
            &state->layers[state->active_layer];
        layer->valid_length += header->logical_m;
        layer->append_count += header->logical_m;
        state->next_position += header->logical_m;
        ++state->completed_step_count;
    }
    result = AEE_SUCCESS;

stop_worker:
    if (hvx_pool_created != 0) {
        int pool_status = qbh_w4f16_pool_destroy(&w4f16_pool);
        if (header->variant == QBH_BLOCK_W4F16) {
            header->w4f16_pool_status = pool_status;
        }
        if (header->mlp_mode != QBH_BLOCK_MLP_CONTROL) {
            header->mlp_pool_status = pool_status;
        }
        if (header->attention_pipeline_mode !=
            QBH_BLOCK_ATTENTION_PIPELINE_CONTROL) {
            header->attention_pool_status = pool_status;
        }
        for (uint32_t worker_index = 0;
             worker_index < w4f16_pool.created_workers;
             ++worker_index) {
            header->w4f16_expand_work_ticks +=
                w4f16_pool.jobs[worker_index].expand_ticks;
            header->w4f16_expand_region_count +=
                w4f16_pool.jobs[worker_index].expand_count;
            header->mlp_silu_worker_work_ticks +=
                w4f16_pool.jobs[worker_index].silu_ticks;
            header->mlp_silu_chunk_count +=
                w4f16_pool.jobs[worker_index].silu_chunk_count;
            header->mlp_stream_worker_work_ticks +=
                w4f16_pool.jobs[worker_index].stream_ticks;
            header->mlp_stream_ready_wait_ticks +=
                w4f16_pool.jobs[worker_index].stream_ready_wait_ticks;
            header->mlp_stream_group_count +=
                w4f16_pool.jobs[worker_index].stream_group_count;
            header->attention_qk_norm_worker_work_ticks +=
                w4f16_pool.jobs[worker_index].attention_qk_norm_ticks;
            if ((header->crouton_boundary_mode &
                 QBH_BLOCK_CROUTON_BOUNDARY_QKV) != 0U) {
                header->crouton_qkv_transform_ticks +=
                    w4f16_pool.jobs[worker_index].attention_qk_norm_ticks;
            }
            header->attention_qk_norm_task_count +=
                w4f16_pool.jobs[worker_index].attention_qk_norm_task_count;
            header->fp16_qk_norm_pair_task_count +=
                w4f16_pool.jobs[worker_index].fp16_qk_norm_pair_task_count;
            header->attention_softmax_worker_work_ticks +=
                w4f16_pool.jobs[worker_index].attention_softmax_ticks;
            header->attention_softmax_task_count +=
                w4f16_pool.jobs[worker_index].attention_softmax_task_count;
            header->attention_gqa_worker_work_ticks +=
                w4f16_pool.jobs[worker_index].attention_gqa_ticks;
            header->attention_gqa_group_count +=
                w4f16_pool.jobs[worker_index].attention_gqa_group_count;
            header->attention_gqa_hmx_wait_ticks +=
                w4f16_pool.jobs[worker_index].attention_gqa_hmx_wait_ticks;
            header->attention_gqa_queue_wait_ticks +=
                w4f16_pool.jobs[worker_index].attention_gqa_queue_wait_ticks;
            header->crouton_av_o_copy_ticks +=
                w4f16_pool.jobs[worker_index].attention_av_o_copy_ticks;
        }
        if (pool_status != 0 && result == AEE_SUCCESS) {
            header->dsp_status = header->variant == QBH_BLOCK_W4F16
                ? QBH_BLOCK_STATUS_W4F16_PIPELINE_FAILED
                : (header->attention_pipeline_mode !=
                       QBH_BLOCK_ATTENTION_PIPELINE_CONTROL
                       ? QBH_BLOCK_STATUS_ATTENTION_POOL_FAILED
                       : QBH_BLOCK_STATUS_MLP_POOL_FAILED);
            result = AEE_EFAILED;
        }
        hvx_pool_created = 0;
    }
    if (main_hvx_locked != 0) {
        if (qurt_hvx_unlock() != AEE_SUCCESS && result == AEE_SUCCESS) {
            header->dsp_status = QBH_BLOCK_STATUS_HMX_WORKER_FAILED;
            result = AEE_EFAILED;
        }
        main_hvx_locked = 0;
    }
    if (thread_created != 0 && thread_joined == 0) {
        worker.stop = 1U;
        asm volatile("barrier" ::: "memory");
        (void)qurt_sem_up(&worker.command_ready);
        (void)qurt_thread_join(thread, &thread_exit_status);
        thread_joined = 1;
    }
    header->hmx_unlock_status = worker.unlock_status;
    header->hmx_worker_status = thread_exit_status;
    header->hmx_compute_ticks = worker.compute_ticks;
    header->hmx_ready_wait_ticks = worker.ready_wait_ticks;
    header->qtimer_end = HAP_perf_get_qtimer_count();
    header->total_ticks = header->qtimer_end - header->qtimer_start;
    if (header->attribution_enabled != 0U && invocation_start != 0U &&
        teardown_start != 0U) {
        uint64_t named_ticks;
        header->runtime_teardown_ticks =
            header->qtimer_end - teardown_start;
        header->invocation_ticks =
            header->qtimer_end - invocation_start;
        named_ticks = header->runtime_setup_ticks +
                      header->generation_embedding_ticks +
                      header->metadata_stage_ticks +
                      header->input_stage_ticks +
                      header->input_norm_ticks +
                      header->qkv_projection_ticks +
                      header->qk_norm_rope_ticks +
                      header->attention_ticks +
                      header->o_projection_ticks +
                      header->post_attention_residual_ticks +
                      header->post_attention_norm_ticks +
                      header->gate_up_ticks +
                      header->activation_ticks +
                      header->down_ticks +
                      header->final_residual_ticks +
                      header->output_stage_ticks +
                      header->scan_cache_pack_ticks +
                      header->scan_cache_append_ticks +
                      header->block_orchestration_ticks +
                      header->layer_bookkeeping_ticks +
                      header->generation_lm_head_ticks +
                      header->runtime_teardown_ticks;
        if (header->invocation_ticks > named_ticks) {
            header->stage_boundary_ticks =
                header->invocation_ticks - named_ticks;
            named_ticks += header->stage_boundary_ticks;
        }
        header->ledger_named_ticks = named_ticks;
        if (header->invocation_ticks > named_ticks) {
            header->ledger_unattributed_ticks =
                header->invocation_ticks - named_ticks;
        }
    }

destroy_semaphores:
    qurt_sem_destroy(&worker.worker_started);
    qurt_sem_destroy(&worker.command_done);
    qurt_sem_destroy(&worker.command_ready);

publish:
    if (header != NULL) {
        int flush_status = qurt_mem_cache_clean(
            (qurt_addr_t)header, (qurt_size_t)sizeof(*header),
            QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
        if (flush_status == 0 &&
            header->replay_mode == QBH_BLOCK_REPLAY_CONTINUOUS &&
            header->replay_session_bytes ==
                sizeof(struct qbh_decode_session_state) &&
            header->replay_session_offset >= sizeof(*header) &&
            header->replay_session_offset <= shared_bytes &&
            header->replay_session_bytes <=
                shared_bytes - header->replay_session_offset) {
            flush_status = qurt_mem_cache_clean(
                (qurt_addr_t)(shared + header->replay_session_offset),
                (qurt_size_t)header->replay_session_bytes,
                QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
        }
        if (flush_status != 0 && result == AEE_SUCCESS) {
            result = AEE_EFAILED;
        }
    }
    (void)HAP_mmap_put(shared_fd);
    return result;
}
