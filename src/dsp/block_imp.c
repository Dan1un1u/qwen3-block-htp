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
#include "hmx_fp16.h"
#include "hmx_u8s8_projection.h"
#include "hvx_fp16_ops.h"
#include "qbh_user_dma.h"
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
};

#define QBH_BLOCK_W4F16_MIN_REGION_TILES UINT32_C(8)
#define QBH_BLOCK_W4F16_HMX_BATCH_N_TILES UINT32_C(2)
#define QBH_BLOCK_W4F16_MAX_REGIONS \
    (QBH_BLOCK_MAX_K / 32U * QBH_BLOCK_W4F16_HMX_BATCH_N_TILES / \
     QBH_BLOCK_W4F16_MIN_REGION_TILES)
#define QBH_BLOCK_W4F16_HVX_WORKERS UINT32_C(4)
#define QBH_BLOCK_W4F16_HVX_STACK_BYTES UINT32_C(8192)
#define QBH_BLOCK_W4F16_DMA_BATCH_N_TILES UINT32_C(4)
#define QBH_BLOCK_F16F16_BATCH_N_TILES UINT32_C(2)
#define QBH_BLOCK_DMA_DESCRIPTOR_TIMEOUT_TICKS UINT64_C(1920000)
#define QBH_BLOCK_HVX_F16_LANES UINT32_C(64)
#define QBH_BLOCK_MLP_STREAM_CHANNELS UINT32_C(64)
#define QBH_BLOCK_MLP_STREAM_GROUPS \
    (QBH_BLOCK_INTERMEDIATE / QBH_BLOCK_MLP_STREAM_CHANNELS)

enum qbh_block_hvx_pool_job_kind {
    QBH_BLOCK_HVX_POOL_NONE = 0,
    QBH_BLOCK_HVX_POOL_W4_EXPAND = 1,
    QBH_BLOCK_HVX_POOL_SILU = 2,
    QBH_BLOCK_HVX_POOL_MLP_STREAM = 3,
    QBH_BLOCK_HVX_POOL_QK_NORM_ROPE = 4,
    QBH_BLOCK_HVX_POOL_SOFTMAX = 5,
    QBH_BLOCK_HVX_POOL_GQA_ATTENTION = 6,
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
    uint64_t attention_qk_norm_ticks;
    uint32_t attention_softmax_task_count;
    uint64_t attention_softmax_ticks;
    uint32_t attention_gqa_group_count;
    uint64_t attention_gqa_ticks;
    uint64_t attention_gqa_hmx_wait_ticks;
    uint64_t attention_gqa_queue_wait_ticks;
};

struct qbh_block_w4f16_pool {
    qurt_sem_t command_ready[QBH_BLOCK_W4F16_HVX_WORKERS];
    qurt_sem_t command_done[QBH_BLOCK_W4F16_HVX_WORKERS];
    qurt_sem_t worker_started[QBH_BLOCK_W4F16_HVX_WORKERS];
    qurt_thread_t threads[QBH_BLOCK_W4F16_HVX_WORKERS];
    struct qbh_block_w4f16_job jobs[QBH_BLOCK_W4F16_HVX_WORKERS];
    volatile uint32_t stop;
    volatile uint32_t next_region;
    const uint8_t *compressed_weight;
    const float *channel_scale;
    uint8_t *expanded_weight;
    volatile uint32_t *ready_generations;
    uint32_t expected_generation;
    uint32_t region_count;
    uint32_t region_tiles;
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
    uint32_t mlp_stream_generation;
    uint32_t mlp_stream_first_worker;
    uint32_t mlp_stream_worker_count;
    __fp16 *attention_q;
    __fp16 *attention_k;
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
    volatile uint32_t attention_qk_ready[
        QBH_BLOCK_HEADS + QBH_BLOCK_KV_HEADS];
    uint32_t attention_qk_generation;
    volatile uint32_t attention_qk_stream_abort;
    uint32_t attention_qk_streaming;
};

struct qbh_block_w4f16_cross_prefetch {
    struct qbh_dma_aligned_desc_1d descriptor;
    const struct qbh_block_projection_desc *target;
    uint64_t start_ticks;
    uint32_t active;
};

static uint8_t qbh_block_hmx_stack[QBH_BLOCK_HMX_STACK_BYTES]
    __attribute__((aligned(128)));
static uint8_t qbh_block_w4f16_hvx_stacks
    [QBH_BLOCK_W4F16_HVX_WORKERS][QBH_BLOCK_W4F16_HVX_STACK_BYTES]
    __attribute__((aligned(128)));

static int qbh_attention_parallel_qk_norm_enabled(uint32_t mode);
static int qbh_attention_parallel_softmax_enabled(uint32_t mode);
static int qbh_attention_gqa_enabled(uint32_t mode);
static void qbh_attention_gqa_pool_run_tasks(
    struct qbh_block_w4f16_pool *pool,
    struct qbh_block_w4f16_job *job);
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

static int qbh_plan_buffers(uint8_t *vtcm, uint32_t vtcm_bytes,
                            uint32_t variant,
                            uint32_t f16f16_projection_mode,
                            struct qbh_block_buffers *buffers,
                            uint32_t *peak_bytes) {
    struct qbh_block_arena arena = {vtcm, vtcm_bytes, 0U, 0U};
    uint32_t element_bytes = variant == QBH_BLOCK_W4U8 ? 1U : 2U;
    uint32_t hidden_bytes = QBH_BLOCK_M * QBH_BLOCK_HIDDEN * element_bytes;
    uint32_t intermediate_bytes =
        QBH_BLOCK_M * QBH_BLOCK_INTERMEDIATE * element_bytes;
    uint32_t compressed_batch_factor =
        variant == QBH_BLOCK_W4F16
            ? QBH_BLOCK_W4F16_DMA_BATCH_N_TILES : 1U;
    uint32_t expanded_batch_factor =
        variant == QBH_BLOCK_W4F16
            ? QBH_BLOCK_W4F16_HMX_BATCH_N_TILES
            : (variant == QBH_BLOCK_F16F16 &&
               f16f16_projection_mode ==
                   QBH_BLOCK_F16F16_PROJECTION_BATCH2
                   ? QBH_BLOCK_F16F16_BATCH_N_TILES
                   : 1U);
    uint32_t scale_batch_factor =
        variant == QBH_BLOCK_W4F16
            ? 4U : 1U;

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
    buffers->k = qbh_arena_alloc(
        &arena, QBH_BLOCK_M * QBH_BLOCK_KV_HIDDEN * sizeof(uint16_t));
    buffers->v = qbh_arena_alloc(
        &arena, QBH_BLOCK_M * QBH_BLOCK_KV_HIDDEN * sizeof(uint16_t));
    buffers->scores = qbh_arena_alloc(
        &arena, QBH_BLOCK_SCORE_ELEMENTS * sizeof(uint16_t));
    buffers->probability = qbh_arena_alloc(
        &arena, QBH_BLOCK_SCORE_ELEMENTS * sizeof(uint16_t));
    buffers->attention_concat = qbh_arena_alloc(
        &arena, QBH_BLOCK_M * QBH_BLOCK_HIDDEN * sizeof(uint16_t));
    buffers->attention_projection = qbh_arena_alloc(&arena, hidden_bytes);
    buffers->gate = qbh_arena_alloc(&arena, intermediate_bytes);
    buffers->up = qbh_arena_alloc(&arena, intermediate_bytes);
    buffers->middle = qbh_arena_alloc(&arena, intermediate_bytes);
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
    buffers->expanded_weight = qbh_arena_alloc_aligned(
        &arena, QBH_BLOCK_MAX_K * QBH_HMX_OUTPUT_CHANNELS *
                    sizeof(uint16_t) * expanded_batch_factor,
        QBH_HMX_FP16_TILE_BYTES);
    buffers->expanded_weight_alt = qbh_arena_alloc_aligned(
        &arena, QBH_BLOCK_MAX_K * QBH_HMX_OUTPUT_CHANNELS *
                    sizeof(uint16_t) * expanded_batch_factor,
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
    if (variant == QBH_BLOCK_W4F16) {
        buffers->projection_scales = qbh_arena_alloc_aligned(
            &arena, QBH_BLOCK_PROJECTION_SCALE_BYTES,
            QBH_HMX_FP16_SCALE_BYTES);
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
        (variant == QBH_BLOCK_W4F16 &&
         buffers->projection_scales == NULL)) {
        return -1;
    }
    if ((uintptr_t)buffers->attention_projection -
            (uintptr_t)buffers->q <
            QBH_BLOCK_M * QBH_BLOCK_INTERMEDIATE * sizeof(uint16_t) ||
        ((uintptr_t)buffers->q &
         (QBH_HMX_FP16_TILE_BYTES - 1U)) != 0U ||
        ((uintptr_t)buffers->hmx_activation &
         (QBH_HMX_FP16_TILE_BYTES - 1U)) != 0U ||
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
         ((uintptr_t)buffers->projection_scales &
          (QBH_HMX_FP16_SCALE_BYTES - 1U)) != 0U)) {
        return -1;
    }
    *peak_bytes = arena.peak;
    return 0;
}

static int qbh_range_valid(uint32_t offset, uint32_t bytes,
                           uint32_t shared_bytes) {
    return offset >= sizeof(struct qbh_block_header) &&
           offset <= shared_bytes && bytes <= shared_bytes - offset;
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
        header->repeat_count == 0U || header->repeat_count > 100U) {
        return 0;
    }
    if ((header->common_ops_mask &
         ~((uint32_t)QBH_BLOCK_COMMON_OPS_HVX_FP16)) != 0U ||
        (header->variant == QBH_BLOCK_W4U8 &&
         header->common_ops_mask != QBH_BLOCK_COMMON_OPS_SCALAR) ||
        header->attribution_enabled > 1U ||
        header->numerical_audit_enabled > 1U ||
        header->residual_mode >
            QBH_BLOCK_RESIDUAL_HVX_FUSED_POST_NORM ||
        header->f16f16_projection_mode >
            QBH_BLOCK_F16F16_PROJECTION_BATCH2 ||
        header->w4f16_pipeline_mode >
            QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_GATE4_CROSS_PREFETCH ||
        (header->attention_pack_mode &
         ~((uint32_t)QBH_BLOCK_ATTENTION_PACK_HVX)) != 0U ||
        header->attention_pipeline_mode >
            QBH_BLOCK_ATTENTION_PIPELINE_GQA_QKV_OVERLAP ||
        header->attention_hvx_contexts == 0U ||
        header->attention_hvx_contexts >
            QBH_BLOCK_W4F16_HVX_WORKERS ||
        (header->attention_pipeline_mode ==
             QBH_BLOCK_ATTENTION_PIPELINE_CONTROL &&
         header->attention_hvx_contexts != 1U) ||
        (header->attention_pipeline_mode !=
             QBH_BLOCK_ATTENTION_PIPELINE_CONTROL &&
         (header->variant == QBH_BLOCK_W4U8 ||
          header->attention_hvx_contexts != 4U)) ||
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
        (header->attention_pipeline_mode ==
             QBH_BLOCK_ATTENTION_PIPELINE_GQA_QKV_OVERLAP &&
         header->variant == QBH_BLOCK_W4F16 &&
         header->w4f16_requested_hvx_workers != 3U) ||
        header->mlp_mode > QBH_BLOCK_MLP_STREAMING ||
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
         (header->variant == QBH_BLOCK_W4U8 ||
          (header->common_ops_mask & QBH_BLOCK_COMMON_OP_SILU) == 0U)) ||
        (header->mlp_mode == QBH_BLOCK_MLP_STREAMING &&
         (header->mlp_hvx_contexts != 4U ||
          (header->variant == QBH_BLOCK_F16F16 &&
           header->f16f16_projection_mode !=
               QBH_BLOCK_F16F16_PROJECTION_BATCH2) ||
          (header->variant == QBH_BLOCK_W4F16 &&
           header->w4f16_pipeline_mode !=
               QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_CROSS_PREFETCH &&
           header->w4f16_pipeline_mode !=
               QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_GATE16_CROSS_PREFETCH &&
           header->w4f16_pipeline_mode !=
               QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_GATE8_CROSS_PREFETCH &&
           header->w4f16_pipeline_mode !=
               QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_GATE4_CROSS_PREFETCH))) ||
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
              QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_GATE4_CROSS_PREFETCH) &&
         header->w4f16_requested_hvx_workers != 3U) ||
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
              QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_GATE4_CROSS_PREFETCH) &&
         header->w4f16_region_tiles != 32U) ||
        (header->variant == QBH_BLOCK_W4U8 &&
         (header->residual_mode != QBH_BLOCK_RESIDUAL_SCALAR ||
          header->attention_pack_mode !=
              QBH_BLOCK_ATTENTION_PACK_CONTROL ||
          header->attention_pipeline_mode !=
              QBH_BLOCK_ATTENTION_PIPELINE_CONTROL ||
          header->mlp_mode != QBH_BLOCK_MLP_CONTROL))) {
        return 0;
    }
    if (header->w4f16_requested_hvx_workers == 0U ||
        header->w4f16_requested_hvx_workers >
            QBH_BLOCK_W4F16_HVX_WORKERS - 1U ||
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
    if (header->input_bytes !=
            QBH_BLOCK_M * QBH_BLOCK_HIDDEN * element_bytes ||
        header->output_bytes != header->input_bytes ||
        header->reference_bytes != header->output_bytes ||
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
            QBH_BLOCK_M * QBH_BLOCK_HEAD_DIM * sizeof(uint16_t) ||
        header->rope_sin_bytes != header->rope_cos_bytes) {
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
               QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_GATE4_CROSS_PREFETCH;
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
    first_batch_tiles =
        n_tiles < QBH_BLOCK_W4F16_DMA_BATCH_N_TILES
            ? n_tiles : QBH_BLOCK_W4F16_DMA_BATCH_N_TILES;
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
                   worker->n_tiles <= 2U) {
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
                   worker->m_tiles == 1U && worker->n_tiles == 1U) {
            qbh_hmx_begin_u8s8_output(
                (const uint32_t *)worker->scale_or_bias);
            (void)qbh_hmx_accumulate_u8s8_projection(
                (const uint8_t *)worker->activation,
                (const int8_t *)worker->weight, worker->k_tiles);
            qbh_hmx_store_u8_output((uint8_t *)worker->output);
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
        uint32_t first_channel;
        uint64_t wait_start;
        uint64_t work_start;

        if (group >= QBH_BLOCK_MLP_STREAM_GROUPS) {
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

        first_channel = group * QBH_BLOCK_MLP_STREAM_CHANNELS;
        work_start = HAP_perf_get_qtimer_count();
        qbh_hvx_silu_multiply_f16_channel64(
            pool->mlp_gate, pool->mlp_up, pool->mlp_middle,
            QBH_BLOCK_M, QBH_BLOCK_INTERMEDIATE, first_channel);
        qbh_pack_fp16_activation_channel64(
            pool->mlp_middle, QBH_BLOCK_INTERMEDIATE,
            first_channel, pool->mlp_hmx_activation);
        job->stream_ticks +=
            HAP_perf_get_qtimer_count() - work_start;
        ++job->stream_group_count;
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

static void qbh_attention_qk_norm_pool_run_tasks(
    struct qbh_block_w4f16_pool *pool,
    struct qbh_block_w4f16_job *job) {
    for (;;) {
        uint32_t task_offset =
            qbh_atomic_fetch_increment(&pool->next_attention_task);
        uint32_t task;
        uint64_t start;
        if (task_offset >= pool->attention_task_count) {
            break;
        }
        task = pool->attention_task_base + task_offset;
        if (pool->attention_qk_streaming != 0U) {
            while (pool->attention_qk_ready[task] !=
                   pool->attention_qk_generation) {
                if (pool->attention_qk_stream_abort != 0U) {
                    return;
                }
                asm volatile("pause(#8)" : : : "memory");
            }
            asm volatile("barrier" ::: "memory");
        }
        start = HAP_perf_get_qtimer_count();
        if (task < QBH_BLOCK_HEADS) {
            qbh_hvx_qk_norm_rope_f16_head(
                pool->attention_q, QBH_BLOCK_M, QBH_BLOCK_HIDDEN,
                QBH_BLOCK_HEAD_DIM, task,
                pool->attention_q_gamma, pool->attention_rope_cos,
                pool->attention_rope_sin);
        } else {
            uint32_t head = task - QBH_BLOCK_HEADS;
            qbh_hvx_qk_norm_rope_f16_head(
                pool->attention_k, QBH_BLOCK_M,
                QBH_BLOCK_KV_HIDDEN, QBH_BLOCK_HEAD_DIM, head,
                pool->attention_k_gamma, pool->attention_rope_cos,
                pool->attention_rope_sin);
        }
        job->attention_qk_norm_ticks +=
            HAP_perf_get_qtimer_count() - start;
        ++job->attention_qk_norm_task_count;
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
            for (;;) {
                uint32_t region =
                    qbh_atomic_fetch_increment(&pool->next_region);
                uint64_t start;
                if (region >= pool->region_count) {
                    break;
                }
                start = HAP_perf_get_qtimer_count();
                qbh_unpack_w4_to_f16_hvx(
                    pool->compressed_weight +
                        (size_t)region * pool->region_tiles *
                            QBH_W4_PACKED_TILE_BYTES,
                    pool->expanded_weight +
                        (size_t)region * pool->region_tiles *
                            QBH_HMX_FP16_TILE_BYTES,
                    pool->region_tiles);
                job->expand_ticks +=
                    HAP_perf_get_qtimer_count() - start;
                ++job->expand_count;
                pool->ready_generations[region] =
                    pool->expected_generation;
                asm volatile("release(%0):at"
                             :
                             : "r"(&pool->ready_generations[region])
                             : "memory");
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
        worker_count > QBH_BLOCK_W4F16_HVX_WORKERS) {
        return -1;
    }
    memset(pool, 0, sizeof(*pool));
    qurt_mutex_init(&pool->attention_hmx_mutex);
    pool->worker_count = worker_count;
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
    uint32_t region_tiles, uint32_t active_worker_count) {
    pool->compressed_weight = compressed_weight;
    pool->channel_scale = channel_scale;
    pool->expanded_weight = expanded_weight;
    pool->ready_generations = ready_generations;
    pool->expected_generation = expected_generation;
    pool->region_count = region_count;
    pool->region_tiles = region_tiles;
    pool->active_worker_count = active_worker_count;
    pool->next_region = 0U;
    for (uint32_t worker = 0; worker < pool->active_worker_count;
         ++worker) {
        pool->jobs[worker].command_kind =
            QBH_BLOCK_HVX_POOL_W4_EXPAND;
    }
    asm volatile("barrier" ::: "memory");
    for (uint32_t worker = 0; worker < pool->active_worker_count;
         ++worker) {
        (void)qurt_sem_up(&pool->command_ready[worker]);
    }
}

static void qbh_w4f16_pool_wait(
    struct qbh_block_w4f16_pool *pool) {
    for (uint32_t worker = 0; worker < pool->active_worker_count;
         ++worker) {
        qurt_sem_down(&pool->command_done[worker]);
    }
    asm volatile("barrier" ::: "memory");
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
    pool->attention_k = k;
    pool->attention_q_gamma = q_gamma;
    pool->attention_k_gamma = k_gamma;
    pool->attention_rope_cos = cosine;
    pool->attention_rope_sin = sine;
    pool->attention_task_count =
        QBH_BLOCK_HEADS + QBH_BLOCK_KV_HEADS;
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
    asm volatile("barrier" ::: "memory");
    return 0;
}

static int qbh_hvx_pool_qk_norm_rope_start_async(
    struct qbh_block_w4f16_pool *pool,
    __fp16 *q, __fp16 *k, const __fp16 *q_gamma,
    const __fp16 *k_gamma, const __fp16 *cosine,
    const __fp16 *sine, uint32_t task_base,
    uint32_t task_count, uint32_t first_worker,
    uint32_t worker_count) {
    if (pool == NULL || worker_count == 0U ||
        first_worker + worker_count > pool->worker_count ||
        task_base + task_count >
            QBH_BLOCK_HEADS + QBH_BLOCK_KV_HEADS) {
        return -1;
    }
    pool->attention_q = q;
    pool->attention_k = k;
    pool->attention_q_gamma = q_gamma;
    pool->attention_k_gamma = k_gamma;
    pool->attention_rope_cos = cosine;
    pool->attention_rope_sin = sine;
    pool->attention_task_base = task_base;
    pool->attention_task_count = task_count;
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

    if (header->mlp_mode != QBH_BLOCK_MLP_STREAMING) {
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

static int qbh_mlp_stream_pipeline_wait(
    struct qbh_block_header *header,
    struct qbh_block_w4f16_pool *pool, uint32_t abort_pipeline) {
    struct qbh_block_w4f16_job main_job;
    uint64_t wait_start;

    if (header->mlp_mode != QBH_BLOCK_MLP_STREAMING) {
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
        memset(&main_job, 0, sizeof(main_job));
        pool->jobs[0].command_kind =
            QBH_BLOCK_HVX_POOL_MLP_STREAM;
        pool->jobs[1].command_kind =
            QBH_BLOCK_HVX_POOL_MLP_STREAM;
        asm volatile("barrier" ::: "memory");
        (void)qurt_sem_up(&pool->command_ready[0]);
        (void)qurt_sem_up(&pool->command_ready[1]);
        pool->mlp_stream_first_worker = 0U;
        pool->mlp_stream_worker_count = 3U;
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
    uint32_t publish_ready) {
    uint32_t total_regions = k_tiles / region_tiles;
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
            (total_regions + active_worker_count) /
            (active_worker_count + 1U);
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
        active_worker_count);
    main_start = HAP_perf_get_qtimer_count();
    if (publish_ready == 0U) {
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

static __attribute__((noinline)) int qbh_run_f16f16_pipelined_projection(
    struct qbh_block_header *header, const uint8_t *shared,
    const struct qbh_block_projection_desc *desc,
    struct qbh_block_buffers *buffers,
    struct qbh_block_hmx_worker *worker,
    struct qbh_block_w4f16_pool *hvx_pool,
    const void *activation_tiles, void *output) {
    uint32_t k_tiles = desc->k / QBH_HMX_FP16_COLS;
    uint32_t n_tiles = desc->n / QBH_HMX_FP16_COLS;
    uint32_t batch_n_tiles =
        header->f16f16_projection_mode ==
                QBH_BLOCK_F16F16_PROJECTION_BATCH2
            ? QBH_BLOCK_F16F16_BATCH_N_TILES
            : 1U;
    uint32_t weight_bytes_per_tile =
        k_tiles * QBH_HMX_FP16_TILE_BYTES;
    uint8_t *weight_slots[2] = {
        buffers->expanded_weight, buffers->expanded_weight_alt};
    struct qbh_dma_aligned_desc_1d prefetch_descriptor
        __attribute__((aligned(64)));
    int prefetch_active = 0;

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
        int hmx_result;
        int prefetch_result = 0;

        if (group_tiles > batch_n_tiles) {
            group_tiles = batch_n_tiles;
        }
        next_tile = n_tile + group_tiles;

        hmx_start = HAP_perf_get_qtimer_count();
        qbh_hmx_start(
            worker, QBH_BLOCK_HMX_FP16, activation_tiles,
            weight_slots[group_index & 1U], buffers->scale_or_bias,
            buffers->hmx_output, 2U, k_tiles, group_tiles);

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

        if (hmx_result == 0) {
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
             QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_GATE4_CROSS_PREFETCH) &&
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
                QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_GATE4_CROSS_PREFETCH) {
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
    return header->w4f16_pipeline_mode ==
                   QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_GATE4_CROSS_PREFETCH
               ? 4U : QBH_BLOCK_W4F16_HMX_BATCH_N_TILES;
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
            active_workers, 1U);
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
        k_tiles * QBH_BLOCK_W4F16_HMX_BATCH_N_TILES,
        region_tiles, active_workers, 0U);
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
         n_tile += QBH_BLOCK_W4F16_HMX_BATCH_N_TILES) {
        uint32_t group_tiles = n_tiles - n_tile;
        uint32_t group_index =
            n_tile / QBH_BLOCK_W4F16_HMX_BATCH_N_TILES;
        uint32_t current_expanded_slot = group_index & 1U;
        uint32_t next_tile;

        if (group_tiles > QBH_BLOCK_W4F16_HMX_BATCH_N_TILES) {
            group_tiles = QBH_BLOCK_W4F16_HMX_BATCH_N_TILES;
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
            buffers->hmx_output, 2U, k_tiles, group_tiles);
        ++header->w4f16_streamed_command_count;
        if (next_tile < n_tiles) {
            uint32_t next_group_tiles = n_tiles - next_tile;
            uint32_t next_batch =
                next_tile / QBH_BLOCK_W4F16_DMA_BATCH_N_TILES;
            uint32_t next_in_batch =
                next_tile % QBH_BLOCK_W4F16_DMA_BATCH_N_TILES;
            uint32_t next_compressed_slot = next_batch & 1U;
            uint32_t next_expanded_slot =
                (next_tile / QBH_BLOCK_W4F16_HMX_BATCH_N_TILES) & 1U;

            if (next_group_tiles >
                QBH_BLOCK_W4F16_HMX_BATCH_N_TILES) {
                next_group_tiles =
                    QBH_BLOCK_W4F16_HMX_BATCH_N_TILES;
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
                    region_tiles, active_workers, 0U);
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
    first_tile = batch * QBH_BLOCK_W4F16_DMA_BATCH_N_TILES;
    batch_tiles = state->n_tiles - first_tile;
    if (batch_tiles > QBH_BLOCK_W4F16_DMA_BATCH_N_TILES) {
        batch_tiles = QBH_BLOCK_W4F16_DMA_BATCH_N_TILES;
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
        first_tile / QBH_BLOCK_W4F16_DMA_BATCH_N_TILES;
    uint32_t in_batch =
        first_tile % QBH_BLOCK_W4F16_DMA_BATCH_N_TILES;
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
        region_tiles, 2U, 0U);
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

    for (uint32_t tile = 0U;
         tile < state->group_tiles; ++tile) {
        qbh_hmx_fp16_init_channel_scales(
            buffers->scale_or_bias +
                (size_t)tile * QBH_HMX_FP16_SCALE_BYTES,
            state->scales + (size_t)(first_tile + tile) * 32U);
    }
    qbh_hmx_start_fp16_tile_scales(
        worker, activation_tiles,
        state->expanded_slots[expanded_slot],
        buffers->scale_or_bias, buffers->hmx_output, 2U,
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
        QBH_BLOCK_MAX_K * QBH_HMX_OUTPUT_CHANNELS *
        sizeof(uint16_t) * QBH_BLOCK_W4F16_HMX_BATCH_N_TILES;
    uint32_t compressed_batch_bytes;
    uint32_t expanded_group_bytes;
    uint32_t group_count;
    uint32_t group_tiles;
    uint64_t pack_start;
    int result = 0;

    if (pool == NULL || header->variant != QBH_BLOCK_W4F16 ||
        header->mlp_mode != QBH_BLOCK_MLP_STREAMING) {
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
    gate.group_tiles = group_tiles;
    up.group_tiles = group_tiles;
    if (gate.k_tiles != up.k_tiles ||
        gate.n_tiles != up.n_tiles ||
        gate.n_tiles % group_tiles != 0U) {
        return -1;
    }
    gate.compressed_bytes_per_tile =
        gate.k_tiles * QBH_W4_PACKED_TILE_BYTES;
    up.compressed_bytes_per_tile =
        up.k_tiles * QBH_W4_PACKED_TILE_BYTES;
    gate.batch_count =
        (gate.n_tiles + QBH_BLOCK_W4F16_DMA_BATCH_N_TILES - 1U) /
        QBH_BLOCK_W4F16_DMA_BATCH_N_TILES;
    up.batch_count = gate.batch_count;
    compressed_batch_bytes =
        QBH_BLOCK_W4F16_DMA_BATCH_N_TILES *
        gate.compressed_bytes_per_tile;
    expanded_group_bytes =
        group_tiles * gate.k_tiles *
        QBH_HMX_FP16_TILE_BYTES;
    if (2U * compressed_batch_bytes > compressed_capacity ||
        (group_tiles == 4U
             ? expanded_group_bytes > expanded_capacity
             : 2U * expanded_group_bytes > expanded_capacity)) {
        return -1;
    }
    gate.compressed_slots[0] = buffers->compressed_weight;
    gate.compressed_slots[1] =
        buffers->compressed_weight + compressed_batch_bytes;
    up.compressed_slots[0] = buffers->compressed_weight_alt;
    up.compressed_slots[1] =
        buffers->compressed_weight_alt + compressed_batch_bytes;
    gate.expanded_slots[0] = buffers->expanded_weight;
    up.expanded_slots[0] = buffers->expanded_weight_alt;
    if (group_tiles == 4U) {
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
    qbh_w4f16_note_active_workers(header, 2U);
    qbh_w4f16_note_effective_region(
        header, qbh_w4f16_gate_up_region_tiles(header));
    header->w4f16_gate_up_effective_region_tiles =
        qbh_w4f16_gate_up_region_tiles(header);

    pack_start = HAP_perf_get_qtimer_count();
    qbh_pack_fp16_activation(
        (const __fp16 *)buffers->normalized, QBH_BLOCK_HIDDEN,
        QBH_BLOCK_HIDDEN, (__fp16 *)buffers->hmx_activation);
    header->projection_pack_ticks +=
        HAP_perf_get_qtimer_count() - pack_start;

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
        header->mlp_mode == QBH_BLOCK_MLP_STREAMING &&
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

    memset((void *)w4f16_ready, 0, sizeof(w4f16_ready));

    if (activation_tiles_ready != 0U) {
        if (header->variant == QBH_BLOCK_W4U8 ||
            (header->mlp_mode == QBH_BLOCK_MLP_STREAMING &&
             desc == &header->projections[QBH_BLOCK_PROJ_DOWN])) {
            return -1;
        }
    } else if (header->variant == QBH_BLOCK_W4U8) {
        qbh_pack_u8_activation((const uint8_t *)input, desc->k,
                               desc->k, projection_activation);
    } else {
        if (header->mlp_mode == QBH_BLOCK_MLP_STREAMING &&
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
                buffers->scale_or_bias, buffers->hmx_output, 1U,
                k_tiles, 1U);
            header->projection_hmx_wait_ticks +=
                HAP_perf_get_qtimer_count() - phase_start;
            header->hmx_u8s8_tile_pair_count += k_tiles;
            if (result == 0) {
                phase_start = HAP_perf_get_qtimer_count();
                qbh_unpack_u8_output(
                    buffers->hmx_output, (uint8_t *)output, desc->n,
                    n_tile * QBH_HMX_OUTPUT_CHANNELS);
                header->projection_unpack_ticks +=
                    HAP_perf_get_qtimer_count() - phase_start;
                qbh_hvx_pool_qk_norm_rope_publish(
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
                    w4f16_pool->worker_count);
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

        qbh_pack_fp16_weight_rows_hvx(
            (const __fp16 *)buffers->k, QBH_BLOCK_KV_HIDDEN,
            kv_head * QBH_BLOCK_HEAD_DIM,
            QBH_BLOCK_HEAD_DIM, QBH_BLOCK_M, weight);
        for (uint32_t local_head = 0U; local_head < 2U; ++local_head) {
            uint32_t head = first_q_head + local_head;
            qbh_pack_fp16_activation(
                (const __fp16 *)buffers->q +
                    head * QBH_BLOCK_HEAD_DIM,
                QBH_BLOCK_HIDDEN, QBH_BLOCK_HEAD_DIM, activation);
            if (qbh_attention_gqa_submit(
                    pool, job, activation, weight, output, 4U, 2U) != 0) {
                break;
            }
            qbh_unpack_fp16_output(
                output, 2U,
                (__fp16 *)buffers->scores +
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
                    (__fp16 *)buffers->scores +
                        (size_t)head * head_elements,
                    (uint32_t)head_elements);
            }
        }

        for (uint32_t local_head = 0U; local_head < 2U; ++local_head) {
            uint32_t head = first_q_head + local_head;
            qbh_hvx_stable_causal_softmax_f16(
                (__fp16 *)buffers->scores +
                    (size_t)head * head_elements,
                (__fp16 *)buffers->probability +
                    (size_t)head * head_elements,
                1U, QBH_BLOCK_M, QBH_BLOCK_M,
                0.08838834764831845f, NULL);
        }

        qbh_pack_fp16_weight_transposed_hvx(
            (const __fp16 *)buffers->v, QBH_BLOCK_KV_HIDDEN,
            kv_head * QBH_BLOCK_HEAD_DIM, QBH_BLOCK_M,
            QBH_BLOCK_HEAD_DIM, weight);
        for (uint32_t local_head = 0U; local_head < 2U; ++local_head) {
            uint32_t head = first_q_head + local_head;
            qbh_pack_fp16_activation(
                (const __fp16 *)buffers->probability +
                    (size_t)head * head_elements,
                QBH_BLOCK_M, QBH_BLOCK_M, activation);
            if (qbh_attention_gqa_submit(
                    pool, job, activation, weight, output, 2U, 4U) != 0) {
                break;
            }
            qbh_unpack_fp16_output(
                output, 4U, (__fp16 *)buffers->attention_concat,
                QBH_BLOCK_HIDDEN, head * QBH_BLOCK_HEAD_DIM);
        }
        if (pool->attention_gqa_abort != 0U) {
            break;
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
           header->attention_gqa_pipeline_ticks;
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
        header->rope_cos_bytes,
        header->rope_sin_bytes,
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
    if (header->variant == QBH_BLOCK_W4F16) {
        uint8_t *destination = buffers->projection_scales;
        for (uint32_t index = 0;
             index < QBH_BLOCK_PROJECTION_COUNT; ++index) {
            const struct qbh_block_projection_desc *desc =
                &header->projections[index];
            if (qbh_dma_copy(
                    header, destination, shared + desc->scale_offset,
                    desc->scale_bytes, 1U) != 0) {
                return -1;
            }
            header->weight_ddr_read_bytes += desc->scale_bytes;
            ++header->weight_dma_descriptor_count;
            destination += desc->scale_bytes;
        }
    }
    header->metadata_stage_ticks += HAP_perf_get_qtimer_count() - start;
    return 0;
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

static int qbh_run_one_block(struct qbh_block_header *header,
                             const uint8_t *shared,
                             struct qbh_block_buffers *buffers,
                             struct qbh_block_hmx_worker *worker,
                             struct qbh_block_w4f16_pool *w4f16_pool) {
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
    uint32_t qkv_overlap_first_worker =
        header->variant == QBH_BLOCK_W4F16 ? 2U : 0U;
    uint32_t qkv_overlap_worker_count =
        header->variant == QBH_BLOCK_W4F16 ? 1U : 3U;
    int post_attention_norm_fused = 0;

    qbh_hvx_check_reset(&rms_check_metrics);
    qbh_hvx_check_reset(&rope_check_metrics);
    qbh_hvx_check_reset(&softmax_check_metrics);
    qbh_hvx_check_reset(&silu_check_metrics);
    memset(&cross_prefetch, 0, sizeof(cross_prefetch));

    start = HAP_perf_get_qtimer_count();
    if (qbh_dma_copy(header, buffers->residual,
                     shared + header->input_offset,
                     header->input_bytes, 1U) != 0) {
        header->input_dma_status = -1;
        return QBH_BLOCK_STATUS_INPUT_DMA_FAILED;
    }
    header->boundary_ddr_read_bytes += header->input_bytes;
    ++header->boundary_dma_descriptor_count;
    header->input_stage_ticks += HAP_perf_get_qtimer_count() - start;

    start = HAP_perf_get_qtimer_count();
    if (header->variant == QBH_BLOCK_W4U8) {
        qbh_rms_norm_u8(
            buffers->residual,
            &header->qparams[QBH_BLOCK_QP_BLOCK_INPUT],
            (const __fp16 *)buffers->input_norm_weight,
            buffers->normalized,
            &header->qparams[QBH_BLOCK_QP_INPUT_NORM],
            QBH_BLOCK_M, QBH_BLOCK_HIDDEN);
    } else {
        if ((header->common_ops_mask & QBH_BLOCK_COMMON_OP_RMS_NORM) != 0U) {
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
            header, buffers->normalized, hidden_elements,
            QBH_BLOCK_NUMERICAL_INPUT_NORM);
        qbh_attribution_accumulate(
            header, audit_start, &header->input_norm_audit_ticks);
    }
    header->input_norm_ticks += HAP_perf_get_qtimer_count() - start;

    start = HAP_perf_get_qtimer_count();
    if (qkv_overlap_enabled != 0U &&
        qbh_hvx_pool_qk_norm_rope_start_async(
            w4f16_pool, (__fp16 *)buffers->q,
            (__fp16 *)buffers->k,
            (const __fp16 *)buffers->q_norm_weight,
            (const __fp16 *)buffers->k_norm_weight,
            (const __fp16 *)buffers->rope_cos,
            (const __fp16 *)buffers->rope_sin,
            0U, QBH_BLOCK_HEADS + QBH_BLOCK_KV_HEADS,
            qkv_overlap_first_worker,
            qkv_overlap_worker_count) != 0) {
        return QBH_BLOCK_STATUS_QK_NORM_ROPE_FAILED;
    }
    if (qbh_run_projection(
            header, shared, &header->projections[QBH_BLOCK_PROJ_Q],
            buffers, worker, w4f16_pool, buffers->normalized,
            buffers->q, 0U,
            &header->projections[QBH_BLOCK_PROJ_K],
            &cross_prefetch) != 0) {
        if (qkv_overlap_enabled != 0U) {
            qbh_hvx_pool_qk_norm_rope_abort_async(w4f16_pool);
            (void)qbh_hvx_pool_qk_norm_rope_wait_async(
                header, w4f16_pool, qkv_overlap_first_worker,
                qkv_overlap_worker_count);
        }
        return QBH_BLOCK_STATUS_QKV_FAILED;
    }
    if (qbh_run_projection(
            header, shared, &header->projections[QBH_BLOCK_PROJ_K],
            buffers, worker, w4f16_pool, buffers->normalized,
            buffers->k, qkv_overlap_enabled,
            &header->projections[QBH_BLOCK_PROJ_V],
            &cross_prefetch) != 0) {
        if (qkv_overlap_enabled != 0U) {
            qbh_hvx_pool_qk_norm_rope_abort_async(w4f16_pool);
            (void)qbh_hvx_pool_qk_norm_rope_wait_async(
                header, w4f16_pool, qkv_overlap_first_worker,
                qkv_overlap_worker_count);
        }
        return QBH_BLOCK_STATUS_QKV_FAILED;
    }
    if (qbh_run_projection(
            header, shared, &header->projections[QBH_BLOCK_PROJ_V],
            buffers, worker, w4f16_pool, buffers->normalized,
            buffers->v, qkv_overlap_enabled,
            &header->projections[QBH_BLOCK_PROJ_O],
            &cross_prefetch) != 0) {
        return QBH_BLOCK_STATUS_QKV_FAILED;
    }
    if (qkv_overlap_enabled != 0U) {
        if (qbh_hvx_pool_qk_norm_rope_wait_async(
                header, w4f16_pool, qkv_overlap_first_worker,
                qkv_overlap_worker_count) != 0) {
            return QBH_BLOCK_STATUS_QK_NORM_ROPE_FAILED;
        }
        audit_start = qbh_attribution_begin(header);
        qbh_record_f16_nonfinite(
            header, buffers->q, QBH_BLOCK_M * QBH_BLOCK_HIDDEN,
            QBH_BLOCK_NUMERICAL_Q_ROPE);
        qbh_record_f16_nonfinite(
            header, buffers->k,
            QBH_BLOCK_M * QBH_BLOCK_KV_HIDDEN,
            QBH_BLOCK_NUMERICAL_K_ROPE);
        qbh_record_f16_nonfinite(
            header, buffers->v,
            QBH_BLOCK_M * QBH_BLOCK_KV_HIDDEN,
            QBH_BLOCK_NUMERICAL_V);
        qbh_attribution_accumulate(
            header, audit_start, &header->qkv_audit_ticks);
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
    if (header->variant != QBH_BLOCK_W4U8 &&
        qbh_attention_gqa_enabled(
            header->attention_pipeline_mode)) {
        /* GQA performs Q/K normalization and RoPE inside each group. The
         * QKV-overlap mode has already completed the same arithmetic from
         * head-readiness events emitted by the Q/K projections. */
    } else if (header->variant == QBH_BLOCK_W4U8) {
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
            header->attention_pipeline_mode)) {
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

    if (header->attribution_enabled != 0U) {
        attention_attributed_before =
            qbh_attention_attributed_ticks(header);
    }
    start = HAP_perf_get_qtimer_count();
    if (qbh_attention_f16(
            header, buffers, worker, w4f16_pool,
            softmax_check) != 0) {
        return QBH_BLOCK_STATUS_ATTENTION_FAILED;
    }
    if (header->variant == QBH_BLOCK_W4U8) {
        qbh_quantize_f16_buffer(
            (const __fp16 *)buffers->attention_concat,
            buffers->attention_concat, hidden_elements,
            &header->qparams[QBH_BLOCK_QP_ATTENTION_CONCAT]);
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

    start = HAP_perf_get_qtimer_count();
    if (qbh_run_projection(
            header, shared, &header->projections[QBH_BLOCK_PROJ_O],
            buffers, worker, w4f16_pool, buffers->attention_concat,
            buffers->attention_projection, 0U,
            &header->projections[QBH_BLOCK_PROJ_GATE],
            &cross_prefetch) != 0) {
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
        qbh_residual_add_u8(
            buffers->residual,
            &header->qparams[QBH_BLOCK_QP_BLOCK_INPUT],
            buffers->attention_projection,
            &header->qparams[QBH_BLOCK_QP_ATTENTION_PROJECTION],
            buffers->residual,
            &header->qparams[QBH_BLOCK_QP_POST_ATTENTION_RESIDUAL],
            hidden_elements);
    } else {
        if (header->residual_mode ==
            QBH_BLOCK_RESIDUAL_HVX_FUSED_POST_NORM) {
            qbh_hvx_residual_rms_norm_f16(
                (__fp16 *)buffers->residual,
                (const __fp16 *)buffers->attention_projection,
                (const __fp16 *)buffers->post_norm_weight,
                (__fp16 *)buffers->normalized,
                QBH_BLOCK_M, QBH_BLOCK_HIDDEN);
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
    header->post_attention_residual_ticks +=
        HAP_perf_get_qtimer_count() - start;

    start = HAP_perf_get_qtimer_count();
    if (header->variant == QBH_BLOCK_W4U8) {
        qbh_rms_norm_u8(
            buffers->residual,
            &header->qparams[QBH_BLOCK_QP_POST_ATTENTION_RESIDUAL],
            (const __fp16 *)buffers->post_norm_weight,
            buffers->normalized,
            &header->qparams[QBH_BLOCK_QP_POST_ATTENTION_NORM],
            QBH_BLOCK_M, QBH_BLOCK_HIDDEN);
    } else {
        if (post_attention_norm_fused != 0) {
            /* Materialized by the fused residual/RMSNorm first pass. */
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
            header, buffers->normalized, hidden_elements,
            QBH_BLOCK_NUMERICAL_POST_NORM);
        qbh_attribution_accumulate(
            header, audit_start,
            &header->post_attention_norm_audit_ticks);
    }
    header->post_attention_norm_ticks +=
        HAP_perf_get_qtimer_count() - start;

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
    if (header->variant == QBH_BLOCK_W4F16 &&
        header->mlp_mode == QBH_BLOCK_MLP_STREAMING) {
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
                worker, w4f16_pool, buffers->normalized,
                buffers->gate, 0U,
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
                worker, w4f16_pool, buffers->normalized,
                buffers->up, 0U,
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
    if (header->variant != QBH_BLOCK_W4U8) {
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
        audit_start = qbh_attribution_begin(header);
        qbh_record_f16_nonfinite(
            header, buffers->middle, intermediate_elements,
            QBH_BLOCK_NUMERICAL_MIDDLE);
        qbh_attribution_accumulate(
            header, audit_start, &header->activation_audit_ticks);
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

    start = HAP_perf_get_qtimer_count();
    if (header->variant == QBH_BLOCK_W4U8) {
        qbh_residual_add_u8(
            buffers->residual,
            &header->qparams[QBH_BLOCK_QP_POST_ATTENTION_RESIDUAL],
            buffers->down, &header->qparams[QBH_BLOCK_QP_DOWN],
            buffers->residual,
            &header->qparams[QBH_BLOCK_QP_BLOCK_OUTPUT],
            hidden_elements);
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

    if (qbh_plan_buffers(vtcm, vtcm_bytes, header->variant,
                         header->f16f16_projection_mode, &buffers,
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
        (header->variant == QBH_BLOCK_F16F16 &&
         ((header->mlp_mode != QBH_BLOCK_MLP_CONTROL &&
           header->mlp_hvx_contexts > 1U) ||
          (header->attention_pipeline_mode !=
               QBH_BLOCK_ATTENTION_PIPELINE_CONTROL &&
           header->attention_hvx_contexts > 1U)))) {
        uint32_t pool_worker_count =
            header->variant == QBH_BLOCK_W4F16
                ? header->w4f16_requested_hvx_workers
                : header->mlp_hvx_contexts - 1U;
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
    if (qbh_stage_metadata(header, shared, &buffers) != 0) {
        header->dsp_status = QBH_BLOCK_STATUS_METADATA_DMA_FAILED;
        result = AEE_EFAILED;
        goto stop_worker;
    }
    for (uint32_t repeat = 0; repeat < header->repeat_count; ++repeat) {
        int block_status = qbh_run_one_block(
            header, shared, &buffers, &worker,
            hvx_pool_created != 0 ? &w4f16_pool : NULL);
        if (block_status != QBH_BLOCK_STATUS_OK) {
            header->dsp_status = block_status;
            result = AEE_EFAILED;
            goto stop_worker;
        }
        ++header->block_invocation_count;
    }
    {
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
    }
    header->dsp_status = QBH_BLOCK_STATUS_OK;
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
            header->attention_qk_norm_task_count +=
                w4f16_pool.jobs[worker_index].attention_qk_norm_task_count;
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
                      header->runtime_teardown_ticks;
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
        if (flush_status != 0 && result == AEE_SUCCESS) {
            result = AEE_EFAILED;
        }
    }
    (void)HAP_mmap_put(shared_fd);
    return result;
}
