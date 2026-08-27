#include <AEEStdErr.h>
#include <HAP_compute_res.h>
#include <HAP_mem.h>
#include <HAP_perf.h>
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
#include "qbh_user_dma.h"
#include "w4_u8_expand.h"

#define QBH_BLOCK_ALIGNMENT UINT32_C(128)
#define QBH_BLOCK_HMX_STACK_BYTES UINT32_C(16384)
#define QBH_BLOCK_MAX_K QBH_BLOCK_INTERMEDIATE
#define QBH_BLOCK_SCORE_ELEMENTS \
    (QBH_BLOCK_HEADS * QBH_BLOCK_M * QBH_BLOCK_M)
#define QBH_BLOCK_HMX_OUTPUT_MAX_BYTES \
    (2U * 4U * QBH_HMX_FP16_TILE_BYTES)

enum qbh_block_hmx_command_kind {
    QBH_BLOCK_HMX_NONE = 0,
    QBH_BLOCK_HMX_FP16 = 1,
    QBH_BLOCK_HMX_U8S8 = 2,
    QBH_BLOCK_HMX_FP16_STREAMING = 3,
};

#define QBH_BLOCK_W4F16_REGION_TILES UINT32_C(8)
#define QBH_BLOCK_W4F16_MAX_REGIONS \
    (QBH_BLOCK_MAX_K / 32U / QBH_BLOCK_W4F16_REGION_TILES)

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
    uint8_t *expanded_weight;
    uint8_t *hmx_output;
    uint8_t *scale_or_bias;
    uint8_t *channel_scale;
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

static uint8_t qbh_block_hmx_stack[QBH_BLOCK_HMX_STACK_BYTES]
    __attribute__((aligned(128)));

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
                            struct qbh_block_buffers *buffers,
                            uint32_t *peak_bytes) {
    struct qbh_block_arena arena = {vtcm, vtcm_bytes, 0U, 0U};
    uint32_t element_bytes = variant == QBH_BLOCK_W4U8 ? 1U : 2U;
    uint32_t hidden_bytes = QBH_BLOCK_M * QBH_BLOCK_HIDDEN * element_bytes;
    uint32_t intermediate_bytes =
        QBH_BLOCK_M * QBH_BLOCK_INTERMEDIATE * element_bytes;

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
    buffers->q = qbh_arena_alloc(
        &arena, QBH_BLOCK_M * QBH_BLOCK_HIDDEN * sizeof(uint16_t));
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
        &arena, QBH_BLOCK_MAX_K * QBH_HMX_OUTPUT_CHANNELS / 2U);
    buffers->expanded_weight = qbh_arena_alloc_aligned(
        &arena, QBH_BLOCK_MAX_K * QBH_HMX_OUTPUT_CHANNELS *
                    sizeof(uint16_t),
        QBH_HMX_FP16_TILE_BYTES);
    buffers->hmx_output = qbh_arena_alloc_aligned(
        &arena, QBH_BLOCK_HMX_OUTPUT_MAX_BYTES,
        QBH_HMX_FP16_TILE_BYTES);
    buffers->scale_or_bias = qbh_arena_alloc_aligned(
        &arena, QBH_HMX_FP16_SCALE_BYTES,
        QBH_HMX_FP16_SCALE_BYTES);
    buffers->channel_scale = qbh_arena_alloc_aligned(
        &arena, QBH_HMX_FP16_SCALE_BYTES,
        QBH_HMX_FP16_SCALE_BYTES);

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
        buffers->expanded_weight == NULL || buffers->hmx_output == NULL ||
        buffers->scale_or_bias == NULL || buffers->channel_scale == NULL) {
        return -1;
    }
    if (((uintptr_t)buffers->hmx_activation &
         (QBH_HMX_FP16_TILE_BYTES - 1U)) != 0U ||
        ((uintptr_t)buffers->expanded_weight &
         (QBH_HMX_FP16_TILE_BYTES - 1U)) != 0U ||
        ((uintptr_t)buffers->hmx_output &
         (QBH_HMX_FP16_TILE_BYTES - 1U)) != 0U ||
        ((uintptr_t)buffers->scale_or_bias &
         (QBH_HMX_FP16_SCALE_BYTES - 1U)) != 0U ||
        ((uintptr_t)buffers->channel_scale &
         (QBH_HMX_FP16_SCALE_BYTES - 1U)) != 0U) {
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

static int qbh_hmx_submit(struct qbh_block_hmx_worker *worker,
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
    qurt_sem_down(&worker->command_done);
    asm volatile("barrier" ::: "memory");
    return worker->command_status == AEE_SUCCESS ? 0 : -1;
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

static int qbh_hmx_wait(struct qbh_block_hmx_worker *worker) {
    qurt_sem_down(&worker->command_done);
    asm volatile("barrier" ::: "memory");
    return worker->command_status == AEE_SUCCESS ? 0 : -1;
}

static void qbh_pack_fp16_activation(const __fp16 *source,
                                     uint32_t source_stride,
                                     uint32_t k, __fp16 *destination) {
    uint32_t k_tiles = k / QBH_HMX_FP16_COLS;
    for (uint32_t row = 0; row < QBH_BLOCK_M; ++row) {
        uint32_t row_tile = row / QBH_HMX_FP16_ROWS;
        uint32_t local_row = row % QBH_HMX_FP16_ROWS;
        for (uint32_t channel = 0; channel < k; ++channel) {
            uint32_t k_tile = channel / QBH_HMX_FP16_COLS;
            uint32_t local_channel = channel % QBH_HMX_FP16_COLS;
            size_t tile = qbh_hmx_fp16_matrix_tile_offset(
                row_tile, k_tile, k_tiles);
            destination[tile + qbh_hmx_fp16_tile_offset(
                                   local_row, local_channel)] =
                source[(size_t)row * source_stride + channel];
        }
    }
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
    for (uint32_t row = 0; row < QBH_BLOCK_M; ++row) {
        uint32_t row_tile = row / QBH_HMX_FP16_ROWS;
        uint32_t local_row = row % QBH_HMX_FP16_ROWS;
        for (uint32_t column = 0;
             column < n_tiles * QBH_HMX_FP16_COLS; ++column) {
            uint32_t column_tile = column / QBH_HMX_FP16_COLS;
            uint32_t local_column = column % QBH_HMX_FP16_COLS;
            size_t tile = qbh_hmx_fp16_matrix_tile_offset(
                row_tile, column_tile, n_tiles);
            destination[(size_t)row * destination_stride +
                        destination_column + column] =
                source[tile + qbh_hmx_fp16_tile_offset(
                                  local_row, local_column)];
        }
    }
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

static int qbh_run_projection(
    struct qbh_block_header *header, const uint8_t *shared,
    const struct qbh_block_projection_desc *desc,
    struct qbh_block_buffers *buffers,
    struct qbh_block_hmx_worker *worker, const void *input,
    void *output) {
    uint32_t k_tiles = desc->k / 32U;
    uint32_t n_tiles = desc->n / 32U;
    uint32_t element_bytes =
        header->variant == QBH_BLOCK_W4U8 ? 1U : 2U;
    int hvx_locked = 0;
    uint32_t failure_step = 0U;
    uint64_t phase_start = HAP_perf_get_qtimer_count();
    volatile uint32_t w4f16_ready[QBH_BLOCK_W4F16_MAX_REGIONS];

    memset((void *)w4f16_ready, 0, sizeof(w4f16_ready));

    if (header->variant == QBH_BLOCK_W4U8) {
        qbh_pack_u8_activation((const uint8_t *)input, desc->k,
                               desc->k, buffers->hmx_activation);
    } else {
        qbh_pack_fp16_activation((const __fp16 *)input, desc->k,
                                 desc->k,
                                 (__fp16 *)buffers->hmx_activation);
        qbh_hmx_fp16_init_unity_scale(buffers->scale_or_bias);
    }
    header->projection_pack_ticks +=
        HAP_perf_get_qtimer_count() - phase_start;
    if (header->variant == QBH_BLOCK_W4U8 ||
        header->variant == QBH_BLOCK_W4F16) {
        if (qurt_hvx_lock(QURT_HVX_MODE_128B) != AEE_SUCCESS) {
            qbh_record_projection_failure(
                header, desc, 0U, 5U, -1);
            return -1;
        }
        hvx_locked = 1;
    }

    for (uint32_t n_tile = 0; n_tile < n_tiles; ++n_tile) {
        uint32_t compressed_bytes = k_tiles * QBH_W4_PACKED_TILE_BYTES;
        uint32_t fp16_weight_bytes =
            k_tiles * QBH_HMX_FP16_TILE_BYTES;
        int result;
        if (header->variant == QBH_BLOCK_F16F16) {
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
            if (result == 0 && header->variant == QBH_BLOCK_W4F16) {
                failure_step = 2U;
                result = qbh_dma_copy(
                    header, buffers->channel_scale,
                    shared + desc->scale_offset +
                        (size_t)n_tile * 32U * sizeof(float),
                    32U * sizeof(float), 1U);
                header->weight_ddr_read_bytes += 32U * sizeof(float);
                ++header->weight_dma_descriptor_count;
            } else if (result == 0) {
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
            if (hvx_locked != 0) {
                (void)qurt_hvx_unlock();
            }
            return -1;
        }

        if (header->variant == QBH_BLOCK_W4U8) {
            failure_step = 4U;
            phase_start = HAP_perf_get_qtimer_count();
            result = qbh_hmx_submit(
                worker, QBH_BLOCK_HMX_U8S8,
                buffers->hmx_activation, buffers->expanded_weight,
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
            }
        } else {
            failure_step = 4U;
            phase_start = HAP_perf_get_qtimer_count();
            if (header->variant == QBH_BLOCK_W4F16) {
                uint32_t region_count =
                    k_tiles / QBH_BLOCK_W4F16_REGION_TILES;
                uint32_t generation = n_tile + 1U;
                qbh_hmx_start_fp16_streaming(
                    worker, buffers->hmx_activation,
                    buffers->expanded_weight, buffers->scale_or_bias,
                    buffers->hmx_output, 2U, k_tiles,
                    QBH_BLOCK_W4F16_REGION_TILES, w4f16_ready,
                    generation);
                ++header->w4f16_streamed_command_count;
                for (uint32_t region = 0; region < region_count;
                     ++region) {
                    uint64_t expand_start =
                        HAP_perf_get_qtimer_count();
                    qbh_expand_w4_to_f16_hvx(
                        buffers->compressed_weight +
                            (size_t)region *
                                QBH_BLOCK_W4F16_REGION_TILES *
                                QBH_W4_PACKED_TILE_BYTES,
                        (const float *)buffers->channel_scale,
                        buffers->expanded_weight +
                            (size_t)region *
                                QBH_BLOCK_W4F16_REGION_TILES *
                                QBH_HMX_FP16_TILE_BYTES,
                        QBH_BLOCK_W4F16_REGION_TILES);
                    header->w4f16_expand_ticks +=
                        HAP_perf_get_qtimer_count() - expand_start;
                    w4f16_ready[region] = generation;
                    asm volatile("release(%0):at"
                                 :
                                 : "r"(&w4f16_ready[region])
                                 : "memory");
                }
                if (desc == &header->projections[0] &&
                    n_tile == 0U) {
                    header->w4f16_expand_mismatch_count =
                        qbh_audit_w4_to_f16_tile(
                            buffers->compressed_weight,
                            (const float *)buffers->channel_scale,
                            buffers->expanded_weight,
                            &header->w4f16_expand_first_logical_index,
                            &header->w4f16_expand_expected_half_bits,
                            &header->w4f16_expand_actual_half_bits);
                }
                result = qbh_hmx_wait(worker);
            } else {
                result = qbh_hmx_submit(
                    worker, QBH_BLOCK_HMX_FP16,
                    buffers->hmx_activation, buffers->expanded_weight,
                    buffers->scale_or_bias, buffers->hmx_output, 2U,
                    k_tiles, 1U);
            }
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
        ++header->hmx_command_count;
        if (result != 0) {
            qbh_record_projection_failure(
                header, desc, n_tile, failure_step, result);
            if (hvx_locked != 0) {
                (void)qurt_hvx_unlock();
            }
            return -1;
        }
    }
    if (hvx_locked != 0 && qurt_hvx_unlock() != AEE_SUCCESS) {
        qbh_record_projection_failure(
            header, desc, n_tiles, 6U, -1);
        return -1;
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

static void qbh_pack_fp16_weight_rows(
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

static void qbh_pack_fp16_weight_transposed(
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

static void qbh_record_f16_nonfinite(struct qbh_block_header *header,
                                     const void *data,
                                     uint32_t elements,
                                     int32_t stage);

static float qbh_f16_max_abs(const __fp16 *data, uint32_t elements) {
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

static int qbh_attention_f16(struct qbh_block_header *header,
                             struct qbh_block_buffers *buffers,
                             struct qbh_block_hmx_worker *worker) {
    const __fp16 *q = (const __fp16 *)buffers->q;
    const __fp16 *k = (const __fp16 *)buffers->k;
    const __fp16 *v = (const __fp16 *)buffers->v;
    __fp16 *scores = (__fp16 *)buffers->scores;
    __fp16 *probability = (__fp16 *)buffers->probability;
    __fp16 *attention = (__fp16 *)buffers->attention_concat;
    qbh_hmx_fp16_init_unity_scale(buffers->scale_or_bias);

    for (uint32_t head = 0; head < QBH_BLOCK_HEADS; ++head) {
        uint32_t kv_head = head / (QBH_BLOCK_HEADS / QBH_BLOCK_KV_HEADS);
        qbh_pack_fp16_activation(
            q + head * QBH_BLOCK_HEAD_DIM, QBH_BLOCK_HIDDEN,
            QBH_BLOCK_HEAD_DIM, (__fp16 *)buffers->hmx_activation);
        qbh_pack_fp16_weight_rows(
            k, QBH_BLOCK_KV_HIDDEN,
            kv_head * QBH_BLOCK_HEAD_DIM, QBH_BLOCK_HEAD_DIM,
            QBH_BLOCK_M, (__fp16 *)buffers->expanded_weight);
        if (qbh_hmx_submit(worker, QBH_BLOCK_HMX_FP16,
                           buffers->hmx_activation,
                           buffers->expanded_weight,
                           buffers->scale_or_bias,
                           buffers->hmx_output, 2U, 4U, 2U) != 0) {
            return -1;
        }
        ++header->hmx_command_count;
        header->hmx_fp16_tile_pair_count += 16U;
        qbh_unpack_fp16_output(
            (const __fp16 *)buffers->hmx_output, 2U,
            scores + (size_t)head * QBH_BLOCK_M * QBH_BLOCK_M,
            QBH_BLOCK_M, 0U);
    }
    qbh_record_f16_nonfinite(
        header, scores, QBH_BLOCK_SCORE_ELEMENTS,
        QBH_BLOCK_NUMERICAL_ATTENTION_QK);
    header->attention_qk_max_abs = qbh_f16_max_abs(
        scores, QBH_BLOCK_SCORE_ELEMENTS);

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
    qbh_record_f16_nonfinite(
        header, probability, QBH_BLOCK_SCORE_ELEMENTS,
        QBH_BLOCK_NUMERICAL_ATTENTION_SOFTMAX);
    header->attention_probability_max_abs = qbh_f16_max_abs(
        probability, QBH_BLOCK_SCORE_ELEMENTS);

    for (uint32_t head = 0; head < QBH_BLOCK_HEADS; ++head) {
        uint32_t kv_head = head / (QBH_BLOCK_HEADS / QBH_BLOCK_KV_HEADS);
        qbh_pack_fp16_activation(
            probability + (size_t)head * QBH_BLOCK_M * QBH_BLOCK_M,
            QBH_BLOCK_M, QBH_BLOCK_M,
            (__fp16 *)buffers->hmx_activation);
        qbh_pack_fp16_weight_transposed(
            v, QBH_BLOCK_KV_HIDDEN,
            kv_head * QBH_BLOCK_HEAD_DIM, QBH_BLOCK_M,
            QBH_BLOCK_HEAD_DIM, (__fp16 *)buffers->expanded_weight);
        if (qbh_hmx_submit(worker, QBH_BLOCK_HMX_FP16,
                           buffers->hmx_activation,
                           buffers->expanded_weight,
                           buffers->scale_or_bias,
                           buffers->hmx_output, 2U, 2U, 4U) != 0) {
            return -1;
        }
        ++header->hmx_command_count;
        header->hmx_fp16_tile_pair_count += 16U;
        qbh_unpack_fp16_output(
            (const __fp16 *)buffers->hmx_output, 4U, attention,
            QBH_BLOCK_HIDDEN, head * QBH_BLOCK_HEAD_DIM);
    }
    qbh_record_f16_nonfinite(
        header, attention, QBH_BLOCK_M * QBH_BLOCK_HIDDEN,
        QBH_BLOCK_NUMERICAL_ATTENTION_AV);
    header->attention_av_max_abs = qbh_f16_max_abs(
        attention, QBH_BLOCK_M * QBH_BLOCK_HIDDEN);
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
    header->metadata_stage_ticks += HAP_perf_get_qtimer_count() - start;
    return 0;
}

static void qbh_record_f16_nonfinite(struct qbh_block_header *header,
                                     const void *data,
                                     uint32_t elements,
                                     int32_t stage) {
    const uint16_t *bits = (const uint16_t *)data;
    if (header->numerical_status != QBH_BLOCK_NUMERICAL_UNCHECKED) {
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
                             struct qbh_block_hmx_worker *worker) {
    uint32_t hidden_elements = QBH_BLOCK_M * QBH_BLOCK_HIDDEN;
    uint32_t intermediate_elements =
        QBH_BLOCK_M * QBH_BLOCK_INTERMEDIATE;
    uint64_t start;

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
        qbh_rms_norm_f16(
            (const __fp16 *)buffers->residual,
            (const __fp16 *)buffers->input_norm_weight,
            (__fp16 *)buffers->normalized,
            QBH_BLOCK_M, QBH_BLOCK_HIDDEN);
        qbh_record_f16_nonfinite(
            header, buffers->normalized, hidden_elements,
            QBH_BLOCK_NUMERICAL_INPUT_NORM);
    }
    header->input_norm_ticks += HAP_perf_get_qtimer_count() - start;

    start = HAP_perf_get_qtimer_count();
    if (qbh_run_projection(
            header, shared, &header->projections[QBH_BLOCK_PROJ_Q],
            buffers, worker, buffers->normalized, buffers->q) != 0 ||
        qbh_run_projection(
            header, shared, &header->projections[QBH_BLOCK_PROJ_K],
            buffers, worker, buffers->normalized, buffers->k) != 0 ||
        qbh_run_projection(
            header, shared, &header->projections[QBH_BLOCK_PROJ_V],
            buffers, worker, buffers->normalized, buffers->v) != 0) {
        return QBH_BLOCK_STATUS_QKV_FAILED;
    }
    if (header->variant != QBH_BLOCK_W4U8) {
        qbh_record_f16_nonfinite(
            header, buffers->q, QBH_BLOCK_M * QBH_BLOCK_HIDDEN,
            QBH_BLOCK_NUMERICAL_Q);
        qbh_record_f16_nonfinite(
            header, buffers->k, QBH_BLOCK_M * QBH_BLOCK_KV_HIDDEN,
            QBH_BLOCK_NUMERICAL_K);
        qbh_record_f16_nonfinite(
            header, buffers->v, QBH_BLOCK_M * QBH_BLOCK_KV_HIDDEN,
            QBH_BLOCK_NUMERICAL_V);
    }
    header->qkv_projection_ticks += HAP_perf_get_qtimer_count() - start;

    start = HAP_perf_get_qtimer_count();
    if (header->variant == QBH_BLOCK_W4U8) {
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
        qbh_qk_norm_rope_f16(
            (__fp16 *)buffers->q, QBH_BLOCK_HEADS, QBH_BLOCK_HIDDEN,
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
    qbh_record_f16_nonfinite(
        header, buffers->q, QBH_BLOCK_M * QBH_BLOCK_HIDDEN,
        QBH_BLOCK_NUMERICAL_Q_ROPE);
    qbh_record_f16_nonfinite(
        header, buffers->k, QBH_BLOCK_M * QBH_BLOCK_KV_HIDDEN,
        QBH_BLOCK_NUMERICAL_K_ROPE);
    header->qk_norm_rope_ticks += HAP_perf_get_qtimer_count() - start;

    start = HAP_perf_get_qtimer_count();
    if (qbh_attention_f16(header, buffers, worker) != 0) {
        return QBH_BLOCK_STATUS_ATTENTION_FAILED;
    }
    if (header->variant == QBH_BLOCK_W4U8) {
        qbh_quantize_f16_buffer(
            (const __fp16 *)buffers->attention_concat,
            buffers->attention_concat, hidden_elements,
            &header->qparams[QBH_BLOCK_QP_ATTENTION_CONCAT]);
    }
    header->attention_ticks += HAP_perf_get_qtimer_count() - start;

    start = HAP_perf_get_qtimer_count();
    if (qbh_run_projection(
            header, shared, &header->projections[QBH_BLOCK_PROJ_O],
            buffers, worker, buffers->attention_concat,
            buffers->attention_projection) != 0) {
        return QBH_BLOCK_STATUS_O_PROJECTION_FAILED;
    }
    if (header->variant != QBH_BLOCK_W4U8) {
        qbh_record_f16_nonfinite(
            header, buffers->attention_projection, hidden_elements,
            QBH_BLOCK_NUMERICAL_O);
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
        qbh_residual_add_f16(
            (__fp16 *)buffers->residual,
            (const __fp16 *)buffers->attention_projection,
            hidden_elements);
        qbh_record_f16_nonfinite(
            header, buffers->residual, hidden_elements,
            QBH_BLOCK_NUMERICAL_POST_RESIDUAL);
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
        qbh_rms_norm_f16(
            (const __fp16 *)buffers->residual,
            (const __fp16 *)buffers->post_norm_weight,
            (__fp16 *)buffers->normalized,
            QBH_BLOCK_M, QBH_BLOCK_HIDDEN);
        qbh_record_f16_nonfinite(
            header, buffers->normalized, hidden_elements,
            QBH_BLOCK_NUMERICAL_POST_NORM);
    }
    header->post_attention_norm_ticks +=
        HAP_perf_get_qtimer_count() - start;

    start = HAP_perf_get_qtimer_count();
    if (qbh_run_projection(
            header, shared,
            &header->projections[QBH_BLOCK_PROJ_GATE], buffers,
            worker, buffers->normalized, buffers->gate) != 0 ||
        qbh_run_projection(
            header, shared,
            &header->projections[QBH_BLOCK_PROJ_UP], buffers,
            worker, buffers->normalized, buffers->up) != 0) {
        return QBH_BLOCK_STATUS_GATE_UP_FAILED;
    }
    if (header->variant != QBH_BLOCK_W4U8) {
        qbh_record_f16_nonfinite(
            header, buffers->gate, intermediate_elements,
            QBH_BLOCK_NUMERICAL_GATE);
        qbh_record_f16_nonfinite(
            header, buffers->up, intermediate_elements,
            QBH_BLOCK_NUMERICAL_UP);
    }
    header->gate_up_ticks += HAP_perf_get_qtimer_count() - start;

    start = HAP_perf_get_qtimer_count();
    if (header->variant == QBH_BLOCK_W4U8) {
        qbh_silu_multiply_u8(
            buffers->gate, &header->qparams[QBH_BLOCK_QP_GATE],
            buffers->up, &header->qparams[QBH_BLOCK_QP_UP],
            buffers->middle, &header->qparams[QBH_BLOCK_QP_MIDDLE],
            intermediate_elements);
    } else {
        qbh_silu_multiply_f16(
            (const __fp16 *)buffers->gate,
            (const __fp16 *)buffers->up,
            (__fp16 *)buffers->middle, intermediate_elements);
        qbh_record_f16_nonfinite(
            header, buffers->middle, intermediate_elements,
            QBH_BLOCK_NUMERICAL_MIDDLE);
    }
    header->activation_ticks += HAP_perf_get_qtimer_count() - start;

    start = HAP_perf_get_qtimer_count();
    if (qbh_run_projection(
            header, shared,
            &header->projections[QBH_BLOCK_PROJ_DOWN], buffers,
            worker, buffers->middle, buffers->down) != 0) {
        return QBH_BLOCK_STATUS_DOWN_FAILED;
    }
    if (header->variant != QBH_BLOCK_W4U8) {
        qbh_record_f16_nonfinite(
            header, buffers->down, hidden_elements,
            QBH_BLOCK_NUMERICAL_DOWN);
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
        qbh_residual_add_f16((__fp16 *)buffers->residual,
                             (const __fp16 *)buffers->down,
                             hidden_elements);
        qbh_record_f16_nonfinite(
            header, buffers->residual, hidden_elements,
            QBH_BLOCK_NUMERICAL_OUTPUT);
    }
    if (header->numerical_status == QBH_BLOCK_NUMERICAL_UNCHECKED) {
        header->numerical_status = QBH_BLOCK_NUMERICAL_OK;
    }
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
    qurt_thread_attr_t attributes;
    qurt_thread_t thread;
    uint8_t *shared = NULL;
    int cache_status;
    int result;
    int thread_created = 0;
    int thread_joined = 0;
    int thread_exit_status = AEE_EFAILED;

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

    if (qbh_plan_buffers(vtcm, vtcm_bytes, header->variant, &buffers,
                         &header->vtcm_peak_plan_bytes) != 0) {
        header->dsp_status = QBH_BLOCK_STATUS_ARENA_FAILED;
        result = AEE_ENOMEMORY;
        goto publish;
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

    header->qtimer_start = HAP_perf_get_qtimer_count();
    if (qbh_stage_metadata(header, shared, &buffers) != 0) {
        header->dsp_status = QBH_BLOCK_STATUS_METADATA_DMA_FAILED;
        result = AEE_EFAILED;
        goto stop_worker;
    }
    for (uint32_t repeat = 0; repeat < header->repeat_count; ++repeat) {
        int block_status = qbh_run_one_block(
            header, shared, &buffers, &worker);
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
        header->output_stage_ticks +=
            HAP_perf_get_qtimer_count() - output_start;
    }
    header->dsp_status = QBH_BLOCK_STATUS_OK;
    result = AEE_SUCCESS;

stop_worker:
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
