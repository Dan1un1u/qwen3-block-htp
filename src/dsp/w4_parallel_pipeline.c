#include <AEEStdErr.h>
#include <HAP_compute_res.h>
#include <HAP_perf.h>
#include <qurt.h>
#include <qurt_hvx.h>
#include <stdint.h>
#include <string.h>

#include "hmx_u8s8_projection.h"
#include "mlp_protocol.h"
#include "mlp_u8.h"
#include "probe_protocol.h"
#include "qbh_user_dma.h"
#include "w4_parallel_pipeline.h"
#include "w4_u8_expand.h"

#define QBH_HVX_WORKER_STACK_BYTES UINT32_C(8192)
#define QBH_CHUNKED_HMX_STACK_BYTES UINT32_C(16384)
#define QBH_DMA_DESCRIPTOR_TIMEOUT_TICKS UINT64_C(1920000)
#define QBH_STREAM_READY_TIMEOUT_TICKS UINT64_C(1920000)

static uint8_t qbh_hvx_worker_stacks[QBH_MAX_HVX_WORKERS]
                                    [QBH_HVX_WORKER_STACK_BYTES]
    __attribute__((aligned(128)));
static uint8_t qbh_chunked_hmx_stack[QBH_CHUNKED_HMX_STACK_BYTES]
    __attribute__((aligned(128)));

struct qbh_chunk_task {
    uint32_t stop;
    uint32_t mlp_activation;
    uint32_t sequence;
    uint32_t compressed_slot;
    uint32_t expanded_slot;
    uint32_t chunk_index;
    uint32_t chunk_tiles;
    uint32_t stream_region_index;
    uint32_t stream_region_tiles;
    uint32_t stream_generation;
    uint32_t mlp_pair_index;
    uint32_t mlp_pair_slot;
};

struct qbh_chunk_queue {
    struct qbh_chunk_task tasks[QBH_W4_TASK_QUEUE_DEPTH];
    uint32_t head;
    uint32_t tail;
    qurt_mutex_t mutex;
    qurt_sem_t available;
    qurt_sem_t free_entries;
};

struct qbh_parallel_state {
    struct qbh_probe_header *header;
    const struct qbh_projection_layout *layout;
    const uint8_t *activation_tiles;
    uint8_t *compressed_slots[QBH_W4_MAX_COMPRESSED_SLOT_COUNT];
    uint8_t *expanded_slots[QBH_W4_MAX_EXPANDED_CHUNK_SLOT_COUNT];
    uint8_t *output_tiles;

    struct qbh_chunk_queue queue;
    qurt_sem_t compressed_free[QBH_W4_MAX_COMPRESSED_SLOT_COUNT];
    qurt_sem_t expanded_free[QBH_W4_MAX_EXPANDED_CHUNK_SLOT_COUNT];
    qurt_sem_t expanded_ready[QBH_W4_MAX_EXPANDED_CHUNK_SLOT_COUNT];
    qurt_sem_t hmx_started;
    qurt_sem_t hvx_started;
    qurt_sem_t mlp_pair_free[QBH_W4_MAX_COMPRESSED_SLOT_COUNT];
    qurt_mutex_t metrics_mutex;

    volatile uint32_t
        compressed_remaining[QBH_W4_MAX_COMPRESSED_SLOT_COUNT];
    volatile uint32_t stream_ready_generation
        [QBH_W4_MAX_EXPANDED_CHUNK_SLOT_COUNT]
        [QBH_W4_MAX_STREAM_REGIONS];
    volatile uint32_t active_hvx_workers;
    volatile uint32_t streaming_hmx_consumption_started;
    uint32_t max_active_hvx_workers;
    uint32_t hmx_active;
    uint32_t hvx_hmx_overlap_observed;
    uint32_t hvx_parallel_overlap_observed;
    volatile int32_t abort_status;
    const struct qbh_mlp_gate_up_handoff *mlp_handoff;
};

struct qbh_hvx_worker_job {
    struct qbh_parallel_state *state;
    uint32_t worker_index;
    int32_t lock_status;
    int32_t unlock_status;
    uint32_t expand_count;
    uint64_t expand_ticks;
    uint64_t first_expand_start;
    uint64_t last_expand_end;
};

struct qbh_chunked_hmx_job {
    struct qbh_parallel_state *state;
    uint32_t hmx_context_id;
    const struct qbh_w4_hmx_runner *runner;
    int32_t lock_status;
    int32_t unlock_status;
    uint32_t execution_count;
    uint32_t stream_count;
    uint32_t output_tile_count;
    uint32_t max_batch_output_count;
    uint32_t in_command_slot_release_count;
    uint32_t producer_progress_command_count;
    uint64_t compute_ticks;
    uint64_t ready_wait_ticks;
    uint64_t first_compute_start;
    uint64_t last_compute_end;
};

static void qbh_abort_pipeline(struct qbh_parallel_state *state,
                               int32_t status);

static uint32_t qbh_atomic_dec_return(volatile uint32_t *target) {
    uint32_t result;
    __asm__ __volatile__(
        "1:     %0 = memw_locked(%2)\n"
        "       %0 = add(%0, #-1)\n"
        "       memw_locked(%2, p0) = %0\n"
        "       if !p0 jump 1b\n"
        : "=&r"(result), "+m"(*target)
        : "r"(target)
        : "p0");
    return result;
}

static uint32_t qbh_atomic_inc_return(volatile uint32_t *target) {
    uint32_t result;
    __asm__ __volatile__(
        "1:     %0 = memw_locked(%2)\n"
        "       %0 = add(%0, #1)\n"
        "       memw_locked(%2, p0) = %0\n"
        "       if !p0 jump 1b\n"
        : "=&r"(result), "+m"(*target)
        : "r"(target)
        : "p0");
    return result;
}

static uint32_t qbh_stream_region_count(
    const struct qbh_projection_layout *layout, uint32_t chunk_index) {
    uint32_t chunk_tiles =
        qbh_projection_chunk_tiles(layout, chunk_index);
    return chunk_tiles / QBH_W4_STREAM_REGION_TILES;
}

static void qbh_queue_init(struct qbh_chunk_queue *queue) {
    memset(queue, 0, sizeof(*queue));
    qurt_mutex_init(&queue->mutex);
    qurt_sem_init_val(&queue->available, 0);
    qurt_sem_init_val(&queue->free_entries,
                      QBH_W4_TASK_QUEUE_DEPTH);
}

static void qbh_queue_destroy(struct qbh_chunk_queue *queue) {
    qurt_sem_destroy(&queue->free_entries);
    qurt_sem_destroy(&queue->available);
    qurt_mutex_destroy(&queue->mutex);
}

static void qbh_queue_push(struct qbh_chunk_queue *queue,
                           const struct qbh_chunk_task *task) {
    uint32_t slot;
    qurt_sem_down(&queue->free_entries);
    qurt_mutex_lock(&queue->mutex);
    slot = queue->tail % QBH_W4_TASK_QUEUE_DEPTH;
    queue->tasks[slot] = *task;
    ++queue->tail;
    qurt_mutex_unlock(&queue->mutex);
    qurt_sem_up(&queue->available);
}

static void qbh_queue_pop(struct qbh_chunk_queue *queue,
                          struct qbh_chunk_task *task) {
    uint32_t slot;
    qurt_sem_down(&queue->available);
    qurt_mutex_lock(&queue->mutex);
    slot = queue->head % QBH_W4_TASK_QUEUE_DEPTH;
    *task = queue->tasks[slot];
    ++queue->head;
    qurt_mutex_unlock(&queue->mutex);
    qurt_sem_up(&queue->free_entries);
}

static int qbh_record_dma_wait(struct qbh_probe_header *header) {
    int status;
    ++header->dma_wait_count;
    status = qbh_dma_wait_idle();
    if (status != 0 && header->dma_status == 0) {
        header->dma_status = status;
    }
    return status;
}

static int qbh_stage_weight_bundles(struct qbh_probe_header *header,
                                    const uint8_t *source,
                                    uint8_t *destination,
                                    uint32_t bundle_bytes,
                                    uint32_t bundle_count) {
    struct qbh_dma_desc_1d descriptor __attribute__((aligned(64)));
    int status;

    memset(&descriptor, 0, sizeof(descriptor));
    if (qbh_record_dma_wait(header) != 0) {
        return -1;
    }
    descriptor.next = 0;
    descriptor.length = bundle_bytes * bundle_count;
    descriptor.type = QBH_DMA_TYPE_1D;
    descriptor.src_bypass = 1;
    descriptor.dst_bypass = 0;
    descriptor.ordered = 1;
    descriptor.dstate = QBH_DMA_DESC_PENDING;
    descriptor.src = (uint32_t)(uintptr_t)source;
    descriptor.dst = (uint32_t)(uintptr_t)destination;

    status = qbh_dma_start(&descriptor);
    ++header->dma_submit_count;
    ++header->dma_descriptor_count;
    if (status != 0) {
        header->dma_status = status;
        return -1;
    }
    if (qbh_record_dma_wait(header) != 0) {
        return -1;
    }
    ++header->dma_descriptor_completion_count;
    asm volatile("barrier" : : : "memory");
    header->weight_bundle_stage_count += bundle_count;
    return 0;
}

static int qbh_start_linked_weight_bundles(
    struct qbh_probe_header *header,
    struct qbh_dma_aligned_desc_1d *descriptors,
    const uint8_t *source, uint8_t *destination,
    uint32_t bundle_bytes, uint32_t bundle_count) {
    if (bundle_count < 2U || bundle_count > 4U ||
        qbh_record_dma_wait(header) != 0) {
        return -1;
    }
    memset(descriptors, 0,
           sizeof(*descriptors) * bundle_count);
    for (uint32_t index = 0; index < bundle_count; ++index) {
        struct qbh_dma_desc_1d *descriptor =
            &descriptors[index].descriptor;
        descriptor->next = index + 1U < bundle_count
                               ? (uint32_t)(uintptr_t)(
                                     &descriptors[index + 1U].descriptor)
                               : 0U;
        descriptor->length = bundle_bytes;
        descriptor->type = QBH_DMA_TYPE_1D;
        descriptor->src_bypass = 1;
        descriptor->dst_bypass = 0;
        descriptor->ordered = 1;
        descriptor->dstate = QBH_DMA_DESC_PENDING;
        descriptor->src = (uint32_t)(uintptr_t)(
            source + (size_t)index * bundle_bytes);
        descriptor->dst = (uint32_t)(uintptr_t)(
            destination + (size_t)index * bundle_bytes);
        asm volatile("release(%0):at"
                     :
                     : "r"(descriptor)
                     : "memory");
    }
    if (qbh_dma_start(&descriptors[0].descriptor) != 0) {
        return -1;
    }
    ++header->dma_submit_count;
    header->dma_descriptor_count += bundle_count;
    ++header->dma_chain_count;
    return 0;
}

static int qbh_wait_linked_descriptor(
    struct qbh_probe_header *header,
    struct qbh_dma_desc_1d *descriptor) {
    uint64_t start = HAP_perf_get_qtimer_count();
    uint32_t spins = 0;

    ++header->dma_wait_count;
    for (;;) {
        uint32_t status = Q6_R_dmpoll() & QBH_DMA_STATUS_MASK;
        if (((volatile struct qbh_dma_desc_1d *)descriptor)->dstate ==
            QBH_DMA_DESC_COMPLETE) {
            asm volatile("barrier" : : : "memory");
            ++header->dma_descriptor_completion_count;
            return 0;
        }
        if (status == QBH_DMA_STATUS_ERROR) {
            header->dma_status = (int32_t)status;
            return -1;
        }
        ++spins;
        if ((spins & UINT32_C(255)) == 0U &&
            HAP_perf_get_qtimer_count() - start >
                QBH_DMA_DESCRIPTOR_TIMEOUT_TICKS) {
            ++header->dma_descriptor_timeout_count;
            header->dma_status = -1;
            return -1;
        }
    }
}

static int qbh_hvx_region_begin(struct qbh_parallel_state *state) {
    uint32_t active;
    for (;;) {
        active = qbh_atomic_inc_return(&state->active_hvx_workers);
        if (!qbh_physical_plan_is_capped_streaming(
                state->layout->physical_plan) ||
            state->streaming_hmx_consumption_started == 0U ||
            active <= 2U) {
            break;
        }
        (void)qbh_atomic_dec_return(&state->active_hvx_workers);
        if (state->abort_status != 0) {
            return -1;
        }
        asm volatile("pause(#8)" : : : "memory");
    }
    if (active > state->max_active_hvx_workers) {
        qurt_mutex_lock(&state->metrics_mutex);
        if (active > state->max_active_hvx_workers) {
            state->max_active_hvx_workers = active;
        }
        qurt_mutex_unlock(&state->metrics_mutex);
    }
    if (active > 1U) {
        state->hvx_parallel_overlap_observed = 1U;
    }
    if (state->hmx_active != 0U) {
        state->hvx_hmx_overlap_observed = 1U;
    }
    return 0;
}

static void qbh_hvx_region_end(struct qbh_parallel_state *state) {
    if (state->hmx_active != 0U) {
        state->hvx_hmx_overlap_observed = 1U;
    }
    (void)qbh_atomic_dec_return(&state->active_hvx_workers);
}

static void qbh_hmx_region_begin(struct qbh_parallel_state *state) {
    state->hmx_active = 1U;
    asm volatile("barrier" : : : "memory");
    if (state->active_hvx_workers != 0U) {
        state->hvx_hmx_overlap_observed = 1U;
    }
}

static void qbh_hmx_region_end(struct qbh_parallel_state *state) {
    if (state->active_hvx_workers != 0U) {
        state->hvx_hmx_overlap_observed = 1U;
    }
    asm volatile("barrier" : : : "memory");
    state->hmx_active = 0U;
}

static int qbh_hvx_worker_run(struct qbh_hvx_worker_job *job,
                              uint32_t manage_hvx_context) {
    struct qbh_parallel_state *state = job->state;
    const struct qbh_projection_layout *layout = state->layout;
    int exit_status = AEE_SUCCESS;
    const uint32_t stream_fence_mode =
        state->mlp_handoff != NULL
            ? state->mlp_handoff->stream_fence_mode
            : QBH_W4_STREAM_FENCE_CONTROL;

    job->lock_status = manage_hvx_context != 0U
        ? qurt_hvx_lock(QURT_HVX_MODE_128B) : AEE_SUCCESS;
    qurt_sem_up(&state->hvx_started);
    if (job->lock_status != AEE_SUCCESS) {
        return job->lock_status;
    }

    for (;;) {
        struct qbh_chunk_task task;
        uint64_t start;
        uint64_t end;
        const uint8_t *compressed;
        uint8_t *expanded;
        uint32_t region_tiles;
        size_t source_offset;
        size_t expanded_offset;
        int admission_status;

        qbh_queue_pop(&state->queue, &task);
        if (task.stop != 0U) {
            break;
        }

        if (task.mlp_activation != 0U) {
            uint64_t activation_start = HAP_perf_get_qtimer_count();
            uint32_t activation_mismatches = 0U;
            uint8_t *pair = state->output_tiles +
                (size_t)task.mlp_pair_slot * 2U *
                    QBH_HMX_OUTPUT_BYTES;
            uint8_t *middle =
                state->mlp_handoff->middle_activation +
                (size_t)task.mlp_pair_index * QBH_HMX_OUTPUT_BYTES;
            if (qbh_hvx_region_begin(state) != 0) {
                exit_status = AEE_EFAILED;
                break;
            }
            const uint8_t *multipliers =
                state->mlp_handoff->output_multipliers != NULL
                    ? state->mlp_handoff->output_multipliers +
                          (size_t)task.mlp_pair_index * 2U *
                              QBH_HMX_OUTPUT_CHANNELS
                    : NULL;
            if (multipliers != NULL &&
                state->mlp_handoff->verify_activation_elements != 0U) {
                qbh_mlp_gate_up_requant_lut_hvx(
                    pair, pair + QBH_HMX_OUTPUT_BYTES,
                    middle,
                    QBH_HMX_OUTPUT_BYTES,
                    state->mlp_handoff->activation_lut,
                    state->mlp_handoff->activation_gather_scratch +
                        (size_t)job->worker_index *
                            QBH_MLP_GATHER_SCRATCH_BYTES,
                    multipliers,
                    multipliers + QBH_HMX_OUTPUT_CHANNELS,
                    QBH_MLP_GATE_ZERO_POINT, QBH_MLP_UP_ZERO_POINT);
                qbh_mlp_gate_up_requant_lut_hvx(
                    pair, pair + QBH_HMX_OUTPUT_BYTES, pair,
                    state->mlp_handoff->activation_elements,
                    state->mlp_handoff->activation_lut,
                    state->mlp_handoff->activation_gather_scratch +
                        (size_t)job->worker_index *
                            QBH_MLP_GATHER_SCRATCH_BYTES,
                    multipliers,
                    multipliers + QBH_HMX_OUTPUT_CHANNELS,
                    QBH_MLP_GATE_ZERO_POINT, QBH_MLP_UP_ZERO_POINT);
            } else if (multipliers != NULL) {
                qbh_mlp_gate_up_requant_lut_hvx(
                    pair, pair + QBH_HMX_OUTPUT_BYTES, middle,
                    state->mlp_handoff->activation_elements,
                    state->mlp_handoff->activation_lut,
                    state->mlp_handoff->activation_gather_scratch +
                        (size_t)job->worker_index *
                            QBH_MLP_GATHER_SCRATCH_BYTES,
                    multipliers,
                    multipliers + QBH_HMX_OUTPUT_CHANNELS,
                    QBH_MLP_GATE_ZERO_POINT, QBH_MLP_UP_ZERO_POINT);
            } else if (
                state->mlp_handoff->verify_activation_elements != 0U) {
                qbh_mlp_gate_up_lut_hvx(
                    pair, pair + QBH_HMX_OUTPUT_BYTES,
                    middle,
                    QBH_HMX_OUTPUT_BYTES,
                    state->mlp_handoff->activation_lut,
                    state->mlp_handoff->activation_gather_scratch +
                        (size_t)job->worker_index *
                            QBH_MLP_GATHER_SCRATCH_BYTES);
                qbh_mlp_gate_up_lut_hvx(
                    pair, pair + QBH_HMX_OUTPUT_BYTES, pair,
                    state->mlp_handoff->activation_elements,
                    state->mlp_handoff->activation_lut,
                    state->mlp_handoff->activation_gather_scratch +
                        (size_t)job->worker_index *
                            QBH_MLP_GATHER_SCRATCH_BYTES);
            } else {
                qbh_mlp_gate_up_lut_hvx(
                    pair, pair + QBH_HMX_OUTPUT_BYTES, middle,
                    state->mlp_handoff->activation_elements,
                    state->mlp_handoff->activation_lut,
                    state->mlp_handoff->activation_gather_scratch +
                        (size_t)job->worker_index *
                            QBH_MLP_GATHER_SCRATCH_BYTES);
            }
            qbh_hvx_region_end(state);
            if (state->mlp_handoff->verify_activation_elements != 0U) {
                for (uint32_t index = 0U;
                     index < state->mlp_handoff->activation_elements;
                     ++index) {
                    activation_mismatches += pair[index] != middle[index];
                }
            }
            qurt_mutex_lock(&state->metrics_mutex);
            *state->mlp_handoff->activation_ticks +=
                HAP_perf_get_qtimer_count() - activation_start;
            if (state->mlp_handoff->activation_mismatch_count != NULL) {
                *state->mlp_handoff->activation_mismatch_count +=
                    activation_mismatches;
            }
            ++*state->mlp_handoff->pair_consume_count;
            qurt_mutex_unlock(&state->metrics_mutex);
            asm volatile("barrier" : : : "memory");
            qurt_sem_up(&state->mlp_pair_free[task.mlp_pair_slot]);
            continue;
        }

        compressed = state->compressed_slots[task.compressed_slot];
        expanded = state->expanded_slots[task.expanded_slot];
        if (qbh_physical_plan_is_streaming(layout->physical_plan)) {
            region_tiles = task.stream_region_tiles;
            source_offset =
                ((size_t)task.chunk_index * layout->chunk_tiles +
                 (size_t)task.stream_region_index *
                     QBH_W4_STREAM_REGION_TILES) *
                (layout->weight_storage_variant ==
                         QBH_WEIGHT_EXPANDED_S8
                     ? QBH_HMX_WEIGHT_BYTES
                     : QBH_W4_PACKED_TILE_BYTES);
            expanded_offset =
                (size_t)task.stream_region_index *
                QBH_W4_STREAM_REGION_TILES * QBH_HMX_WEIGHT_BYTES;
        } else {
            region_tiles = task.chunk_tiles;
            source_offset =
                (size_t)task.chunk_index * layout->chunk_tiles *
                (layout->weight_storage_variant ==
                         QBH_WEIGHT_EXPANDED_S8
                     ? QBH_HMX_WEIGHT_BYTES
                     : QBH_W4_PACKED_TILE_BYTES);
            expanded_offset = 0U;
        }
        start = HAP_perf_get_qtimer_count();
        if (job->first_expand_start == 0U) {
            job->first_expand_start = start;
        }
        admission_status = qbh_hvx_region_begin(state);
        if (admission_status != 0) {
            exit_status = AEE_EFAILED;
            break;
        }
        if (layout->weight_storage_variant ==
            QBH_WEIGHT_EXPANDED_S8) {
            qbh_copy_s8_hmx_tiles_hvx(
                (const int8_t *)(compressed + source_offset),
                (int8_t *)(expanded + expanded_offset), region_tiles);
        } else if (layout->weight_storage_variant ==
            QBH_WEIGHT_PACKED_W4_HMX_SCALE) {
            if (stream_fence_mode >=
                QBH_W4_STREAM_FENCE_SINGLE) {
                qbh_unpack_w4_to_s8_hvx_relaxed(
                    compressed + source_offset,
                    (int8_t *)(expanded + expanded_offset),
                    region_tiles);
            } else {
                qbh_unpack_w4_to_s8_hvx(
                    compressed + source_offset,
                    (int8_t *)(expanded + expanded_offset),
                    region_tiles);
            }
        } else {
            qbh_expand_w4_to_s8_hvx(
                compressed + source_offset,
                compressed + layout->w4_scale_offset,
                (int8_t *)(expanded + expanded_offset), region_tiles);
        }
        if (task.chunk_index == 0U &&
            (!qbh_physical_plan_is_streaming(layout->physical_plan) ||
             task.stream_region_index == 0U)) {
            const uint8_t *bias_source = compressed +
                (layout->weight_storage_variant ==
                         QBH_WEIGHT_EXPANDED_S8
                     ? layout->expanded_weight_chunk_bytes
                     : layout->w4_bias_offset);
            if (stream_fence_mode >=
                QBH_W4_STREAM_FENCE_SINGLE) {
                qbh_copy_hmx_bias_hvx_relaxed(
                    bias_source,
                    expanded + layout->expanded_chunk_weight_bytes);
            } else {
                qbh_copy_hmx_bias_hvx(
                    bias_source,
                    expanded + layout->expanded_chunk_weight_bytes);
            }
        }
        qbh_hvx_region_end(state);
        end = HAP_perf_get_qtimer_count();
        job->last_expand_end = end;
        job->expand_ticks += end - start;
        ++job->expand_count;

        if (stream_fence_mode !=
            QBH_W4_STREAM_FENCE_RELEASE_ONLY) {
            asm volatile("barrier" : : : "memory");
        }
        if (qbh_physical_plan_is_streaming(layout->physical_plan)) {
            volatile uint32_t *ready =
                &state->stream_ready_generation[task.expanded_slot]
                                                     [task.stream_region_index];
            *ready = task.stream_generation;
            asm volatile("release(%0):at"
                         :
                         : "r"(ready)
                         : "memory");
            (void)qbh_atomic_inc_return(
                &state->header->streaming_region_publish_count);
        } else {
            qurt_sem_up(&state->expanded_ready[task.expanded_slot]);
        }
        if (qbh_atomic_dec_return(
                &state->compressed_remaining[task.compressed_slot]) == 0U) {
            qurt_sem_up(&state->compressed_free[task.compressed_slot]);
        }
    }

    job->unlock_status = manage_hvx_context != 0U
        ? qurt_hvx_unlock() : AEE_SUCCESS;
    if (job->unlock_status != AEE_SUCCESS) {
        exit_status = job->unlock_status;
    }
    return exit_status;
}

int qbh_run_chunked_w4_external_hvx_worker(void *worker_context) {
    if (worker_context == NULL) {
        return AEE_EBADPARM;
    }
    return qbh_hvx_worker_run(
        (struct qbh_hvx_worker_job *)worker_context, 0U);
}

int qbh_run_chunked_w4_managed_hvx_worker(void *worker_context) {
    if (worker_context == NULL) {
        return AEE_EBADPARM;
    }
    return qbh_hvx_worker_run(
        (struct qbh_hvx_worker_job *)worker_context, 1U);
}

static void qbh_hvx_worker_main(void *opaque) {
    int exit_status =
        qbh_run_chunked_w4_managed_hvx_worker(opaque);
    qurt_thread_exit(exit_status);
}

static void qbh_chunked_hmx_main(void *opaque) {
    struct qbh_chunked_hmx_job *job =
        (struct qbh_chunked_hmx_job *)opaque;
    struct qbh_parallel_state *state = job->state;
    const struct qbh_projection_layout *layout = state->layout;
    int exit_status = AEE_SUCCESS;

    job->lock_status = job->runner == NULL
        ? HAP_compute_res_hmx_lock2(
              job->hmx_context_id, HAP_COMPUTE_RES_HMX_SHARED)
        : AEE_SUCCESS;
    qurt_sem_up(&state->hmx_started);
    if (job->lock_status != AEE_SUCCESS) {
        qurt_thread_exit(job->lock_status);
    }

    if (state->mlp_handoff != NULL && job->runner != NULL &&
        job->runner->max_batch_outputs > 1U &&
        job->runner->max_batch_outputs <=
            QBH_W4_HMX_MAX_BATCH_OUTPUTS &&
        qbh_physical_plan_is_streaming(layout->physical_plan) &&
        layout->chunks_per_output == 1U &&
        state->header->repeat_count == 1U) {
        uint32_t batch_capacity = job->runner->max_batch_outputs;
        for (uint32_t output_base = 0U;
             output_base < layout->n_tiles;
             output_base += batch_capacity) {
            uint32_t batch_outputs = layout->n_tiles - output_base;
            uint32_t chunk_tiles =
                qbh_projection_chunk_tiles(layout, 0U);
            uint32_t stream_regions =
                qbh_stream_region_count(layout, 0U);
            uint64_t stream_ready_wait = 0U;
            uint64_t core_start;
            uint64_t core_end;
            uint32_t executed_streams = 0U;
            struct qbh_w4_hmx_request request;

            if (batch_outputs > batch_capacity) {
                batch_outputs = batch_capacity;
            }
            if (stream_regions == 0U ||
                stream_regions > QBH_W4_MAX_STREAM_REGIONS ||
                chunk_tiles != stream_regions *
                                   QBH_W4_STREAM_REGION_TILES) {
                exit_status = AEE_EBADPARM;
                state->abort_status = exit_status;
                goto unlock;
            }
            memset(&request, 0, sizeof(request));
            request.activation_tiles = state->activation_tiles;
            request.chunk_tiles = chunk_tiles;
            request.begin_output = 1U;
            request.store_output = 1U;
            request.streaming = 1U;
            request.stream_count = stream_regions;
            request.abort_status = &state->abort_status;
            request.timeout_ticks = QBH_STREAM_READY_TIMEOUT_TICKS;
            request.ready_wait_ticks = &stream_ready_wait;
            request.hmx_consumption_started =
                &state->streaming_hmx_consumption_started;
            request.executed_stream_count = &executed_streams;
            request.batch_output_count = batch_outputs;

            for (uint32_t batch_index = 0U;
                 batch_index < batch_outputs; ++batch_index) {
                uint32_t output_tile = output_base + batch_index;
                uint32_t sequence = output_tile;
                uint32_t expanded_slot =
                    sequence % layout->expanded_slot_count;
                uint32_t pair_index = output_tile / 2U;
                uint32_t pair_slot = pair_index %
                    state->mlp_handoff->pair_slot_count;
                uint32_t pair_lane = output_tile & 1U;

                if (pair_lane == 0U) {
                    uint64_t pair_wait =
                        HAP_perf_get_qtimer_count();
                    qurt_sem_down(&state->mlp_pair_free[pair_slot]);
                    state->header->producer_slot_wait_ticks +=
                        HAP_perf_get_qtimer_count() - pair_wait;
                }
                request.batch_outputs[batch_index]
                    .expanded_weight_tiles = (const int8_t *)
                        state->expanded_slots[expanded_slot];
                request.batch_outputs[batch_index].bias_words =
                    (const uint32_t *)(
                        state->expanded_slots[expanded_slot] +
                        layout->expanded_chunk_weight_bytes);
                request.batch_outputs[batch_index].output_tiles =
                    state->output_tiles +
                    (size_t)pair_slot * 2U * QBH_HMX_OUTPUT_BYTES +
                    (size_t)pair_lane * QBH_HMX_OUTPUT_BYTES;
                request.batch_outputs[batch_index]
                    .ready_generations =
                        state->stream_ready_generation[expanded_slot];
                request.batch_outputs[batch_index]
                    .expected_generation = sequence + 1U;
            }

            core_start = HAP_perf_get_qtimer_count();
            if (job->first_compute_start == 0U) {
                job->first_compute_start = core_start;
            }
            qbh_hmx_region_begin(state);
            if (job->runner->submit(job->runner->context, &request) != 0) {
                qbh_hmx_region_end(state);
                exit_status = AEE_EFAILED;
                qbh_abort_pipeline(state, exit_status);
                goto unlock;
            }
            qbh_hmx_region_end(state);
            core_end = HAP_perf_get_qtimer_count();
            job->ready_wait_ticks += stream_ready_wait;
            job->stream_count += executed_streams;
            job->execution_count += batch_outputs * chunk_tiles;
            job->output_tile_count += batch_outputs;
            job->last_compute_end = core_end;
            job->compute_ticks +=
                core_end - core_start - stream_ready_wait;

            for (uint32_t batch_index = 0U;
                 batch_index < batch_outputs; ++batch_index) {
                uint32_t output_tile = output_base + batch_index;
                uint32_t expanded_slot =
                    output_tile % layout->expanded_slot_count;
                qurt_sem_up(&state->expanded_free[expanded_slot]);
                if ((output_tile & 1U) != 0U) {
                    uint32_t pair_index = output_tile / 2U;
                    struct qbh_chunk_task activation_task;
                    memset(&activation_task, 0, sizeof(activation_task));
                    activation_task.mlp_activation = 1U;
                    activation_task.mlp_pair_index = pair_index;
                    activation_task.mlp_pair_slot = pair_index %
                        state->mlp_handoff->pair_slot_count;
                    qbh_queue_push(&state->queue, &activation_task);
                    ++*state->mlp_handoff->pair_publish_count;
                }
            }
        }
        goto unlock;
    }

    if (job->runner != NULL &&
        job->runner->max_nonstreaming_batch_outputs > 1U &&
        job->runner->max_nonstreaming_batch_outputs <=
            QBH_W4_HMX_MAX_BATCH_OUTPUTS &&
        !qbh_physical_plan_is_streaming(layout->physical_plan) &&
        layout->chunks_per_output == 2U) {
        uint32_t batch_capacity =
            job->runner->max_nonstreaming_batch_outputs;
        for (uint32_t repeat = 0U;
             repeat < state->header->repeat_count; ++repeat) {
            for (uint32_t output_base = 0U;
                 output_base < layout->n_tiles;
                 output_base += batch_capacity) {
                uint32_t batch_outputs = layout->n_tiles - output_base;
                uint32_t first_chunk_tiles =
                    qbh_projection_chunk_tiles(layout, 0U);
                uint32_t second_chunk_tiles =
                    qbh_projection_chunk_tiles(layout, 1U);
                uint32_t first_linear_output =
                    repeat * layout->n_tiles + output_base;
                uint32_t first_sequence = first_linear_output * 2U;
                uint32_t first_slot =
                    first_sequence % layout->expanded_slot_count;
                uint32_t staged_before;
                uint64_t wait_start = HAP_perf_get_qtimer_count();
                uint64_t command_ready_wait = 0U;
                uint64_t core_start;
                uint64_t core_end;
                uint32_t executed_streams = 0U;
                struct qbh_w4_hmx_request request;

                if (batch_outputs > batch_capacity) {
                    batch_outputs = batch_capacity;
                }
                qurt_sem_down(&state->expanded_ready[first_slot]);
                job->ready_wait_ticks +=
                    HAP_perf_get_qtimer_count() - wait_start;
                if (state->abort_status != 0) {
                    exit_status = state->abort_status;
                    goto unlock;
                }

                memset(&request, 0, sizeof(request));
                request.activation_tiles = state->activation_tiles;
                request.chunk_tiles = first_chunk_tiles;
                request.begin_output = 1U;
                request.store_output = 1U;
                request.abort_status = &state->abort_status;
                request.ready_wait_ticks = &command_ready_wait;
                request.executed_stream_count = &executed_streams;
                request.batch_output_count = batch_outputs;
                request.continuation_chunk_count = 1U;
                request.continuation_chunks[0].activation_tiles =
                    state->activation_tiles +
                    (size_t)layout->chunk_tiles *
                        QBH_HMX_ACTIVATION_BYTES;
                request.continuation_chunks[0].chunk_tiles =
                    second_chunk_tiles;
                request.in_command_slot_release_count =
                    &job->in_command_slot_release_count;

                for (uint32_t batch_index = 0U;
                     batch_index < batch_outputs; ++batch_index) {
                    uint32_t linear_output =
                        first_linear_output + batch_index;
                    uint32_t sequence = linear_output * 2U;
                    uint32_t output_tile = output_base + batch_index;
                    uint32_t output_first_slot =
                        sequence % layout->expanded_slot_count;
                    uint32_t output_second_slot =
                        (sequence + 1U) % layout->expanded_slot_count;

                    request.batch_outputs[batch_index]
                        .expanded_weight_tiles = (const int8_t *)
                            state->expanded_slots[output_first_slot];
                    request.batch_outputs[batch_index].bias_words =
                        (const uint32_t *)(
                            state->expanded_slots[output_first_slot] +
                            layout->expanded_chunk_weight_bytes);
                    request.batch_outputs[batch_index].output_tiles =
                        state->output_tiles +
                        (size_t)output_tile * QBH_HMX_OUTPUT_BYTES;
                    request.batch_outputs[batch_index].ready_semaphore =
                        batch_index == 0U
                            ? NULL
                            : &state->expanded_ready[output_first_slot];
                    request.batch_outputs[batch_index].free_semaphore =
                        &state->expanded_free[output_first_slot];
                    request.batch_outputs[batch_index]
                        .continuation_expanded_weight_tiles =
                            (const int8_t *)
                                state->expanded_slots[output_second_slot];
                    request.batch_outputs[batch_index]
                        .continuation_ready_semaphore =
                            &state->expanded_ready[output_second_slot];
                    request.batch_outputs[batch_index]
                        .continuation_free_semaphore =
                            &state->expanded_free[output_second_slot];
                }

                staged_before = state->header->weight_bundle_stage_count;
                core_start = HAP_perf_get_qtimer_count();
                if (job->first_compute_start == 0U) {
                    job->first_compute_start = core_start;
                }
                qbh_hmx_region_begin(state);
                if (job->runner->submit(job->runner->context,
                                        &request) != 0) {
                    qbh_hmx_region_end(state);
                    exit_status = AEE_EFAILED;
                    qbh_abort_pipeline(state, exit_status);
                    goto unlock;
                }
                qbh_hmx_region_end(state);
                core_end = HAP_perf_get_qtimer_count();
                asm volatile("barrier" : : : "memory");
                if (state->header->weight_bundle_stage_count >
                    staged_before) {
                    ++job->producer_progress_command_count;
                }
                if (batch_outputs > job->max_batch_output_count) {
                    job->max_batch_output_count = batch_outputs;
                }
                job->ready_wait_ticks += command_ready_wait;
                job->stream_count += executed_streams;
                job->execution_count += batch_outputs *
                    (first_chunk_tiles + second_chunk_tiles);
                job->output_tile_count += batch_outputs;
                job->last_compute_end = core_end;
                job->compute_ticks +=
                    core_end - core_start - command_ready_wait;
            }
        }
        goto unlock;
    }

    if (job->runner != NULL &&
        job->runner->max_chunks_per_command >= 2U &&
        !qbh_physical_plan_is_streaming(layout->physical_plan) &&
        layout->chunks_per_output == 2U) {
        for (uint32_t repeat = 0U;
             repeat < state->header->repeat_count; ++repeat) {
            for (uint32_t output_tile = 0U;
                 output_tile < layout->n_tiles; ++output_tile) {
                uint32_t linear_output =
                    repeat * layout->n_tiles + output_tile;
                uint32_t first_sequence = linear_output * 2U;
                uint32_t first_slot =
                    first_sequence % layout->expanded_slot_count;
                uint32_t second_slot =
                    (first_sequence + 1U) % layout->expanded_slot_count;
                uint32_t first_chunk_tiles =
                    qbh_projection_chunk_tiles(layout, 0U);
                uint32_t second_chunk_tiles =
                    qbh_projection_chunk_tiles(layout, 1U);
                uint64_t wait_start = HAP_perf_get_qtimer_count();
                uint64_t continuation_ready_wait = 0U;
                uint64_t core_start;
                uint64_t core_end;
                uint32_t executed_streams = 0U;
                struct qbh_w4_hmx_request request;

                qurt_sem_down(&state->expanded_ready[first_slot]);
                job->ready_wait_ticks +=
                    HAP_perf_get_qtimer_count() - wait_start;
                if (state->abort_status != 0) {
                    exit_status = state->abort_status;
                    goto unlock;
                }

                memset(&request, 0, sizeof(request));
                request.activation_tiles = state->activation_tiles;
                request.expanded_weight_tiles =
                    (const int8_t *)state->expanded_slots[first_slot];
                request.bias_words = (const uint32_t *)(
                    state->expanded_slots[first_slot] +
                    layout->expanded_chunk_weight_bytes);
                request.output_tiles = state->output_tiles +
                    (size_t)output_tile * QBH_HMX_OUTPUT_BYTES;
                request.chunk_tiles = first_chunk_tiles;
                request.begin_output = 1U;
                request.store_output = 1U;
                request.abort_status = &state->abort_status;
                request.ready_wait_ticks = &continuation_ready_wait;
                request.executed_stream_count = &executed_streams;
                request.continuation_chunk_count = 1U;
                request.continuation_chunks[0].activation_tiles =
                    state->activation_tiles +
                    (size_t)layout->chunk_tiles *
                        QBH_HMX_ACTIVATION_BYTES;
                request.continuation_chunks[0].expanded_weight_tiles =
                    (const int8_t *)state->expanded_slots[second_slot];
                request.continuation_chunks[0].chunk_tiles =
                    second_chunk_tiles;
                request.continuation_chunks[0].ready_semaphore =
                    &state->expanded_ready[second_slot];

                core_start = HAP_perf_get_qtimer_count();
                if (job->first_compute_start == 0U) {
                    job->first_compute_start = core_start;
                }
                qbh_hmx_region_begin(state);
                if (job->runner->submit(job->runner->context,
                                        &request) != 0) {
                    qbh_hmx_region_end(state);
                    exit_status = AEE_EFAILED;
                    qbh_abort_pipeline(state, exit_status);
                    goto unlock;
                }
                qbh_hmx_region_end(state);
                core_end = HAP_perf_get_qtimer_count();
                job->ready_wait_ticks += continuation_ready_wait;
                job->stream_count += executed_streams;
                job->execution_count +=
                    first_chunk_tiles + second_chunk_tiles;
                ++job->output_tile_count;
                job->last_compute_end = core_end;
                job->compute_ticks +=
                    core_end - core_start - continuation_ready_wait;

                qurt_sem_up(&state->expanded_free[first_slot]);
                qurt_sem_up(&state->expanded_free[second_slot]);
            }
        }
        goto unlock;
    }

    for (uint32_t repeat = 0; repeat < state->header->repeat_count;
         ++repeat) {
        for (uint32_t output_tile = 0; output_tile < layout->n_tiles;
             ++output_tile) {
            uint32_t linear_output = repeat * layout->n_tiles + output_tile;
            for (uint32_t chunk_index = 0;
                 chunk_index < layout->chunks_per_output; ++chunk_index) {
                uint32_t sequence =
                    linear_output * layout->chunks_per_output + chunk_index;
                uint32_t expanded_slot =
                    sequence % layout->expanded_slot_count;
                uint32_t chunk_tiles =
                    qbh_projection_chunk_tiles(layout, chunk_index);
                uint64_t wait_start = HAP_perf_get_qtimer_count();
                uint64_t core_start;
                uint64_t core_end;

                if (qbh_physical_plan_is_streaming(
                        layout->physical_plan)) {
                    uint32_t stream_regions =
                        qbh_stream_region_count(layout, chunk_index);
                    uint32_t generation = sequence + 1U;
                    uint32_t final_chunk =
                        chunk_index + 1U == layout->chunks_per_output;
                    uint32_t pair_lane = output_tile & 1U;
                    uint32_t pair_index = output_tile / 2U;
                    uint32_t pair_slot = 0U;
                    uint8_t *store_output = NULL;
                    uint64_t stream_ready_wait = 0U;
                    int32_t streams;

                    if (stream_regions == 0U ||
                        stream_regions > QBH_W4_MAX_STREAM_REGIONS ||
                        chunk_tiles != stream_regions *
                                           QBH_W4_STREAM_REGION_TILES) {
                        exit_status = AEE_EBADPARM;
                        state->abort_status = exit_status;
                        goto unlock;
                    }
                    if (final_chunk != 0U) {
                        if (state->mlp_handoff != NULL) {
                            pair_slot = pair_index %
                                state->mlp_handoff->pair_slot_count;
                            if (pair_lane == 0U) {
                                uint64_t pair_wait =
                                    HAP_perf_get_qtimer_count();
                                qurt_sem_down(
                                    &state->mlp_pair_free[pair_slot]);
                                state->header->producer_slot_wait_ticks +=
                                    HAP_perf_get_qtimer_count() - pair_wait;
                            }
                            store_output = state->output_tiles +
                                (size_t)pair_slot * 2U *
                                    QBH_HMX_OUTPUT_BYTES +
                                (size_t)pair_lane * QBH_HMX_OUTPUT_BYTES;
                        } else {
                            store_output = state->output_tiles +
                                (size_t)output_tile * QBH_HMX_OUTPUT_BYTES;
                        }
                    }
                    core_start = HAP_perf_get_qtimer_count();
                    if (job->first_compute_start == 0U) {
                        job->first_compute_start = core_start;
                    }
                    qbh_hmx_region_begin(state);
                    if (job->runner != NULL) {
                        uint32_t executed_streams = 0U;
                        struct qbh_w4_hmx_request request = {
                            .activation_tiles = state->activation_tiles +
                                (size_t)chunk_index * layout->chunk_tiles *
                                    QBH_HMX_ACTIVATION_BYTES,
                            .expanded_weight_tiles = (const int8_t *)state
                                ->expanded_slots[expanded_slot],
                            .bias_words = (const uint32_t *)(
                                state->expanded_slots[expanded_slot] +
                                layout->expanded_chunk_weight_bytes),
                            .output_tiles = store_output,
                            .chunk_tiles = chunk_tiles,
                            .begin_output = chunk_index == 0U,
                            .store_output = final_chunk,
                            .streaming = 1U,
                            .ready_generations =
                                state->stream_ready_generation[expanded_slot],
                            .expected_generation = generation,
                            .stream_count = stream_regions,
                            .abort_status = &state->abort_status,
                            .timeout_ticks = QBH_STREAM_READY_TIMEOUT_TICKS,
                            .ready_wait_ticks = &stream_ready_wait,
                            .hmx_consumption_started =
                                &state->streaming_hmx_consumption_started,
                            .executed_stream_count = &executed_streams,
                        };
                        streams = job->runner->submit(
                                      job->runner->context, &request) == 0
                                      ? (int32_t)executed_streams : -1;
                    } else {
                        streams = qbh_hmx_accumulate_u8s8_streaming(
                            state->activation_tiles +
                                (size_t)chunk_index * layout->chunk_tiles *
                                    QBH_HMX_ACTIVATION_BYTES,
                            (const int8_t *)state
                                ->expanded_slots[expanded_slot],
                            (const uint32_t *)(
                                state->expanded_slots[expanded_slot] +
                                layout->expanded_chunk_weight_bytes),
                            chunk_index == 0U,
                            state->stream_ready_generation[expanded_slot],
                            generation, stream_regions,
                            &state->abort_status,
                            QBH_STREAM_READY_TIMEOUT_TICKS,
                            &stream_ready_wait,
                            &state->streaming_hmx_consumption_started);
                        if (streams >= 0 && final_chunk != 0U) {
                            qbh_hmx_store_u8_output(store_output);
                        }
                    }
                    qbh_hmx_region_end(state);
                    core_end = HAP_perf_get_qtimer_count();
                    job->ready_wait_ticks += stream_ready_wait;
                    if (streams < 0) {
                        if (streams == -2) {
                            ++state->header
                                  ->streaming_ready_timeout_count;
                        }
                        exit_status = AEE_EFAILED;
                        qbh_abort_pipeline(state, exit_status);
                        goto unlock;
                    }
                    job->stream_count += (uint32_t)streams;
                    job->execution_count += chunk_tiles;
                    if (final_chunk != 0U) {
                        if (state->mlp_handoff != NULL) {
                            if (pair_lane != 0U) {
                                struct qbh_chunk_task activation_task;
                                memset(&activation_task, 0,
                                       sizeof(activation_task));
                                activation_task.mlp_activation = 1U;
                                activation_task.mlp_pair_index = pair_index;
                                activation_task.mlp_pair_slot = pair_slot;
                                qbh_queue_push(&state->queue,
                                               &activation_task);
                                ++*state->mlp_handoff->pair_publish_count;
                            }
                        }
                    }
                    job->last_compute_end = core_end;
                    job->compute_ticks +=
                        core_end - core_start - stream_ready_wait;
                    qurt_sem_up(
                        &state->expanded_free[expanded_slot]);
                    continue;
                }

                qurt_sem_down(&state->expanded_ready[expanded_slot]);
                job->ready_wait_ticks +=
                    HAP_perf_get_qtimer_count() - wait_start;
                if (state->abort_status != 0) {
                    exit_status = state->abort_status;
                    goto unlock;
                }

                core_start = HAP_perf_get_qtimer_count();
                if (job->first_compute_start == 0U) {
                    job->first_compute_start = core_start;
                }
                qbh_hmx_region_begin(state);
                if (job->runner != NULL) {
                    uint32_t executed_streams = 0U;
                    struct qbh_w4_hmx_request request = {
                        .activation_tiles = state->activation_tiles +
                            (size_t)chunk_index * layout->chunk_tiles *
                                QBH_HMX_ACTIVATION_BYTES,
                        .expanded_weight_tiles = (const int8_t *)
                            state->expanded_slots[expanded_slot],
                        .bias_words = (const uint32_t *)(
                            state->expanded_slots[expanded_slot] +
                            layout->expanded_chunk_weight_bytes),
                        .output_tiles =
                            chunk_index + 1U == layout->chunks_per_output
                                ? state->output_tiles +
                                      (size_t)output_tile *
                                          QBH_HMX_OUTPUT_BYTES
                                : NULL,
                        .chunk_tiles = chunk_tiles,
                        .begin_output = chunk_index == 0U,
                        .store_output =
                            chunk_index + 1U == layout->chunks_per_output,
                        .executed_stream_count = &executed_streams,
                    };
                    if (job->runner->submit(job->runner->context,
                                            &request) != 0) {
                        exit_status = AEE_EFAILED;
                        qbh_abort_pipeline(state, exit_status);
                        goto unlock;
                    }
                    job->stream_count += executed_streams;
                } else {
                    if (chunk_index == 0U) {
                        qbh_hmx_begin_u8s8_output((const uint32_t *)(
                            state->expanded_slots[expanded_slot] +
                            layout->expanded_chunk_weight_bytes));
                    }
                    job->stream_count += qbh_hmx_accumulate_u8s8_projection(
                        state->activation_tiles +
                            (size_t)chunk_index * layout->chunk_tiles *
                                QBH_HMX_ACTIVATION_BYTES,
                        (const int8_t *)state->expanded_slots[expanded_slot],
                        chunk_tiles);
                }
                job->execution_count += chunk_tiles;
                if (chunk_index + 1U == layout->chunks_per_output) {
                    if (state->mlp_handoff != NULL) {
                        exit_status = AEE_EBADPARM;
                        state->abort_status = exit_status;
                        goto unlock;
                    }
                    if (job->runner == NULL) {
                        qbh_hmx_store_u8_output(
                            state->output_tiles +
                            (size_t)output_tile * QBH_HMX_OUTPUT_BYTES);
                    }
                }
                qbh_hmx_region_end(state);
                core_end = HAP_perf_get_qtimer_count();
                job->last_compute_end = core_end;
                job->compute_ticks += core_end - core_start;

                qurt_sem_up(&state->expanded_free[expanded_slot]);
            }
            ++job->output_tile_count;
        }
    }

unlock:
    job->unlock_status = job->runner == NULL
        ? HAP_compute_res_hmx_unlock2(
              job->hmx_context_id, HAP_COMPUTE_RES_HMX_SHARED)
        : AEE_SUCCESS;
    if (exit_status == AEE_SUCCESS &&
        job->unlock_status != AEE_SUCCESS) {
        exit_status = job->unlock_status;
    }
    qurt_thread_exit(exit_status);
}

static void qbh_abort_pipeline(struct qbh_parallel_state *state,
                               int32_t status) {
    const struct qbh_projection_layout *layout = state->layout;
    state->abort_status = status != 0 ? status : AEE_EFAILED;
    for (uint32_t slot = 0;
         slot < layout->expanded_slot_count; ++slot) {
        qurt_sem_up(&state->expanded_ready[slot]);
        qurt_sem_up(&state->expanded_free[slot]);
    }
    for (uint32_t slot = 0;
         slot < layout->compressed_slot_count; ++slot) {
        qurt_sem_up(&state->compressed_free[slot]);
    }
}

static void qbh_publish_w4_bundle(
    struct qbh_parallel_state *state, uint32_t linear_output,
    uint32_t compressed_slot) {
    const struct qbh_projection_layout *layout = state->layout;
    struct qbh_probe_header *header = state->header;

    state->compressed_remaining[compressed_slot] =
        qbh_physical_plan_is_streaming(layout->physical_plan)
            ? layout->k_tiles / QBH_W4_STREAM_REGION_TILES
            : layout->chunks_per_output;
    for (uint32_t chunk_index = 0;
         chunk_index < layout->chunks_per_output; ++chunk_index) {
        uint32_t sequence =
            linear_output * layout->chunks_per_output + chunk_index;
        uint32_t expanded_slot =
            sequence % layout->expanded_slot_count;
        uint64_t wait_start = HAP_perf_get_qtimer_count();

        qurt_sem_down(&state->expanded_free[expanded_slot]);
        header->expanded_slot_wait_ticks +=
            HAP_perf_get_qtimer_count() - wait_start;
        if (sequence >= layout->expanded_slot_count) {
            ++header->expanded_chunk_slot_reuse_count;
        }
        if (qbh_physical_plan_is_streaming(layout->physical_plan)) {
            uint32_t regions =
                qbh_stream_region_count(layout, chunk_index);
            uint32_t generation = sequence + 1U;
            for (uint32_t region = 0;
                 region < QBH_W4_MAX_STREAM_REGIONS; ++region) {
                state->stream_ready_generation[expanded_slot][region] = 0U;
            }
            asm volatile("barrier" : : : "memory");
            for (uint32_t region = 0; region < regions; ++region) {
                struct qbh_chunk_task task;
                memset(&task, 0, sizeof(task));
                task.sequence = sequence;
                task.compressed_slot = compressed_slot;
                task.expanded_slot = expanded_slot;
                task.chunk_index = chunk_index;
                task.chunk_tiles =
                    qbh_projection_chunk_tiles(layout, chunk_index);
                task.stream_region_index = region;
                task.stream_region_tiles =
                    QBH_W4_STREAM_REGION_TILES;
                task.stream_generation = generation;
                qbh_queue_push(&state->queue, &task);
            }
        } else {
            struct qbh_chunk_task task;
            memset(&task, 0, sizeof(task));
            task.sequence = sequence;
            task.compressed_slot = compressed_slot;
            task.expanded_slot = expanded_slot;
            task.chunk_index = chunk_index;
            task.chunk_tiles =
                qbh_projection_chunk_tiles(layout, chunk_index);
            qbh_queue_push(&state->queue, &task);
        }
    }
}

static int qbh_run_chunked_w4_pipeline_impl(
    struct qbh_probe_header *header,
    const struct qbh_projection_layout *layout,
    const uint8_t *stored_weights, const uint8_t *activation_tiles,
    uint8_t *vtcm, uint32_t hmx_context_id,
    const struct qbh_mlp_gate_up_handoff *handoff,
    const struct qbh_w4_hmx_runner *runner,
    const struct qbh_w4_hvx_dispatch_runner *hvx_runner) {
    struct qbh_parallel_state state;
    struct qbh_hvx_worker_job hvx_jobs[QBH_MAX_HVX_WORKERS];
    struct qbh_chunked_hmx_job hmx_job;
    qurt_thread_t hvx_threads[QBH_MAX_HVX_WORKERS];
    qurt_thread_t hmx_thread = 0;
    uint32_t created_hvx_workers = 0;
    uint32_t joined_hvx_workers = 0;
    uint32_t successful_hvx_locks = 0;
    uint32_t dma_bundle_batch;
    int hmx_thread_created = 0;
    int hmx_thread_joined = 0;
    int external_hvx_started = 0;
    int hmx_exit_status = 0;
    int result = AEE_SUCCESS;

    if (header == NULL || layout == NULL || stored_weights == NULL ||
        activation_tiles == NULL || vtcm == NULL ||
        (hmx_context_id == 0U && runner == NULL) ||
        (runner != NULL && runner->submit == NULL) ||
        (hvx_runner != NULL &&
         (hvx_runner->start == NULL || hvx_runner->wait == NULL ||
          hvx_runner->max_workers < header->requested_hvx_workers)) ||
        !qbh_physical_plan_is_chunked(layout->physical_plan) ||
        (layout->weight_storage_variant != QBH_WEIGHT_EXPANDED_S8 &&
         !qbh_weight_storage_is_packed_w4(
             layout->weight_storage_variant)) ||
        header->requested_hvx_workers == 0U ||
        header->requested_hvx_workers > QBH_MAX_HVX_WORKERS) {
        return AEE_EBADPARM;
    }
    if (handoff != NULL &&
        (layout->variant != QBH_PROJECTION_GATE_UP_PAIR ||
         !qbh_physical_plan_is_streaming(layout->physical_plan) ||
         handoff->middle_activation == NULL ||
         handoff->activation_lut == NULL ||
         handoff->activation_gather_scratch == NULL ||
         handoff->pair_publish_count == NULL ||
         handoff->pair_consume_count == NULL ||
         handoff->activation_ticks == NULL ||
         (handoff->activation_elements != QBH_MLP_HVX_VECTOR_BYTES &&
          handoff->activation_elements != QBH_HMX_OUTPUT_BYTES) ||
         (handoff->verify_activation_elements != 0U &&
          handoff->activation_mismatch_count == NULL) ||
         handoff->pair_slot_count == 0U ||
         handoff->pair_slot_count > QBH_W4_MAX_COMPRESSED_SLOT_COUNT ||
         (layout->n_tiles & 1U) != 0U)) {
        return AEE_EBADPARM;
    }
    if (qbh_physical_plan_is_streaming(layout->physical_plan) &&
        ((layout->chunk_tiles % QBH_W4_STREAM_REGION_TILES) != 0U ||
         (layout->k_tiles % QBH_W4_STREAM_REGION_TILES) != 0U ||
         layout->chunk_tiles / QBH_W4_STREAM_REGION_TILES >
             QBH_W4_MAX_STREAM_REGIONS)) {
        return AEE_EBADPARM;
    }
    dma_bundle_batch = qbh_physical_plan_dma_bundle_batch(
        layout->physical_plan);
    if (dma_bundle_batch > layout->compressed_slot_count ||
        layout->compressed_slot_count % dma_bundle_batch != 0U ||
        layout->n_tiles % layout->compressed_slot_count != 0U) {
        return AEE_EBADPARM;
    }

    memset(&state, 0, sizeof(state));
    memset(hvx_jobs, 0, sizeof(hvx_jobs));
    memset(&hmx_job, 0, sizeof(hmx_job));
    memset(hvx_threads, 0, sizeof(hvx_threads));
    state.header = header;
    state.layout = layout;
    state.mlp_handoff = handoff;
    state.activation_tiles = activation_tiles;
    state.output_tiles = vtcm + layout->vtcm_output_offset;
    for (uint32_t slot = 0; slot < layout->compressed_slot_count;
         ++slot) {
        state.compressed_slots[slot] =
            vtcm + qbh_projection_compressed_slot_offset(layout, slot);
    }
    for (uint32_t slot = 0;
         slot < layout->expanded_slot_count; ++slot) {
        state.expanded_slots[slot] =
            vtcm + qbh_projection_expanded_chunk_offset(layout, slot);
    }
    if (handoff != NULL) {
        for (uint32_t slot = 0; slot < handoff->pair_slot_count; ++slot) {
            qurt_sem_init_val(&state.mlp_pair_free[slot], 1);
        }
    }

    qbh_queue_init(&state.queue);
    qurt_mutex_init(&state.metrics_mutex);
    qurt_sem_init_val(&state.hmx_started, 0);
    qurt_sem_init_val(&state.hvx_started, 0);
    for (uint32_t slot = 0; slot < layout->compressed_slot_count;
         ++slot) {
        qurt_sem_init_val(&state.compressed_free[slot], 1);
    }
    for (uint32_t slot = 0;
         slot < layout->expanded_slot_count; ++slot) {
        qurt_sem_init_val(&state.expanded_free[slot], 1);
        qurt_sem_init_val(&state.expanded_ready[slot], 0);
    }

    hmx_job.state = &state;
    hmx_job.hmx_context_id = hmx_context_id;
    hmx_job.runner = runner;
    qurt_thread_attr_t hmx_attributes;
    qurt_thread_attr_init(&hmx_attributes);
    qurt_thread_attr_set_name(&hmx_attributes, "qbh-hmx-chunk");
    qurt_thread_attr_set_stack_addr(&hmx_attributes,
                                    qbh_chunked_hmx_stack);
    qurt_thread_attr_set_stack_size(&hmx_attributes,
                                    QBH_CHUNKED_HMX_STACK_BYTES);
    qurt_thread_attr_set_priority(
        &hmx_attributes,
        qurt_thread_get_priority(qurt_thread_get_id()));
    qurt_thread_attr_set_detachstate(&hmx_attributes,
                                     QURT_THREAD_ATTR_CREATE_JOINABLE);
    header->hmx_thread_create_status = qurt_thread_create(
        &hmx_thread, &hmx_attributes, qbh_chunked_hmx_main, &hmx_job);
    if (header->hmx_thread_create_status != QURT_EOK) {
        result = AEE_EFAILED;
        goto cleanup;
    }
    hmx_thread_created = 1;

    for (uint32_t worker = 0; worker < header->requested_hvx_workers;
         ++worker) {
        hvx_jobs[worker].state = &state;
        hvx_jobs[worker].worker_index = worker;
    }
    if (hvx_runner != NULL) {
        void *worker_contexts[QBH_MAX_HVX_WORKERS];
        for (uint32_t worker = 0;
             worker < header->requested_hvx_workers; ++worker) {
            worker_contexts[worker] = &hvx_jobs[worker];
        }
        if (hvx_runner->start(
                hvx_runner->context, worker_contexts,
                header->requested_hvx_workers) != AEE_SUCCESS) {
            result = AEE_EFAILED;
        } else {
            created_hvx_workers = header->requested_hvx_workers;
            external_hvx_started = 1;
        }
    } else {
        for (uint32_t worker = 0;
             worker < header->requested_hvx_workers; ++worker) {
            qurt_thread_attr_t attributes;
            char name[16] = "qbh-hvx0";
            name[7] = (char)('0' + worker);
            qurt_thread_attr_init(&attributes);
            qurt_thread_attr_set_name(&attributes, name);
            qurt_thread_attr_set_stack_addr(
                &attributes, qbh_hvx_worker_stacks[worker]);
            qurt_thread_attr_set_stack_size(
                &attributes, QBH_HVX_WORKER_STACK_BYTES);
            qurt_thread_attr_set_priority(
                &attributes,
                qurt_thread_get_priority(qurt_thread_get_id()));
            qurt_thread_attr_set_detachstate(
                &attributes, QURT_THREAD_ATTR_CREATE_JOINABLE);
            int create_status = qurt_thread_create(
                &hvx_threads[worker], &attributes,
                qbh_hvx_worker_main, &hvx_jobs[worker]);
            if (create_status != QURT_EOK) {
                header->hvx_thread_create_status = create_status;
                result = AEE_EFAILED;
                break;
            }
            ++created_hvx_workers;
        }
    }
    header->hvx_workers_created = created_hvx_workers;

    qurt_sem_down(&state.hmx_started);
    for (uint32_t worker = 0; worker < created_hvx_workers; ++worker) {
        qurt_sem_down(&state.hvx_started);
        if (hvx_jobs[worker].lock_status == AEE_SUCCESS) {
            ++successful_hvx_locks;
        }
    }
    header->hvx_workers_locked = successful_hvx_locks;
    if (result != AEE_SUCCESS || hmx_job.lock_status != AEE_SUCCESS ||
        successful_hvx_locks != header->requested_hvx_workers) {
        result = AEE_EFAILED;
        qbh_abort_pipeline(&state, result);
        goto stop_workers;
    }

    header->qtimer_start = HAP_perf_get_qtimer_count();
    header->pcycles_start = HAP_perf_get_pcycles();
    for (uint32_t repeat = 0; repeat < header->repeat_count; ++repeat) {
        for (uint32_t output_base = 0; output_base < layout->n_tiles;
             output_base += dma_bundle_batch) {
            struct qbh_dma_aligned_desc_1d linked_descriptors[4];
            uint32_t linear_base = repeat * layout->n_tiles + output_base;
            uint32_t first_compressed_slot =
                linear_base % layout->compressed_slot_count;

            if (state.abort_status != 0) {
                result = state.abort_status;
                goto stop_workers;
            }

            for (uint32_t batch_index = 0;
                 batch_index < dma_bundle_batch; ++batch_index) {
                uint32_t linear_output = linear_base + batch_index;
                uint32_t compressed_slot =
                    first_compressed_slot + batch_index;
                uint64_t wait_start = HAP_perf_get_qtimer_count();
                qurt_sem_down(&state.compressed_free[compressed_slot]);
                header->producer_slot_wait_ticks +=
                    HAP_perf_get_qtimer_count() - wait_start;
                if (state.abort_status != 0) {
                    result = state.abort_status;
                    goto stop_workers;
                }
                if (linear_output >= layout->compressed_slot_count) {
                    ++header->weight_slot_reuse_count;
                }
            }

            if (qbh_physical_plan_uses_linked_dma(
                    layout->physical_plan)) {
                if (qbh_start_linked_weight_bundles(
                        header, linked_descriptors,
                        stored_weights + (size_t)output_base *
                            layout->stored_weight_bundle_bytes,
                        state.compressed_slots[first_compressed_slot],
                        layout->stored_weight_bundle_bytes,
                        dma_bundle_batch) != 0) {
                    result = AEE_EFAILED;
                    qbh_abort_pipeline(&state, result);
                    goto stop_workers;
                }
                for (uint32_t batch_index = 0;
                     batch_index < dma_bundle_batch; ++batch_index) {
                    uint64_t wait_start = HAP_perf_get_qtimer_count();
                    if (qbh_wait_linked_descriptor(
                            header,
                            &linked_descriptors[batch_index].descriptor) !=
                        0) {
                        result = AEE_EFAILED;
                        qbh_abort_pipeline(&state, result);
                        goto stop_workers;
                    }
                    header->weight_stage_ticks +=
                        HAP_perf_get_qtimer_count() - wait_start;
                    ++header->weight_bundle_stage_count;
                    qbh_publish_w4_bundle(
                        &state, linear_base + batch_index,
                        first_compressed_slot + batch_index);
                }
                if (qbh_record_dma_wait(header) != 0) {
                    result = AEE_EFAILED;
                    qbh_abort_pipeline(&state, result);
                    goto stop_workers;
                }
            } else {
                uint64_t stage_start = HAP_perf_get_qtimer_count();
                if (qbh_stage_weight_bundles(
                        header,
                        stored_weights + (size_t)output_base *
                            layout->stored_weight_bundle_bytes,
                        state.compressed_slots[first_compressed_slot],
                        layout->stored_weight_bundle_bytes,
                        dma_bundle_batch) != 0) {
                    result = AEE_EFAILED;
                    qbh_abort_pipeline(&state, result);
                    goto stop_workers;
                }
                header->weight_stage_ticks +=
                    HAP_perf_get_qtimer_count() - stage_start;
                for (uint32_t batch_index = 0;
                     batch_index < dma_bundle_batch; ++batch_index) {
                    qbh_publish_w4_bundle(
                        &state, linear_base + batch_index,
                        first_compressed_slot + batch_index);
                }
            }
        }
    }

stop_workers:
    if (handoff != NULL && result == AEE_SUCCESS &&
        hmx_thread_created && !hmx_thread_joined) {
        header->hmx_thread_join_status =
            qurt_thread_join(hmx_thread, &hmx_exit_status);
        hmx_thread_joined = 1;
    }
    for (uint32_t worker = 0; worker < created_hvx_workers; ++worker) {
        struct qbh_chunk_task stop_task;
        memset(&stop_task, 0, sizeof(stop_task));
        stop_task.stop = 1U;
        qbh_queue_push(&state.queue, &stop_task);
    }

    if (hmx_thread_created && !hmx_thread_joined) {
        header->hmx_thread_join_status =
            qurt_thread_join(hmx_thread, &hmx_exit_status);
        hmx_thread_joined = 1;
    }
    if (external_hvx_started != 0) {
        int wait_status = hvx_runner->wait(
            hvx_runner->context, created_hvx_workers);
        joined_hvx_workers = created_hvx_workers;
        if (wait_status != AEE_SUCCESS) {
            header->hvx_thread_join_status = wait_status;
        }
    } else {
        for (uint32_t worker = 0; worker < created_hvx_workers; ++worker) {
            int exit_status = 0;
            int join_status =
                qurt_thread_join(hvx_threads[worker], &exit_status);
            ++joined_hvx_workers;
            if ((join_status != QURT_EOK ||
                 exit_status != AEE_SUCCESS) &&
                header->hvx_thread_join_status == 0) {
                header->hvx_thread_join_status =
                    join_status != QURT_EOK
                        ? join_status : exit_status;
            }
        }
    }
    header->pcycles_end = HAP_perf_get_pcycles();
    header->qtimer_end = HAP_perf_get_qtimer_count();
    if (header->qtimer_start != 0U) {
        header->pipeline_ticks =
            header->qtimer_end - header->qtimer_start;
        header->qtimer_elapsed = header->pipeline_ticks;
    }

    header->hmx_lock_status = hmx_job.lock_status;
    header->hmx_unlock_status = hmx_job.unlock_status;
    header->hmx_execution_count = hmx_job.execution_count;
    header->hmx_stream_count = hmx_job.stream_count;
    header->output_tile_count = hmx_job.output_tile_count;
    header->hmx_compute_ticks = hmx_job.compute_ticks;
    header->hmx_ready_wait_ticks = hmx_job.ready_wait_ticks;
    header->hmx_batch_output_count =
        hmx_job.max_batch_output_count;
    header->hmx_in_command_slot_release_count =
        hmx_job.in_command_slot_release_count;
    header->hmx_producer_progress_command_count =
        hmx_job.producer_progress_command_count;
    header->hmx_window_start = hmx_job.first_compute_start;
    header->hmx_window_end = hmx_job.last_compute_end;
    header->hvx_max_active_workers = state.max_active_hvx_workers;
    header->hvx_hmx_overlap_observed =
        state.hvx_hmx_overlap_observed;
    header->hvx_parallel_overlap_observed =
        state.hvx_parallel_overlap_observed;

    for (uint32_t worker = 0; worker < created_hvx_workers; ++worker) {
        header->hvx_worker_lock_status[worker] =
            hvx_jobs[worker].lock_status;
        header->hvx_worker_unlock_status[worker] =
            hvx_jobs[worker].unlock_status;
        header->hvx_worker_ticks[worker] = hvx_jobs[worker].expand_ticks;
        header->weight_expand_ticks += hvx_jobs[worker].expand_ticks;
        header->weight_expand_count += hvx_jobs[worker].expand_count;
        header->chunk_expand_count += hvx_jobs[worker].expand_count;
        if (hvx_jobs[worker].first_expand_start != 0U &&
            (header->expand_window_start == 0U ||
             hvx_jobs[worker].first_expand_start <
                 header->expand_window_start)) {
            header->expand_window_start =
                hvx_jobs[worker].first_expand_start;
        }
        if (hvx_jobs[worker].last_expand_end >
            header->expand_window_end) {
            header->expand_window_end =
                hvx_jobs[worker].last_expand_end;
        }
        if (header->hvx_lock_status == AEE_SUCCESS &&
            hvx_jobs[worker].lock_status != AEE_SUCCESS) {
            header->hvx_lock_status = hvx_jobs[worker].lock_status;
        }
        if (header->hvx_unlock_status == AEE_SUCCESS &&
            hvx_jobs[worker].unlock_status != AEE_SUCCESS) {
            header->hvx_unlock_status = hvx_jobs[worker].unlock_status;
        }
    }

    if (result == AEE_SUCCESS &&
        (header->hmx_thread_join_status != QURT_EOK ||
         hmx_exit_status != AEE_SUCCESS ||
         header->hmx_lock_status != AEE_SUCCESS ||
         header->hmx_unlock_status != AEE_SUCCESS ||
         header->hvx_thread_join_status != 0 ||
         joined_hvx_workers != header->requested_hvx_workers ||
         (handoff != NULL &&
          (*handoff->pair_publish_count != layout->n_tiles / 2U ||
           *handoff->pair_consume_count != layout->n_tiles / 2U)) ||
         header->weight_expand_count !=
             header->repeat_count * layout->n_tiles *
                 (qbh_physical_plan_is_streaming(
                      layout->physical_plan)
                      ? layout->k_tiles /
                            QBH_W4_STREAM_REGION_TILES
                      : layout->chunks_per_output))) {
        result = AEE_EFAILED;
    }

cleanup:
    if (hmx_thread_created && !hmx_thread_joined) {
        qbh_abort_pipeline(&state, AEE_EFAILED);
        header->hmx_thread_join_status =
            qurt_thread_join(hmx_thread, &hmx_exit_status);
    }
    if (external_hvx_started != 0 &&
        joined_hvx_workers < created_hvx_workers) {
        for (uint32_t worker = joined_hvx_workers;
             worker < created_hvx_workers; ++worker) {
            struct qbh_chunk_task stop_task;
            memset(&stop_task, 0, sizeof(stop_task));
            stop_task.stop = 1U;
            qbh_queue_push(&state.queue, &stop_task);
        }
        (void)hvx_runner->wait(
            hvx_runner->context, created_hvx_workers);
    } else {
        for (uint32_t worker = joined_hvx_workers;
             worker < created_hvx_workers; ++worker) {
            int exit_status = 0;
            struct qbh_chunk_task stop_task;
            memset(&stop_task, 0, sizeof(stop_task));
            stop_task.stop = 1U;
            qbh_queue_push(&state.queue, &stop_task);
            (void)qurt_thread_join(
                hvx_threads[worker], &exit_status);
        }
    }

    qurt_sem_destroy(&state.hvx_started);
    qurt_sem_destroy(&state.hmx_started);
    for (uint32_t slot = 0;
         slot < layout->expanded_slot_count; ++slot) {
        qurt_sem_destroy(&state.expanded_ready[slot]);
        qurt_sem_destroy(&state.expanded_free[slot]);
    }
    for (uint32_t slot = 0; slot < layout->compressed_slot_count;
         ++slot) {
        qurt_sem_destroy(&state.compressed_free[slot]);
    }
    if (handoff != NULL) {
        for (uint32_t slot = 0; slot < handoff->pair_slot_count; ++slot) {
            qurt_sem_destroy(&state.mlp_pair_free[slot]);
        }
    }
    qurt_mutex_destroy(&state.metrics_mutex);
    qbh_queue_destroy(&state.queue);
    return result;
}

int qbh_run_chunked_w4_pipeline(
    struct qbh_probe_header *header,
    const struct qbh_projection_layout *layout,
    const uint8_t *stored_weights, const uint8_t *activation_tiles,
    uint8_t *vtcm, uint32_t hmx_context_id) {
    return qbh_run_chunked_w4_pipeline_impl(
        header, layout, stored_weights, activation_tiles, vtcm,
        hmx_context_id, NULL, NULL, NULL);
}

int qbh_run_chunked_w4_pipeline_mlp(
    struct qbh_probe_header *header,
    const struct qbh_projection_layout *layout,
    const uint8_t *stored_weights, const uint8_t *activation_tiles,
    uint8_t *vtcm, uint32_t hmx_context_id,
    const struct qbh_mlp_gate_up_handoff *handoff) {
    return qbh_run_chunked_w4_pipeline_impl(
        header, layout, stored_weights, activation_tiles, vtcm,
        hmx_context_id, handoff, NULL, NULL);
}

int qbh_run_chunked_w4_pipeline_external(
    struct qbh_probe_header *header,
    const struct qbh_projection_layout *layout,
    const uint8_t *stored_weights, const uint8_t *activation_tiles,
    uint8_t *vtcm,
    const struct qbh_mlp_gate_up_handoff *handoff,
    const struct qbh_w4_hmx_runner *runner) {
    return qbh_run_chunked_w4_pipeline_impl(
        header, layout, stored_weights, activation_tiles, vtcm,
        0U, handoff, runner, NULL);
}

int qbh_run_chunked_w4_pipeline_external_hvx(
    struct qbh_probe_header *header,
    const struct qbh_projection_layout *layout,
    const uint8_t *stored_weights, const uint8_t *activation_tiles,
    uint8_t *vtcm,
    const struct qbh_mlp_gate_up_handoff *handoff,
    const struct qbh_w4_hmx_runner *hmx_runner,
    const struct qbh_w4_hvx_dispatch_runner *hvx_runner) {
    return qbh_run_chunked_w4_pipeline_impl(
        header, layout, stored_weights, activation_tiles, vtcm,
        0U, handoff, hmx_runner, hvx_runner);
}
