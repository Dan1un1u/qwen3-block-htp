#include <AEEStdErr.h>
#include <HAP_compute_res.h>
#include <HAP_perf.h>
#include <qurt.h>
#include <qurt_hvx.h>
#include <stdint.h>
#include <string.h>

#include "hmx_u8s8_projection.h"
#include "probe_protocol.h"
#include "qbh_user_dma.h"
#include "w4_parallel_pipeline.h"
#include "w4_u8_expand.h"

#define QBH_HVX_WORKER_STACK_BYTES UINT32_C(8192)
#define QBH_CHUNKED_HMX_STACK_BYTES UINT32_C(16384)
#define QBH_DMA_DESCRIPTOR_TIMEOUT_TICKS UINT64_C(1920000)

static uint8_t qbh_hvx_worker_stacks[QBH_MAX_HVX_WORKERS]
                                    [QBH_HVX_WORKER_STACK_BYTES]
    __attribute__((aligned(128)));
static uint8_t qbh_chunked_hmx_stack[QBH_CHUNKED_HMX_STACK_BYTES]
    __attribute__((aligned(128)));

struct qbh_chunk_task {
    uint32_t stop;
    uint32_t sequence;
    uint32_t compressed_slot;
    uint32_t expanded_slot;
    uint32_t chunk_index;
    uint32_t chunk_tiles;
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
    uint8_t *expanded_slots[QBH_W4_EXPANDED_CHUNK_SLOT_COUNT];
    uint8_t *output_tiles;

    struct qbh_chunk_queue queue;
    qurt_sem_t compressed_free[QBH_W4_MAX_COMPRESSED_SLOT_COUNT];
    qurt_sem_t expanded_free[QBH_W4_EXPANDED_CHUNK_SLOT_COUNT];
    qurt_sem_t expanded_ready[QBH_W4_EXPANDED_CHUNK_SLOT_COUNT];
    qurt_sem_t hmx_started;
    qurt_sem_t hvx_started;
    qurt_mutex_t metrics_mutex;

    volatile uint32_t
        compressed_remaining[QBH_W4_MAX_COMPRESSED_SLOT_COUNT];
    volatile uint32_t active_hvx_workers;
    uint32_t max_active_hvx_workers;
    uint32_t hmx_active;
    uint32_t hvx_hmx_overlap_observed;
    uint32_t hvx_parallel_overlap_observed;
    volatile int32_t abort_status;
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
    int32_t lock_status;
    int32_t unlock_status;
    uint32_t execution_count;
    uint32_t stream_count;
    uint32_t output_tile_count;
    uint64_t compute_ticks;
    uint64_t ready_wait_ticks;
    uint64_t first_compute_start;
    uint64_t last_compute_end;
};

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

static void qbh_hvx_region_begin(struct qbh_parallel_state *state) {
    uint32_t active = qbh_atomic_inc_return(&state->active_hvx_workers);
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

static void qbh_hvx_worker_main(void *opaque) {
    struct qbh_hvx_worker_job *job =
        (struct qbh_hvx_worker_job *)opaque;
    struct qbh_parallel_state *state = job->state;
    const struct qbh_projection_layout *layout = state->layout;
    int exit_status = AEE_SUCCESS;

    job->lock_status = qurt_hvx_lock(QURT_HVX_MODE_128B);
    qurt_sem_up(&state->hvx_started);
    if (job->lock_status != AEE_SUCCESS) {
        qurt_thread_exit(job->lock_status);
    }

    for (;;) {
        struct qbh_chunk_task task;
        uint64_t start;
        uint64_t end;
        const uint8_t *compressed;
        uint8_t *expanded;

        qbh_queue_pop(&state->queue, &task);
        if (task.stop != 0U) {
            break;
        }

        compressed = state->compressed_slots[task.compressed_slot];
        expanded = state->expanded_slots[task.expanded_slot];
        start = HAP_perf_get_qtimer_count();
        if (job->first_expand_start == 0U) {
            job->first_expand_start = start;
        }
        qbh_hvx_region_begin(state);
        if (layout->weight_storage_variant ==
            QBH_WEIGHT_PACKED_W4_HMX_SCALE) {
            qbh_unpack_w4_to_s8_hvx(
                compressed +
                    (size_t)task.chunk_index * layout->chunk_tiles *
                        QBH_W4_PACKED_TILE_BYTES,
                (int8_t *)expanded, task.chunk_tiles);
        } else {
            qbh_expand_w4_to_s8_hvx(
                compressed +
                    (size_t)task.chunk_index * layout->chunk_tiles *
                        QBH_W4_PACKED_TILE_BYTES,
                compressed + layout->w4_scale_offset,
                (int8_t *)expanded, task.chunk_tiles);
        }
        if (task.chunk_index == 0U) {
            qbh_copy_hmx_bias_hvx(
                compressed + layout->w4_bias_offset,
                expanded + layout->expanded_chunk_weight_bytes);
        }
        qbh_hvx_region_end(state);
        end = HAP_perf_get_qtimer_count();
        job->last_expand_end = end;
        job->expand_ticks += end - start;
        ++job->expand_count;

        asm volatile("barrier" : : : "memory");
        qurt_sem_up(&state->expanded_ready[task.expanded_slot]);
        if (qbh_atomic_dec_return(
                &state->compressed_remaining[task.compressed_slot]) == 0U) {
            qurt_sem_up(&state->compressed_free[task.compressed_slot]);
        }
    }

    job->unlock_status = qurt_hvx_unlock();
    if (job->unlock_status != AEE_SUCCESS) {
        exit_status = job->unlock_status;
    }
    qurt_thread_exit(exit_status);
}

static void qbh_chunked_hmx_main(void *opaque) {
    struct qbh_chunked_hmx_job *job =
        (struct qbh_chunked_hmx_job *)opaque;
    struct qbh_parallel_state *state = job->state;
    const struct qbh_projection_layout *layout = state->layout;
    int exit_status = AEE_SUCCESS;

    job->lock_status = HAP_compute_res_hmx_lock2(
        job->hmx_context_id, HAP_COMPUTE_RES_HMX_SHARED);
    qurt_sem_up(&state->hmx_started);
    if (job->lock_status != AEE_SUCCESS) {
        qurt_thread_exit(job->lock_status);
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
                job->execution_count += chunk_tiles;
                if (chunk_index + 1U == layout->chunks_per_output) {
                    qbh_hmx_store_u8_output(
                        state->output_tiles +
                        (size_t)output_tile * QBH_HMX_OUTPUT_BYTES);
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
    job->unlock_status = HAP_compute_res_hmx_unlock2(
        job->hmx_context_id, HAP_COMPUTE_RES_HMX_SHARED);
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
    }
}

static void qbh_publish_w4_bundle(
    struct qbh_parallel_state *state, uint32_t linear_output,
    uint32_t compressed_slot) {
    const struct qbh_projection_layout *layout = state->layout;
    struct qbh_probe_header *header = state->header;

    state->compressed_remaining[compressed_slot] =
        layout->chunks_per_output;
    for (uint32_t chunk_index = 0;
         chunk_index < layout->chunks_per_output; ++chunk_index) {
        struct qbh_chunk_task task;
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

int qbh_run_chunked_w4_pipeline(
    struct qbh_probe_header *header,
    const struct qbh_projection_layout *layout,
    const uint8_t *stored_weights, const uint8_t *activation_tiles,
    uint8_t *vtcm, uint32_t hmx_context_id) {
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
    int hmx_exit_status = 0;
    int result = AEE_SUCCESS;

    if (header == NULL || layout == NULL || stored_weights == NULL ||
        activation_tiles == NULL || vtcm == NULL || hmx_context_id == 0U ||
        !qbh_physical_plan_is_chunked(layout->physical_plan) ||
        !qbh_weight_storage_is_packed_w4(
            layout->weight_storage_variant) ||
        header->requested_hvx_workers == 0U ||
        header->requested_hvx_workers > QBH_MAX_HVX_WORKERS) {
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
        qurt_thread_attr_t attributes;
        char name[16] = "qbh-hvx0";
        name[7] = (char)('0' + worker);
        hvx_jobs[worker].state = &state;
        hvx_jobs[worker].worker_index = worker;
        qurt_thread_attr_init(&attributes);
        qurt_thread_attr_set_name(&attributes, name);
        qurt_thread_attr_set_stack_addr(
            &attributes, qbh_hvx_worker_stacks[worker]);
        qurt_thread_attr_set_stack_size(&attributes,
                                        QBH_HVX_WORKER_STACK_BYTES);
        qurt_thread_attr_set_priority(
            &attributes,
            qurt_thread_get_priority(qurt_thread_get_id()));
        qurt_thread_attr_set_detachstate(
            &attributes, QURT_THREAD_ATTR_CREATE_JOINABLE);
        int create_status = qurt_thread_create(
            &hvx_threads[worker], &attributes, qbh_hvx_worker_main,
            &hvx_jobs[worker]);
        if (create_status != QURT_EOK) {
            header->hvx_thread_create_status = create_status;
            result = AEE_EFAILED;
            break;
        }
        ++created_hvx_workers;
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

            for (uint32_t batch_index = 0;
                 batch_index < dma_bundle_batch; ++batch_index) {
                uint32_t linear_output = linear_base + batch_index;
                uint32_t compressed_slot =
                    first_compressed_slot + batch_index;
                uint64_t wait_start = HAP_perf_get_qtimer_count();
                qurt_sem_down(&state.compressed_free[compressed_slot]);
                header->producer_slot_wait_ticks +=
                    HAP_perf_get_qtimer_count() - wait_start;
                if (linear_output >= layout->compressed_slot_count) {
                    ++header->weight_slot_reuse_count;
                }
            }

            if (qbh_physical_plan_uses_linked_dma(
                    layout->physical_plan)) {
                if (qbh_start_linked_weight_bundles(
                        header, linked_descriptors,
                        stored_weights +
                            (size_t)output_base * layout->w4_bundle_bytes,
                        state.compressed_slots[first_compressed_slot],
                        layout->w4_bundle_bytes, dma_bundle_batch) != 0) {
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
                        stored_weights +
                            (size_t)output_base * layout->w4_bundle_bytes,
                        state.compressed_slots[first_compressed_slot],
                        layout->w4_bundle_bytes, dma_bundle_batch) != 0) {
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
    for (uint32_t worker = 0; worker < created_hvx_workers; ++worker) {
        int exit_status = 0;
        int join_status =
            qurt_thread_join(hvx_threads[worker], &exit_status);
        ++joined_hvx_workers;
        if ((join_status != QURT_EOK || exit_status != AEE_SUCCESS) &&
            header->hvx_thread_join_status == 0) {
            header->hvx_thread_join_status =
                join_status != QURT_EOK ? join_status : exit_status;
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
         header->weight_expand_count !=
             header->repeat_count * layout->n_tiles *
                 layout->chunks_per_output)) {
        result = AEE_EFAILED;
    }

cleanup:
    if (hmx_thread_created && !hmx_thread_joined) {
        qbh_abort_pipeline(&state, AEE_EFAILED);
        header->hmx_thread_join_status =
            qurt_thread_join(hmx_thread, &hmx_exit_status);
    }
    for (uint32_t worker = joined_hvx_workers;
         worker < created_hvx_workers; ++worker) {
        int exit_status = 0;
        struct qbh_chunk_task stop_task;
        memset(&stop_task, 0, sizeof(stop_task));
        stop_task.stop = 1U;
        qbh_queue_push(&state.queue, &stop_task);
        (void)qurt_thread_join(hvx_threads[worker], &exit_status);
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
    qurt_mutex_destroy(&state.metrics_mutex);
    qbh_queue_destroy(&state.queue);
    return result;
}
