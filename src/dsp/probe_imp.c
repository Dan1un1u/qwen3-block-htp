#include <AEEStdErr.h>
#include <HAP_compute_res.h>
#include <HAP_farf.h>
#include <HAP_mem.h>
#include <HAP_perf.h>
#include <HAP_power.h>
#include <qurt.h>
#include <qurt_hvx.h>
#include <remote.h>
#include <stdint.h>
#include <string.h>

#include <hexagon_types.h>

#include "hmx_u8s8_projection.h"
#include "probe_protocol.h"
#include "qbh_user_dma.h"
#include "qwen3_probe.h"

#define QBH_VTCM_ACTIVATION_OFFSET UINT32_C(0)
#define QBH_VTCM_WEIGHT_SLOT0_OFFSET UINT32_C(8192)
#define QBH_VTCM_WEIGHT_SLOT1_OFFSET UINT32_C(12544)
#define QBH_VTCM_OUTPUT_OFFSET UINT32_C(18432)
#define QBH_HMX_WORKER_STACK_BYTES UINT32_C(16384)

_Static_assert(QBH_VTCM_WEIGHT_SLOT0_OFFSET +
                       QBH_PROJ_WEIGHT_BUNDLE_BYTES <=
                   QBH_VTCM_WEIGHT_SLOT1_OFFSET,
               "weight slot 0 overlaps slot 1");
_Static_assert(QBH_VTCM_WEIGHT_SLOT1_OFFSET +
                       QBH_PROJ_WEIGHT_BUNDLE_BYTES <=
                   QBH_VTCM_OUTPUT_OFFSET,
               "weight slot 1 overlaps output");
_Static_assert(QBH_VTCM_OUTPUT_OFFSET + QBH_PROJ_OUTPUT_TILES_BYTES <=
                   QBH_PROJ_VTCM_BYTES,
               "projection VTCM layout exceeds allocation");

static uint32_t probe_session_token;
static uint8_t hmx_worker_stack[QBH_HMX_WORKER_STACK_BYTES]
    __attribute__((aligned(128)));

struct qbh_projection_worker_job {
    uint32_t hmx_context_id;
    const uint8_t *activation_tiles;
    const uint8_t *weight_slots[2];
    uint8_t *output_tiles;
    uint32_t repeat_count;
    qurt_sem_t *ready[2];
    qurt_sem_t *free_slot[2];
    qurt_sem_t *started;
    volatile int32_t abort_status;
    int32_t lock_status;
    int32_t unlock_status;
    int32_t sync_status;
    uint32_t hmx_execution_count;
    uint32_t output_tile_count;
    uint64_t hmx_compute_ticks;
    uint64_t ready_wait_ticks;
};

AEEResult qwen3_probe_open(const char *uri, remote_handle64 *handle) {
    (void)uri;
    if (handle == NULL) {
        return AEE_EBADPARM;
    }
    *handle = (remote_handle64)&probe_session_token;
    return AEE_SUCCESS;
}

AEEResult qwen3_probe_close(remote_handle64 handle) {
    (void)handle;
    return AEE_SUCCESS;
}

static int range_is_valid(uint32_t offset, uint32_t bytes,
                          uint32_t total_bytes) {
    return offset <= total_bytes && bytes <= total_bytes - offset;
}

static int header_is_valid(const struct qbh_probe_header *header,
                           uint32_t shared_bytes) {
    if (header->magic != QBH_PROBE_MAGIC ||
        header->abi_version != QBH_PROBE_ABI_VERSION ||
        header->header_bytes != sizeof(*header) ||
        header->total_bytes > shared_bytes ||
        header->pattern < QBH_PATTERN_IDENTITY ||
        header->pattern > QBH_PATTERN_BOUNDARY ||
        header->input_zero_point > UINT8_MAX ||
        header->repeat_count == 0 ||
        header->repeat_count > QBH_HMX_MAX_REPEATS) {
        return 0;
    }
    return range_is_valid(header->activation_offset,
                          QBH_PROJ_ACTIVATION_BYTES,
                          header->total_bytes) &&
           range_is_valid(header->weight_offset,
                          QBH_PROJ_PACKED_WEIGHT_BYTES,
                          header->total_bytes) &&
           range_is_valid(header->output_offset, QBH_PROJ_OUTPUT_BYTES,
                          header->total_bytes);
}

static int vtcm_layout_is_aligned(const uint8_t *vtcm) {
    uintptr_t base = (uintptr_t)vtcm;
    return (base & UINT32_C(2047)) == 0 &&
           ((base + QBH_VTCM_WEIGHT_SLOT0_OFFSET) & UINT32_C(255)) == 0 &&
           ((base + QBH_VTCM_WEIGHT_SLOT1_OFFSET) & UINT32_C(255)) == 0 &&
           ((base + QBH_VTCM_OUTPUT_OFFSET) & UINT32_C(2047)) == 0;
}

static int record_dma_wait(struct qbh_probe_header *header) {
    int status;
    ++header->dma_wait_count;
    status = qbh_dma_wait_idle();
    if (status != 0 && header->dma_status == 0) {
        header->dma_status = status;
    }
    return status;
}

static int stage_activation_tiles(struct qbh_probe_header *header,
                                  const uint8_t *source,
                                  uint8_t *destination) {
    struct qbh_dma_desc_2d descriptor __attribute__((aligned(64)));
    uint64_t start = HAP_perf_get_qtimer_count();

    for (uint32_t input_tile = 0; input_tile < QBH_PROJ_K_TILES;
         ++input_tile) {
        int status;
        memset(&descriptor, 0, sizeof(descriptor));
        if (record_dma_wait(header) != 0) {
            return -1;
        }
        descriptor.next = 0;
        descriptor.length = 0;
        descriptor.type = QBH_DMA_TYPE_2D;
        descriptor.src_bypass = 1;
        descriptor.dst_bypass = 0;
        descriptor.ordered = 1;
        descriptor.dstate = QBH_DMA_DESC_PENDING;
        descriptor.src = (uint32_t)(uintptr_t)(
            source + input_tile * QBH_HMX_INPUT_CHANNELS);
        descriptor.dst = (uint32_t)(uintptr_t)(
            destination + input_tile * QBH_HMX_ACTIVATION_BYTES);
        descriptor.roi_width = (uint16_t)QBH_HMX_INPUT_CHANNELS;
        descriptor.roi_height = (uint16_t)QBH_PROJ_M;
        descriptor.src_stride = (uint16_t)QBH_PROJ_K;
        descriptor.dst_stride = (uint16_t)QBH_HMX_INPUT_CHANNELS;
        descriptor.src_width_offset = 0;
        descriptor.dst_width_offset = 0;

        status = qbh_dma_start(&descriptor);
        ++header->dma_submit_count;
        if (status != 0) {
            header->dma_status = status;
            return -1;
        }
        if (record_dma_wait(header) != 0) {
            return -1;
        }
        asm volatile("barrier" : : : "memory");
        for (uint32_t row = 0; row < QBH_PROJ_M; ++row) {
            for (uint32_t channel = 0;
                 channel < QBH_HMX_INPUT_CHANNELS; ++channel) {
                uint8_t expected = source[
                    (size_t)row * QBH_PROJ_K +
                    input_tile * QBH_HMX_INPUT_CHANNELS + channel];
                uint8_t actual = destination[
                    (size_t)input_tile * QBH_HMX_ACTIVATION_BYTES +
                    (size_t)row * QBH_HMX_INPUT_CHANNELS + channel];
                if (actual != expected) {
                    header->dma_status = QBH_DMA_CONTENT_MISMATCH;
                    FARF(ALWAYS,
                         "EXP0003 activation DMA mismatch tile=%u row=%u "
                         "channel=%u expected=%u actual=%u",
                         input_tile, row, channel, expected, actual);
                    return -1;
                }
            }
        }
        ++header->activation_stage_count;
    }
    header->activation_stage_ticks =
        HAP_perf_get_qtimer_count() - start;
    return 0;
}

static int stage_weight_bundle(struct qbh_probe_header *header,
                               const uint8_t *source,
                               uint8_t *destination) {
    struct qbh_dma_desc_1d descriptor __attribute__((aligned(64)));
    int status;

    memset(&descriptor, 0, sizeof(descriptor));
    if (record_dma_wait(header) != 0) {
        return -1;
    }
    descriptor.next = 0;
    descriptor.length = QBH_PROJ_WEIGHT_BUNDLE_BYTES;
    descriptor.type = QBH_DMA_TYPE_1D;
    descriptor.src_bypass = 1;
    descriptor.dst_bypass = 0;
    descriptor.ordered = 1;
    descriptor.dstate = QBH_DMA_DESC_PENDING;
    descriptor.src = (uint32_t)(uintptr_t)source;
    descriptor.dst = (uint32_t)(uintptr_t)destination;

    status = qbh_dma_start(&descriptor);
    ++header->dma_submit_count;
    if (status != 0) {
        header->dma_status = status;
        return -1;
    }
    if (record_dma_wait(header) != 0) {
        return -1;
    }
    asm volatile("barrier" : : : "memory");
    for (uint32_t index = 0; index < QBH_PROJ_WEIGHT_BUNDLE_BYTES;
         ++index) {
        if (destination[index] != source[index]) {
            header->dma_status = QBH_DMA_CONTENT_MISMATCH;
            FARF(ALWAYS,
                 "EXP0003 weight DMA mismatch index=%u expected=%u "
                 "actual=%u",
                 index, source[index], destination[index]);
            return -1;
        }
    }
    ++header->weight_bundle_stage_count;
    return 0;
}

__attribute__((noinline)) static void assemble_row_major_output(
    uint8_t *destination, const uint8_t *source_tiles) {
    for (uint32_t row = 0; row < QBH_PROJ_M; ++row) {
        for (uint32_t output_tile = 0;
             output_tile < QBH_PROJ_N_TILES; ++output_tile) {
            memcpy(destination + qbh_projection_output_offset(
                                     row,
                                     output_tile *
                                         QBH_HMX_OUTPUT_CHANNELS),
                   source_tiles +
                       (size_t)output_tile * QBH_HMX_OUTPUT_BYTES +
                       (size_t)row * QBH_HMX_OUTPUT_CHANNELS,
                   QBH_HMX_OUTPUT_CHANNELS);
        }
    }
    asm volatile("barrier" : : : "memory");
}

static void hmx_worker_main(void *opaque) {
    struct qbh_projection_worker_job *job =
        (struct qbh_projection_worker_job *)opaque;
    int exit_status = AEE_SUCCESS;

    job->lock_status = HAP_compute_res_hmx_lock2(
        job->hmx_context_id, HAP_COMPUTE_RES_HMX_SHARED);
    (void)qurt_sem_up(job->started);
    if (job->lock_status != AEE_SUCCESS) {
        qurt_thread_exit(job->lock_status);
    }

    for (uint32_t repeat = 0; repeat < job->repeat_count; ++repeat) {
        for (uint32_t output_tile = 0;
             output_tile < QBH_PROJ_N_TILES; ++output_tile) {
            uint32_t linear_tile = repeat * QBH_PROJ_N_TILES + output_tile;
            uint32_t slot = linear_tile & 1U;
            const uint8_t *bundle;
            const uint32_t *bias_words;
            uint64_t wait_start = HAP_perf_get_qtimer_count();
            qurt_sem_down(job->ready[slot]);
            job->ready_wait_ticks +=
                HAP_perf_get_qtimer_count() - wait_start;
            if (job->abort_status != 0) {
                exit_status = job->abort_status;
                goto unlock;
            }

            bundle = job->weight_slots[slot];
            bias_words = (const uint32_t *)(
                bundle + QBH_PROJ_WEIGHT_CHUNK_BYTES);
            uint64_t core_start = HAP_perf_get_qtimer_count();
            qbh_hmx_begin_u8s8_output(bias_words);
            qbh_hmx_accumulate_u8s8_projection(
                job->activation_tiles, (const int8_t *)bundle);
            job->hmx_execution_count += QBH_PROJ_K_TILES;
            qbh_hmx_store_u8_output(
                job->output_tiles +
                (size_t)output_tile * QBH_HMX_OUTPUT_BYTES);
            job->hmx_compute_ticks +=
                HAP_perf_get_qtimer_count() - core_start;
            ++job->output_tile_count;

            (void)qurt_sem_up(job->free_slot[slot]);
        }
    }

unlock:
    job->unlock_status = HAP_compute_res_hmx_unlock2(
        job->hmx_context_id, HAP_COMPUTE_RES_HMX_SHARED);
    if (exit_status == AEE_SUCCESS && job->unlock_status != AEE_SUCCESS) {
        exit_status = job->unlock_status;
    }
    qurt_thread_exit(exit_status);
}

AEEResult qwen3_probe_run(remote_handle64 handle, int32 shared_fd,
                          uint32 shared_bytes) {
    struct qbh_probe_header *header = NULL;
    compute_res_attr_t vtcm_attributes;
    compute_res_attr_t hmx_attributes;
    uint8_t *shared = NULL;
    uint8_t *vtcm = NULL;
    uint8_t *activation_tiles;
    uint8_t *weight_slots[2];
    uint8_t *output_tiles;
    uint32_t vtcm_context_id = 0;
    uint32_t hmx_context_id = 0;
    struct qbh_projection_worker_job hmx_job;
    qurt_thread_attr_t hmx_thread_attributes;
    qurt_thread_t hmx_thread;
    qurt_sem_t ready[2];
    qurt_sem_t free_slot[2];
    qurt_sem_t worker_started;
    int semaphores_initialized = 0;
    int hmx_thread_created = 0;
    int hmx_thread_joined = 0;
    int hmx_thread_exit_status = 0;
    HAP_power_request_t hmx_power_request;
    int hmx_power_context = 0;
    int hmx_powered = 0;
    int cache_result;
    int result = AEE_SUCCESS;
    uint64_t dsp_total_start = 0;

    (void)handle;

    result = HAP_mmap_get(shared_fd, (void **)&shared, NULL);
    if (result != AEE_SUCCESS || shared == NULL) {
        FARF(ERROR, "HAP_mmap_get failed: %d", result);
        return result != AEE_SUCCESS ? result : AEE_EFAILED;
    }
    if (shared_bytes < sizeof(struct qbh_probe_header)) {
        (void)HAP_mmap_put(shared_fd);
        return AEE_EBADSIZE;
    }

    cache_result = qurt_mem_cache_clean((qurt_addr_t)shared,
                                        (qurt_size_t)shared_bytes,
                                        QURT_MEM_CACHE_INVALIDATE,
                                        QURT_MEM_DCACHE);
    header = (struct qbh_probe_header *)shared;
    if (cache_result != 0) {
        header->dsp_status = QBH_PROBE_STATUS_CACHE_INVALIDATE_FAILED;
        header->cache_status = cache_result;
        result = AEE_EFAILED;
        goto cleanup;
    }
    if (!header_is_valid(header, shared_bytes)) {
        header->dsp_status = QBH_PROBE_STATUS_BAD_HEADER;
        result = AEE_EBADPARM;
        goto cleanup;
    }

    header->dsp_status = QBH_PROBE_STATUS_DSP_RUNNING;
    header->cache_status = 0;
    header->vtcm_requested_bytes = QBH_PROJ_VTCM_BYTES;
    header->vtcm_acquired_bytes = 0;
    header->hmx_resource_status = 0;
    header->hmx_lock_status = 0;
    header->hmx_unlock_status = 0;
    header->hmx_release_status = 0;
    header->hmx_thread_create_status = 0;
    header->hmx_thread_join_status = 0;
    header->hmx_power_up_status = 0;
    header->hmx_power_down_status = 0;
    header->hmx_execution_count = 0;
    header->hvx_lock_status = 0;
    header->hvx_unlock_status = 0;
    header->projection_m = QBH_PROJ_M;
    header->projection_k = QBH_PROJ_K;
    header->projection_n = QBH_PROJ_N;
    header->k_tile_count = QBH_PROJ_K_TILES;
    header->n_tile_count = QBH_PROJ_N_TILES;
    header->activation_stage_count = 0;
    header->weight_bundle_stage_count = 0;
    header->output_tile_count = 0;
    header->dma_submit_count = 0;
    header->dma_wait_count = 0;
    header->weight_slot_reuse_count = 0;
    header->dma_status = 0;
    header->sync_status = 0;
    header->qtimer_start = 0;
    header->qtimer_end = 0;
    header->qtimer_elapsed = 0;
    header->pcycles_start = 0;
    header->pcycles_end = 0;
    header->activation_stage_ticks = 0;
    header->weight_stage_ticks = 0;
    header->hmx_compute_ticks = 0;
    header->hmx_ready_wait_ticks = 0;
    header->producer_slot_wait_ticks = 0;
    header->pipeline_ticks = 0;
    header->output_assembly_ticks = 0;
    header->dsp_total_ticks = 0;
    dsp_total_start = HAP_perf_get_qtimer_count();
    FARF(ALWAYS, "EXP0003 stage=header_valid");

    memset(&hmx_power_request, 0, sizeof(hmx_power_request));
    hmx_power_request.type = HAP_power_set_HMX;
    hmx_power_request.hmx.power_up = 1;
    header->hmx_power_up_status = HAP_power_set(
        &hmx_power_context, &hmx_power_request);
    if (header->hmx_power_up_status != AEE_SUCCESS) {
        header->dsp_status = QBH_PROBE_STATUS_HMX_POWER_FAILED;
        result = AEE_EFAILED;
        goto cleanup;
    }
    hmx_powered = 1;

    result = HAP_compute_res_attr_init(&vtcm_attributes);
    if (result != AEE_SUCCESS ||
        HAP_compute_res_attr_set_serialize(&vtcm_attributes, 1) !=
            AEE_SUCCESS ||
        HAP_compute_res_attr_set_vtcm_param(&vtcm_attributes,
                                            QBH_PROJ_VTCM_BYTES, 1) !=
            AEE_SUCCESS) {
        header->dsp_status = QBH_PROBE_STATUS_VTCM_CONFIG_FAILED;
        result = AEE_EFAILED;
        goto cleanup;
    }
    vtcm_context_id = HAP_compute_res_acquire(&vtcm_attributes, 100000);
    if (vtcm_context_id == 0) {
        header->dsp_status = QBH_PROBE_STATUS_VTCM_ACQUIRE_FAILED;
        result = AEE_ERESOURCENOTFOUND;
        goto cleanup;
    }
    vtcm = (uint8_t *)HAP_compute_res_attr_get_vtcm_ptr(&vtcm_attributes);
    if (vtcm == NULL) {
        header->dsp_status = QBH_PROBE_STATUS_VTCM_POINTER_FAILED;
        result = AEE_EFAILED;
        goto cleanup;
    }
    header->vtcm_acquired_bytes = QBH_PROJ_VTCM_BYTES;
    if (!vtcm_layout_is_aligned(vtcm)) {
        header->dsp_status = QBH_PROBE_STATUS_VTCM_ALIGNMENT_FAILED;
        result = AEE_EFAILED;
        goto cleanup;
    }

    result = HAP_compute_res_attr_init(&hmx_attributes);
    if (result != AEE_SUCCESS) {
        header->dsp_status = QBH_PROBE_STATUS_HMX_CONFIG_FAILED;
        header->hmx_resource_status = result;
        goto cleanup;
    }
    result = HAP_compute_res_attr_set_hmx_param(&hmx_attributes, 1);
    header->hmx_resource_status = result;
    if (result != AEE_SUCCESS) {
        header->dsp_status = QBH_PROBE_STATUS_HMX_CONFIG_FAILED;
        goto cleanup;
    }
    hmx_context_id = HAP_compute_res_acquire(&hmx_attributes, 100000);
    if (hmx_context_id == 0) {
        header->dsp_status = QBH_PROBE_STATUS_HMX_CONFIG_FAILED;
        header->hmx_resource_status = AEE_ERESOURCENOTFOUND;
        result = AEE_ERESOURCENOTFOUND;
        goto cleanup;
    }

    activation_tiles = vtcm + QBH_VTCM_ACTIVATION_OFFSET;
    weight_slots[0] = vtcm + QBH_VTCM_WEIGHT_SLOT0_OFFSET;
    weight_slots[1] = vtcm + QBH_VTCM_WEIGHT_SLOT1_OFFSET;
    output_tiles = vtcm + QBH_VTCM_OUTPUT_OFFSET;

    if (stage_activation_tiles(header,
                               shared + header->activation_offset,
                               activation_tiles) != 0) {
        header->dsp_status = QBH_PROBE_STATUS_DMA_FAILED;
        result = AEE_EFAILED;
        goto cleanup;
    }

    qurt_sem_init_val(&ready[0], 0);
    qurt_sem_init_val(&ready[1], 0);
    qurt_sem_init_val(&free_slot[0], 1);
    qurt_sem_init_val(&free_slot[1], 1);
    qurt_sem_init_val(&worker_started, 0);
    semaphores_initialized = 1;

    memset(&hmx_job, 0, sizeof(hmx_job));
    hmx_job.hmx_context_id = hmx_context_id;
    hmx_job.activation_tiles = activation_tiles;
    hmx_job.weight_slots[0] = weight_slots[0];
    hmx_job.weight_slots[1] = weight_slots[1];
    hmx_job.output_tiles = output_tiles;
    hmx_job.repeat_count = header->repeat_count;
    hmx_job.ready[0] = &ready[0];
    hmx_job.ready[1] = &ready[1];
    hmx_job.free_slot[0] = &free_slot[0];
    hmx_job.free_slot[1] = &free_slot[1];
    hmx_job.started = &worker_started;

    qurt_thread_attr_init(&hmx_thread_attributes);
    qurt_thread_attr_set_name(&hmx_thread_attributes, "qbh-hmx-proj");
    qurt_thread_attr_set_stack_addr(&hmx_thread_attributes,
                                    hmx_worker_stack);
    qurt_thread_attr_set_stack_size(&hmx_thread_attributes,
                                    QBH_HMX_WORKER_STACK_BYTES);
    qurt_thread_attr_set_priority(
        &hmx_thread_attributes,
        qurt_thread_get_priority(qurt_thread_get_id()));
    qurt_thread_attr_set_detachstate(&hmx_thread_attributes,
                                     QURT_THREAD_ATTR_CREATE_JOINABLE);

    header->hmx_thread_create_status = qurt_thread_create(
        &hmx_thread, &hmx_thread_attributes, hmx_worker_main, &hmx_job);
    if (header->hmx_thread_create_status != QURT_EOK) {
        header->dsp_status = QBH_PROBE_STATUS_HMX_THREAD_FAILED;
        result = AEE_EFAILED;
        goto cleanup;
    }
    hmx_thread_created = 1;

    qurt_sem_down(&worker_started);
    if (hmx_job.lock_status != AEE_SUCCESS) {
        header->hmx_lock_status = hmx_job.lock_status;
        header->dsp_status = QBH_PROBE_STATUS_HMX_LOCK_FAILED;
        result = AEE_EFAILED;
        goto join_worker;
    }

    header->qtimer_start = HAP_perf_get_qtimer_count();
    header->pcycles_start = HAP_perf_get_pcycles();
    for (uint32_t repeat = 0; repeat < header->repeat_count; ++repeat) {
        for (uint32_t output_tile = 0;
             output_tile < QBH_PROJ_N_TILES; ++output_tile) {
            uint32_t linear_tile = repeat * QBH_PROJ_N_TILES + output_tile;
            uint32_t slot = linear_tile & 1U;
            uint64_t wait_start = HAP_perf_get_qtimer_count();
            qurt_sem_down(&free_slot[slot]);
            header->producer_slot_wait_ticks +=
                HAP_perf_get_qtimer_count() - wait_start;
            if (linear_tile >= 2U) {
                ++header->weight_slot_reuse_count;
            }

            uint64_t stage_start = HAP_perf_get_qtimer_count();
            if (stage_weight_bundle(
                    header,
                    shared + header->weight_offset +
                        qbh_projection_bundle_offset(output_tile),
                    weight_slots[slot]) != 0) {
                header->dsp_status = QBH_PROBE_STATUS_DMA_FAILED;
                result = AEE_EFAILED;
                goto abort_worker;
            }
            header->weight_stage_ticks +=
                HAP_perf_get_qtimer_count() - stage_start;

            (void)qurt_sem_up(&ready[slot]);
        }
    }
    goto join_worker;

abort_worker:
    hmx_job.abort_status = result != AEE_SUCCESS ? result : AEE_EFAILED;
    (void)qurt_sem_up(&ready[0]);
    (void)qurt_sem_up(&ready[1]);

join_worker:
    if (hmx_thread_created && !hmx_thread_joined) {
        header->hmx_thread_join_status = qurt_thread_join(
            hmx_thread, &hmx_thread_exit_status);
        hmx_thread_joined = 1;
    }
    header->pcycles_end = HAP_perf_get_pcycles();
    header->qtimer_end = HAP_perf_get_qtimer_count();
    header->pipeline_ticks = header->qtimer_end - header->qtimer_start;
    header->qtimer_elapsed = header->pipeline_ticks;
    header->hmx_lock_status = hmx_job.lock_status;
    header->hmx_unlock_status = hmx_job.unlock_status;
    header->hmx_execution_count = hmx_job.hmx_execution_count;
    header->output_tile_count = hmx_job.output_tile_count;
    header->hmx_compute_ticks = hmx_job.hmx_compute_ticks;
    header->hmx_ready_wait_ticks = hmx_job.ready_wait_ticks;
    if (header->sync_status == 0 && hmx_job.sync_status != 0) {
        header->sync_status = hmx_job.sync_status;
    }
    if (result != AEE_SUCCESS ||
        header->hmx_thread_join_status != QURT_EOK ||
        hmx_thread_exit_status != AEE_SUCCESS ||
        header->hmx_lock_status != AEE_SUCCESS ||
        header->hmx_unlock_status != AEE_SUCCESS ||
        header->sync_status != 0) {
        if (header->dsp_status == QBH_PROBE_STATUS_DSP_RUNNING) {
            header->dsp_status =
                header->hmx_lock_status != AEE_SUCCESS
                    ? QBH_PROBE_STATUS_HMX_LOCK_FAILED
                    : (header->sync_status != 0
                           ? QBH_PROBE_STATUS_SYNC_FAILED
                           : QBH_PROBE_STATUS_HMX_THREAD_FAILED);
        }
        result = AEE_EFAILED;
        goto cleanup;
    }

    header->hvx_lock_status = qurt_hvx_lock(QURT_HVX_MODE_128B);
    if (header->hvx_lock_status != AEE_SUCCESS) {
        header->dsp_status = QBH_PROBE_STATUS_HVX_LOCK_FAILED;
        result = AEE_EFAILED;
        goto cleanup;
    }
    uint64_t output_start = HAP_perf_get_qtimer_count();
    assemble_row_major_output(shared + header->output_offset, output_tiles);
    header->output_assembly_ticks =
        HAP_perf_get_qtimer_count() - output_start;
    header->hvx_unlock_status = qurt_hvx_unlock();
    if (header->hvx_unlock_status != AEE_SUCCESS) {
        header->dsp_status = QBH_PROBE_STATUS_HVX_LOCK_FAILED;
        result = AEE_EFAILED;
        goto cleanup;
    }

    header->dsp_status = QBH_PROBE_STATUS_OK;
    result = AEE_SUCCESS;

cleanup:
    if (hmx_thread_created && !hmx_thread_joined) {
        hmx_job.abort_status = AEE_EFAILED;
        if (semaphores_initialized) {
            (void)qurt_sem_up(&ready[0]);
            (void)qurt_sem_up(&ready[1]);
        }
        header->hmx_thread_join_status = qurt_thread_join(
            hmx_thread, &hmx_thread_exit_status);
        hmx_thread_joined = 1;
    }
    if (semaphores_initialized) {
        qurt_sem_destroy(&worker_started);
        qurt_sem_destroy(&ready[0]);
        qurt_sem_destroy(&ready[1]);
        qurt_sem_destroy(&free_slot[0]);
        qurt_sem_destroy(&free_slot[1]);
    }
    if (hmx_context_id != 0) {
        header->hmx_release_status = HAP_compute_res_release(hmx_context_id);
        if (header->hmx_release_status != AEE_SUCCESS) {
            header->dsp_status = QBH_PROBE_STATUS_HMX_RELEASE_FAILED;
            if (result == AEE_SUCCESS) {
                result = header->hmx_release_status;
            }
        }
    }
    if (vtcm_context_id != 0) {
        int release_result = HAP_compute_res_release(vtcm_context_id);
        if (release_result != AEE_SUCCESS && result == AEE_SUCCESS) {
            result = release_result;
        }
    }
    if (hmx_powered) {
        memset(&hmx_power_request, 0, sizeof(hmx_power_request));
        hmx_power_request.type = HAP_power_set_HMX;
        hmx_power_request.hmx.power_up = 0;
        header->hmx_power_down_status = HAP_power_set(
            &hmx_power_context, &hmx_power_request);
        if (header->hmx_power_down_status != AEE_SUCCESS) {
            header->dsp_status = QBH_PROBE_STATUS_HMX_POWER_FAILED;
            if (result == AEE_SUCCESS) {
                result = AEE_EFAILED;
            }
        }
    }

    if (dsp_total_start != 0) {
        header->dsp_total_ticks =
            HAP_perf_get_qtimer_count() - dsp_total_start;
    }
    cache_result = qurt_mem_cache_clean((qurt_addr_t)shared,
                                        (qurt_size_t)shared_bytes,
                                        QURT_MEM_CACHE_FLUSH,
                                        QURT_MEM_DCACHE);
    if (cache_result != 0 && header != NULL) {
        header->dsp_status = QBH_PROBE_STATUS_CACHE_FLUSH_FAILED;
        header->cache_status = cache_result;
        if (result == AEE_SUCCESS) {
            result = AEE_EFAILED;
        }
    }

    (void)HAP_mmap_put(shared_fd);
    return result;
}
