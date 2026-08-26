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

#include "hmx_int8_tile.h"
#include "probe_protocol.h"
#include "qwen3_probe.h"

#define QBH_VTCM_ACTIVATION_OFFSET UINT32_C(0)
#define QBH_VTCM_OUTPUT_OFFSET UINT32_C(2048)
#define QBH_VTCM_WEIGHT_OFFSET UINT32_C(4096)
#define QBH_VTCM_BIAS_OFFSET UINT32_C(5120)
#define QBH_HMX_WORKER_STACK_BYTES UINT32_C(16384)

static uint32_t probe_session_token;
static uint8_t hmx_worker_stack[QBH_HMX_WORKER_STACK_BYTES]
    __attribute__((aligned(128)));

struct qbh_hmx_worker_job {
    uint32_t hmx_context_id;
    const uint8_t *activation;
    const int8_t *packed_weight;
    const uint32_t *bias_words;
    uint8_t *output;
    uint32_t repeat_count;
    int32_t lock_status;
    int32_t unlock_status;
    uint32_t execution_count;
    uint64_t qtimer_start;
    uint64_t qtimer_end;
    uint64_t pcycles_start;
    uint64_t pcycles_end;
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
                          QBH_HMX_ACTIVATION_BYTES,
                          header->total_bytes) &&
           range_is_valid(header->weight_offset, QBH_HMX_WEIGHT_BYTES,
                          header->total_bytes) &&
           range_is_valid(header->output_offset, QBH_HMX_OUTPUT_BYTES,
                          header->total_bytes);
}

static int vtcm_layout_is_aligned(const uint8_t *vtcm) {
    uintptr_t base = (uintptr_t)vtcm;
    return (base & UINT32_C(2047)) == 0 &&
           ((base + QBH_VTCM_OUTPUT_OFFSET) & UINT32_C(2047)) == 0 &&
           ((base + QBH_VTCM_WEIGHT_OFFSET) & UINT32_C(127)) == 0 &&
           ((base + QBH_VTCM_BIAS_OFFSET) & UINT32_C(255)) == 0;
}

__attribute__((noinline)) static void copy_output_from_vtcm_hvx(
    uint8_t *destination, const uint8_t *source) {
    uint32_t index;
    HVX_Vector *destination_vector = (HVX_Vector *)destination;
    const HVX_Vector *source_vector = (const HVX_Vector *)source;

    for (index = 0; index < QBH_HMX_OUTPUT_BYTES / sizeof(HVX_Vector);
         ++index) {
        destination_vector[index] = source_vector[index];
    }
    asm volatile("barrier" ::: "memory");
}

static void hmx_worker_main(void *opaque) {
    struct qbh_hmx_worker_job *job =
        (struct qbh_hmx_worker_job *)opaque;
    uint32_t repeat;

    job->lock_status = HAP_compute_res_hmx_lock2(
        job->hmx_context_id, HAP_COMPUTE_RES_HMX_SHARED);
    if (job->lock_status != AEE_SUCCESS) {
        qurt_thread_exit(job->lock_status);
    }

    job->qtimer_start = HAP_perf_get_qtimer_count();
    job->pcycles_start = HAP_perf_get_pcycles();
    for (repeat = 0; repeat < job->repeat_count; ++repeat) {
        qbh_execute_u8s8_tile(job->activation, job->packed_weight,
                              job->bias_words, job->output);
    }
    job->pcycles_end = HAP_perf_get_pcycles();
    job->qtimer_end = HAP_perf_get_qtimer_count();
    job->execution_count = job->repeat_count;
    job->unlock_status = HAP_compute_res_hmx_unlock2(
        job->hmx_context_id, HAP_COMPUTE_RES_HMX_SHARED);
    qurt_thread_exit(job->unlock_status);
}

AEEResult qwen3_probe_run(remote_handle64 handle, int32 shared_fd,
                          uint32 shared_bytes) {
    struct qbh_probe_header *header = NULL;
    compute_res_attr_t vtcm_attributes;
    compute_res_attr_t hmx_attributes;
    uint8_t *shared = NULL;
    uint8_t *vtcm = NULL;
    uint8_t *activation;
    uint8_t *output;
    int8_t *packed_weight;
    uint32_t *bias_words;
    uint32_t vtcm_context_id = 0;
    uint32_t hmx_context_id = 0;
    struct qbh_hmx_worker_job hmx_job;
    qurt_thread_attr_t hmx_thread_attributes;
    qurt_thread_t hmx_thread;
    int hmx_thread_exit_status = 0;
    HAP_power_request_t hmx_power_request;
    int hmx_power_context = 0;
    int hmx_powered = 0;
    int cache_result;
    int result = AEE_SUCCESS;

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
    header->vtcm_requested_bytes = QBH_HMX_VTCM_BYTES;
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
    FARF(ALWAYS, "EXP0002 stage=header_valid");

    memset(&hmx_power_request, 0, sizeof(hmx_power_request));
    hmx_power_request.type = HAP_power_set_HMX;
    hmx_power_request.hmx.power_up = 1;
    FARF(ALWAYS, "EXP0002 stage=hmx_power_up_begin");
    header->hmx_power_up_status = HAP_power_set(
        &hmx_power_context, &hmx_power_request);
    FARF(ALWAYS, "EXP0002 stage=hmx_power_up_end status=%d",
         header->hmx_power_up_status);
    if (header->hmx_power_up_status != AEE_SUCCESS) {
        header->dsp_status = QBH_PROBE_STATUS_HMX_POWER_FAILED;
        result = AEE_EFAILED;
        goto cleanup;
    }
    hmx_powered = 1;

    result = HAP_compute_res_attr_init(&vtcm_attributes);
    if (result != AEE_SUCCESS) {
        header->dsp_status = QBH_PROBE_STATUS_VTCM_CONFIG_FAILED;
        goto cleanup;
    }
    result = HAP_compute_res_attr_set_serialize(&vtcm_attributes, 1);
    if (result != AEE_SUCCESS) {
        header->dsp_status = QBH_PROBE_STATUS_VTCM_CONFIG_FAILED;
        goto cleanup;
    }
    result = HAP_compute_res_attr_set_vtcm_param(&vtcm_attributes,
                                                 QBH_HMX_VTCM_BYTES, 1);
    if (result != AEE_SUCCESS) {
        header->dsp_status = QBH_PROBE_STATUS_VTCM_CONFIG_FAILED;
        goto cleanup;
    }
    FARF(ALWAYS, "EXP0002 stage=vtcm_acquire_begin");
    vtcm_context_id = HAP_compute_res_acquire(&vtcm_attributes, 100000);
    FARF(ALWAYS, "EXP0002 stage=vtcm_acquire_end context=%u",
         vtcm_context_id);
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
    header->vtcm_acquired_bytes = QBH_HMX_VTCM_BYTES;
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

    FARF(ALWAYS, "EXP0002 stage=hmx_acquire_begin");
    hmx_context_id = HAP_compute_res_acquire(&hmx_attributes, 10000);
    FARF(ALWAYS, "EXP0002 stage=hmx_acquire_end context=%u",
         hmx_context_id);
    if (hmx_context_id == 0) {
        header->dsp_status = QBH_PROBE_STATUS_HMX_CONFIG_FAILED;
        result = AEE_ERESOURCENOTFOUND;
        goto cleanup;
    }

    activation = vtcm + QBH_VTCM_ACTIVATION_OFFSET;
    output = vtcm + QBH_VTCM_OUTPUT_OFFSET;
    packed_weight = (int8_t *)(vtcm + QBH_VTCM_WEIGHT_OFFSET);
    bias_words = (uint32_t *)(vtcm + QBH_VTCM_BIAS_OFFSET);

    memcpy(activation, shared + header->activation_offset,
           QBH_HMX_ACTIVATION_BYTES);
    qbh_pack_s8_weight((const int8_t *)(shared + header->weight_offset),
                       packed_weight);
    qbh_fill_asymmetric_bias(packed_weight,
                             (int32_t)header->input_zero_point, bias_words);

    memset(&hmx_job, 0, sizeof(hmx_job));
    hmx_job.hmx_context_id = hmx_context_id;
    hmx_job.activation = activation;
    hmx_job.packed_weight = packed_weight;
    hmx_job.bias_words = bias_words;
    hmx_job.output = output;
    hmx_job.repeat_count = header->repeat_count;

    qurt_thread_attr_init(&hmx_thread_attributes);
    qurt_thread_attr_set_name(&hmx_thread_attributes, "qbh-hmx");
    qurt_thread_attr_set_stack_addr(&hmx_thread_attributes,
                                    hmx_worker_stack);
    qurt_thread_attr_set_stack_size(&hmx_thread_attributes,
                                    QBH_HMX_WORKER_STACK_BYTES);
    qurt_thread_attr_set_priority(
        &hmx_thread_attributes,
        qurt_thread_get_priority(qurt_thread_get_id()));
    qurt_thread_attr_set_detachstate(&hmx_thread_attributes,
                                     QURT_THREAD_ATTR_CREATE_JOINABLE);

    FARF(ALWAYS, "EXP0002 stage=hmx_thread_create_begin");
    header->hmx_thread_create_status = qurt_thread_create(
        &hmx_thread, &hmx_thread_attributes, hmx_worker_main, &hmx_job);
    FARF(ALWAYS, "EXP0002 stage=hmx_thread_create_end status=%d",
         header->hmx_thread_create_status);
    if (header->hmx_thread_create_status != QURT_EOK) {
        header->dsp_status = QBH_PROBE_STATUS_HMX_THREAD_FAILED;
        result = AEE_EFAILED;
        goto cleanup;
    }

    FARF(ALWAYS, "EXP0002 stage=hmx_thread_join_begin");
    header->hmx_thread_join_status = qurt_thread_join(
        hmx_thread, &hmx_thread_exit_status);
    FARF(ALWAYS,
         "EXP0002 stage=hmx_thread_join_end status=%d exit_status=%d",
         header->hmx_thread_join_status, hmx_thread_exit_status);

    header->hmx_lock_status = hmx_job.lock_status;
    header->hmx_unlock_status = hmx_job.unlock_status;
    header->hmx_execution_count = hmx_job.execution_count;
    header->qtimer_start = hmx_job.qtimer_start;
    header->qtimer_end = hmx_job.qtimer_end;
    header->qtimer_elapsed = hmx_job.qtimer_end - hmx_job.qtimer_start;
    header->pcycles_start = hmx_job.pcycles_start;
    header->pcycles_end = hmx_job.pcycles_end;
    if (header->hmx_thread_join_status != QURT_EOK ||
        hmx_thread_exit_status != AEE_SUCCESS ||
        header->hmx_lock_status != AEE_SUCCESS ||
        header->hmx_unlock_status != AEE_SUCCESS) {
        header->dsp_status =
            header->hmx_lock_status != AEE_SUCCESS
                ? QBH_PROBE_STATUS_HMX_LOCK_FAILED
                : QBH_PROBE_STATUS_HMX_THREAD_FAILED;
        result = AEE_EFAILED;
        goto cleanup;
    }

    FARF(ALWAYS, "EXP0002 stage=output_copy_begin");
    header->hvx_lock_status = qurt_hvx_lock(QURT_HVX_MODE_128B);
    FARF(ALWAYS, "EXP0002 stage=hvx_lock_end status=%d",
         header->hvx_lock_status);
    if (header->hvx_lock_status != AEE_SUCCESS) {
        header->dsp_status = QBH_PROBE_STATUS_HVX_LOCK_FAILED;
        result = AEE_EFAILED;
        goto cleanup;
    }
    copy_output_from_vtcm_hvx(shared + header->output_offset, output);
    header->hvx_unlock_status = qurt_hvx_unlock();
    FARF(ALWAYS, "EXP0002 stage=hvx_unlock_end status=%d",
         header->hvx_unlock_status);
    if (header->hvx_unlock_status != AEE_SUCCESS) {
        result = AEE_EFAILED;
        goto cleanup;
    }
    FARF(ALWAYS, "EXP0002 stage=output_copy_end");
    header->dsp_status = QBH_PROBE_STATUS_OK;
    FARF(ALWAYS, "EXP0002 stage=success");

cleanup:
    if (hmx_context_id != 0) {
        FARF(ALWAYS, "EXP0002 stage=hmx_release_begin");
        header->hmx_release_status = HAP_compute_res_release(hmx_context_id);
        FARF(ALWAYS, "EXP0002 stage=hmx_release_end status=%d",
             header->hmx_release_status);
        if (header->hmx_release_status != AEE_SUCCESS) {
            header->dsp_status = QBH_PROBE_STATUS_HMX_RELEASE_FAILED;
            if (result == AEE_SUCCESS) {
                result = header->hmx_release_status;
            }
        }
    }
    if (vtcm_context_id != 0) {
        FARF(ALWAYS, "EXP0002 stage=vtcm_release_begin");
        int release_result = HAP_compute_res_release(vtcm_context_id);
        FARF(ALWAYS, "EXP0002 stage=vtcm_release_end status=%d",
             release_result);
        if (release_result != AEE_SUCCESS && result == AEE_SUCCESS) {
            result = release_result;
        }
    }
    if (hmx_powered) {
        memset(&hmx_power_request, 0, sizeof(hmx_power_request));
        hmx_power_request.type = HAP_power_set_HMX;
        hmx_power_request.hmx.power_up = 0;
        FARF(ALWAYS, "EXP0002 stage=hmx_power_down_begin");
        header->hmx_power_down_status = HAP_power_set(
            &hmx_power_context, &hmx_power_request);
        FARF(ALWAYS, "EXP0002 stage=hmx_power_down_end status=%d",
             header->hmx_power_down_status);
        if (header->hmx_power_down_status != AEE_SUCCESS) {
            header->dsp_status = QBH_PROBE_STATUS_HMX_POWER_FAILED;
            if (result == AEE_SUCCESS) {
                result = AEE_EFAILED;
            }
        }
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
