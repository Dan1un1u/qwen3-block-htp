#include <AEEStdErr.h>
#include <HAP_compute_res.h>
#include <HAP_farf.h>
#include <HAP_mem.h>
#include <HAP_perf.h>
#include <qurt.h>
#include <remote.h>
#include <stdint.h>
#include <string.h>

#include "probe_protocol.h"
#include "qwen3_probe.h"

static uint32_t probe_session_token;

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

static uint32_t align_up_u32(uint32_t value, uint32_t alignment) {
    return (value + alignment - 1U) / alignment * alignment;
}

static int header_is_valid(const struct qbh_probe_header *header,
                           uint32_t shared_bytes) {
    if (header->magic != QBH_PROBE_MAGIC ||
        header->abi_version != QBH_PROBE_ABI_VERSION ||
        header->header_bytes != sizeof(*header) ||
        header->total_bytes > shared_bytes ||
        header->vector_length == 0) {
        return 0;
    }
    if (header->input_offset > header->total_bytes ||
        header->vector_length > header->total_bytes - header->input_offset) {
        return 0;
    }
    if (header->output_offset > header->total_bytes ||
        header->vector_length > header->total_bytes - header->output_offset) {
        return 0;
    }
    if (header->addend > UINT8_MAX) {
        return 0;
    }
    return 1;
}

AEEResult qwen3_probe_run(remote_handle64 handle, int32 shared_fd,
                          uint32 shared_bytes) {
    struct qbh_probe_header *header = NULL;
    compute_res_attr_t resource_attributes;
    uint8_t *shared = NULL;
    uint8_t *vtcm = NULL;
    uint32_t context_id = 0;
    uint32_t vector_bytes;
    uint32_t vtcm_half_bytes;
    uint32_t vtcm_bytes;
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

    vector_bytes = header->vector_length;
    vtcm_half_bytes = align_up_u32(vector_bytes, QBH_PROBE_ALIGNMENT);
    if (vtcm_half_bytes > UINT32_MAX / 2U) {
        header->dsp_status = QBH_PROBE_STATUS_BAD_HEADER;
        result = AEE_EBADSIZE;
        goto cleanup;
    }
    vtcm_bytes = 2U * vtcm_half_bytes;
    header->vtcm_requested_bytes = vtcm_bytes;

    result = HAP_compute_res_attr_init(&resource_attributes);
    if (result != AEE_SUCCESS) {
        header->dsp_status = QBH_PROBE_STATUS_VTCM_CONFIG_FAILED;
        goto cleanup;
    }
    result = HAP_compute_res_attr_set_serialize(&resource_attributes, 1);
    if (result != AEE_SUCCESS) {
        header->dsp_status = QBH_PROBE_STATUS_VTCM_CONFIG_FAILED;
        goto cleanup;
    }
    result = HAP_compute_res_attr_set_vtcm_param(&resource_attributes,
                                                 vtcm_bytes, 0);
    if (result != AEE_SUCCESS) {
        header->dsp_status = QBH_PROBE_STATUS_VTCM_CONFIG_FAILED;
        goto cleanup;
    }

    context_id = HAP_compute_res_acquire(&resource_attributes, 100000);
    if (context_id == 0) {
        header->dsp_status = QBH_PROBE_STATUS_VTCM_ACQUIRE_FAILED;
        result = AEE_ERESOURCENOTFOUND;
        goto cleanup;
    }
    vtcm = (uint8_t *)HAP_compute_res_attr_get_vtcm_ptr(&resource_attributes);
    if (vtcm == NULL) {
        header->dsp_status = QBH_PROBE_STATUS_VTCM_POINTER_FAILED;
        result = AEE_EFAILED;
        goto cleanup;
    }
    header->vtcm_acquired_bytes = vtcm_bytes;

    memcpy(vtcm, shared + header->input_offset, vector_bytes);
    header->qtimer_start = HAP_perf_get_qtimer_count();
    header->pcycles_start = HAP_perf_get_pcycles();
    for (uint32_t index = 0; index < vector_bytes; ++index) {
        vtcm[vtcm_half_bytes + index] =
            (uint8_t)(vtcm[index] + (uint8_t)header->addend);
    }
    header->pcycles_end = HAP_perf_get_pcycles();
    header->qtimer_end = HAP_perf_get_qtimer_count();
    header->qtimer_elapsed = header->qtimer_end - header->qtimer_start;
    memcpy(shared + header->output_offset, vtcm + vtcm_half_bytes,
           vector_bytes);
    header->dsp_status = QBH_PROBE_STATUS_OK;

cleanup:
    if (context_id != 0) {
        int release_result = HAP_compute_res_release(context_id);
        if (release_result != AEE_SUCCESS && result == AEE_SUCCESS) {
            result = release_result;
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
