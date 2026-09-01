#include <AEEStdErr.h>
#include <HAP_mem.h>
#include <qurt.h>
#include <remote.h>
#include <stdint.h>

#include "qwen3_probe.h"
#include "rpcmem2_protocol.h"

static int qbh_rpcmem2_range_valid(uint32_t offset, uint32_t bytes,
                                   uint32_t total_bytes) {
    return offset <= total_bytes && bytes <= total_bytes - offset;
}

AEEResult qwen3_probe_run_rpcmem2_probe(remote_handle64 handle,
                                        int32 shared_fd,
                                        uint32 shared_bytes) {
    struct qbh_rpcmem2_header *header = NULL;
    uint8_t *shared = NULL;
    uint64 physical_base = 0U;
    uintptr_t base_address;
    uint32_t i;
    int result;
    int put_result;

    if (handle == (remote_handle64)-1 || shared_fd < 0 ||
        shared_bytes != QBH_RPCMEM2_REQUEST_BYTES) {
        return AEE_EBADPARM;
    }

    result = HAP_mmap_get(shared_fd, (void **)&shared, &physical_base);
    if (result != AEE_SUCCESS || shared == NULL) {
        return result != AEE_SUCCESS ? result : AEE_ENOSUCHMAP;
    }
    header = (struct qbh_rpcmem2_header *)shared;
    header->mmap_get_result = result;

    result = qurt_mem_cache_clean(
        (qurt_addr_t)header, (qurt_size_t)sizeof(*header),
        QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
    header->cache_invalidate_result = result;
    if (result != 0) {
        header->dsp_status = QBH_RPCMEM2_STATUS_CACHE_FAILED;
        result = AEE_EFAILED;
        goto publish;
    }

    if (header->magic != QBH_RPCMEM2_MAGIC ||
        header->abi_version != QBH_RPCMEM2_ABI_VERSION ||
        header->experiment != QBH_RPCMEM2_EXPERIMENT ||
        header->header_bytes != sizeof(*header) ||
        header->requested_bytes != shared_bytes ||
        header->sentinel_count != QBH_RPCMEM2_SENTINEL_COUNT) {
        header->dsp_status = QBH_RPCMEM2_STATUS_BAD_HEADER;
        result = AEE_EBADPARM;
        goto publish;
    }

    base_address = (uintptr_t)shared;
    header->address_range_valid = 1U;
    for (i = 0U; i < QBH_RPCMEM2_SENTINEL_COUNT; ++i) {
        uint32_t offset = header->sentinel_offsets[i];
        if ((offset & UINT32_C(4095)) != 0U ||
            !qbh_rpcmem2_range_valid(offset, sizeof(uint32_t),
                                     shared_bytes) ||
            base_address > UINTPTR_MAX - offset - sizeof(uint32_t)) {
            header->address_range_valid = 0U;
            header->dsp_status = QBH_RPCMEM2_STATUS_BAD_RANGE;
            result = AEE_EBADPARM;
            goto publish;
        }
    }

    header->dsp_virtual_base = (uint32_t)base_address;
    header->dsp_virtual_end =
        (uint32_t)(base_address + QBH_RPCMEM2_END_OFFSET + sizeof(uint32_t));
    header->dsp_physical_base = (uint64_t)physical_base;
    header->sentinel_mismatch_count = 0U;

    for (i = 0U; i < QBH_RPCMEM2_SENTINEL_COUNT; ++i) {
        volatile uint32_t *sentinel = (volatile uint32_t *)(
            base_address + header->sentinel_offsets[i]);
        int cache_result = qurt_mem_cache_clean(
            (qurt_addr_t)sentinel, (qurt_size_t)sizeof(*sentinel),
            QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
        if (cache_result != 0) {
            header->cache_invalidate_result = cache_result;
            header->dsp_status = QBH_RPCMEM2_STATUS_CACHE_FAILED;
            result = AEE_EFAILED;
            goto publish;
        }
        header->dsp_observed_values[i] = *sentinel;
        if (header->dsp_observed_values[i] != header->host_values[i]) {
            ++header->sentinel_mismatch_count;
        }
        *sentinel = header->dsp_write_values[i];
        cache_result = qurt_mem_cache_clean(
            (qurt_addr_t)sentinel, (qurt_size_t)sizeof(*sentinel),
            QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
        if (cache_result != 0) {
            header->cache_flush_result = cache_result;
            header->dsp_status = QBH_RPCMEM2_STATUS_CACHE_FAILED;
            result = AEE_EFAILED;
            goto publish;
        }
    }

    if (header->sentinel_mismatch_count != 0U) {
        header->dsp_status = QBH_RPCMEM2_STATUS_SENTINEL_MISMATCH;
        result = AEE_EFAILED;
    } else {
        header->dsp_status = QBH_RPCMEM2_STATUS_OK;
        result = AEE_SUCCESS;
    }

publish:
    header->cache_flush_result = qurt_mem_cache_clean(
        (qurt_addr_t)header, (qurt_size_t)sizeof(*header),
        QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
    if (header->cache_flush_result != 0 && result == AEE_SUCCESS) {
        header->dsp_status = QBH_RPCMEM2_STATUS_CACHE_FAILED;
        result = AEE_EFAILED;
    }
    put_result = HAP_mmap_put(shared_fd);
    header->mmap_put_result = put_result;
    (void)qurt_mem_cache_clean(
        (qurt_addr_t)header, (qurt_size_t)sizeof(*header),
        QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
    return result == AEE_SUCCESS && put_result != AEE_SUCCESS
               ? put_result
               : result;
}
