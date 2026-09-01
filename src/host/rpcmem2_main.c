#include <AEEStdErr.h>
#include <errno.h>
#include <inttypes.h>
#include <remote.h>
#include <rpcmem.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "host/session.h"
#include "qwen3_probe.h"
#include "rpcmem2_protocol.h"

#pragma weak rpcmem_alloc2

static uint64_t qbh_monotonic_ns(void) {
    struct timespec timestamp;
    if (clock_gettime(CLOCK_MONOTONIC, &timestamp) != 0) {
        return 0U;
    }
    return (uint64_t)timestamp.tv_sec * UINT64_C(1000000000) +
           (uint64_t)timestamp.tv_nsec;
}

int main(void) {
    static const uint32_t offsets[QBH_RPCMEM2_SENTINEL_COUNT] = {
        QBH_RPCMEM2_BEGIN_OFFSET,
        QBH_RPCMEM2_MIDDLE_OFFSET,
        QBH_RPCMEM2_END_OFFSET,
    };
    static const uint32_t host_values[QBH_RPCMEM2_SENTINEL_COUNT] = {
        UINT32_C(0x13579bdf),
        UINT32_C(0x2468ace0),
        UINT32_C(0x0badc0de),
    };
    static const uint32_t dsp_values[QBH_RPCMEM2_SENTINEL_COUNT] = {
        UINT32_C(0xa5a50151),
        UINT32_C(0x5a5a0151),
        UINT32_C(0xc0010151),
    };
    struct qbh_session session = {(remote_handle64)-1, 0};
    struct qbh_rpcmem2_header *header = NULL;
    uint8_t *shared = NULL;
    int shared_fd = -1;
    int symbol_present = rpcmem_alloc2 != NULL;
    int allocation_errno = 0;
    int open_result = AEE_EFAILED;
    int map_result = AEE_EFAILED;
    int run_result = AEE_EFAILED;
    int close_result = AEE_EFAILED;
    int unmap_result = AEE_EFAILED;
    int mapped = 0;
    int host_mismatch_count = 0;
    uint32_t i;
    uint64_t allocation_start = 0U;
    uint64_t allocation_end = 0U;
    uint64_t map_start = 0U;
    uint64_t map_end = 0U;
    uint64_t run_start = 0U;
    uint64_t run_end = 0U;
    int exit_code = 1;

    if (!symbol_present) {
        fprintf(stderr, "rpcmem_alloc2 symbol is unavailable\n");
        goto report;
    }

    allocation_start = qbh_monotonic_ns();
    shared = rpcmem_alloc2(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_FLAG_UNCACHED,
                           (size_t)QBH_RPCMEM2_REQUEST_BYTES);
    allocation_end = qbh_monotonic_ns();
    if (shared == NULL) {
        allocation_errno = errno;
        fprintf(stderr, "rpcmem_alloc2 failed: errno=%d\n", allocation_errno);
        goto report;
    }

    shared_fd = rpcmem_to_fd(shared);
    if (shared_fd < 0) {
        fprintf(stderr, "rpcmem_to_fd failed\n");
        goto report;
    }

    header = (struct qbh_rpcmem2_header *)shared;
    memset(header, 0, sizeof(*header));
    header->magic = QBH_RPCMEM2_MAGIC;
    header->abi_version = QBH_RPCMEM2_ABI_VERSION;
    header->experiment = QBH_RPCMEM2_EXPERIMENT;
    header->header_bytes = (uint32_t)sizeof(*header);
    header->requested_bytes = QBH_RPCMEM2_REQUEST_BYTES;
    header->sentinel_count = QBH_RPCMEM2_SENTINEL_COUNT;
    header->dsp_status = QBH_RPCMEM2_STATUS_HOST_READY;
    for (i = 0U; i < QBH_RPCMEM2_SENTINEL_COUNT; ++i) {
        volatile uint32_t *sentinel =
            (volatile uint32_t *)(shared + offsets[i]);
        header->sentinel_offsets[i] = offsets[i];
        header->host_values[i] = host_values[i];
        header->dsp_write_values[i] = dsp_values[i];
        *sentinel = host_values[i];
    }

    open_result = qbh_session_open(&session);
    if (open_result != AEE_SUCCESS) {
        goto report;
    }

    map_start = qbh_monotonic_ns();
    map_result = fastrpc_mmap(CDSP_DOMAIN_ID, shared_fd, shared, 0,
                              (size_t)QBH_RPCMEM2_REQUEST_BYTES,
                              FASTRPC_MAP_FD);
    map_end = qbh_monotonic_ns();
    if (map_result != AEE_SUCCESS) {
        fprintf(stderr, "fastrpc_mmap failed: 0x%08x\n",
                (unsigned int)map_result);
        goto report;
    }
    mapped = 1;

    run_start = qbh_monotonic_ns();
    run_result = qwen3_probe_run_rpcmem2_probe(
        session.handle, shared_fd, QBH_RPCMEM2_REQUEST_BYTES);
    run_end = qbh_monotonic_ns();
    if (run_result != AEE_SUCCESS) {
        fprintf(stderr, "run_rpcmem2_probe failed: 0x%08x\n",
                (unsigned int)run_result);
        goto report;
    }

    for (i = 0U; i < QBH_RPCMEM2_SENTINEL_COUNT; ++i) {
        volatile uint32_t *sentinel =
            (volatile uint32_t *)(shared + offsets[i]);
        if (*sentinel != dsp_values[i] ||
            header->dsp_observed_values[i] != host_values[i]) {
            ++host_mismatch_count;
        }
    }

    exit_code = header->dsp_status == QBH_RPCMEM2_STATUS_OK &&
                        header->address_range_valid == 1U &&
                        header->sentinel_mismatch_count == 0U &&
                        host_mismatch_count == 0
                    ? 0
                    : 1;

report:
    if (session.handle != (remote_handle64)-1) {
        close_result = qbh_session_close(&session);
    }
    if (mapped) {
        unmap_result = fastrpc_munmap(
            CDSP_DOMAIN_ID, shared_fd, shared,
            (size_t)QBH_RPCMEM2_REQUEST_BYTES);
    }

    printf("{\"experiment\":\"EXP-0151\"," 
           "\"requested_bytes\":%" PRIu32 ","
           "\"rpcmem_alloc2_symbol\":%d,"
           "\"allocation_nonnull\":%d,\"allocation_errno\":%d,"
           "\"shared_fd\":%d,\"open_result\":%d,"
           "\"map_result\":%d,\"run_result\":%d,"
           "\"close_result\":%d,\"unmap_result\":%d,"
           "\"dsp_status\":%d,\"dsp_mmap_get_result\":%d,"
           "\"dsp_mmap_put_result\":%d,"
           "\"dsp_virtual_base\":%" PRIu32 ","
           "\"dsp_virtual_end\":%" PRIu32 ","
           "\"dsp_physical_base\":%" PRIu64 ","
           "\"address_range_valid\":%" PRIu32 ","
           "\"dsp_sentinel_mismatches\":%" PRIu32 ","
           "\"host_sentinel_mismatches\":%d,"
           "\"allocation_ns\":%" PRIu64 ","
           "\"map_ns\":%" PRIu64 ",\"run_ns\":%" PRIu64 ","
           "\"gate_pass\":%d}\n",
           QBH_RPCMEM2_REQUEST_BYTES, symbol_present, shared != NULL,
           allocation_errno, shared_fd, open_result, map_result, run_result,
           close_result, unmap_result,
           header != NULL ? header->dsp_status : 0,
           header != NULL ? header->mmap_get_result : AEE_EFAILED,
           header != NULL ? header->mmap_put_result : AEE_EFAILED,
           header != NULL ? header->dsp_virtual_base : 0U,
           header != NULL ? header->dsp_virtual_end : 0U,
           header != NULL ? header->dsp_physical_base : 0U,
           header != NULL ? header->address_range_valid : 0U,
           header != NULL ? header->sentinel_mismatch_count : 0U,
           host_mismatch_count,
           allocation_end >= allocation_start
               ? allocation_end - allocation_start
               : 0U,
           map_end >= map_start ? map_end - map_start : 0U,
           run_end >= run_start ? run_end - run_start : 0U,
           exit_code == 0 && close_result == AEE_SUCCESS &&
                   unmap_result == AEE_SUCCESS
               ? 1
               : 0);

    if (shared != NULL) {
        rpcmem_free(shared);
    }
    if (exit_code == 0 &&
        (close_result != AEE_SUCCESS || unmap_result != AEE_SUCCESS)) {
        exit_code = 1;
    }
    return exit_code;
}
