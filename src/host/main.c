#include <AEEStdErr.h>
#include <inttypes.h>
#include <remote.h>
#include <rpcmem.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "host/session.h"
#include "probe_protocol.h"
#include "qwen3_probe.h"

static size_t align_up(size_t value, size_t alignment) {
    return (value + alignment - 1U) / alignment * alignment;
}

static uint64_t monotonic_ns(void) {
    struct timespec now;
    (void)clock_gettime(CLOCK_MONOTONIC, &now);
    return (uint64_t)now.tv_sec * UINT64_C(1000000000) +
           (uint64_t)now.tv_nsec;
}

static int parse_u32(const char *text, uint32_t *value) {
    char *end = NULL;
    unsigned long parsed = strtoul(text, &end, 0);
    if (text[0] == '\0' || end == NULL || *end != '\0' ||
        parsed > UINT32_MAX) {
        return -1;
    }
    *value = (uint32_t)parsed;
    return 0;
}

int main(int argc, char **argv) {
    struct qbh_session session = {(remote_handle64)-1};
    struct qbh_probe_header *header = NULL;
    uint8_t *shared = NULL;
    uint8_t *input;
    uint8_t *output;
    uint32_t length = QBH_PROBE_DEFAULT_LENGTH;
    uint32_t addend = 29;
    size_t input_offset;
    size_t output_offset;
    size_t total_bytes;
    uint64_t host_start;
    uint64_t host_end;
    uint32_t mismatches = 0;
    int shared_fd = -1;
    int mapped = 0;
    int result = EXIT_FAILURE;
    int rpc_result;

    if (argc > 1 && parse_u32(argv[1], &length) != 0) {
        fprintf(stderr, "invalid vector length: %s\n", argv[1]);
        return EXIT_FAILURE;
    }
    if (argc > 2 && parse_u32(argv[2], &addend) != 0) {
        fprintf(stderr, "invalid addend: %s\n", argv[2]);
        return EXIT_FAILURE;
    }
    if (length == 0 || addend > UINT8_MAX) {
        fprintf(stderr, "length must be positive and addend must fit U8\n");
        return EXIT_FAILURE;
    }

    input_offset = align_up(sizeof(*header), 4096);
    output_offset = input_offset + align_up(length, 4096);
    total_bytes = output_offset + align_up(length, 4096);
    if (total_bytes > UINT32_MAX) {
        fprintf(stderr, "probe allocation is too large\n");
        return EXIT_FAILURE;
    }

    shared = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_FLAG_UNCACHED,
                          (int)total_bytes);
    if (shared == NULL) {
        fprintf(stderr, "rpcmem_alloc failed for %zu bytes\n", total_bytes);
        goto cleanup;
    }
    shared_fd = rpcmem_to_fd(shared);
    if (shared_fd < 0) {
        fprintf(stderr, "rpcmem_to_fd failed\n");
        goto cleanup;
    }

    memset(shared, 0, total_bytes);
    header = (struct qbh_probe_header *)shared;
    header->magic = QBH_PROBE_MAGIC;
    header->abi_version = QBH_PROBE_ABI_VERSION;
    header->header_bytes = (uint32_t)sizeof(*header);
    header->total_bytes = (uint32_t)total_bytes;
    header->vector_length = length;
    header->input_offset = (uint32_t)input_offset;
    header->output_offset = (uint32_t)output_offset;
    header->addend = addend;
    header->dsp_status = QBH_PROBE_STATUS_HOST_INITIALIZED;

    input = shared + input_offset;
    output = shared + output_offset;
    for (uint32_t index = 0; index < length; ++index) {
        input[index] = (uint8_t)((index * UINT32_C(131) + UINT32_C(17)) &
                                 UINT32_C(0xff));
        output[index] = UINT8_C(0xa5);
    }

    rpc_result = qbh_session_open(&session);
    if (rpc_result != AEE_SUCCESS) {
        goto cleanup;
    }

    rpc_result = fastrpc_mmap(CDSP_DOMAIN_ID, shared_fd, shared, 0,
                              total_bytes, FASTRPC_MAP_FD);
    if (rpc_result != AEE_SUCCESS) {
        fprintf(stderr, "fastrpc_mmap failed: 0x%08x\n",
                (unsigned int)rpc_result);
        goto cleanup;
    }
    mapped = 1;

    host_start = monotonic_ns();
    rpc_result = qwen3_probe_run(session.handle, shared_fd,
                                 (uint32_t)total_bytes);
    host_end = monotonic_ns();
    if (rpc_result != AEE_SUCCESS) {
        fprintf(stderr, "qwen3_probe_run failed: 0x%08x\n",
                (unsigned int)rpc_result);
        goto cleanup;
    }

    for (uint32_t index = 0; index < length; ++index) {
        uint8_t expected = (uint8_t)(input[index] + (uint8_t)addend);
        if (output[index] != expected) {
            if (mismatches < 8) {
                fprintf(stderr,
                        "mismatch index=%" PRIu32 " expected=%u actual=%u\n",
                        index, (unsigned int)expected,
                        (unsigned int)output[index]);
            }
            ++mismatches;
        }
    }

    printf("{\"experiment\":\"EXP-0001\","
           "\"length\":%" PRIu32 ",\"addend\":%" PRIu32 ","
           "\"rpc_result\":%d,\"dsp_status\":%d,"
           "\"mismatches\":%" PRIu32 ","
           "\"host_wall_ns\":%" PRIu64 ","
           "\"qtimer_ticks\":%" PRIu64 ","
           "\"pcycles\":%" PRIu64 ","
           "\"vtcm_requested_bytes\":%" PRIu32 ","
           "\"vtcm_acquired_bytes\":%" PRIu32 "}\n",
           length, addend, rpc_result, header->dsp_status, mismatches,
           host_end - host_start, header->qtimer_elapsed,
           header->pcycles_end - header->pcycles_start,
           header->vtcm_requested_bytes, header->vtcm_acquired_bytes);

    if (header->dsp_status == QBH_PROBE_STATUS_OK && mismatches == 0 &&
        header->qtimer_end > header->qtimer_start &&
        header->vtcm_acquired_bytes >= 2U * align_up(length,
                                                     QBH_PROBE_ALIGNMENT)) {
        result = EXIT_SUCCESS;
    }

cleanup:
    if (mapped) {
        int unmap_result = fastrpc_munmap(CDSP_DOMAIN_ID, shared_fd, shared,
                                          total_bytes);
        if (unmap_result != AEE_SUCCESS) {
            fprintf(stderr, "fastrpc_munmap failed: 0x%08x\n",
                    (unsigned int)unmap_result);
            result = EXIT_FAILURE;
        }
    }
    qbh_session_close(&session);
    if (shared != NULL) {
        rpcmem_free(shared);
    }
    return result;
}
