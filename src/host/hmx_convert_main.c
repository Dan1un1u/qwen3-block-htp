#include <AEEStdErr.h>
#include <inttypes.h>
#include <limits.h>
#include <remote.h>
#include <rpcmem.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hmx_convert_protocol.h"
#include "hmx_int8_tile.h"
#include "host/session.h"
#include "qwen3_probe.h"

static size_t align_up(size_t value, size_t alignment) {
    return (value + alignment - 1U) / alignment * alignment;
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

static int parse_i32(const char *text, int32_t *value) {
    char *end = NULL;
    long parsed = strtol(text, &end, 0);
    if (text[0] == '\0' || end == NULL || *end != '\0' ||
        parsed < INT32_MIN || parsed > INT32_MAX) {
        return -1;
    }
    *value = (int32_t)parsed;
    return 0;
}

static int write_file(const char *path, const void *data, size_t bytes) {
    FILE *stream = fopen(path, "wb");
    if (stream == NULL) {
        perror(path);
        return -1;
    }
    if (fwrite(data, 1U, bytes, stream) != bytes || fclose(stream) != 0) {
        fprintf(stderr, "failed to write %s\n", path);
        return -1;
    }
    return 0;
}

int main(int argc, char **argv) {
    struct qbh_session session = {(remote_handle64)-1, 0};
    struct qbh_hmx_convert_header *header;
    uint8_t *shared = NULL;
    uint8_t *activation;
    int8_t *packed_weight;
    uint32_t *bias_words;
    uint8_t *output;
    uint32_t lower_base;
    uint32_t lower_step;
    int32_t upper_bias;
    size_t activation_offset;
    size_t weight_offset;
    size_t bias_offset;
    size_t output_offset;
    size_t total_bytes;
    int shared_fd = -1;
    int mapped = 0;
    int rpc_result;
    int result = EXIT_FAILURE;

    if (argc != 5 || parse_u32(argv[1], &lower_base) != 0 ||
        parse_u32(argv[2], &lower_step) != 0 ||
        parse_i32(argv[3], &upper_bias) != 0) {
        fprintf(stderr,
                "usage: %s LOWER_BASE LOWER_STEP UPPER_BIAS OUTPUT_BIN\n",
                argv[0]);
        return EXIT_FAILURE;
    }

    activation_offset = align_up(sizeof(*header),
                                 QBH_HMX_CONVERT_ALIGNMENT);
    weight_offset = activation_offset + QBH_HMX_ACTIVATION_BYTES;
    bias_offset = align_up(weight_offset + QBH_HMX_WEIGHT_BYTES, 256U);
    output_offset = align_up(bias_offset + QBH_HMX_BIAS_BYTES,
                             QBH_HMX_OUTPUT_BYTES);
    total_bytes = output_offset + QBH_HMX_OUTPUT_BYTES;
    if (total_bytes > INT_MAX || total_bytes > UINT32_MAX) {
        return EXIT_FAILURE;
    }

    shared = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_FLAG_UNCACHED,
                          (int)total_bytes);
    if (shared == NULL) {
        fprintf(stderr, "rpcmem_alloc failed\n");
        goto cleanup;
    }
    shared_fd = rpcmem_to_fd(shared);
    if (shared_fd < 0) {
        fprintf(stderr, "rpcmem_to_fd failed\n");
        goto cleanup;
    }
    memset(shared, 0, total_bytes);
    header = (struct qbh_hmx_convert_header *)shared;
    activation = shared + activation_offset;
    packed_weight = (int8_t *)(shared + weight_offset);
    bias_words = (uint32_t *)(shared + bias_offset);
    output = shared + output_offset;

    header->magic = QBH_HMX_CONVERT_MAGIC;
    header->abi_version = QBH_HMX_CONVERT_ABI_VERSION;
    header->header_bytes = (uint32_t)sizeof(*header);
    header->total_bytes = (uint32_t)total_bytes;
    header->activation_offset = (uint32_t)activation_offset;
    header->weight_offset = (uint32_t)weight_offset;
    header->bias_offset = (uint32_t)bias_offset;
    header->output_offset = (uint32_t)output_offset;
    header->dsp_status = QBH_HMX_CONVERT_STATUS_HOST_INITIALIZED;

    for (uint32_t row = 0; row < QBH_HMX_SPATIAL; ++row) {
        for (uint32_t channel = 0;
             channel < QBH_HMX_INPUT_CHANNELS; ++channel) {
            activation[qbh_activation_offset(row, channel)] =
                (uint8_t)((row * 4U + channel / 8U) & UINT8_MAX);
        }
    }
    for (uint32_t channel = 0;
         channel < QBH_HMX_OUTPUT_CHANNELS; ++channel) {
        packed_weight[qbh_packed_weight_offset(channel, channel)] = 1;
        bias_words[channel] = lower_base + channel * lower_step;
        bias_words[QBH_HMX_OUTPUT_CHANNELS + channel] =
            (uint32_t)upper_bias;
    }
    memset(output, 0xa5, QBH_HMX_OUTPUT_BYTES);

    rpc_result = qbh_session_open(&session);
    if (rpc_result != AEE_SUCCESS) {
        goto cleanup;
    }
    rpc_result = qbh_session_prepare(&session);
    if (rpc_result != AEE_SUCCESS) {
        fprintf(stderr, "qbh_session_prepare failed: 0x%08x\n",
                (unsigned int)rpc_result);
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
    rpc_result = qwen3_probe_run_hmx_convert(
        session.handle, shared_fd, (uint32_t)total_bytes);
    if (rpc_result != AEE_SUCCESS ||
        header->dsp_status != QBH_HMX_CONVERT_STATUS_OK) {
        fprintf(stderr,
                "run_hmx_convert failed: rpc=0x%08x dsp=%d cache=%d "
                "lock=%d unlock=%d\n",
                (unsigned int)rpc_result, header->dsp_status,
                header->cache_status, header->hmx_lock_status,
                header->hmx_unlock_status);
        goto cleanup;
    }
    if (write_file(argv[4], output, QBH_HMX_OUTPUT_BYTES) != 0) {
        goto cleanup;
    }
    printf("{\"lower_base\":%" PRIu32
           ",\"lower_step\":%" PRIu32
           ",\"upper_bias\":%" PRId32
           ",\"vtcm_address\":%" PRIu32
           ",\"hmx_context_id\":%" PRIu32
           ",\"hmx_ticks\":%" PRIu64
           ",\"output\":\"%s\"}\n",
           lower_base, lower_step, upper_bias, header->vtcm_address,
           header->hmx_context_id, header->hmx_ticks, argv[4]);
    result = EXIT_SUCCESS;

cleanup:
    if (mapped) {
        (void)fastrpc_munmap(CDSP_DOMAIN_ID, shared_fd, shared,
                             total_bytes);
    }
    (void)qbh_session_close(&session);
    if (shared != NULL) {
        rpcmem_free(shared);
    }
    return result;
}
