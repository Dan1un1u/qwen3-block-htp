#include <AEEStdErr.h>
#include <inttypes.h>
#include <limits.h>
#include <remote.h>
#include <rpcmem.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "hmx_int8_tile.h"
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

static const char *pattern_name(uint32_t pattern) {
    switch (pattern) {
        case QBH_PATTERN_IDENTITY:
            return "identity";
        case QBH_PATTERN_SIGNED:
            return "signed";
        case QBH_PATTERN_STRUCTURED:
            return "structured";
        case QBH_PATTERN_BOUNDARY:
            return "boundary";
        default:
            return "invalid";
    }
}

static int parse_pattern(const char *text, uint32_t *pattern) {
    uint32_t parsed;
    if (strcmp(text, "identity") == 0) {
        *pattern = QBH_PATTERN_IDENTITY;
        return 0;
    }
    if (strcmp(text, "signed") == 0) {
        *pattern = QBH_PATTERN_SIGNED;
        return 0;
    }
    if (strcmp(text, "structured") == 0) {
        *pattern = QBH_PATTERN_STRUCTURED;
        return 0;
    }
    if (strcmp(text, "boundary") == 0) {
        *pattern = QBH_PATTERN_BOUNDARY;
        return 0;
    }
    if (parse_u32(text, &parsed) == 0 &&
        parsed >= QBH_PATTERN_IDENTITY && parsed <= QBH_PATTERN_BOUNDARY) {
        *pattern = parsed;
        return 0;
    }
    return -1;
}

static void fill_pattern(uint32_t pattern, uint8_t *activation,
                         int8_t *weight) {
    uint32_t spatial;
    uint32_t input_channel;
    uint32_t output_channel;

    for (spatial = 0; spatial < QBH_HMX_SPATIAL; ++spatial) {
        for (input_channel = 0; input_channel < QBH_HMX_INPUT_CHANNELS;
             ++input_channel) {
            uint8_t value;
            switch (pattern) {
                case QBH_PATTERN_IDENTITY:
                    value = (uint8_t)(QBH_HMX_DEFAULT_ZERO_POINT +
                                      ((spatial + input_channel) & 7U));
                    break;
                case QBH_PATTERN_SIGNED:
                    value = (uint8_t)(QBH_HMX_DEFAULT_ZERO_POINT +
                                      (int32_t)((spatial * 5U +
                                                 input_channel * 3U) % 17U) -
                                      8);
                    break;
                case QBH_PATTERN_STRUCTURED:
                    value = (uint8_t)(96U +
                                      ((spatial * 11U + input_channel * 13U) %
                                       65U));
                    break;
                default:
                    value = ((spatial + input_channel) & 1U) != 0U
                                ? UINT8_MAX
                                : UINT8_C(0);
                    break;
            }
            activation[qbh_activation_offset(spatial, input_channel)] = value;
        }
    }

    for (input_channel = 0; input_channel < QBH_HMX_INPUT_CHANNELS;
         ++input_channel) {
        for (output_channel = 0; output_channel < QBH_HMX_OUTPUT_CHANNELS;
             ++output_channel) {
            int8_t value;
            switch (pattern) {
                case QBH_PATTERN_IDENTITY:
                    value = input_channel == output_channel ? INT8_C(1)
                                                            : INT8_C(0);
                    break;
                case QBH_PATTERN_SIGNED:
                    value = (int8_t)((int32_t)((input_channel * 7U +
                                                output_channel * 5U) % 7U) -
                                     3);
                    break;
                case QBH_PATTERN_STRUCTURED:
                    value = ((input_channel + 3U * output_channel) % 5U) == 0U
                                ? (int8_t)((int32_t)((input_channel +
                                                     output_channel) % 5U) -
                                           2)
                                : INT8_C(0);
                    break;
                default:
                    value = ((input_channel + output_channel) & 1U) != 0U
                                ? INT8_C(127)
                                : INT8_MIN;
                    break;
            }
            weight[qbh_logical_weight_offset(input_channel, output_channel)] =
                value;
        }
    }
}

static uint8_t clamp_to_u8(int32_t value) {
    if (value < 0) {
        return UINT8_C(0);
    }
    if (value > UINT8_MAX) {
        return UINT8_MAX;
    }
    return (uint8_t)value;
}

static uint32_t validate_output(const uint8_t *activation,
                                const int8_t *weight, const uint8_t *output,
                                int32_t input_zero_point,
                                uint8_t *reference_min,
                                uint8_t *reference_max,
                                uint64_t *reference_checksum) {
    uint32_t mismatches = 0;
    uint32_t spatial;
    uint32_t output_channel;

    *reference_min = UINT8_MAX;
    *reference_max = 0;
    *reference_checksum = 0;
    for (spatial = 0; spatial < QBH_HMX_SPATIAL; ++spatial) {
        for (output_channel = 0; output_channel < QBH_HMX_OUTPUT_CHANNELS;
             ++output_channel) {
            int32_t accumulator = 0;
            uint32_t input_channel;
            uint8_t expected;

            for (input_channel = 0;
                 input_channel < QBH_HMX_INPUT_CHANNELS; ++input_channel) {
                accumulator +=
                    ((int32_t)activation[qbh_activation_offset(
                         spatial, input_channel)] -
                     input_zero_point) *
                    (int32_t)weight[qbh_logical_weight_offset(
                        input_channel, output_channel)];
            }
            expected = clamp_to_u8(accumulator);
            if (expected < *reference_min) {
                *reference_min = expected;
            }
            if (expected > *reference_max) {
                *reference_max = expected;
            }
            *reference_checksum += expected;
            if (output[qbh_output_offset(spatial, output_channel)] != expected) {
                if (mismatches < 8U) {
                    fprintf(stderr,
                            "mismatch spatial=%" PRIu32 " channel=%" PRIu32
                            " expected=%u actual=%u accumulator=%" PRId32 "\n",
                            spatial, output_channel, (unsigned int)expected,
                            (unsigned int)output[qbh_output_offset(
                                spatial, output_channel)],
                            accumulator);
                }
                ++mismatches;
            }
        }
    }
    return mismatches;
}

int main(int argc, char **argv) {
    struct qbh_session session = {(remote_handle64)-1};
    struct qbh_probe_header *header = NULL;
    uint8_t *shared = NULL;
    uint8_t *activation;
    int8_t *weight;
    uint8_t *output;
    uint32_t pattern = QBH_PATTERN_IDENTITY;
    uint32_t repeats = QBH_HMX_DEFAULT_REPEATS;
    size_t activation_offset;
    size_t weight_offset;
    size_t output_offset;
    size_t total_bytes;
    uint64_t host_start;
    uint64_t host_end;
    uint64_t reference_checksum;
    uint8_t reference_min;
    uint8_t reference_max;
    uint32_t mismatches;
    int shared_fd = -1;
    int mapped = 0;
    int result = EXIT_FAILURE;
    int rpc_result;

    if (argc > 1 && parse_pattern(argv[1], &pattern) != 0) {
        fprintf(stderr, "invalid pattern: %s\n", argv[1]);
        return EXIT_FAILURE;
    }
    if (argc > 2 && parse_u32(argv[2], &repeats) != 0) {
        fprintf(stderr, "invalid repeat count: %s\n", argv[2]);
        return EXIT_FAILURE;
    }
    if (repeats == 0 || repeats > QBH_HMX_MAX_REPEATS) {
        fprintf(stderr, "repeat count must be in [1, %u]\n",
                (unsigned int)QBH_HMX_MAX_REPEATS);
        return EXIT_FAILURE;
    }

    activation_offset = align_up(sizeof(*header), QBH_PROBE_ALIGNMENT);
    weight_offset = activation_offset +
                    align_up(QBH_HMX_ACTIVATION_BYTES, QBH_PROBE_ALIGNMENT);
    output_offset = weight_offset +
                    align_up(QBH_HMX_WEIGHT_BYTES, QBH_PROBE_ALIGNMENT);
    total_bytes = output_offset +
                  align_up(QBH_HMX_OUTPUT_BYTES, QBH_PROBE_ALIGNMENT);

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
    header->pattern = pattern;
    header->activation_offset = (uint32_t)activation_offset;
    header->weight_offset = (uint32_t)weight_offset;
    header->output_offset = (uint32_t)output_offset;
    header->input_zero_point = QBH_HMX_DEFAULT_ZERO_POINT;
    header->repeat_count = repeats;
    header->dsp_status = QBH_PROBE_STATUS_HOST_INITIALIZED;

    activation = shared + activation_offset;
    weight = (int8_t *)(shared + weight_offset);
    output = shared + output_offset;
    fill_pattern(pattern, activation, weight);
    memset(output, 0xa5, QBH_HMX_OUTPUT_BYTES);

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
        fprintf(stderr, "qwen3_probe_run failed: 0x%08x dsp_status=%d "
                        "hmx_resource=%d hmx_lock=%d\n",
                (unsigned int)rpc_result, header->dsp_status,
                header->hmx_resource_status, header->hmx_lock_status);
        goto cleanup;
    }

    mismatches = validate_output(
        activation, weight, output, (int32_t)header->input_zero_point,
        &reference_min, &reference_max, &reference_checksum);

    printf("{\"experiment\":\"EXP-0002\","
           "\"pattern\":\"%s\",\"repeat_count\":%" PRIu32 ","
           "\"rpc_result\":%d,\"dsp_status\":%d,"
           "\"mismatches\":%" PRIu32 ","
           "\"reference_min\":%u,\"reference_max\":%u,"
           "\"reference_checksum\":%" PRIu64 ","
           "\"host_wall_ns\":%" PRIu64 ","
           "\"qtimer_ticks\":%" PRIu64 ","
           "\"pcycles\":%" PRIu64 ","
           "\"vtcm_requested_bytes\":%" PRIu32 ","
           "\"vtcm_acquired_bytes\":%" PRIu32 ","
           "\"hmx_resource_status\":%d,\"hmx_lock_status\":%d,"
           "\"hmx_unlock_status\":%d,\"hmx_release_status\":%d,"
           "\"hmx_thread_create_status\":%d,"
           "\"hmx_thread_join_status\":%d,"
           "\"hmx_power_up_status\":%d,"
           "\"hmx_power_down_status\":%d,"
           "\"hmx_execution_count\":%" PRIu32 ","
           "\"hvx_lock_status\":%d,\"hvx_unlock_status\":%d}\n",
           pattern_name(pattern), repeats, rpc_result, header->dsp_status,
           mismatches, (unsigned int)reference_min,
           (unsigned int)reference_max, reference_checksum,
           host_end - host_start, header->qtimer_elapsed,
           header->pcycles_end - header->pcycles_start,
           header->vtcm_requested_bytes, header->vtcm_acquired_bytes,
           header->hmx_resource_status, header->hmx_lock_status,
           header->hmx_unlock_status, header->hmx_release_status,
           header->hmx_thread_create_status,
           header->hmx_thread_join_status,
           header->hmx_power_up_status,
           header->hmx_power_down_status,
           header->hmx_execution_count,
           header->hvx_lock_status, header->hvx_unlock_status);

    if (header->dsp_status == QBH_PROBE_STATUS_OK && mismatches == 0 &&
        header->qtimer_end > header->qtimer_start &&
        header->vtcm_acquired_bytes >= QBH_HMX_VTCM_BYTES &&
        header->hmx_resource_status == 0 && header->hmx_lock_status == 0 &&
        header->hmx_unlock_status == 0 && header->hmx_release_status == 0 &&
        header->hmx_thread_create_status == 0 &&
        header->hmx_thread_join_status == 0 &&
        header->hmx_power_up_status == 0 &&
        header->hmx_power_down_status == 0 &&
        header->hmx_execution_count == repeats &&
        header->hvx_lock_status == 0 && header->hvx_unlock_status == 0) {
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
