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

#include "hmx_u8s8_projection.h"
#include "host/session.h"
#include "mlp_protocol.h"
#include "mlp_u8.h"
#include "qwen3_probe.h"

static const uint16_t qbh_hmx_integer_scale_words[19] = {
    0x0000, 0x6000, 0x6400, 0x6600, 0x6800,
    0x6900, 0x6a00, 0x6b00, 0x6c00, 0x6c80,
    0x6d00, 0x6d80, 0x6e00, 0x6e80, 0x6f00,
    0x6f80, 0x7000, 0x7040, 0x7080,
};

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
    if (text == NULL || text[0] == '\0' || end == NULL || *end != '\0' ||
        parsed > UINT32_MAX) {
        return -1;
    }
    *value = (uint32_t)parsed;
    return 0;
}

static uint8_t output_scale(uint32_t channel) {
    static const uint8_t scales[4] = {1U, 2U, 4U, 8U};
    return scales[channel & 3U];
}

static uint8_t clamp_u8(int32_t value) {
    if (value < 0) {
        return UINT8_C(0);
    }
    if (value > UINT8_MAX) {
        return UINT8_MAX;
    }
    return (uint8_t)value;
}

static void fill_input(uint8_t *input) {
    for (uint32_t row = 0; row < QBH_PROJ_M; ++row) {
        for (uint32_t channel = 0; channel < QBH_GATE_UP_K; ++channel) {
            input[(size_t)row * QBH_GATE_UP_K + channel] =
                (uint8_t)(QBH_MLP_ACTIVATION_ZERO_POINT +
                          (int32_t)((row * 5U + channel * 3U) % 17U) - 8);
        }
    }
}

static void fill_identity_weights(
    const struct qbh_projection_layout *layout, int8_t *logical_w4,
    int gate_up_pair) {
    memset(logical_w4, 0, layout->logical_weight_bytes);
    for (uint32_t output = 0; output < layout->n; ++output) {
        uint32_t input;
        int8_t value;
        if (gate_up_pair != 0) {
            uint32_t local = output % QBH_GATE_UP_N;
            if (output < QBH_GATE_UP_N) {
                input = local % QBH_GATE_UP_K;
                value = (local & 1U) == 0U ? INT8_C(1) : INT8_C(-1);
            } else {
                input = (local * 7U + 3U) % QBH_GATE_UP_K;
                value = (local % 3U) == 0U ? INT8_C(-1) : INT8_C(1);
            }
        } else {
            input = (output * 3U + 5U) % QBH_DOWN_K;
            value = (output & 1U) == 0U ? INT8_C(1) : INT8_C(-1);
        }
        logical_w4[(size_t)input * layout->n + output] = value;
    }
}

static void pack_w4_postscale(
    const struct qbh_projection_layout *layout,
    const int8_t *logical_w4, int gate_up_interleaved,
    uint8_t *stored_weights) {
    memset(stored_weights, 0, layout->stored_weight_bytes);
    for (uint32_t output_tile = 0; output_tile < layout->n_tiles;
         ++output_tile) {
        size_t bundle = qbh_projection_w4_bundle_offset(
            layout, output_tile);
        uint8_t *scales = stored_weights +
            qbh_projection_w4_scale_offset(layout, output_tile);
        uint32_t *bias = (uint32_t *)(stored_weights +
            qbh_projection_w4_bias_offset(layout, output_tile));
        uint32_t logical_output_tile = output_tile;
        if (gate_up_interleaved != 0) {
            logical_output_tile = output_tile / 2U +
                (output_tile & 1U) *
                    (QBH_GATE_UP_N / QBH_HMX_OUTPUT_CHANNELS);
        }
        for (uint32_t output_lane = 0;
             output_lane < QBH_HMX_OUTPUT_CHANNELS; ++output_lane) {
            uint32_t logical_n =
                logical_output_tile * QBH_HMX_OUTPUT_CHANNELS +
                output_lane;
            uint32_t scale = output_scale(logical_n);
            int32_t sum = 0;
            scales[output_lane] = (uint8_t)scale;
            for (uint32_t input = 0; input < layout->k; ++input) {
                int8_t q4 = logical_w4[(size_t)input * layout->n +
                                       logical_n];
                uint32_t input_tile = input / QBH_HMX_INPUT_CHANNELS;
                uint32_t input_lane = input % QBH_HMX_INPUT_CHANNELS;
                size_t physical_s8 =
                    (size_t)input_tile * QBH_HMX_WEIGHT_BYTES +
                    qbh_packed_weight_offset(input_lane, output_lane);
                size_t packed = bundle + physical_s8 / 2U;
                uint8_t nibble = (uint8_t)q4 & UINT8_C(0x0f);
                if ((physical_s8 & 1U) == 0U) {
                    stored_weights[packed] |= nibble;
                } else {
                    stored_weights[packed] |= (uint8_t)(nibble << 4U);
                }
                sum += q4;
            }
            bias[output_lane] = qbh_hmx_integer_scale_words[scale];
            bias[QBH_HMX_OUTPUT_CHANNELS + output_lane] =
                (uint32_t)(-QBH_MLP_ACTIVATION_ZERO_POINT * sum +
                           QBH_MLP_ACTIVATION_ZERO_POINT /
                               (int32_t)scale);
        }
    }
}

static void identity_reference(
    const uint8_t *input, uint8_t *middle, uint8_t *output) {
    for (uint32_t row = 0; row < QBH_PROJ_M; ++row) {
        for (uint32_t channel = 0; channel < QBH_GATE_UP_N; ++channel) {
            uint32_t gate_input = channel % QBH_GATE_UP_K;
            uint32_t up_input = (channel * 7U + 3U) % QBH_GATE_UP_K;
            int32_t gate_sign = (channel & 1U) == 0U ? 1 : -1;
            int32_t up_sign = (channel % 3U) == 0U ? -1 : 1;
            int32_t scale = output_scale(channel);
            uint8_t gate = clamp_u8(
                ((int32_t)input[(size_t)row * QBH_GATE_UP_K + gate_input] -
                 QBH_MLP_ACTIVATION_ZERO_POINT) * gate_sign * scale +
                QBH_MLP_ACTIVATION_ZERO_POINT);
            uint8_t up = clamp_u8(
                ((int32_t)input[(size_t)row * QBH_GATE_UP_K + up_input] -
                 QBH_MLP_ACTIVATION_ZERO_POINT) * up_sign * scale +
                QBH_MLP_ACTIVATION_ZERO_POINT);
            middle[(size_t)row * QBH_GATE_UP_N + channel] =
                qbh_mlp_gate_up_scalar(gate, up);
        }
    }
    for (uint32_t row = 0; row < QBH_PROJ_M; ++row) {
        for (uint32_t channel = 0; channel < QBH_DOWN_N; ++channel) {
            uint32_t input_channel = (channel * 3U + 5U) % QBH_DOWN_K;
            int32_t sign = (channel & 1U) == 0U ? 1 : -1;
            int32_t scale = output_scale(channel);
            output[(size_t)row * QBH_DOWN_N + channel] = clamp_u8(
                ((int32_t)middle[(size_t)row * QBH_DOWN_K + input_channel] -
                 QBH_MLP_ACTIVATION_ZERO_POINT) * sign * scale +
                QBH_MLP_ACTIVATION_ZERO_POINT);
        }
    }
}

static uint64_t checksum(const uint8_t *data, size_t bytes) {
    uint64_t value = UINT64_C(1469598103934665603);
    for (size_t index = 0; index < bytes; ++index) {
        value ^= data[index];
        value *= UINT64_C(1099511628211);
    }
    return value;
}

int main(int argc, char **argv) {
    struct qbh_projection_layout gate_layout;
    struct qbh_projection_layout down_layout;
    struct qbh_session session = {(remote_handle64)-1, 0};
    struct qbh_mlp_header *header = NULL;
    uint8_t *shared = NULL;
    uint8_t *input;
    uint8_t *gate_weights;
    uint8_t *down_weights;
    uint8_t *output;
    int8_t *gate_logical = NULL;
    int8_t *down_logical = NULL;
    uint8_t *reference_middle = NULL;
    uint8_t *reference_output = NULL;
    uint32_t repeats = 1U;
    uint32_t self_test = 1U;
    size_t input_offset;
    size_t gate_offset;
    size_t down_offset;
    size_t output_offset;
    size_t total_bytes;
    int shared_fd = -1;
    int mapped = 0;
    int rpc_result = AEE_EFAILED;
    int prepare_result = AEE_EFAILED;
    int release_result = AEE_EFAILED;
    int close_result = AEE_EFAILED;
    uint64_t host_start;
    uint64_t host_end;
    uint64_t reference_start;
    uint64_t reference_end;
    uint32_t mismatches = 0U;
    int exit_code = 1;

    if ((argc > 1 && parse_u32(argv[1], &repeats) != 0) ||
        (argc > 2 && parse_u32(argv[2], &self_test) != 0) || argc > 3 ||
        repeats == 0U || repeats > QBH_HMX_MAX_REPEATS) {
        fprintf(stderr, "usage: %s [repeat_count] [self_test_0_or_1]\n",
                argv[0]);
        return 2;
    }
    if (qbh_projection_layout_init(
            QBH_PROJECTION_GATE_UP_PAIR,
            QBH_WEIGHT_PACKED_W4_HMX_SCALE,
            QBH_PHYSICAL_PLAN_STREAMING_E7_DMA_CHAIN4, 8U,
            QBH_W4_COARSE_CHUNK_TILES, &gate_layout) != 0 ||
        qbh_projection_layout_init(
            QBH_PROJECTION_DOWN, QBH_WEIGHT_PACKED_W4_HMX_SCALE,
            QBH_PHYSICAL_PLAN_CHUNKED_DMA_BATCH2, 4U,
            QBH_W4_WIDE_CHUNK_TILES, &down_layout) != 0) {
        fprintf(stderr, "MLP projection layout initialization failed\n");
        return 2;
    }

    input_offset = align_up(sizeof(struct qbh_mlp_header),
                            QBH_PROBE_ALIGNMENT);
    gate_offset = input_offset + align_up(gate_layout.activation_bytes,
                                          QBH_PROBE_ALIGNMENT);
    down_offset = gate_offset + align_up(gate_layout.stored_weight_bytes,
                                         QBH_PROBE_ALIGNMENT);
    output_offset = down_offset + align_up(down_layout.stored_weight_bytes,
                                           QBH_PROBE_ALIGNMENT);
    total_bytes = output_offset + align_up(down_layout.output_bytes,
                                           QBH_PROBE_ALIGNMENT);
    if (total_bytes > UINT32_MAX || total_bytes > INT_MAX) {
        fprintf(stderr, "MLP rpcmem region too large: %zu\n", total_bytes);
        return 2;
    }

    gate_logical = malloc(gate_layout.logical_weight_bytes);
    down_logical = malloc(down_layout.logical_weight_bytes);
    reference_middle = malloc(QBH_MLP_INTERMEDIATE_BYTES);
    reference_output = malloc(down_layout.output_bytes);
    if (gate_logical == NULL || down_logical == NULL ||
        reference_middle == NULL || reference_output == NULL) {
        fprintf(stderr, "host allocation failed\n");
        goto cleanup;
    }
    fill_identity_weights(&gate_layout, gate_logical, 1);
    fill_identity_weights(&down_layout, down_logical, 0);

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
    header = (struct qbh_mlp_header *)shared;
    input = shared + input_offset;
    gate_weights = shared + gate_offset;
    down_weights = shared + down_offset;
    output = shared + output_offset;

    header->magic = QBH_MLP_MAGIC;
    header->abi_version = QBH_MLP_ABI_VERSION;
    header->experiment = QBH_MLP_EXPERIMENT;
    header->header_bytes = (uint32_t)sizeof(*header);
    header->shared_bytes = (uint32_t)total_bytes;
    header->repeat_count = repeats;
    header->run_activation_self_test = self_test != 0U;
    header->pattern = 1U;
    header->input_offset = (uint32_t)input_offset;
    header->gate_up_weight_offset = (uint32_t)gate_offset;
    header->down_weight_offset = (uint32_t)down_offset;
    header->output_offset = (uint32_t)output_offset;
    header->input_bytes = gate_layout.activation_bytes;
    header->gate_up_weight_bytes = gate_layout.stored_weight_bytes;
    header->down_weight_bytes = down_layout.stored_weight_bytes;
    header->output_bytes = down_layout.output_bytes;
    header->dsp_status = QBH_MLP_STATUS_HOST_READY;

    fill_input(input);
    pack_w4_postscale(&gate_layout, gate_logical, 1, gate_weights);
    pack_w4_postscale(&down_layout, down_logical, 0, down_weights);
    memset(output, 0xa5, down_layout.output_bytes);

    reference_start = monotonic_ns();
    identity_reference(input, reference_middle, reference_output);
    reference_end = monotonic_ns();

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
    prepare_result = qbh_session_prepare(&session);
    if (prepare_result != AEE_SUCCESS) {
        fprintf(stderr, "prepare failed: 0x%08x\n",
                (unsigned int)prepare_result);
        goto cleanup;
    }

    host_start = monotonic_ns();
    rpc_result = qwen3_probe_run_mlp(
        session.handle, shared_fd, (uint32_t)total_bytes);
    host_end = monotonic_ns();
    if (rpc_result != AEE_SUCCESS) {
        fprintf(stderr, "run_mlp failed: 0x%08x dsp_status=%d\n",
                (unsigned int)rpc_result, header->dsp_status);
        goto cleanup;
    }
    for (size_t index = 0; index < down_layout.output_bytes; ++index) {
        if (output[index] != reference_output[index]) {
            if (mismatches < 8U) {
                fprintf(stderr,
                        "mismatch index=%zu expected=%u actual=%u\n",
                        index, (unsigned int)reference_output[index],
                        (unsigned int)output[index]);
            }
            ++mismatches;
        }
    }

    release_result = qbh_session_release(&session);
    close_result = qbh_session_close(&session);
    if (release_result != AEE_SUCCESS || close_result != AEE_SUCCESS) {
        goto cleanup;
    }

    printf("{\"experiment\":\"EXP-0021\","
           "\"implementation\":\"vtcm_resident_mlp_bringup\","
           "\"repeat_count\":%" PRIu32 ","
           "\"rpc_result\":%d,\"dsp_status\":%d,"
           "\"mismatches\":%" PRIu32 ","
           "\"reference_checksum\":%" PRIu64 ","
           "\"output_checksum\":%" PRIu64 ","
           "\"host_wall_ns\":%" PRIu64 ","
           "\"reference_wall_ns\":%" PRIu64 ","
           "\"vtcm_requested_bytes\":%" PRIu32 ","
           "\"vtcm_acquired_bytes\":%" PRIu32 ","
           "\"vtcm_peak_plan_bytes\":%" PRIu32 ","
           "\"gate_up_output_vtcm_bytes\":%" PRIu32 ","
           "\"middle_vtcm_bytes\":%" PRIu32 ","
           "\"final_output_vtcm_bytes\":%" PRIu32 ","
           "\"gate_up_pair_slot_count\":%" PRIu32 ","
           "\"gate_up_pair_publish_count\":%" PRIu32 ","
           "\"gate_up_pair_consume_count\":%" PRIu32 ","
           "\"gate_up_full_tensor_materialized\":%" PRIu32 ","
           "\"activation_self_test_cases\":%" PRIu32 ","
           "\"activation_self_test_mismatches\":%" PRIu32 ","
           "\"intermediate_ddr_read_bytes\":%" PRIu32 ","
           "\"intermediate_ddr_write_bytes\":%" PRIu32 ","
           "\"intermediate_dma_descriptor_count\":%" PRIu32 ","
           "\"intermediate_spill_fill_count\":%" PRIu32 ","
           "\"gate_up_output_dma_descriptor_count\":%" PRIu32 ","
           "\"middle_dma_descriptor_count\":%" PRIu32 ","
           "\"final_output_dma_descriptor_count\":%" PRIu32 ","
           "\"final_output_dma_timeout_count\":%" PRIu32 ","
           "\"gate_up_hvx_hmx_overlap\":%" PRIu32 ","
           "\"down_hvx_hmx_overlap\":%" PRIu32 ","
           "\"gate_up_ticks\":%" PRIu64 ","
           "\"activation_ticks\":%" PRIu64 ","
           "\"down_ticks\":%" PRIu64 ","
           "\"total_ticks\":%" PRIu64 "}\n",
           repeats, rpc_result, header->dsp_status, mismatches,
           checksum(reference_output, down_layout.output_bytes),
           checksum(output, down_layout.output_bytes), host_end - host_start,
           reference_end - reference_start, header->vtcm_requested_bytes,
           header->vtcm_acquired_bytes, header->vtcm_peak_plan_bytes,
           header->gate_up_output_vtcm_bytes, header->middle_vtcm_bytes,
           header->final_output_vtcm_bytes,
           header->gate_up_pair_slot_count,
           header->gate_up_pair_publish_count,
           header->gate_up_pair_consume_count,
           header->gate_up_full_tensor_materialized,
           header->activation_self_test_cases,
           header->activation_self_test_mismatches,
           header->intermediate_ddr_read_bytes,
           header->intermediate_ddr_write_bytes,
           header->intermediate_dma_descriptor_count,
           header->intermediate_spill_fill_count,
           header->gate_up_output_dma_descriptor_count,
           header->middle_dma_descriptor_count,
           header->final_output_dma_descriptor_count,
           header->final_output_dma_timeout_count,
           header->gate_up_phase.hvx_hmx_overlap_observed,
           header->down_phase.hvx_hmx_overlap_observed,
           header->gate_up_ticks, header->activation_ticks,
           header->down_ticks, header->total_ticks);
    exit_code = mismatches == 0U &&
                        header->activation_self_test_mismatches == 0U &&
                        header->intermediate_ddr_read_bytes == 0U &&
                        header->intermediate_ddr_write_bytes == 0U &&
                        header->intermediate_dma_descriptor_count == 0U &&
                        header->intermediate_spill_fill_count == 0U &&
                        header->gate_up_full_tensor_materialized == 0U &&
                        header->gate_up_pair_publish_count ==
                            repeats * QBH_GATE_UP_N /
                                QBH_HMX_OUTPUT_CHANNELS &&
                        header->gate_up_pair_consume_count ==
                            header->gate_up_pair_publish_count
                    ? 0
                    : 1;

cleanup:
    if (session.handle != (remote_handle64)-1) {
        (void)qbh_session_close(&session);
    }
    if (mapped) {
        (void)fastrpc_munmap(CDSP_DOMAIN_ID, shared_fd, shared,
                             total_bytes);
    }
    if (shared != NULL) {
        rpcmem_free(shared);
    }
    free(reference_output);
    free(reference_middle);
    free(down_logical);
    free(gate_logical);
    return exit_code;
}
