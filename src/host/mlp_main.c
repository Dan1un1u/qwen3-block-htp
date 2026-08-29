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

static size_t qbh_align_up(size_t value, size_t alignment) {
    return (value + alignment - 1U) / alignment * alignment;
}

static uint64_t qbh_monotonic_ns(void) {
    struct timespec now;
    (void)clock_gettime(CLOCK_MONOTONIC, &now);
    return (uint64_t)now.tv_sec * UINT64_C(1000000000) +
           (uint64_t)now.tv_nsec;
}

static int qbh_parse_u32(const char *text, uint32_t *value) {
    char *end = NULL;
    unsigned long parsed;
    if (text == NULL || value == NULL) {
        return -1;
    }
    parsed = strtoul(text, &end, 0);
    if (text[0] == '\0' || end == NULL || *end != '\0' ||
        parsed > UINT32_MAX) {
        return -1;
    }
    *value = (uint32_t)parsed;
    return 0;
}

static int qbh_parse_storage(const char *text, uint32_t *storage) {
    if (strcmp(text, "packed_w4") == 0 ||
        strcmp(text, "packed") == 0) {
        *storage = QBH_WEIGHT_PACKED_W4_HMX_SCALE;
        return 0;
    }
    if (strcmp(text, "expanded_s8") == 0 ||
        strcmp(text, "expanded") == 0) {
        *storage = QBH_WEIGHT_EXPANDED_S8;
        return 0;
    }
    return -1;
}

static const char *qbh_storage_name(uint32_t storage) {
    return storage == QBH_WEIGHT_EXPANDED_S8
               ? "expanded_s8_control"
               : "packed_w4_candidate";
}

static int qbh_read_file(const char *package, const char *name,
                         void *destination, size_t expected_bytes) {
    char path[PATH_MAX];
    FILE *stream;
    long bytes;
    int written = snprintf(path, sizeof(path), "%s/%s", package, name);
    if (written < 0 || (size_t)written >= sizeof(path)) {
        return -1;
    }
    stream = fopen(path, "rb");
    if (stream == NULL || fseek(stream, 0, SEEK_END) != 0) {
        if (stream != NULL) {
            fclose(stream);
        }
        return -1;
    }
    bytes = ftell(stream);
    if (bytes < 0 || (size_t)bytes != expected_bytes ||
        fseek(stream, 0, SEEK_SET) != 0 ||
        fread(destination, 1, expected_bytes, stream) != expected_bytes ||
        fclose(stream) != 0) {
        return -1;
    }
    return 0;
}

static uint64_t qbh_checksum(const uint8_t *data, size_t bytes) {
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
    uint8_t *reference = NULL;
    uint8_t *input;
    uint8_t *gate_weights;
    uint8_t *down_weights;
    uint8_t *activation_lut;
    uint8_t *gate_up_multipliers;
    uint8_t *down_multipliers;
    uint8_t *output;
    uint32_t storage = 0U;
    uint32_t repeats = 1U;
    uint32_t self_test = 1U;
    uint32_t gate_up_workers = 2U;
    size_t input_offset;
    size_t gate_offset;
    size_t down_offset;
    size_t lut_offset;
    size_t gate_up_multiplier_offset;
    size_t down_multiplier_offset;
    size_t output_offset;
    size_t total_bytes = 0U;
    const char *gate_file;
    const char *down_file;
    int shared_fd = -1;
    int mapped = 0;
    int warmup_result = AEE_EFAILED;
    int run_result = AEE_EFAILED;
    int prepare_result = AEE_EFAILED;
    int release_result = AEE_EFAILED;
    int close_result = AEE_EFAILED;
    uint64_t warmup_start;
    uint64_t warmup_end;
    uint64_t run_start;
    uint64_t run_end;
    uint32_t warmup_mismatches = 0U;
    uint32_t mismatches = 0U;
    uint32_t warmup_run_index = 0U;
    uint64_t warmup_checksum = 0U;
    int exit_code = 1;

    if (argc < 3 || argc > 6 ||
        qbh_parse_storage(argv[2], &storage) != 0 ||
        (argc >= 4 && qbh_parse_u32(argv[3], &repeats) != 0) ||
        (argc >= 5 && qbh_parse_u32(argv[4], &self_test) != 0) ||
        (argc >= 6 && qbh_parse_u32(argv[5], &gate_up_workers) != 0) ||
        (gate_up_workers != 2U && gate_up_workers != 3U &&
         gate_up_workers != 4U && gate_up_workers != 6U) ||
        repeats == 0U || repeats > QBH_HMX_MAX_REPEATS) {
        fprintf(stderr,
                "usage: %s PACKAGE_DIR packed_w4|expanded_s8 "
                "[repeat_count] [self_test_0_or_1] "
                "[gate_up_workers_2_3_4_6]\n",
                argv[0]);
        return 2;
    }

    if (qbh_projection_layout_init(
            QBH_PROJECTION_GATE_UP_PAIR, storage,
            QBH_PHYSICAL_PLAN_STREAMING_E7_DMA_CHAIN4, 8U,
            QBH_W4_COARSE_CHUNK_TILES, &gate_layout) != 0 ||
        qbh_projection_layout_init(
            QBH_PROJECTION_DOWN, storage,
            QBH_PHYSICAL_PLAN_CHUNKED_DMA_BATCH2, 4U,
            QBH_W4_WIDE_CHUNK_TILES, &down_layout) != 0) {
        fprintf(stderr, "EXP-0040 projection layout initialization failed\n");
        return 2;
    }

    input_offset = qbh_align_up(sizeof(struct qbh_mlp_header),
                                QBH_PROBE_ALIGNMENT);
    gate_offset = input_offset + qbh_align_up(gate_layout.activation_bytes,
                                              QBH_PROBE_ALIGNMENT);
    down_offset = gate_offset + qbh_align_up(gate_layout.stored_weight_bytes,
                                             QBH_PROBE_ALIGNMENT);
    lut_offset = down_offset + qbh_align_up(down_layout.stored_weight_bytes,
                                            QBH_PROBE_ALIGNMENT);
    gate_up_multiplier_offset = lut_offset + qbh_align_up(
        QBH_MLP_LUT_BYTES, QBH_PROBE_ALIGNMENT);
    down_multiplier_offset = gate_up_multiplier_offset + qbh_align_up(
        QBH_MLP_GATE_UP_MULTIPLIER_BYTES, QBH_PROBE_ALIGNMENT);
    output_offset = down_multiplier_offset + qbh_align_up(
        QBH_MLP_DOWN_MULTIPLIER_BYTES, QBH_PROBE_ALIGNMENT);
    total_bytes = output_offset + qbh_align_up(
        down_layout.output_bytes, QBH_PROBE_ALIGNMENT);
    if (total_bytes > INT_MAX || total_bytes > UINT32_MAX) {
        fprintf(stderr, "EXP-0040 rpcmem package too large: %zu\n",
                total_bytes);
        return 2;
    }

    reference = malloc(down_layout.output_bytes);
    if (reference == NULL) {
        return 2;
    }
    shared = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_FLAG_UNCACHED,
                          (int)total_bytes);
    if (shared == NULL) {
        fprintf(stderr, "rpcmem_alloc failed for %zu bytes\n", total_bytes);
        goto cleanup;
    }
    shared_fd = rpcmem_to_fd(shared);
    if (shared_fd < 0) {
        goto cleanup;
    }
    memset(shared, 0, total_bytes);
    header = (struct qbh_mlp_header *)shared;
    input = shared + input_offset;
    gate_weights = shared + gate_offset;
    down_weights = shared + down_offset;
    activation_lut = shared + lut_offset;
    gate_up_multipliers = shared + gate_up_multiplier_offset;
    down_multipliers = shared + down_multiplier_offset;
    output = shared + output_offset;
    gate_file = storage == QBH_WEIGHT_EXPANDED_S8
                    ? "gate_up_expanded_s8_bundles.bin"
                    : "gate_up_packed_w4_bundles.bin";
    down_file = storage == QBH_WEIGHT_EXPANDED_S8
                    ? "down_expanded_s8_bundles.bin"
                    : "down_packed_w4_bundles.bin";
    if (qbh_read_file(argv[1],
                      "reference_w4u8_post_attention_norm_u8.bin",
                      input, gate_layout.activation_bytes) != 0 ||
        qbh_read_file(argv[1], gate_file, gate_weights,
                      gate_layout.stored_weight_bytes) != 0 ||
        qbh_read_file(argv[1], down_file, down_weights,
                      down_layout.stored_weight_bytes) != 0 ||
        qbh_read_file(argv[1], "silu_up_lut_u16.bin", activation_lut,
                      QBH_MLP_LUT_BYTES) != 0 ||
        qbh_read_file(argv[1], "gate_up_output_multipliers_u8.bin",
                      gate_up_multipliers,
                      QBH_MLP_GATE_UP_MULTIPLIER_BYTES) != 0 ||
        qbh_read_file(argv[1], "down_output_multipliers_u8.bin",
                      down_multipliers,
                      QBH_MLP_DOWN_MULTIPLIER_BYTES) != 0 ||
        qbh_read_file(argv[1], "reference_integer_hmx_down_u8.bin", reference,
                      down_layout.output_bytes) != 0) {
        fprintf(stderr, "EXP-0040 package audit/read failed\n");
        goto cleanup;
    }

    header->magic = QBH_MLP_MAGIC;
    header->abi_version = QBH_MLP_ABI_VERSION;
    header->experiment = QBH_MLP_EXPERIMENT;
    header->header_bytes = (uint32_t)sizeof(*header);
    header->shared_bytes = (uint32_t)total_bytes;
    header->repeat_count = repeats;
    header->run_activation_self_test = self_test != 0U;
    header->pattern = 1U;
    header->weight_storage_variant = storage;
    header->gate_up_worker_count = gate_up_workers;
    header->input_offset = (uint32_t)input_offset;
    header->gate_up_weight_offset = (uint32_t)gate_offset;
    header->down_weight_offset = (uint32_t)down_offset;
    header->activation_lut_offset = (uint32_t)lut_offset;
    header->gate_up_multiplier_offset =
        (uint32_t)gate_up_multiplier_offset;
    header->down_multiplier_offset =
        (uint32_t)down_multiplier_offset;
    header->output_offset = (uint32_t)output_offset;
    header->input_bytes = gate_layout.activation_bytes;
    header->gate_up_weight_bytes = gate_layout.stored_weight_bytes;
    header->down_weight_bytes = down_layout.stored_weight_bytes;
    header->activation_lut_bytes = QBH_MLP_LUT_BYTES;
    header->gate_up_multiplier_bytes =
        QBH_MLP_GATE_UP_MULTIPLIER_BYTES;
    header->down_multiplier_bytes = QBH_MLP_DOWN_MULTIPLIER_BYTES;
    header->output_bytes = down_layout.output_bytes;
    header->dsp_status = QBH_MLP_STATUS_HOST_READY;

    run_result = qbh_session_open(&session);
    if (run_result != AEE_SUCCESS) {
        goto cleanup;
    }
    run_result = fastrpc_mmap(CDSP_DOMAIN_ID, shared_fd, shared, 0,
                              total_bytes, FASTRPC_MAP_FD);
    if (run_result != AEE_SUCCESS) {
        fprintf(stderr, "fastrpc_mmap failed: 0x%08x\n",
                (unsigned int)run_result);
        goto cleanup;
    }
    mapped = 1;
    prepare_result = qbh_session_prepare(&session);
    if (prepare_result != AEE_SUCCESS) {
        fprintf(stderr, "prepare failed: 0x%08x\n",
                (unsigned int)prepare_result);
        goto cleanup;
    }

    header->repeat_count = 1U;
    header->run_activation_self_test = self_test != 0U;
    header->dsp_status = QBH_MLP_STATUS_HOST_READY;
    memset(output, 0xA5, down_layout.output_bytes);
    warmup_start = qbh_monotonic_ns();
    warmup_result = qwen3_probe_run_mlp(
        session.handle, shared_fd, (uint32_t)total_bytes);
    warmup_end = qbh_monotonic_ns();
    if (warmup_result != AEE_SUCCESS) {
        fprintf(stderr, "warmup failed: 0x%08x dsp_status=%d\n",
                (unsigned int)warmup_result, header->dsp_status);
        goto cleanup;
    }
    warmup_run_index = header->prepared_session_run_index;
    warmup_checksum = qbh_checksum(output, down_layout.output_bytes);
    for (size_t index = 0; index < down_layout.output_bytes; ++index) {
        warmup_mismatches += output[index] != reference[index];
    }
    if (warmup_mismatches != 0U ||
        header->activation_self_test_mismatches != 0U) {
        FILE *diagnostic = fopen("actual_exp0040_warmup_u8.bin", "wb");
        if (diagnostic != NULL) {
            (void)fwrite(output, 1, down_layout.output_bytes, diagnostic);
            (void)fclose(diagnostic);
        }
        for (size_t index = 0, shown = 0;
             index < down_layout.output_bytes && shown < 8U; ++index) {
            if (output[index] != reference[index]) {
                fprintf(stderr,
                        "warmup mismatch index=%zu expected=%u actual=%u\n",
                        index, (unsigned int)reference[index],
                        (unsigned int)output[index]);
                ++shown;
            }
        }
        fprintf(stderr,
                "warmup correctness failed: output=%" PRIu32
                " activation=%" PRIu32 " actual_checksum=%" PRIu64
                " reference_checksum=%" PRIu64 "\n",
                warmup_mismatches,
                header->activation_self_test_mismatches,
                qbh_checksum(output, down_layout.output_bytes),
                qbh_checksum(reference, down_layout.output_bytes));
        goto cleanup;
    }

    header->repeat_count = repeats;
    header->run_activation_self_test = 0U;
    header->dsp_status = QBH_MLP_STATUS_HOST_READY;
    memset(output, 0xA5, down_layout.output_bytes);
    run_start = qbh_monotonic_ns();
    run_result = qwen3_probe_run_mlp(
        session.handle, shared_fd, (uint32_t)total_bytes);
    run_end = qbh_monotonic_ns();
    if (run_result != AEE_SUCCESS) {
        fprintf(stderr, "measured run failed: 0x%08x dsp_status=%d\n",
                (unsigned int)run_result, header->dsp_status);
        goto cleanup;
    }
    for (size_t index = 0; index < down_layout.output_bytes; ++index) {
        if (output[index] != reference[index]) {
            if (mismatches < 8U) {
                fprintf(stderr, "mismatch index=%zu expected=%u actual=%u\n",
                        index, (unsigned int)reference[index],
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

    printf("{\"experiment\":\"EXP-0040\","
           "\"stage\":\"A_latest_layout_real_mlp\","
           "\"weight_storage\":\"%s\","
           "\"hmx_contract\":\"U8xS8_integer\","
           "\"activation\":\"asymmetric_U8\","
           "\"silu_contract\":\"exact_U8_pair_LUT_HVX_vgather\","
           "\"intermediate_residency\":\"VTCM\","
           "\"repeat_count\":%" PRIu32 ","
           "\"gate_up_worker_count\":%" PRIu32 ","
           "\"warmup_result\":%d,\"rpc_result\":%d,"
           "\"dsp_status\":%d,"
           "\"warmup_prepared_session_run_index\":%" PRIu32 ","
           "\"prepared_session_run_index\":%" PRIu32 ","
           "\"warmup_mismatches\":%" PRIu32 ","
           "\"mismatches\":%" PRIu32 ","
           "\"activation_self_test_cases\":%" PRIu32 ","
           "\"activation_self_test_mismatches\":%" PRIu32 ","
           "\"reference_checksum\":%" PRIu64 ","
           "\"warmup_output_checksum\":%" PRIu64 ","
           "\"output_checksum\":%" PRIu64 ","
           "\"warmup_host_wall_ns\":%" PRIu64 ","
           "\"host_wall_ns\":%" PRIu64 ","
           "\"total_ticks\":%" PRIu64 ","
           "\"input_stage_ticks\":%" PRIu64 ","
           "\"activation_lut_stage_ticks\":%" PRIu64 ","
           "\"requant_metadata_stage_ticks\":%" PRIu64 ","
           "\"gate_up_ticks\":%" PRIu64 ","
           "\"activation_ticks\":%" PRIu64 ","
           "\"down_ticks\":%" PRIu64 ","
           "\"down_requant_ticks\":%" PRIu64 ","
           "\"final_output_ticks\":%" PRIu64 ","
           "\"gate_up_weight_stage_ticks\":%" PRIu64 ","
           "\"gate_up_weight_transform_ticks\":%" PRIu64 ","
           "\"gate_up_hmx_compute_ticks\":%" PRIu64 ","
           "\"gate_up_hmx_ready_wait_ticks\":%" PRIu64 ","
           "\"down_weight_stage_ticks\":%" PRIu64 ","
           "\"down_weight_transform_ticks\":%" PRIu64 ","
           "\"down_hmx_compute_ticks\":%" PRIu64 ","
           "\"down_hmx_ready_wait_ticks\":%" PRIu64 ","
           "\"gate_up_hvx_hmx_overlap\":%" PRIu32 ","
           "\"down_hvx_hmx_overlap\":%" PRIu32 ","
           "\"gate_up_hmx_execution_count\":%" PRIu32 ","
           "\"down_hmx_execution_count\":%" PRIu32 ","
           "\"vtcm_requested_bytes\":%" PRIu32 ","
           "\"vtcm_acquired_bytes\":%" PRIu32 ","
           "\"vtcm_peak_plan_bytes\":%" PRIu32 ","
           "\"activation_lut_vtcm_bytes\":%" PRIu32 ","
           "\"requant_metadata_vtcm_bytes\":%" PRIu32 ","
           "\"activation_gather_scratch_vtcm_bytes\":%" PRIu32 ","
           "\"gate_up_full_tensor_materialized\":%" PRIu32 ","
           "\"intermediate_ddr_read_bytes\":%" PRIu32 ","
           "\"intermediate_ddr_write_bytes\":%" PRIu32 ","
           "\"intermediate_dma_descriptor_count\":%" PRIu32 ","
           "\"intermediate_spill_fill_count\":%" PRIu32 "}\n",
           qbh_storage_name(storage), repeats, gate_up_workers,
           warmup_result, run_result,
           header->dsp_status, warmup_run_index,
           header->prepared_session_run_index, warmup_mismatches,
           mismatches, header->activation_self_test_cases,
           header->activation_self_test_mismatches,
           qbh_checksum(reference, down_layout.output_bytes),
           warmup_checksum, qbh_checksum(output, down_layout.output_bytes),
           warmup_end - warmup_start, run_end - run_start,
           header->total_ticks, header->input_stage_ticks,
           header->activation_lut_stage_ticks,
           header->requant_metadata_stage_ticks,
           header->gate_up_ticks, header->activation_ticks,
           header->down_ticks, header->down_requant_ticks,
           header->final_output_ticks,
           header->gate_up_phase.weight_stage_ticks,
           header->gate_up_phase.weight_expand_ticks,
           header->gate_up_phase.hmx_compute_ticks,
           header->gate_up_phase.hmx_ready_wait_ticks,
           header->down_phase.weight_stage_ticks,
           header->down_phase.weight_expand_ticks,
           header->down_phase.hmx_compute_ticks,
           header->down_phase.hmx_ready_wait_ticks,
           header->gate_up_phase.hvx_hmx_overlap_observed,
           header->down_phase.hvx_hmx_overlap_observed,
           header->gate_up_phase.hmx_execution_count,
           header->down_phase.hmx_execution_count,
           header->vtcm_requested_bytes, header->vtcm_acquired_bytes,
           header->vtcm_peak_plan_bytes,
           header->activation_lut_vtcm_bytes,
           header->requant_metadata_vtcm_bytes,
           header->activation_gather_scratch_vtcm_bytes,
           header->gate_up_full_tensor_materialized,
           header->intermediate_ddr_read_bytes,
           header->intermediate_ddr_write_bytes,
           header->intermediate_dma_descriptor_count,
           header->intermediate_spill_fill_count);

    exit_code = warmup_result == AEE_SUCCESS &&
                        run_result == AEE_SUCCESS &&
                        warmup_mismatches == 0U && mismatches == 0U &&
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
    if (mapped != 0) {
        (void)fastrpc_munmap(CDSP_DOMAIN_ID, shared_fd, shared,
                             total_bytes);
    }
    if (shared != NULL) {
        rpcmem_free(shared);
    }
    free(reference);
    return exit_code;
}
