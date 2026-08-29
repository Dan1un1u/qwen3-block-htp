#include <AEEStdErr.h>
#include <inttypes.h>
#include <limits.h>
#include <stddef.h>
#include <remote.h>
#include <rpcmem.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "attention_protocol.h"
#include "host/session.h"
#include "qwen3_probe.h"

static size_t qbh_attn_align_up(size_t value, size_t alignment) {
    return (value + alignment - 1U) / alignment * alignment;
}

static uint64_t qbh_attn_monotonic_ns(void) {
    struct timespec now;
    (void)clock_gettime(CLOCK_MONOTONIC, &now);
    return (uint64_t)now.tv_sec * UINT64_C(1000000000) +
           (uint64_t)now.tv_nsec;
}

static int qbh_attn_parse_u32(const char *text, uint32_t *value) {
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

static int qbh_attn_read_file(const char *package, const char *name,
                              void *destination,
                              size_t expected_bytes) {
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
        fread(destination, 1U, expected_bytes, stream) != expected_bytes ||
        fclose(stream) != 0) {
        return -1;
    }
    return 0;
}

static uint64_t qbh_attn_checksum(const uint8_t *data, size_t bytes) {
    uint64_t value = UINT64_C(1469598103934665603);
    for (size_t index = 0U; index < bytes; ++index) {
        value ^= data[index];
        value *= UINT64_C(1099511628211);
    }
    return value;
}

static void qbh_attn_reset_telemetry(
    struct qbh_attention_header *header) {
    memset((uint8_t *)header + offsetof(struct qbh_attention_header,
                                       dsp_status),
           0, sizeof(*header) -
                  offsetof(struct qbh_attention_header, dsp_status));
    header->dsp_status = QBH_ATTENTION_STATUS_HOST_READY;
}

static uint32_t qbh_attn_host_mismatches(const uint8_t *actual,
                                         const uint8_t *reference,
                                         size_t bytes,
                                         uint32_t *max_abs) {
    uint32_t mismatches = 0U;
    *max_abs = 0U;
    for (size_t index = 0U; index < bytes; ++index) {
        uint32_t left = actual[index];
        uint32_t right = reference[index];
        uint32_t difference = left > right ? left - right : right - left;
        if (difference != 0U) {
            ++mismatches;
        }
        if (difference > *max_abs) {
            *max_abs = difference;
        }
    }
    return mismatches;
}

int main(int argc, char **argv) {
    struct qbh_session session = {(remote_handle64)-1, 0};
    struct qbh_attention_header *header = NULL;
    struct qbh_attention_config config;
    uint8_t *shared = NULL;
    uint8_t *q;
    uint8_t *k;
    uint8_t *v;
    uint8_t *reference_score;
    uint8_t *reference_probability;
    uint8_t *reference_output;
    uint8_t *output;
    uint32_t repeats = 1U;
    uint32_t self_test = 1U;
    size_t q_offset;
    size_t k_offset;
    size_t v_offset;
    size_t reference_score_offset;
    size_t reference_probability_offset;
    size_t reference_output_offset;
    size_t output_offset;
    size_t total_bytes;
    int shared_fd = -1;
    int mapped = 0;
    int warmup_result = AEE_EFAILED;
    int run_result = AEE_EFAILED;
    uint64_t warmup_start;
    uint64_t warmup_end;
    uint64_t run_start;
    uint64_t run_end;
    uint32_t host_mismatches = 0U;
    uint32_t host_max_abs = 0U;
    int exit_code = EXIT_FAILURE;

    if (argc < 2 || argc > 4 ||
        (argc >= 3 && qbh_attn_parse_u32(argv[2], &repeats) != 0) ||
        (argc >= 4 && qbh_attn_parse_u32(argv[3], &self_test) != 0) ||
        repeats == 0U || repeats > QBH_HMX_MAX_REPEATS ||
        self_test > 1U) {
        fprintf(stderr,
                "usage: %s PACKAGE_DIR [repeat_count] "
                "[self_test_0_or_1]\n",
                argv[0]);
        return 2;
    }
    if (qbh_attn_read_file(argv[1], "attention_config.bin", &config,
                           sizeof(config)) != 0 ||
        config.abi_version != QBH_ATTENTION_ABI_VERSION) {
        fprintf(stderr, "EXP-0042 config audit/read failed\n");
        return 2;
    }

    q_offset = qbh_attn_align_up(sizeof(*header),
                                 QBH_ATTENTION_ALIGNMENT);
    k_offset = q_offset + qbh_attn_align_up(
        QBH_ATTENTION_Q_BYTES, QBH_ATTENTION_ALIGNMENT);
    v_offset = k_offset + qbh_attn_align_up(
        QBH_ATTENTION_KV_BYTES, QBH_ATTENTION_ALIGNMENT);
    reference_score_offset = v_offset + qbh_attn_align_up(
        QBH_ATTENTION_KV_BYTES, QBH_ATTENTION_ALIGNMENT);
    reference_probability_offset = reference_score_offset +
        qbh_attn_align_up(QBH_ATTENTION_SCORE_BYTES,
                          QBH_ATTENTION_ALIGNMENT);
    reference_output_offset = reference_probability_offset +
        qbh_attn_align_up(QBH_ATTENTION_SCORE_BYTES,
                          QBH_ATTENTION_ALIGNMENT);
    output_offset = reference_output_offset + qbh_attn_align_up(
        QBH_ATTENTION_OUTPUT_BYTES, QBH_ATTENTION_ALIGNMENT);
    total_bytes = output_offset + qbh_attn_align_up(
        QBH_ATTENTION_OUTPUT_BYTES, QBH_ATTENTION_ALIGNMENT);
    if (total_bytes > INT_MAX || total_bytes > UINT32_MAX) {
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
    header = (struct qbh_attention_header *)shared;
    q = shared + q_offset;
    k = shared + k_offset;
    v = shared + v_offset;
    reference_score = shared + reference_score_offset;
    reference_probability = shared + reference_probability_offset;
    reference_output = shared + reference_output_offset;
    output = shared + output_offset;

    if (qbh_attn_read_file(argv[1], "q_group_u8.bin", q,
                           QBH_ATTENTION_Q_BYTES) != 0 ||
        qbh_attn_read_file(argv[1], "k_group_u8.bin", k,
                           QBH_ATTENTION_KV_BYTES) != 0 ||
        qbh_attn_read_file(argv[1], "v_group_u8.bin", v,
                           QBH_ATTENTION_KV_BYTES) != 0 ||
        qbh_attn_read_file(argv[1], "reference_score_u8.bin",
                           reference_score,
                           QBH_ATTENTION_SCORE_BYTES) != 0 ||
        qbh_attn_read_file(argv[1], "reference_probability_u8.bin",
                           reference_probability,
                           QBH_ATTENTION_SCORE_BYTES) != 0 ||
        qbh_attn_read_file(argv[1], "reference_output_u8.bin",
                           reference_output,
                           QBH_ATTENTION_OUTPUT_BYTES) != 0) {
        fprintf(stderr, "EXP-0042 package audit/read failed\n");
        goto cleanup;
    }

    header->magic = QBH_ATTENTION_MAGIC;
    header->abi_version = QBH_ATTENTION_ABI_VERSION;
    header->experiment = QBH_ATTENTION_EXPERIMENT;
    header->header_bytes = (uint32_t)sizeof(*header);
    header->shared_bytes = (uint32_t)total_bytes;
    header->repeat_count = repeats;
    header->run_self_test = self_test;
    header->q_offset = (uint32_t)q_offset;
    header->k_offset = (uint32_t)k_offset;
    header->v_offset = (uint32_t)v_offset;
    header->reference_score_offset = (uint32_t)reference_score_offset;
    header->reference_probability_offset =
        (uint32_t)reference_probability_offset;
    header->reference_output_offset =
        (uint32_t)reference_output_offset;
    header->output_offset = (uint32_t)output_offset;
    header->q_bytes = QBH_ATTENTION_Q_BYTES;
    header->k_bytes = QBH_ATTENTION_KV_BYTES;
    header->v_bytes = QBH_ATTENTION_KV_BYTES;
    header->reference_score_bytes = QBH_ATTENTION_SCORE_BYTES;
    header->reference_probability_bytes = QBH_ATTENTION_SCORE_BYTES;
    header->reference_output_bytes = QBH_ATTENTION_OUTPUT_BYTES;
    header->output_bytes = QBH_ATTENTION_OUTPUT_BYTES;
    header->config = config;

    if (qbh_session_open(&session) != AEE_SUCCESS ||
        qbh_session_prepare(&session) != AEE_SUCCESS) {
        fprintf(stderr, "prepared session setup failed\n");
        goto cleanup;
    }
    if (fastrpc_mmap(CDSP_DOMAIN_ID, shared_fd, shared, 0,
                     total_bytes, FASTRPC_MAP_FD) != AEE_SUCCESS) {
        fprintf(stderr, "fastrpc_mmap failed\n");
        goto cleanup;
    }
    mapped = 1;

    qbh_attn_reset_telemetry(header);
    warmup_start = qbh_attn_monotonic_ns();
    warmup_result = qwen3_probe_run_attention(
        session.handle, shared_fd, (uint32_t)total_bytes);
    warmup_end = qbh_attn_monotonic_ns();
    if (warmup_result != AEE_SUCCESS ||
        header->dsp_status != QBH_ATTENTION_STATUS_OK) {
        fprintf(stderr,
                "attention warmup failed: rpc=0x%08x dsp=%d "
                "score_mismatch=%" PRIu32 " prob_mismatch=%" PRIu32
                " output_mismatch=%" PRIu32
                " first_score=%" PRIu32 ":%" PRIu32 "/%" PRIu32
                " qk0_acc=%" PRId32 " qk0_mid=%" PRIu32
                "/%" PRIu32 " qk0_post=%" PRIu32
                " qk10=%" PRId32 ":%" PRIu32 "/%" PRIu32
                ":%" PRIu32 " qk032=%" PRId32 ":%" PRIu32
                "/%" PRIu32 ":%" PRIu32
                " packed=%" PRId32
                " bias=%" PRId32 "/%" PRId32 "\n",
                (unsigned int)warmup_result, header->dsp_status,
                header->score_mismatch_count,
                header->probability_mismatch_count,
                header->output_mismatch_count,
                header->first_score_mismatch_index,
                header->first_score_actual,
                header->first_score_expected,
                header->debug_qk_accumulator0,
                header->debug_qk_intermediate0,
                header->debug_qk_expected_intermediate0,
                header->debug_qk_post0,
                header->debug_qk_accumulator_row1_col0,
                header->debug_qk_intermediate_row1_col0,
                header->debug_qk_expected_intermediate_row1_col0,
                header->debug_qk_post_row1_col0,
                header->debug_qk_accumulator_col32,
                header->debug_qk_intermediate_col32,
                header->debug_qk_expected_intermediate_col32,
                header->debug_qk_post_col32,
                header->debug_qk_packed_accumulator_col32,
                header->debug_qk_bias_upper0,
                header->debug_qk_bias_upper32);
        goto cleanup;
    }

    qbh_attn_reset_telemetry(header);
    run_start = qbh_attn_monotonic_ns();
    run_result = qwen3_probe_run_attention(
        session.handle, shared_fd, (uint32_t)total_bytes);
    run_end = qbh_attn_monotonic_ns();
    if (run_result != AEE_SUCCESS ||
        header->dsp_status != QBH_ATTENTION_STATUS_OK) {
        fprintf(stderr,
                "attention run failed: rpc=0x%08x dsp=%d "
                "score_mismatch=%" PRIu32 " prob_mismatch=%" PRIu32
                " output_mismatch=%" PRIu32
                " first_score=%" PRIu32 ":%" PRIu32 "/%" PRIu32
                " qk0_acc=%" PRId32 " qk0_mid=%" PRIu32
                "/%" PRIu32 " qk0_post=%" PRIu32 "\n",
                (unsigned int)run_result, header->dsp_status,
                header->score_mismatch_count,
                header->probability_mismatch_count,
                header->output_mismatch_count,
                header->first_score_mismatch_index,
                header->first_score_actual,
                header->first_score_expected,
                header->debug_qk_accumulator0,
                header->debug_qk_intermediate0,
                header->debug_qk_expected_intermediate0,
                header->debug_qk_post0);
        goto cleanup;
    }
    host_mismatches = qbh_attn_host_mismatches(
        output, reference_output, QBH_ATTENTION_OUTPUT_BYTES,
        &host_max_abs);

    printf(
        "{\"experiment\":\"EXP-0042\","
        "\"group\":%" PRIu32 ",\"fraction_bits\":%" PRIu32
        ",\"division_mode\":%" PRIu32
        ",\"repeat_count\":%" PRIu32
        ",\"self_test\":%" PRIu32
        ",\"warmup_host_ns\":%" PRIu64
        ",\"host_ns\":%" PRIu64
        ",\"dsp_ticks\":%" PRIu64
        ",\"input_stage_ticks\":%" PRIu64
        ",\"pack_ticks\":%" PRIu64
        ",\"qk_hmx_ticks\":%" PRIu64
        ",\"qk_requant_ticks\":%" PRIu64
        ",\"softmax_ticks\":%" PRIu64
        ",\"av_hmx_ticks\":%" PRIu64
        ",\"av_requant_ticks\":%" PRIu64
        ",\"self_test_ticks\":%" PRIu64
        ",\"output_stage_ticks\":%" PRIu64
        ",\"qk_hmx_executions\":%" PRIu32
        ",\"av_hmx_executions\":%" PRIu32
        ",\"score_saturations\":%" PRIu32
        ",\"v_recenter_saturations\":%" PRIu32
        ",\"probability_row_sum_min\":%" PRIu32
        ",\"probability_row_sum_max\":%" PRIu32
        ",\"score_mismatches\":%" PRIu32
        ",\"probability_mismatches\":%" PRIu32
        ",\"output_mismatches\":%" PRIu32
        ",\"output_max_abs_lsb\":%" PRIu32
        ",\"host_output_mismatches\":%" PRIu32
        ",\"host_output_max_abs_lsb\":%" PRIu32
        ",\"intermediate_ddr_read_bytes\":%" PRIu32
        ",\"intermediate_ddr_write_bytes\":%" PRIu32
        ",\"intermediate_spill_fill_count\":%" PRIu32
        ",\"graph_split_count\":%" PRIu32
        ",\"cpu_fallback_count\":%" PRIu32
        ",\"vtcm_plan_bytes\":%" PRIu32
        ",\"output_checksum\":\"%016" PRIx64 "\"}\n",
        config.group_index, config.fraction_bits, config.division_mode,
        repeats, self_test, warmup_end - warmup_start,
        run_end - run_start, header->total_ticks,
        header->input_stage_ticks, header->pack_ticks,
        header->qk_hmx_ticks, header->qk_requant_ticks,
        header->softmax_ticks, header->av_hmx_ticks,
        header->av_requant_ticks, header->self_test_ticks,
        header->output_stage_ticks,
        header->hmx_qk_execution_count,
        header->hmx_av_execution_count,
        header->score_saturation_count,
        header->v_recenter_saturation_count,
        header->probability_row_sum_min,
        header->probability_row_sum_max,
        header->score_mismatch_count,
        header->probability_mismatch_count,
        header->output_mismatch_count,
        header->output_max_abs_lsb,
        host_mismatches, host_max_abs,
        header->intermediate_ddr_read_bytes,
        header->intermediate_ddr_write_bytes,
        header->intermediate_spill_fill_count,
        header->graph_split_count, header->cpu_fallback_count,
        header->vtcm_peak_plan_bytes,
        qbh_attn_checksum(output, QBH_ATTENTION_OUTPUT_BYTES));
    exit_code = host_mismatches == 0U ? EXIT_SUCCESS : EXIT_FAILURE;

cleanup:
    if (mapped) {
        (void)fastrpc_munmap(CDSP_DOMAIN_ID, shared_fd, shared,
                             total_bytes);
    }
    (void)qbh_session_close(&session);
    if (shared != NULL) {
        rpcmem_free(shared);
    }
    return exit_code;
}
