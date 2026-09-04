#include <AEEStdErr.h>
#include <errno.h>
#include <float.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <remote.h>
#include <rpcmem.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#include "block_protocol.h"
#include "hmx_fp16.h"
#include "hmx_u8s8_projection.h"
#include "host/session.h"
#include "mlp_u8.h"
#include "qwen3_probe.h"

#pragma weak rpcmem_alloc2

#define QBH_HOST_ALIGNMENT ((size_t)128)
#define QBH_HOST_PATH_BYTES ((size_t)1024)
#define QBH_REPLAY_FP16_ATOL (0.0625)
#define QBH_REPLAY_FP16_RTOL (0.002)
#define QBH_REPLAY_FP16_MIN_COSINE (0.99999)
#define QBH_REPLAY_FP16_MAX_CACHE_VIOLATION_FRACTION (0.01)
#define QBH_REPLAY_FP16_MAX_COMPOSED_NRMSE (0.003)

struct qbh_file_slot {
    char path[QBH_HOST_PATH_BYTES];
    uint32_t expected_bytes;
    uint32_t offset;
};

struct qbh_vertical_layer_slots {
    struct qbh_file_slot qparam;
    struct qbh_file_slot attention_config;
    struct qbh_file_slot norms[4];
    struct qbh_file_slot caches[2];
    struct qbh_file_slot cache_references[2];
    struct qbh_file_slot silu_lut;
    struct qbh_file_slot weights[QBH_BLOCK_PROJECTION_COUNT];
    struct qbh_file_slot scales[QBH_BLOCK_PROJECTION_COUNT];
    size_t gate_up_bundle_offset;
    size_t down_bundle_offset;
    size_t gate_up_scale_cache_offset;
};

struct qbh_qparam_record {
    char name[32];
    float scale;
    int32_t zero_point;
    float minimum;
    float maximum;
};

struct qbh_error_metrics {
    double max_abs;
    double mean_abs;
    double rmse;
    double nrmse;
    double cosine;
    uint64_t elements;
    uint64_t mismatches;
    uint32_t max_lsb;
    uint64_t mixed_tolerance_violations;
    uint64_t nonfinite_count;
    double max_required_rtol_after_atol;
};


struct qbh_replay_step_result {
    uint64_t host_wall_ns;
    struct qbh_error_metrics output;
    uint64_t cache_prefix_mismatches;
    uint64_t cache_mismatches;
    uint64_t cache_structure_mismatches;
    double cache_min_cosine;
    double cache_max_mixed_tolerance_violation_fraction;
    double cache_max_nrmse;
    uint64_t cache_compared_elements;
    uint64_t cache_mixed_tolerance_violations;
    uint64_t cache_nonfinite_count;
    uint32_t cache_tensor_count;
    uint32_t cache_composed_cosine_diagnostic_failure_count;
    uint32_t cache_legacy_mixed_bound_failure_count;
    uint32_t step_index;
    uint32_t first_position;
    uint32_t valid_length;
    uint32_t dsp_status;
    uint32_t numerical_status;
    uint32_t vtcm_requested_bytes;
    uint32_t vtcm_acquired_bytes;
    uint64_t intermediate_ddr_read_bytes;
    uint64_t intermediate_ddr_write_bytes;
    uint32_t intermediate_spill_fill_count;
    uint64_t scan_cache_ddr_read_bytes;
    uint64_t scan_cache_ddr_write_bytes;
    uint64_t scan_dynamic_attention_ticks;
};

static int qbh_write_named_tensor(
    const char *root, const char *name,
    const void *data, uint32_t bytes) {
    char path[512];
    FILE *stream;
    size_t written;
    int path_bytes = snprintf(path, sizeof(path), "%s/%s", root, name);
    if (path_bytes < 0 || (size_t)path_bytes >= sizeof(path)) {
        return -1;
    }
    stream = fopen(path, "wb");
    if (stream == NULL) {
        return -1;
    }
    written = fwrite(data, 1U, bytes, stream);
    return fclose(stream) == 0 && written == bytes ? 0 : -1;
}

_Static_assert(sizeof(struct qbh_qparam_record) ==
                   QBH_BLOCK_QPARAM_RECORD_BYTES,
               "qparam package record changed");

static const char *qbh_projection_names[QBH_BLOCK_PROJECTION_COUNT] = {
    "q", "k", "v", "o", "gate", "up", "down",
};

static const uint32_t qbh_projection_k[QBH_BLOCK_PROJECTION_COUNT] = {
    QBH_BLOCK_HIDDEN, QBH_BLOCK_HIDDEN, QBH_BLOCK_HIDDEN,
    QBH_BLOCK_HIDDEN, QBH_BLOCK_HIDDEN, QBH_BLOCK_HIDDEN,
    QBH_BLOCK_INTERMEDIATE,
};

static const uint32_t qbh_projection_n[QBH_BLOCK_PROJECTION_COUNT] = {
    QBH_BLOCK_HIDDEN, QBH_BLOCK_KV_HIDDEN, QBH_BLOCK_KV_HIDDEN,
    QBH_BLOCK_HIDDEN, QBH_BLOCK_INTERMEDIATE,
    QBH_BLOCK_INTERMEDIATE, QBH_BLOCK_HIDDEN,
};

static const char *qbh_qparam_names[QBH_BLOCK_QPARAM_COUNT] = {
    "block_input",
    "input_norm",
    "q_projection",
    "k_projection",
    "v",
    "q_rope",
    "k_rope",
    "attention_probability",
    "attention_concat",
    "attention_projection",
    "post_attention_residual",
    "post_attention_norm",
    "gate",
    "up",
    "middle",
    "down",
    "block_output",
};

static const uint32_t qbh_projection_input_qparam[
    QBH_BLOCK_PROJECTION_COUNT] = {
    QBH_BLOCK_QP_INPUT_NORM,
    QBH_BLOCK_QP_INPUT_NORM,
    QBH_BLOCK_QP_INPUT_NORM,
    QBH_BLOCK_QP_ATTENTION_CONCAT,
    QBH_BLOCK_QP_POST_ATTENTION_NORM,
    QBH_BLOCK_QP_POST_ATTENTION_NORM,
    QBH_BLOCK_QP_MIDDLE,
};

static const uint32_t qbh_projection_output_qparam[
    QBH_BLOCK_PROJECTION_COUNT] = {
    QBH_BLOCK_QP_Q_PROJECTION,
    QBH_BLOCK_QP_K_PROJECTION,
    QBH_BLOCK_QP_V,
    QBH_BLOCK_QP_ATTENTION_PROJECTION,
    QBH_BLOCK_QP_GATE,
    QBH_BLOCK_QP_UP,
    QBH_BLOCK_QP_DOWN,
};

static size_t qbh_align_up_size(size_t value, size_t alignment) {
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
    errno = 0;
    parsed = strtoul(text, &end, 0);
    if (errno != 0 || text[0] == '\0' || end == NULL || *end != '\0' ||
        parsed > UINT32_MAX) {
        return -1;
    }
    *value = (uint32_t)parsed;
    return 0;
}

static const char *qbh_variant_name(uint32_t variant) {
    switch (variant) {
        case QBH_BLOCK_F16F16:
            return "F16F16";
        case QBH_BLOCK_W4F16:
            return "W4F16";
        case QBH_BLOCK_W4U8:
            return "W4U8";
        default:
            return "invalid";
    }
}

static int qbh_generation_w4f16_enabled(uint32_t mode) {
    return mode >= QBH_BLOCK_GENERATION_GREEDY_W4F16 &&
           mode <= QBH_BLOCK_GENERATION_GREEDY_W4F16_COARSE_PIPELINE;
}

static int qbh_generation_w4u8_enabled(uint32_t mode) {
    return mode == QBH_BLOCK_GENERATION_GREEDY_W4U8_COARSE_PIPELINE ||
           mode ==
               QBH_BLOCK_GENERATION_GREEDY_W4U8_BATCH8_RESIDENT_BIAS;
}

static const char *qbh_scan_mode_name(uint32_t mode) {
    switch (mode) {
        case QBH_BLOCK_SCAN_DISABLED:
            return "disabled";
        case QBH_BLOCK_SCAN_PREFILL:
            return "prefill";
        case QBH_BLOCK_SCAN_DECODE:
            return "decode";
        default:
            return "invalid";
    }
}

static int qbh_parse_variant(const char *text, uint32_t *variant) {
    if (strcmp(text, "F16F16") == 0 || strcmp(text, "f16f16") == 0 ||
        strcmp(text, "W16A16") == 0 || strcmp(text, "w16a16") == 0) {
        *variant = QBH_BLOCK_F16F16;
        return 0;
    }
    if (strcmp(text, "W4F16") == 0 || strcmp(text, "w4f16") == 0 ||
        strcmp(text, "W4A16") == 0 || strcmp(text, "w4a16") == 0) {
        *variant = QBH_BLOCK_W4F16;
        return 0;
    }
    if (strcmp(text, "W4U8") == 0 || strcmp(text, "w4u8") == 0 ||
        strcmp(text, "W4A8") == 0 || strcmp(text, "w4a8") == 0) {
        *variant = QBH_BLOCK_W4U8;
        return 0;
    }
    return -1;
}

static int qbh_parse_common_ops_mode(const char *text, uint32_t *mask) {
    if (strcmp(text, "scalar") == 0) {
        *mask = QBH_BLOCK_COMMON_OPS_SCALAR;
        return 0;
    }
    if (strcmp(text, "rms") == 0) {
        *mask = QBH_BLOCK_COMMON_OP_RMS_NORM;
        return 0;
    }
    if (strcmp(text, "rope") == 0) {
        *mask = QBH_BLOCK_COMMON_OP_ROPE;
        return 0;
    }
    if (strcmp(text, "softmax") == 0) {
        *mask = QBH_BLOCK_COMMON_OP_SOFTMAX;
        return 0;
    }
    if (strcmp(text, "silu") == 0) {
        *mask = QBH_BLOCK_COMMON_OP_SILU;
        return 0;
    }
    if (strcmp(text, "rms_silu") == 0) {
        *mask = QBH_BLOCK_COMMON_OP_RMS_NORM |
                QBH_BLOCK_COMMON_OP_SILU;
        return 0;
    }
    if (strcmp(text, "rms_silu_rope") == 0) {
        *mask = QBH_BLOCK_COMMON_OP_RMS_NORM |
                QBH_BLOCK_COMMON_OP_SILU |
                QBH_BLOCK_COMMON_OP_ROPE;
        return 0;
    }
    if (strcmp(text, "rms_rope") == 0) {
        *mask = QBH_BLOCK_COMMON_OP_RMS_NORM |
                QBH_BLOCK_COMMON_OP_ROPE;
        return 0;
    }
    if (strcmp(text, "rms_rope_softmax") == 0 ||
        strcmp(text, "u8_all") == 0) {
        *mask = QBH_BLOCK_COMMON_OP_RMS_NORM |
                QBH_BLOCK_COMMON_OP_ROPE |
                QBH_BLOCK_COMMON_OP_SOFTMAX;
        return 0;
    }
    if (strcmp(text, "hvx") == 0 || strcmp(text, "hvx_fp16") == 0 ||
        strcmp(text, "all") == 0) {
        *mask = QBH_BLOCK_COMMON_OPS_HVX_FP16;
        return 0;
    }
    return -1;
}

static int qbh_parse_attribution_mode(const char *text,
                                      uint32_t *enabled) {
    if (strcmp(text, "off") == 0) {
        *enabled = 0U;
        return 0;
    }
    if (strcmp(text, "on") == 0) {
        *enabled = 1U;
        return 0;
    }
    return -1;
}

static int qbh_parse_residual_mode(const char *text, uint32_t *mode) {
    if (strcmp(text, "scalar") == 0) {
        *mode = QBH_BLOCK_RESIDUAL_SCALAR;
        return 0;
    }
    if (strcmp(text, "hvx") == 0 || strcmp(text, "hvx_fp16") == 0) {
        *mode = QBH_BLOCK_RESIDUAL_HVX;
        return 0;
    }
    if (strcmp(text, "fused") == 0 ||
        strcmp(text, "hvx_fused_post_norm") == 0) {
        *mode = QBH_BLOCK_RESIDUAL_HVX_FUSED_POST_NORM;
        return 0;
    }
    if (strcmp(text, "fused_pool4") == 0 ||
        strcmp(text, "hvx_fused_post_norm_pool4") == 0) {
        *mode = QBH_BLOCK_RESIDUAL_HVX_FUSED_POST_NORM_POOL4;
        return 0;
    }
    if (strcmp(text, "fused_pool6") == 0 ||
        strcmp(text, "hvx_fused_post_norm_pool6") == 0) {
        *mode = QBH_BLOCK_RESIDUAL_HVX_FUSED_POST_NORM_POOL6;
        return 0;
    }
    if (strcmp(text, "fused_pool6_shuffle4") == 0 ||
        strcmp(text, "hvx_fused_post_norm_pool6_shuffle4") == 0) {
        *mode =
            QBH_BLOCK_RESIDUAL_HVX_FUSED_POST_NORM_POOL6_SHUFFLE4;
        return 0;
    }
    return -1;
}

static const char *qbh_residual_mode_name(uint32_t mode) {
    if (mode ==
        QBH_BLOCK_RESIDUAL_HVX_FUSED_POST_NORM_POOL6_SHUFFLE4) {
        return "hvx_fused_post_norm_pool6_shuffle4";
    }
    if (mode == QBH_BLOCK_RESIDUAL_HVX_FUSED_POST_NORM_POOL6) {
        return "hvx_fused_post_norm_pool6";
    }
    if (mode == QBH_BLOCK_RESIDUAL_HVX_FUSED_POST_NORM_POOL4) {
        return "hvx_fused_post_norm_pool4";
    }
    if (mode == QBH_BLOCK_RESIDUAL_HVX_FUSED_POST_NORM) {
        return "hvx_fused_post_norm";
    }
    return mode == QBH_BLOCK_RESIDUAL_HVX ? "hvx_fp16" : "scalar";
}

static int qbh_parse_f16f16_projection_mode(const char *text,
                                             uint32_t *mode) {
    if (strcmp(text, "serial") == 0) {
        *mode = QBH_BLOCK_F16F16_PROJECTION_SERIAL;
        return 0;
    }
    if (strcmp(text, "async") == 0 ||
        strcmp(text, "async_single") == 0) {
        *mode = QBH_BLOCK_F16F16_PROJECTION_ASYNC_SINGLE;
        return 0;
    }
    if (strcmp(text, "batch2") == 0 ||
        strcmp(text, "double_buffer_batch2") == 0) {
        *mode = QBH_BLOCK_F16F16_PROJECTION_BATCH2;
        return 0;
    }
    if (strcmp(text, "gate4") == 0 ||
        strcmp(text, "gate_up_batch4") == 0) {
        *mode = QBH_BLOCK_F16F16_PROJECTION_GATE4;
        return 0;
    }
    if (strcmp(text, "gate8") == 0 ||
        strcmp(text, "gate_up_batch8") == 0) {
        *mode = QBH_BLOCK_F16F16_PROJECTION_GATE8;
        return 0;
    }
    if (strcmp(text, "gate8_interleaved") == 0 ||
        strcmp(text, "gate_up_batch8_interleaved") == 0) {
        *mode = QBH_BLOCK_F16F16_PROJECTION_GATE8_INTERLEAVED;
        return 0;
    }
    return -1;
}

static const char *qbh_f16f16_projection_mode_name(uint32_t mode) {
    if (mode == QBH_BLOCK_F16F16_PROJECTION_GATE8_INTERLEAVED) {
        return "gate_up_batch8_interleaved";
    }
    if (mode == QBH_BLOCK_F16F16_PROJECTION_GATE8) {
        return "gate_up_batch8";
    }
    if (mode == QBH_BLOCK_F16F16_PROJECTION_GATE4) {
        return "gate_up_batch4";
    }
    if (mode == QBH_BLOCK_F16F16_PROJECTION_BATCH2) {
        return "double_buffer_batch2";
    }
    if (mode == QBH_BLOCK_F16F16_PROJECTION_ASYNC_SINGLE) {
        return "async_single";
    }
    return "serial";
}

static int qbh_parse_w4f16_pipeline_mode(const char *text,
                                         uint32_t *mode) {
    if (strcmp(text, "control") == 0) {
        *mode = QBH_BLOCK_W4F16_PIPELINE_CONTROL;
        return 0;
    }
    if (strcmp(text, "early") == 0 ||
        strcmp(text, "early_region") == 0) {
        *mode = QBH_BLOCK_W4F16_PIPELINE_EARLY_REGION;
        return 0;
    }
    if (strcmp(text, "hybrid") == 0 ||
        strcmp(text, "hybrid_workers") == 0) {
        *mode = QBH_BLOCK_W4F16_PIPELINE_HYBRID_WORKERS;
        return 0;
    }
    if (strcmp(text, "main_half") == 0) {
        *mode = QBH_BLOCK_W4F16_PIPELINE_MAIN_HALF;
        return 0;
    }
    if (strcmp(text, "main_two_thirds") == 0) {
        *mode = QBH_BLOCK_W4F16_PIPELINE_MAIN_TWO_THIRDS;
        return 0;
    }
    if (strcmp(text, "cross") == 0 ||
        strcmp(text, "cross_prefetch") == 0) {
        *mode = QBH_BLOCK_W4F16_PIPELINE_CROSS_PREFETCH;
        return 0;
    }
    if (strcmp(text, "hybrid_cross") == 0 ||
        strcmp(text, "hybrid_cross_prefetch") == 0) {
        *mode = QBH_BLOCK_W4F16_PIPELINE_HYBRID_CROSS_PREFETCH;
        return 0;
    }
    if (strcmp(text, "adaptive_down64_cross") == 0 ||
        strcmp(text, "adaptive_down64_cross_prefetch") == 0) {
        *mode =
            QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN64_CROSS_PREFETCH;
        return 0;
    }
    if (strcmp(text, "adaptive_down48_cross") == 0 ||
        strcmp(text, "adaptive_down48_cross_prefetch") == 0) {
        *mode =
            QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN48_CROSS_PREFETCH;
        return 0;
    }
    if (strcmp(text, "adaptive_down96_cross") == 0 ||
        strcmp(text, "adaptive_down96_cross_prefetch") == 0) {
        *mode =
            QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_CROSS_PREFETCH;
        return 0;
    }
    if (strcmp(text, "adaptive_down96_gate16_cross") == 0 ||
        strcmp(text, "adaptive_down96_gate16_cross_prefetch") == 0) {
        *mode =
            QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_GATE16_CROSS_PREFETCH;
        return 0;
    }
    if (strcmp(text, "adaptive_down96_gate8_cross") == 0 ||
        strcmp(text, "adaptive_down96_gate8_cross_prefetch") == 0) {
        *mode =
            QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_GATE8_CROSS_PREFETCH;
        return 0;
    }
    if (strcmp(text, "adaptive_down96_gate4_cross") == 0 ||
        strcmp(text, "adaptive_down96_gate4_cross_prefetch") == 0) {
        *mode =
            QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_GATE4_CROSS_PREFETCH;
        return 0;
    }
    if (strcmp(text, "adaptive_down96_gate4_dma8_cross") == 0 ||
        strcmp(text, "adaptive_down96_gate4_dma8_cross_prefetch") == 0) {
        *mode =
            QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_GATE4_DMA8_CROSS_PREFETCH;
        return 0;
    }
    return -1;
}

static const char *qbh_w4f16_pipeline_mode_name(uint32_t mode) {
    if (mode == QBH_BLOCK_W4F16_PIPELINE_EARLY_REGION) {
        return "early_region";
    }
    if (mode == QBH_BLOCK_W4F16_PIPELINE_HYBRID_WORKERS) {
        return "hybrid_workers";
    }
    if (mode == QBH_BLOCK_W4F16_PIPELINE_MAIN_HALF) {
        return "main_half";
    }
    if (mode == QBH_BLOCK_W4F16_PIPELINE_MAIN_TWO_THIRDS) {
        return "main_two_thirds";
    }
    if (mode == QBH_BLOCK_W4F16_PIPELINE_CROSS_PREFETCH) {
        return "cross_prefetch";
    }
    if (mode == QBH_BLOCK_W4F16_PIPELINE_HYBRID_CROSS_PREFETCH) {
        return "hybrid_cross_prefetch";
    }
    if (mode ==
        QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN64_CROSS_PREFETCH) {
        return "adaptive_down64_cross_prefetch";
    }
    if (mode ==
        QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN48_CROSS_PREFETCH) {
        return "adaptive_down48_cross_prefetch";
    }
    if (mode ==
        QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_CROSS_PREFETCH) {
        return "adaptive_down96_cross_prefetch";
    }
    if (mode ==
        QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_GATE16_CROSS_PREFETCH) {
        return "adaptive_down96_gate16_cross_prefetch";
    }
    if (mode ==
        QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_GATE8_CROSS_PREFETCH) {
        return "adaptive_down96_gate8_cross_prefetch";
    }
    if (mode ==
        QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_GATE4_CROSS_PREFETCH) {
        return "adaptive_down96_gate4_cross_prefetch";
    }
    if (mode ==
        QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_GATE4_DMA8_CROSS_PREFETCH) {
        return "adaptive_down96_gate4_dma8_cross_prefetch";
    }
    return "control";
}

static int qbh_parse_attention_pack_mode(const char *text,
                                         uint32_t *mode) {
    if (strcmp(text, "control") == 0 || strcmp(text, "scalar") == 0) {
        *mode = QBH_BLOCK_ATTENTION_PACK_CONTROL;
        return 0;
    }
    if (strcmp(text, "qk_hvx") == 0) {
        *mode = QBH_BLOCK_ATTENTION_PACK_QK_HVX;
        return 0;
    }
    if (strcmp(text, "av_hvx") == 0) {
        *mode = QBH_BLOCK_ATTENTION_PACK_AV_HVX;
        return 0;
    }
    if (strcmp(text, "hvx") == 0 || strcmp(text, "combined_hvx") == 0) {
        *mode = QBH_BLOCK_ATTENTION_PACK_HVX;
        return 0;
    }
    return -1;
}

static const char *qbh_attention_pack_mode_name(uint32_t mode) {
    if (mode == QBH_BLOCK_ATTENTION_PACK_QK_HVX) {
        return "qk_hvx";
    }
    if (mode == QBH_BLOCK_ATTENTION_PACK_AV_HVX) {
        return "av_hvx";
    }
    if (mode == QBH_BLOCK_ATTENTION_PACK_HVX) {
        return "combined_hvx";
    }
    return "control";
}

static int qbh_parse_attention_pipeline_mode(
    const char *text, uint32_t *mode) {
    if (strcmp(text, "control") == 0) {
        *mode = QBH_BLOCK_ATTENTION_PIPELINE_CONTROL;
        return 0;
    }
    if (strcmp(text, "parallel_qk_norm_rope") == 0 ||
        strcmp(text, "parallel_qk") == 0) {
        *mode = QBH_BLOCK_ATTENTION_PIPELINE_PARALLEL_QK_NORM_ROPE;
        return 0;
    }
    if (strcmp(text, "parallel_softmax") == 0) {
        *mode = QBH_BLOCK_ATTENTION_PIPELINE_PARALLEL_SOFTMAX;
        return 0;
    }
    if (strcmp(text, "parallel_hvx") == 0 ||
        strcmp(text, "parallel_both") == 0) {
        *mode = QBH_BLOCK_ATTENTION_PIPELINE_PARALLEL_HVX;
        return 0;
    }
    if (strcmp(text, "gqa_pipeline") == 0 ||
        strcmp(text, "gqa") == 0) {
        *mode = QBH_BLOCK_ATTENTION_PIPELINE_GQA;
        return 0;
    }
    if (strcmp(text, "gqa_qkv_overlap") == 0 ||
        strcmp(text, "qkv_overlap") == 0) {
        *mode = QBH_BLOCK_ATTENTION_PIPELINE_GQA_QKV_OVERLAP;
        return 0;
    }
    if (strcmp(text, "u8_log2_gqa") == 0 ||
        strcmp(text, "integer_gqa") == 0) {
        *mode = QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA;
        return 0;
    }
    if (strcmp(text, "u8_log2_gqa_fused_k") == 0 ||
        strcmp(text, "integer_gqa_fused_k") == 0) {
        *mode =
            QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_FUSED_K;
        return 0;
    }
    if (strcmp(text, "u8_log2_gqa_qkv_overlap") == 0 ||
        strcmp(text, "integer_gqa_qkv_overlap") == 0) {
        *mode =
            QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP;
        return 0;
    }
    if (strcmp(text, "u8_log2_gqa_qkv_overlap_vgather") == 0 ||
        strcmp(text, "integer_gqa_qkv_overlap_vgather") == 0) {
        *mode =
            QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER;
        return 0;
    }
    if (strcmp(text,
               "u8_log2_gqa_qkv_overlap_vgather_vdeal") == 0 ||
        strcmp(text,
               "integer_gqa_qkv_overlap_vgather_vdeal") == 0) {
        *mode =
            QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER_VDEAL;
        return 0;
    }
    if (strcmp(text,
               "u8_log2_gqa_qkv_overlap_vgather_vdeal_fused_qk_requant") == 0 ||
        strcmp(text,
               "integer_gqa_qkv_overlap_vgather_vdeal_fused_qk_requant") == 0) {
        *mode =
            QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER_VDEAL_FUSED_QK_REQUANT;
        return 0;
    }
    if (strcmp(text,
               "u8_log2_gqa_qkv_overlap_vgather_vdeal_fused_qk_requant_hmx_batch") == 0 ||
        strcmp(text,
               "integer_gqa_qkv_overlap_vgather_vdeal_fused_qk_requant_hmx_batch") == 0) {
        *mode =
            QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER_VDEAL_FUSED_QK_REQUANT_HMX_BATCH;
        return 0;
    }
    if (strcmp(text,
               "u8_log2_gqa_qkv_overlap_vgather_vdeal_fused_qk_requant_hmx_batch_lut_templates") == 0 ||
        strcmp(text,
               "integer_gqa_qkv_overlap_vgather_vdeal_fused_qk_requant_hmx_batch_lut_templates") == 0) {
        *mode =
            QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER_VDEAL_FUSED_QK_REQUANT_HMX_BATCH_LUT_TEMPLATES;
        return 0;
    }
    if (strcmp(text,
               "u8_log2_gqa_qkv_overlap_vgather_vdeal_fused_qk_requant_hmx_batch_lut_templates_gqa_batch") == 0 ||
        strcmp(text,
               "integer_gqa_qkv_overlap_vgather_vdeal_fused_qk_requant_hmx_batch_lut_templates_gqa_batch") == 0) {
        *mode =
            QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER_VDEAL_FUSED_QK_REQUANT_HMX_BATCH_LUT_TEMPLATES_GQA_BATCH;
        return 0;
    }
    if (strcmp(text,
               "u8_log2_gqa_qkv_overlap_vgather_vdeal_fused_qk_requant_hmx_batch_lut_templates_gqa_batch_dependency_stream") == 0 ||
        strcmp(text,
               "integer_gqa_qkv_overlap_vgather_vdeal_fused_qk_requant_hmx_batch_lut_templates_gqa_batch_dependency_stream") == 0) {
        *mode =
            QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER_VDEAL_FUSED_QK_REQUANT_HMX_BATCH_LUT_TEMPLATES_GQA_BATCH_DEPENDENCY_STREAM;
        return 0;
    }
    if (strcmp(text,
               "u8_log2_gqa_qkv_overlap_vgather_vdeal_fused_qk_requant_hmx_batch_lut_templates_gqa_batch_dependency_stream_softmax_shuffle4") == 0 ||
        strcmp(text,
               "integer_gqa_qkv_overlap_vgather_vdeal_fused_qk_requant_hmx_batch_lut_templates_gqa_batch_dependency_stream_softmax_shuffle4") == 0) {
        *mode =
            QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER_VDEAL_FUSED_QK_REQUANT_HMX_BATCH_LUT_TEMPLATES_GQA_BATCH_DEPENDENCY_STREAM_SOFTMAX_SHUFFLE4;
        return 0;
    }
    return -1;
}

static uint32_t qbh_attention_u8_base_mode(uint32_t mode) {
    return mode ==
                   QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER_VDEAL_FUSED_QK_REQUANT_HMX_BATCH_LUT_TEMPLATES_GQA_BATCH_DEPENDENCY_STREAM_SOFTMAX_SHUFFLE4
               ? QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER_VDEAL_FUSED_QK_REQUANT_HMX_BATCH_LUT_TEMPLATES_GQA_BATCH_DEPENDENCY_STREAM
               : mode;
}

static int qbh_attention_u8_enabled(uint32_t mode) {
    mode = qbh_attention_u8_base_mode(mode);
    return mode == QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA ||
           mode ==
               QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_FUSED_K ||
           mode ==
               QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP ||
           mode ==
               QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER ||
           mode ==
               QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER_VDEAL ||
           mode ==
               QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER_VDEAL_FUSED_QK_REQUANT ||
           mode ==
               QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER_VDEAL_FUSED_QK_REQUANT_HMX_BATCH ||
           mode ==
               QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER_VDEAL_FUSED_QK_REQUANT_HMX_BATCH_LUT_TEMPLATES ||
           mode ==
               QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER_VDEAL_FUSED_QK_REQUANT_HMX_BATCH_LUT_TEMPLATES_GQA_BATCH ||
           mode ==
               QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER_VDEAL_FUSED_QK_REQUANT_HMX_BATCH_LUT_TEMPLATES_GQA_BATCH_DEPENDENCY_STREAM;
}

static int qbh_attention_u8_qkv_overlap_enabled(uint32_t mode) {
    mode = qbh_attention_u8_base_mode(mode);
    return mode ==
               QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP ||
           mode ==
               QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER ||
           mode ==
               QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER_VDEAL ||
           mode ==
               QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER_VDEAL_FUSED_QK_REQUANT ||
           mode ==
               QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER_VDEAL_FUSED_QK_REQUANT_HMX_BATCH ||
           mode ==
               QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER_VDEAL_FUSED_QK_REQUANT_HMX_BATCH_LUT_TEMPLATES ||
           mode ==
               QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER_VDEAL_FUSED_QK_REQUANT_HMX_BATCH_LUT_TEMPLATES_GQA_BATCH ||
           mode ==
               QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER_VDEAL_FUSED_QK_REQUANT_HMX_BATCH_LUT_TEMPLATES_GQA_BATCH_DEPENDENCY_STREAM;
}

static const char *qbh_attention_pipeline_mode_name(uint32_t mode) {
    if (mode ==
        QBH_BLOCK_ATTENTION_PIPELINE_PARALLEL_QK_NORM_ROPE) {
        return "parallel_qk_norm_rope";
    }
    if (mode == QBH_BLOCK_ATTENTION_PIPELINE_PARALLEL_SOFTMAX) {
        return "parallel_softmax";
    }
    if (mode == QBH_BLOCK_ATTENTION_PIPELINE_PARALLEL_HVX) {
        return "parallel_hvx";
    }
    if (mode == QBH_BLOCK_ATTENTION_PIPELINE_GQA) {
        return "gqa_pipeline";
    }
    if (mode == QBH_BLOCK_ATTENTION_PIPELINE_GQA_QKV_OVERLAP) {
        return "gqa_qkv_overlap";
    }
    if (mode == QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA) {
        return "u8_log2_gqa";
    }
    if (mode ==
        QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_FUSED_K) {
        return "u8_log2_gqa_fused_k";
    }
    if (mode ==
        QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP) {
        return "u8_log2_gqa_qkv_overlap";
    }
    if (mode ==
        QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER) {
        return "u8_log2_gqa_qkv_overlap_vgather";
    }
    if (mode ==
        QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER_VDEAL) {
        return "u8_log2_gqa_qkv_overlap_vgather_vdeal";
    }
    if (mode ==
        QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER_VDEAL_FUSED_QK_REQUANT) {
        return "u8_log2_gqa_qkv_overlap_vgather_vdeal_fused_qk_requant";
    }
    if (mode ==
        QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER_VDEAL_FUSED_QK_REQUANT_HMX_BATCH) {
        return "u8_log2_gqa_qkv_overlap_vgather_vdeal_fused_qk_requant_hmx_batch";
    }
    if (mode ==
        QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER_VDEAL_FUSED_QK_REQUANT_HMX_BATCH_LUT_TEMPLATES) {
        return "u8_log2_gqa_qkv_overlap_vgather_vdeal_fused_qk_requant_hmx_batch_lut_templates";
    }
    if (mode ==
        QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER_VDEAL_FUSED_QK_REQUANT_HMX_BATCH_LUT_TEMPLATES_GQA_BATCH) {
        return "u8_log2_gqa_qkv_overlap_vgather_vdeal_fused_qk_requant_hmx_batch_lut_templates_gqa_batch";
    }
    if (mode ==
        QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER_VDEAL_FUSED_QK_REQUANT_HMX_BATCH_LUT_TEMPLATES_GQA_BATCH_DEPENDENCY_STREAM) {
        return "u8_log2_gqa_qkv_overlap_vgather_vdeal_fused_qk_requant_hmx_batch_lut_templates_gqa_batch_dependency_stream";
    }
    if (mode ==
        QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER_VDEAL_FUSED_QK_REQUANT_HMX_BATCH_LUT_TEMPLATES_GQA_BATCH_DEPENDENCY_STREAM_SOFTMAX_SHUFFLE4) {
        return "u8_log2_gqa_qkv_overlap_vgather_vdeal_fused_qk_requant_hmx_batch_lut_templates_gqa_batch_dependency_stream_softmax_shuffle4";
    }
    return "control";
}

static int qbh_parse_mlp_mode(const char *text, uint32_t *mode) {
    if (strcmp(text, "control") == 0) {
        *mode = QBH_BLOCK_MLP_CONTROL;
        return 0;
    }
    if (strcmp(text, "parallel_silu") == 0 ||
        strcmp(text, "multi_worker_silu") == 0) {
        *mode = QBH_BLOCK_MLP_MULTI_WORKER_SILU;
        return 0;
    }
    if (strcmp(text, "streaming") == 0) {
        *mode = QBH_BLOCK_MLP_STREAMING;
        return 0;
    }
    if (strcmp(text, "crouton_native") == 0 ||
        strcmp(text, "crouton") == 0) {
        *mode = QBH_BLOCK_MLP_CROUTON_NATIVE;
        return 0;
    }
    if (strcmp(text, "crouton_native_batch8") == 0 ||
        strcmp(text, "crouton_batch8") == 0) {
        *mode = QBH_BLOCK_MLP_CROUTON_NATIVE_BATCH8;
        return 0;
    }
    if (strcmp(text, "w4u8_streaming") == 0 ||
        strcmp(text, "w4u8") == 0) {
        *mode = QBH_BLOCK_MLP_W4U8_STREAMING;
        return 0;
    }
    if (strcmp(
            text,
            "w4u8_streaming_persistent_gate_up_hvx") == 0 ||
        strcmp(text, "w4u8_persistent_gate_up_hvx") == 0) {
        *mode =
            QBH_BLOCK_MLP_W4U8_STREAMING_PERSISTENT_GATE_UP_HVX;
        return 0;
    }
    if (strcmp(
            text,
            "w4u8_streaming_persistent_mlp_hvx") == 0 ||
        strcmp(text, "w4u8_persistent_mlp_hvx") == 0) {
        *mode = QBH_BLOCK_MLP_W4U8_STREAMING_PERSISTENT_MLP_HVX;
        return 0;
    }
    return -1;
}

static const char *qbh_mlp_mode_name(uint32_t mode) {
    if (mode == QBH_BLOCK_MLP_MULTI_WORKER_SILU) {
        return "multi_worker_silu";
    }
    if (mode == QBH_BLOCK_MLP_STREAMING) {
        return "streaming";
    }
    if (mode == QBH_BLOCK_MLP_CROUTON_NATIVE) {
        return "crouton_native";
    }
    if (mode == QBH_BLOCK_MLP_CROUTON_NATIVE_BATCH8) {
        return "crouton_native_batch8";
    }
    if (mode == QBH_BLOCK_MLP_W4U8_STREAMING) {
        return "w4u8_streaming";
    }
    if (mode ==
        QBH_BLOCK_MLP_W4U8_STREAMING_PERSISTENT_GATE_UP_HVX) {
        return "w4u8_streaming_persistent_gate_up_hvx";
    }
    if (mode == QBH_BLOCK_MLP_W4U8_STREAMING_PERSISTENT_MLP_HVX) {
        return "w4u8_streaming_persistent_mlp_hvx";
    }
    return "control";
}

static int qbh_parse_crouton_boundary_mode(
    const char *text, uint32_t *mode) {
    if (strcmp(text, "control") == 0) {
        *mode = QBH_BLOCK_CROUTON_BOUNDARY_CONTROL;
        return 0;
    }
    if (strcmp(text, "qkv") == 0) {
        *mode = QBH_BLOCK_CROUTON_BOUNDARY_QKV;
        return 0;
    }
    if (strcmp(text, "av_to_o") == 0 || strcmp(text, "av_o") == 0) {
        *mode = QBH_BLOCK_CROUTON_BOUNDARY_AV_TO_O;
        return 0;
    }
    if (strcmp(text, "input_norm") == 0) {
        *mode = QBH_BLOCK_CROUTON_BOUNDARY_INPUT_NORM;
        return 0;
    }
    if (strcmp(text, "post_norm") == 0) {
        *mode = QBH_BLOCK_CROUTON_BOUNDARY_POST_NORM;
        return 0;
    }
    if (strcmp(text, "norms") == 0 ||
        strcmp(text, "input_post_norm") == 0) {
        *mode = QBH_BLOCK_CROUTON_BOUNDARY_INPUT_NORM |
                QBH_BLOCK_CROUTON_BOUNDARY_POST_NORM;
        return 0;
    }
    if (strcmp(text, "qkv_norms") == 0 ||
        strcmp(text, "qkv_input_post_norm") == 0) {
        *mode = QBH_BLOCK_CROUTON_BOUNDARY_QKV |
                QBH_BLOCK_CROUTON_BOUNDARY_INPUT_NORM |
                QBH_BLOCK_CROUTON_BOUNDARY_POST_NORM;
        return 0;
    }
    if (strcmp(text, "all") == 0) {
        *mode = QBH_BLOCK_CROUTON_BOUNDARY_QKV |
                QBH_BLOCK_CROUTON_BOUNDARY_AV_TO_O |
                QBH_BLOCK_CROUTON_BOUNDARY_INPUT_NORM |
                QBH_BLOCK_CROUTON_BOUNDARY_POST_NORM;
        return 0;
    }
    if (strcmp(text, "w4u8_mlp_input") == 0) {
        *mode = QBH_BLOCK_CROUTON_BOUNDARY_W4U8_MLP_INPUT;
        return 0;
    }
    if (strcmp(text, "w4u8_mlp_io") == 0) {
        *mode = QBH_BLOCK_CROUTON_BOUNDARY_W4U8_MLP_INPUT |
                QBH_BLOCK_CROUTON_BOUNDARY_W4U8_MLP_OUTPUT;
        return 0;
    }
    if (strcmp(text, "w4u8_mlp_io_qkv_input") == 0) {
        *mode = QBH_BLOCK_CROUTON_BOUNDARY_W4U8_MLP_INPUT |
                QBH_BLOCK_CROUTON_BOUNDARY_W4U8_MLP_OUTPUT |
                QBH_BLOCK_CROUTON_BOUNDARY_W4U8_QKV_INPUT;
        return 0;
    }
    if (strcmp(text, "w4u8_mlp_io_qkv_o") == 0) {
        *mode = QBH_BLOCK_CROUTON_BOUNDARY_W4U8_MLP_INPUT |
                QBH_BLOCK_CROUTON_BOUNDARY_W4U8_MLP_OUTPUT |
                QBH_BLOCK_CROUTON_BOUNDARY_W4U8_QKV_INPUT |
                QBH_BLOCK_CROUTON_BOUNDARY_W4U8_O_OUTPUT;
        return 0;
    }
    return -1;
}

static const char *qbh_crouton_boundary_mode_name(uint32_t mode) {
    switch (mode) {
        case QBH_BLOCK_CROUTON_BOUNDARY_QKV:
            return "qkv";
        case QBH_BLOCK_CROUTON_BOUNDARY_AV_TO_O:
            return "av_to_o";
        case QBH_BLOCK_CROUTON_BOUNDARY_INPUT_NORM:
            return "input_norm";
        case QBH_BLOCK_CROUTON_BOUNDARY_POST_NORM:
            return "post_norm";
        case QBH_BLOCK_CROUTON_BOUNDARY_INPUT_NORM |
             QBH_BLOCK_CROUTON_BOUNDARY_POST_NORM:
            return "norms";
        case QBH_BLOCK_CROUTON_BOUNDARY_QKV |
             QBH_BLOCK_CROUTON_BOUNDARY_INPUT_NORM |
             QBH_BLOCK_CROUTON_BOUNDARY_POST_NORM:
            return "qkv_norms";
        case QBH_BLOCK_CROUTON_BOUNDARY_QKV |
             QBH_BLOCK_CROUTON_BOUNDARY_AV_TO_O |
             QBH_BLOCK_CROUTON_BOUNDARY_INPUT_NORM |
             QBH_BLOCK_CROUTON_BOUNDARY_POST_NORM:
            return "all";
        case QBH_BLOCK_CROUTON_BOUNDARY_W4U8_MLP_INPUT:
            return "w4u8_mlp_input";
        case QBH_BLOCK_CROUTON_BOUNDARY_W4U8_MLP_INPUT |
             QBH_BLOCK_CROUTON_BOUNDARY_W4U8_MLP_OUTPUT:
            return "w4u8_mlp_io";
        case QBH_BLOCK_CROUTON_BOUNDARY_W4U8_MLP_INPUT |
             QBH_BLOCK_CROUTON_BOUNDARY_W4U8_MLP_OUTPUT |
             QBH_BLOCK_CROUTON_BOUNDARY_W4U8_QKV_INPUT:
            return "w4u8_mlp_io_qkv_input";
        case QBH_BLOCK_CROUTON_BOUNDARY_W4U8_MLP_INPUT |
             QBH_BLOCK_CROUTON_BOUNDARY_W4U8_MLP_OUTPUT |
             QBH_BLOCK_CROUTON_BOUNDARY_W4U8_QKV_INPUT |
             QBH_BLOCK_CROUTON_BOUNDARY_W4U8_O_OUTPUT:
            return "w4u8_mlp_io_qkv_o";
        default:
            return "control";
    }
}

static int qbh_parse_w4u8_qkvo_pipeline_mode(
    const char *text, uint32_t *mode) {
    if (strcmp(text, "serial") == 0 || strcmp(text, "control") == 0) {
        *mode = QBH_BLOCK_W4U8_QKVO_SERIAL;
        return 0;
    }
    if (strcmp(text, "qkv_batch2") == 0 ||
        strcmp(text, "batch2") == 0) {
        *mode = QBH_BLOCK_W4U8_QKV_BATCH2;
        return 0;
    }
    if (strcmp(text, "qkv_batch4") == 0 ||
        strcmp(text, "batch4") == 0) {
        *mode = QBH_BLOCK_W4U8_QKV_BATCH4;
        return 0;
    }
    if (strcmp(text, "qkvo_batch4") == 0) {
        *mode = QBH_BLOCK_W4U8_QKVO_BATCH4;
        return 0;
    }
    if (strcmp(text, "qkvo_batch4_qk_head_tasks") == 0 ||
        strcmp(text, "qk_head_tasks") == 0) {
        *mode = QBH_BLOCK_W4U8_QKVO_BATCH4_QK_HEAD_TASKS;
        return 0;
    }
    if (strcmp(text, "qkvo_batch4_qk_head_pairs") == 0 ||
        strcmp(text, "qk_head_pairs") == 0) {
        *mode = QBH_BLOCK_W4U8_QKVO_BATCH4_QK_HEAD_PAIRS;
        return 0;
    }
    return -1;
}

static const char *qbh_w4u8_qkvo_pipeline_mode_name(uint32_t mode) {
    switch (mode) {
        case QBH_BLOCK_W4U8_QKV_BATCH2:
            return "qkv_batch2";
        case QBH_BLOCK_W4U8_QKV_BATCH4:
            return "qkv_batch4";
        case QBH_BLOCK_W4U8_QKVO_BATCH4:
            return "qkvo_batch4";
        case QBH_BLOCK_W4U8_QKVO_BATCH4_QK_HEAD_TASKS:
            return "qkvo_batch4_qk_head_tasks";
        case QBH_BLOCK_W4U8_QKVO_BATCH4_QK_HEAD_PAIRS:
            return "qkvo_batch4_qk_head_pairs";
        default:
            return "serial";
    }
}

static int qbh_parse_u8_norm_reduction_mode(
    const char *text, uint32_t *mode) {
    if (strcmp(text, "scalar") == 0 || strcmp(text, "control") == 0) {
        *mode = QBH_BLOCK_U8_NORM_REDUCTION_SCALAR;
        return 0;
    }
    if (strcmp(text, "hvx_tree") == 0 ||
        strcmp(text, "vector_tree") == 0) {
        *mode = QBH_BLOCK_U8_NORM_REDUCTION_HVX_TREE;
        return 0;
    }
    if (strcmp(text, "hvx_tree_qk_batched_rsqrt") == 0 ||
        strcmp(text, "qk_batched_rsqrt") == 0) {
        *mode =
            QBH_BLOCK_U8_NORM_REDUCTION_HVX_TREE_QK_BATCHED_RSQRT;
        return 0;
    }
    if (strcmp(text,
               "hvx_tree_qk_batched_rsqrt_shared_rope") == 0 ||
        strcmp(text, "qk_batched_rsqrt_shared_rope") == 0) {
        *mode =
            QBH_BLOCK_U8_NORM_REDUCTION_HVX_TREE_QK_BATCHED_RSQRT_SHARED_ROPE;
        return 0;
    }
    if (strcmp(text,
               "hvx_tree_qk_batched_rsqrt_shared_rope_parallel_input") == 0 ||
        strcmp(text, "parallel_input_rmsnorm") == 0) {
        *mode =
            QBH_BLOCK_U8_NORM_REDUCTION_HVX_TREE_QK_BATCHED_RSQRT_SHARED_ROPE_PARALLEL_INPUT;
        return 0;
    }
    return -1;
}

static const char *qbh_u8_norm_reduction_mode_name(uint32_t mode) {
    switch (mode) {
        case QBH_BLOCK_U8_NORM_REDUCTION_HVX_TREE:
            return "hvx_tree";
        case QBH_BLOCK_U8_NORM_REDUCTION_HVX_TREE_QK_BATCHED_RSQRT:
            return "hvx_tree_qk_batched_rsqrt";
        case QBH_BLOCK_U8_NORM_REDUCTION_HVX_TREE_QK_BATCHED_RSQRT_SHARED_ROPE:
            return "hvx_tree_qk_batched_rsqrt_shared_rope";
        case QBH_BLOCK_U8_NORM_REDUCTION_HVX_TREE_QK_BATCHED_RSQRT_SHARED_ROPE_PARALLEL_INPUT:
            return "hvx_tree_qk_batched_rsqrt_shared_rope_parallel_input";
        default:
            return "scalar";
    }
}

static int qbh_parse_fp16_common_schedule_mode(
    const char *text, uint32_t *mode) {
    if (strcmp(text, "control") == 0) {
        *mode = QBH_BLOCK_FP16_COMMON_SCHEDULE_CONTROL;
        return 0;
    }
    if (strcmp(text, "qk_head_pairs") == 0) {
        *mode = QBH_BLOCK_FP16_COMMON_SCHEDULE_QK_HEAD_PAIRS;
        return 0;
    }
    if (strcmp(text, "input_norm_pool") == 0) {
        *mode = QBH_BLOCK_FP16_COMMON_SCHEDULE_INPUT_NORM_POOL;
        return 0;
    }
    if (strcmp(text, "post_norm_pool") == 0) {
        *mode = QBH_BLOCK_FP16_COMMON_SCHEDULE_POST_RESIDUAL_NORM_POOL;
        return 0;
    }
    if (strcmp(text, "input_norm_pool_post_norm_pool") == 0) {
        *mode = QBH_BLOCK_FP16_COMMON_SCHEDULE_INPUT_NORM_POOL |
                QBH_BLOCK_FP16_COMMON_SCHEDULE_POST_RESIDUAL_NORM_POOL;
        return 0;
    }
    if (strcmp(text, "qk_head_pairs_input_norm_pool") == 0) {
        *mode = QBH_BLOCK_FP16_COMMON_SCHEDULE_QK_HEAD_PAIRS |
                QBH_BLOCK_FP16_COMMON_SCHEDULE_INPUT_NORM_POOL;
        return 0;
    }
    if (strcmp(text, "all") == 0 ||
        strcmp(text,
               "qk_head_pairs_input_norm_pool_post_norm_pool") == 0) {
        *mode = QBH_BLOCK_FP16_COMMON_SCHEDULE_ALL;
        return 0;
    }
    return -1;
}

static const char *qbh_fp16_common_schedule_mode_name(uint32_t mode) {
    switch (mode) {
        case QBH_BLOCK_FP16_COMMON_SCHEDULE_QK_HEAD_PAIRS:
            return "qk_head_pairs";
        case QBH_BLOCK_FP16_COMMON_SCHEDULE_INPUT_NORM_POOL:
            return "input_norm_pool";
        case QBH_BLOCK_FP16_COMMON_SCHEDULE_POST_RESIDUAL_NORM_POOL:
            return "post_norm_pool";
        case QBH_BLOCK_FP16_COMMON_SCHEDULE_INPUT_NORM_POOL |
             QBH_BLOCK_FP16_COMMON_SCHEDULE_POST_RESIDUAL_NORM_POOL:
            return "input_norm_pool_post_norm_pool";
        case QBH_BLOCK_FP16_COMMON_SCHEDULE_QK_HEAD_PAIRS |
             QBH_BLOCK_FP16_COMMON_SCHEDULE_INPUT_NORM_POOL:
            return "qk_head_pairs_input_norm_pool";
        case QBH_BLOCK_FP16_COMMON_SCHEDULE_ALL:
            return "qk_head_pairs_input_norm_pool_post_norm_pool";
        default:
            return "control";
    }
}

static const char *qbh_qkv_schedule_mode_name(uint32_t mode) {
    switch (mode) {
        case QBH_BLOCK_QKV_SCHEDULE_Q_PREFIX4_K_ALL:
            return "q_prefix4_k_all";
        case QBH_BLOCK_QKV_SCHEDULE_HEAD_ALIGNED_BATCH4:
            return "head_aligned_batch4";
        case QBH_BLOCK_QKV_SCHEDULE_V_BATCH4:
            return "v_batch4";
        case QBH_BLOCK_QKV_SCHEDULE_KV_BATCH4:
            return "kv_batch4";
        default:
            return "control";
    }
}

static const char *qbh_w4f16_group_fence_mode_name(uint32_t mode) {
    switch (mode) {
        case QBH_BLOCK_W4F16_GROUP_FENCE_JOIN_ONLY:
            return "join_only";
        case QBH_BLOCK_W4F16_GROUP_FENCE_JOIN_ONLY_DOWN:
            return "join_only_down";
        default:
            return "control";
    }
}

static const char *qbh_w4u8_stream_fence_mode_name(uint32_t mode) {
    switch (mode) {
        case QBH_BLOCK_W4U8_STREAM_FENCE_SINGLE:
            return "single_fence";
        case QBH_BLOCK_W4U8_STREAM_FENCE_RELEASE_ONLY:
            return "release_only";
        default:
            return "control";
    }
}

static const char *qbh_w4u8_decode_softmax_mode_name(uint32_t mode) {
    return mode == QBH_BLOCK_W4U8_DECODE_SOFTMAX_HVX_TILE4
        ? "hvx_tile4" : "scalar";
}

static uint64_t qbh_fnv1a64(const uint8_t *data, size_t bytes) {
    uint64_t hash = UINT64_C(1469598103934665603);
    for (size_t index = 0; index < bytes; ++index) {
        hash ^= data[index];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static const char *qbh_common_ops_mode_name(uint32_t mask) {
    switch (mask) {
        case QBH_BLOCK_COMMON_OPS_SCALAR:
            return "scalar";
        case QBH_BLOCK_COMMON_OP_RMS_NORM:
            return "rms";
        case QBH_BLOCK_COMMON_OP_ROPE:
            return "rope";
        case QBH_BLOCK_COMMON_OP_SOFTMAX:
            return "softmax";
        case QBH_BLOCK_COMMON_OP_SILU:
            return "silu";
        case QBH_BLOCK_COMMON_OP_RMS_NORM |
             QBH_BLOCK_COMMON_OP_SILU:
            return "rms_silu";
        case QBH_BLOCK_COMMON_OP_RMS_NORM |
             QBH_BLOCK_COMMON_OP_SILU |
             QBH_BLOCK_COMMON_OP_ROPE:
            return "rms_silu_rope";
        case QBH_BLOCK_COMMON_OP_RMS_NORM |
             QBH_BLOCK_COMMON_OP_ROPE:
            return "rms_rope";
        case QBH_BLOCK_COMMON_OP_RMS_NORM |
             QBH_BLOCK_COMMON_OP_ROPE |
             QBH_BLOCK_COMMON_OP_SOFTMAX:
            return "rms_rope_softmax";
        case QBH_BLOCK_COMMON_OPS_HVX_FP16:
            return "hvx_fp16";
        default:
            return "custom_mask";
    }
}

static int qbh_make_path(char *destination, size_t destination_bytes,
                         const char *root, const char *file) {
    int written = snprintf(destination, destination_bytes, "%s/%s",
                           root, file);
    return written >= 0 && (size_t)written < destination_bytes ? 0 : -1;
}

static int qbh_file_size(const char *path, uint32_t expected_bytes) {
    struct stat status;
    return stat(path, &status) == 0 && status.st_size >= 0 &&
                   (uint64_t)status.st_size == expected_bytes
               ? 0
               : -1;
}

static int qbh_prepare_slot(struct qbh_file_slot *slot,
                            const char *root, const char *file,
                            uint32_t expected_bytes, size_t *cursor) {
    if (slot == NULL || cursor == NULL ||
        qbh_make_path(slot->path, sizeof(slot->path), root, file) != 0 ||
        qbh_file_size(slot->path, expected_bytes) != 0) {
        return -1;
    }
    *cursor = qbh_align_up_size(*cursor, QBH_HOST_ALIGNMENT);
    if (*cursor > UINT32_MAX || expected_bytes > UINT32_MAX - *cursor) {
        return -1;
    }
    slot->offset = (uint32_t)*cursor;
    slot->expected_bytes = expected_bytes;
    *cursor += expected_bytes;
    return 0;
}

static int qbh_read_slot(uint8_t *shared,
                         const struct qbh_file_slot *slot) {
    FILE *stream = fopen(slot->path, "rb");
    size_t read_bytes;
    if (stream == NULL) {
        return -1;
    }
    read_bytes = fread(shared + slot->offset, 1, slot->expected_bytes,
                       stream);
    if (fclose(stream) != 0 || read_bytes != slot->expected_bytes) {
        return -1;
    }
    return 0;
}

static int qbh_hmx_native_u8_cache_formats(uint32_t k_format,
                                            uint32_t v_format) {
    return (k_format == QBH_KV_CACHE_FORMAT_HMX_U8_K_WEIGHT_V1 &&
            v_format == QBH_KV_CACHE_FORMAT_HMX_U8_V_WEIGHT_V1) ||
           (k_format ==
                QBH_KV_CACHE_FORMAT_HMX_U8_K_WEIGHT_DELTA_V2 &&
            v_format ==
                QBH_KV_CACHE_FORMAT_HMX_U8_V_WEIGHT_DELTA_V2) ||
           (k_format ==
                QBH_KV_CACHE_FORMAT_HMX_U8_K_SEGMENTED_V4 &&
            (v_format ==
                 QBH_KV_CACHE_FORMAT_HMX_U8_V_SEGMENTED_V4 ||
             v_format ==
                 QBH_KV_CACHE_FORMAT_HMX_U8_V_QUARTET_TAIL_V5 ||
             v_format ==
                 QBH_KV_CACHE_FORMAT_HMX_U8_V_ATTENTION_PUBLISH_V6 ||
             v_format ==
                 QBH_KV_CACHE_FORMAT_HMX_U8_V_VTCM_TAIL_V7));
}

static int qbh_hmx_native_u8_delta_cache_formats(
    uint32_t k_format, uint32_t v_format) {
    return k_format ==
               QBH_KV_CACHE_FORMAT_HMX_U8_K_WEIGHT_DELTA_V2 &&
           v_format ==
               QBH_KV_CACHE_FORMAT_HMX_U8_V_WEIGHT_DELTA_V2;
}

static int qbh_hmx_native_u8_segmented_cache_formats(
    uint32_t k_format, uint32_t v_format) {
    return k_format ==
               QBH_KV_CACHE_FORMAT_HMX_U8_K_SEGMENTED_V4 &&
           (v_format ==
                QBH_KV_CACHE_FORMAT_HMX_U8_V_SEGMENTED_V4 ||
            v_format ==
                QBH_KV_CACHE_FORMAT_HMX_U8_V_QUARTET_TAIL_V5 ||
            v_format ==
                QBH_KV_CACHE_FORMAT_HMX_U8_V_ATTENTION_PUBLISH_V6 ||
            v_format ==
                QBH_KV_CACHE_FORMAT_HMX_U8_V_VTCM_TAIL_V7);
}

static int qbh_hmx_native_f16_cache_formats(uint32_t k_format,
                                             uint32_t v_format) {
    return k_format == QBH_KV_CACHE_FORMAT_HMX_F16_K_WEIGHT_V1 &&
           v_format == QBH_KV_CACHE_FORMAT_HMX_F16_V_WEIGHT_V1;
}

static int qbh_hmx_native_cache_formats(uint32_t k_format,
                                         uint32_t v_format) {
    return qbh_hmx_native_u8_cache_formats(k_format, v_format) ||
           qbh_hmx_native_f16_cache_formats(k_format, v_format);
}

static uint32_t qbh_host_k_cache_bytes(uint32_t variant,
                                       uint32_t capacity,
                                       uint32_t element_bytes,
                                       uint32_t k_format) {
    if (variant == QBH_BLOCK_W4U8 &&
        k_format == QBH_KV_CACHE_FORMAT_HMX_U8_K_WEIGHT_V1) {
        return QBH_KV_CACHE_HMX_K_BYTES(capacity);
    }
    if (variant == QBH_BLOCK_W4U8 &&
        k_format ==
            QBH_KV_CACHE_FORMAT_HMX_U8_K_WEIGHT_DELTA_V2) {
        return QBH_KV_CACHE_HMX_U8_K_DELTA_BYTES(capacity);
    }
    if (variant == QBH_BLOCK_W4U8 &&
        k_format == QBH_KV_CACHE_FORMAT_HMX_U8_K_SEGMENTED_V4) {
        return QBH_KV_CACHE_HMX_U8_K_SEGMENTED_BYTES(capacity);
    }
    if (variant != QBH_BLOCK_W4U8 &&
        k_format == QBH_KV_CACHE_FORMAT_HMX_F16_K_WEIGHT_V1) {
        return QBH_KV_CACHE_HMX_F16_K_BYTES(capacity);
    }
    return capacity * QBH_BLOCK_KV_HIDDEN * element_bytes;
}

static uint32_t qbh_host_v_cache_bytes(uint32_t variant,
                                       uint32_t capacity,
                                       uint32_t element_bytes,
                                       uint32_t v_format) {
    if (variant == QBH_BLOCK_W4U8 &&
        v_format == QBH_KV_CACHE_FORMAT_HMX_U8_V_WEIGHT_V1) {
        return QBH_KV_CACHE_HMX_V_BYTES(capacity);
    }
    if (variant == QBH_BLOCK_W4U8 &&
        v_format ==
            QBH_KV_CACHE_FORMAT_HMX_U8_V_WEIGHT_DELTA_V2) {
        return QBH_KV_CACHE_HMX_U8_V_DELTA_BYTES(capacity);
    }
    if (variant == QBH_BLOCK_W4U8 &&
        (v_format == QBH_KV_CACHE_FORMAT_HMX_U8_V_SEGMENTED_V4 ||
         v_format ==
             QBH_KV_CACHE_FORMAT_HMX_U8_V_VTCM_TAIL_V7)) {
        return QBH_KV_CACHE_HMX_U8_V_SEGMENTED_BYTES(capacity);
    }
    if (variant == QBH_BLOCK_W4U8 &&
        (v_format == QBH_KV_CACHE_FORMAT_HMX_U8_V_QUARTET_TAIL_V5 ||
         v_format ==
             QBH_KV_CACHE_FORMAT_HMX_U8_V_ATTENTION_PUBLISH_V6)) {
        return QBH_KV_CACHE_HMX_U8_V_QUARTET_BYTES(capacity);
    }
    if (variant != QBH_BLOCK_W4U8 &&
        v_format == QBH_KV_CACHE_FORMAT_HMX_F16_V_WEIGHT_V1) {
        return QBH_KV_CACHE_HMX_F16_V_BYTES(capacity);
    }
    return capacity * QBH_BLOCK_KV_HIDDEN * element_bytes;
}

static int qbh_prepare_vertical_layer_slots(
    struct qbh_vertical_layer_slots *slots, const char *root,
    uint32_t layer_index, uint32_t variant,
    uint32_t attention_pipeline_mode, uint32_t mlp_mode,
    uint32_t cache_capacity, uint32_t element_bytes,
    uint32_t k_cache_format, uint32_t v_cache_format,
    size_t *cursor) {
    char name[160];
    const char *suffix = variant == QBH_BLOCK_W4U8 ? "u8" : "f16";
    const char *norm_names[4] = {
        "input_norm_weight_f16.bin",
        "post_norm_weight_f16.bin",
        "q_norm_weight_f16.bin",
        "k_norm_weight_f16.bin",
    };
    const uint32_t norm_bytes[4] = {
        QBH_BLOCK_HIDDEN * sizeof(uint16_t),
        QBH_BLOCK_HIDDEN * sizeof(uint16_t),
        QBH_BLOCK_HEAD_DIM * sizeof(uint16_t),
        QBH_BLOCK_HEAD_DIM * sizeof(uint16_t),
    };
    int status;

    if (slots == NULL || cursor == NULL) {
        return -1;
    }
    memset(slots, 0, sizeof(*slots));
    status = snprintf(name, sizeof(name),
                      "layer%" PRIu32 "/qparams_u8.bin", layer_index);
    if (status < 0 || (size_t)status >= sizeof(name) ||
        qbh_prepare_slot(
            &slots->qparam, root, name,
            QBH_BLOCK_QPARAM_COUNT * QBH_BLOCK_QPARAM_RECORD_BYTES,
            cursor) != 0) {
        return -1;
    }
    for (uint32_t index = 0U; index < 4U; ++index) {
        status = snprintf(name, sizeof(name), "layer%" PRIu32 "/%s",
                          layer_index, norm_names[index]);
        if (status < 0 || (size_t)status >= sizeof(name) ||
            qbh_prepare_slot(&slots->norms[index], root, name,
                             norm_bytes[index], cursor) != 0) {
            return -1;
        }
    }
    if (qbh_attention_u8_enabled(attention_pipeline_mode)) {
        status = snprintf(
            name, sizeof(name),
            "layer%" PRIu32 "/attention_config_all_groups.bin",
            layer_index);
        if (status < 0 || (size_t)status >= sizeof(name) ||
            qbh_prepare_slot(
                &slots->attention_config, root, name,
                QBH_BLOCK_ATTENTION_CONFIG_BYTES, cursor) != 0) {
            return -1;
        }
    }
    {
        const uint32_t cache_bytes[2] = {
            qbh_host_k_cache_bytes(
                variant, cache_capacity, element_bytes,
                k_cache_format),
            qbh_host_v_cache_bytes(
                variant, cache_capacity, element_bytes,
                v_cache_format),
        };
        const char *cache_kinds[2] = {"k", "v"};
        const int hmx_native_u8 = qbh_hmx_native_u8_cache_formats(
            k_cache_format, v_cache_format);
        const int hmx_native_u8_delta =
            qbh_hmx_native_u8_delta_cache_formats(
                k_cache_format, v_cache_format);
        const int hmx_native_u8_segmented =
            qbh_hmx_native_u8_segmented_cache_formats(
                k_cache_format, v_cache_format);
        const int hmx_native_f16 = qbh_hmx_native_f16_cache_formats(
            k_cache_format, v_cache_format);
        const int hmx_native = hmx_native_u8 || hmx_native_f16;
        const char *hmx_suffix = hmx_native_u8_segmented
            ? "u8_segmented"
            : (hmx_native_u8_delta
                   ? "u8_delta" : (hmx_native_u8 ? "u8" : "f16"));
        for (uint32_t index = 0U; index < 2U; ++index) {
            if (hmx_native) {
                status = snprintf(
                    name, sizeof(name),
                    "layer%" PRIu32 "/kv_cache_%s_hmx_%s.bin",
                    layer_index, cache_kinds[index],
                    hmx_suffix);
            } else {
                status = snprintf(
                    name, sizeof(name),
                    "layer%" PRIu32 "/kv_cache_%s_%s.bin",
                    layer_index, cache_kinds[index], suffix);
            }
            if (status < 0 || (size_t)status >= sizeof(name) ||
                qbh_prepare_slot(&slots->caches[index], root, name,
                                 cache_bytes[index], cursor) != 0) {
                return -1;
            }
            if (hmx_native) {
                status = snprintf(
                    name, sizeof(name),
                    "layer%" PRIu32
                    "/reference_kv_cache_%s_hmx_%s_step00.bin",
                    layer_index, cache_kinds[index],
                    hmx_suffix);
            } else {
                status = snprintf(
                    name, sizeof(name),
                    "layer%" PRIu32
                    "/reference_kv_cache_%s_%s.bin",
                    layer_index, cache_kinds[index], suffix);
            }
            if (status < 0 || (size_t)status >= sizeof(name) ||
                qbh_prepare_slot(
                    &slots->cache_references[index], root, name,
                    cache_bytes[index], cursor) != 0) {
                return -1;
            }
        }
    }
    if (qbh_block_mlp_is_w4u8_streaming(mlp_mode)) {
        status = snprintf(name, sizeof(name),
                          "layer%" PRIu32 "/silu_up_lut_u16.bin",
                          layer_index);
        if (status < 0 || (size_t)status >= sizeof(name) ||
            qbh_prepare_slot(&slots->silu_lut, root, name,
                             QBH_MLP_LUT_BYTES, cursor) != 0) {
            return -1;
        }
    }
    for (uint32_t projection = 0U;
         projection < QBH_BLOCK_PROJECTION_COUNT; ++projection) {
        const uint32_t weight_bytes = variant == QBH_BLOCK_F16F16
            ? qbh_projection_k[projection] *
                  qbh_projection_n[projection] * sizeof(uint16_t)
            : qbh_projection_k[projection] *
                  qbh_projection_n[projection] / 2U;
        status = snprintf(
            name, sizeof(name),
            variant == QBH_BLOCK_F16F16
                ? "layer%" PRIu32 "/%s_weight_f16_hmx.bin"
                : "layer%" PRIu32 "/%s_weight_w4_hmx.bin",
            layer_index, qbh_projection_names[projection]);
        if (status < 0 || (size_t)status >= sizeof(name) ||
            qbh_prepare_slot(&slots->weights[projection], root, name,
                             weight_bytes, cursor) != 0) {
            return -1;
        }
        if (variant != QBH_BLOCK_F16F16) {
            status = snprintf(
                name, sizeof(name),
                "layer%" PRIu32 "/%s_weight_w4_scale_f32.bin",
                layer_index, qbh_projection_names[projection]);
            if (status < 0 || (size_t)status >= sizeof(name) ||
                qbh_prepare_slot(
                    &slots->scales[projection], root, name,
                    qbh_projection_n[projection] * sizeof(float),
                    cursor) != 0) {
                return -1;
            }
        }
    }
    return 0;
}

static int qbh_read_vertical_layer_slots(
    uint8_t *shared, const struct qbh_vertical_layer_slots *slots,
    uint32_t variant, uint32_t attention_pipeline_mode,
    uint32_t mlp_mode) {
    if (qbh_read_slot(shared, &slots->qparam) != 0) {
        return -1;
    }
    for (uint32_t index = 0U; index < 4U; ++index) {
        if (qbh_read_slot(shared, &slots->norms[index]) != 0) {
            return -1;
        }
    }
    if (qbh_attention_u8_enabled(attention_pipeline_mode) &&
        qbh_read_slot(shared, &slots->attention_config) != 0) {
        return -1;
    }
    for (uint32_t index = 0U; index < 2U; ++index) {
        if (qbh_read_slot(shared, &slots->caches[index]) != 0 ||
            qbh_read_slot(shared, &slots->cache_references[index]) != 0) {
            return -1;
        }
    }
    if (qbh_block_mlp_is_w4u8_streaming(mlp_mode) &&
        qbh_read_slot(shared, &slots->silu_lut) != 0) {
        return -1;
    }
    for (uint32_t projection = 0U;
         projection < QBH_BLOCK_PROJECTION_COUNT; ++projection) {
        if (qbh_read_slot(shared, &slots->weights[projection]) != 0 ||
            (variant != QBH_BLOCK_F16F16 &&
             qbh_read_slot(shared, &slots->scales[projection]) != 0)) {
            return -1;
        }
    }
    return 0;
}

static uint16_t qbh_float_to_half_bits(float value) {
    __fp16 converted = (__fp16)value;
    uint16_t bits;
    memcpy(&bits, &converted, sizeof(bits));
    return bits;
}

static int qbh_prepare_gate_up_scale_cache(
    const struct qbh_block_header *header, uint8_t *shared) {
    uint8_t *destination;
    uint32_t cache_bytes = 0U;

    if (header->w4f16_pipeline_mode !=
        QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_GATE4_DMA8_CROSS_PREFETCH) {
        return 0;
    }
    destination = shared + header->w4f16_gate_up_scale_cache_offset;
    for (uint32_t projection = QBH_BLOCK_PROJ_GATE;
         projection <= QBH_BLOCK_PROJ_UP; ++projection) {
        const struct qbh_block_projection_desc *desc =
            &header->projections[projection];
        const float *scales =
            (const float *)(shared + desc->scale_offset);
        uint32_t tile_count = desc->n / QBH_HMX_FP16_COLS;
        uint32_t projection_bytes =
            tile_count * QBH_HMX_FP16_SCALE_BYTES;

        if (desc->n % QBH_HMX_FP16_COLS != 0U ||
            desc->scale_bytes != desc->n * sizeof(float) ||
            projection_bytes >
                header->w4f16_gate_up_scale_cache_bytes - cache_bytes) {
            return -1;
        }
        for (uint32_t tile = 0U; tile < tile_count; ++tile) {
            uint16_t *block = (uint16_t *)(destination + cache_bytes +
                (size_t)tile * QBH_HMX_FP16_SCALE_BYTES);
            memset(block, 0, QBH_HMX_FP16_SCALE_BYTES);
            for (uint32_t channel = 0U;
                 channel < QBH_HMX_FP16_COLS; ++channel) {
                block[channel * 2U] = qbh_float_to_half_bits(
                    scales[(size_t)tile * QBH_HMX_FP16_COLS + channel]);
            }
        }
        cache_bytes += projection_bytes;
    }
    return 0;
}

static float qbh_half_bits_to_float(uint16_t bits) {
    __fp16 converted;
    memcpy(&converted, &bits, sizeof(bits));
    return (float)converted;
}

static int8_t qbh_decode_w4(const uint8_t *packed,
                            uint32_t physical_index) {
    uint8_t byte = packed[physical_index / 2U];
    uint8_t nibble = (physical_index & 1U) != 0U
                         ? (uint8_t)(byte >> 4U)
                         : (uint8_t)(byte & UINT8_C(0x0f));
    return (nibble & UINT8_C(0x08)) != 0U
               ? (int8_t)(nibble | UINT8_C(0xf0))
               : (int8_t)nibble;
}

static int qbh_load_qparams_array(
    struct qbh_block_qparam target_qparams[QBH_BLOCK_QPARAM_COUNT],
    const uint8_t *records) {
    const struct qbh_qparam_record *source =
        (const struct qbh_qparam_record *)records;
    uint32_t found = 0U;
    for (uint32_t target = 0; target < QBH_BLOCK_QPARAM_COUNT; ++target) {
        int matched = 0;
        for (uint32_t record = 0; record < QBH_BLOCK_QPARAM_COUNT;
             ++record) {
            if (strncmp(source[record].name, qbh_qparam_names[target],
                        sizeof(source[record].name)) == 0) {
                target_qparams[target].scale = source[record].scale;
                target_qparams[target].zero_point =
                    source[record].zero_point;
                target_qparams[target].minimum = source[record].minimum;
                target_qparams[target].maximum = source[record].maximum;
                matched = 1;
                ++found;
                break;
            }
        }
        if (matched == 0) {
            return -1;
        }
    }
    return found == QBH_BLOCK_QPARAM_COUNT ? 0 : -1;
}

static int qbh_load_qparams(struct qbh_block_header *header,
                            const uint8_t *records) {
    return qbh_load_qparams_array(header->qparams, records);
}

static int qbh_load_generation_qparams(
    struct qbh_block_header *header, const uint8_t *records) {
    static const char *const names[QBH_GENERATION_QPARAM_COUNT] = {
        "generation_final_norm_output",
        "generation_lm_head_output",
    };
    struct qbh_block_qparam *targets[QBH_GENERATION_QPARAM_COUNT] = {
        &header->generation_final_norm_output_qparam,
        &header->generation_lm_head_output_qparam,
    };
    const struct qbh_qparam_record *source =
        (const struct qbh_qparam_record *)records;

    for (uint32_t target = 0U;
         target < QBH_GENERATION_QPARAM_COUNT; ++target) {
        int found = 0;
        for (uint32_t record = 0U;
             record < QBH_GENERATION_QPARAM_COUNT; ++record) {
            if (strncmp(source[record].name, names[target],
                        sizeof(source[record].name)) == 0) {
                targets[target]->scale = source[record].scale;
                targets[target]->zero_point = source[record].zero_point;
                targets[target]->minimum = source[record].minimum;
                targets[target]->maximum = source[record].maximum;
                found = 1;
                break;
            }
        }
        if (found == 0 || !(targets[target]->scale > 0.0f) ||
            !isfinite(targets[target]->scale) ||
            targets[target]->zero_point < 0 ||
            targets[target]->zero_point > UINT8_MAX) {
            return -1;
        }
    }
    return 0;
}

static void qbh_bind_host_slice_layer(
    struct qbh_block_header *header, uint32_t slice_index) {
    const struct qbh_block_layer_desc *layer =
        &header->slice_layers[slice_index];
    header->qparam_offset = layer->qparam_offset;
    header->qparam_bytes = layer->qparam_bytes;
    header->input_norm_weight_offset = layer->input_norm_weight_offset;
    header->input_norm_weight_bytes = layer->input_norm_weight_bytes;
    header->post_norm_weight_offset = layer->post_norm_weight_offset;
    header->post_norm_weight_bytes = layer->post_norm_weight_bytes;
    header->q_norm_weight_offset = layer->q_norm_weight_offset;
    header->q_norm_weight_bytes = layer->q_norm_weight_bytes;
    header->k_norm_weight_offset = layer->k_norm_weight_offset;
    header->k_norm_weight_bytes = layer->k_norm_weight_bytes;
    header->w4u8_gate_up_bundle_offset =
        layer->w4u8_gate_up_bundle_offset;
    header->w4u8_gate_up_bundle_bytes =
        layer->w4u8_gate_up_bundle_bytes;
    header->w4u8_down_bundle_offset = layer->w4u8_down_bundle_offset;
    header->w4u8_down_bundle_bytes = layer->w4u8_down_bundle_bytes;
    header->w4u8_silu_lut_offset = layer->w4u8_silu_lut_offset;
    header->w4u8_silu_lut_bytes = layer->w4u8_silu_lut_bytes;
    header->attention_config_offset = layer->attention_config_offset;
    header->attention_config_bytes = layer->attention_config_bytes;
    header->kv_cache_k_format = layer->kv_cache_k_format;
    header->kv_cache_v_format = layer->kv_cache_v_format;
    header->kv_cache_padded_capacity =
        layer->kv_cache_padded_capacity;
    header->kv_cache_k_offset = layer->kv_cache_k_offset;
    header->kv_cache_k_bytes = layer->kv_cache_k_bytes;
    header->kv_cache_v_offset = layer->kv_cache_v_offset;
    header->kv_cache_v_bytes = layer->kv_cache_v_bytes;
    header->w4f16_gate_up_scale_cache_offset =
        layer->w4f16_gate_up_scale_cache_offset;
    header->w4f16_gate_up_scale_cache_bytes =
        layer->w4f16_gate_up_scale_cache_bytes;
    memcpy(header->projections, layer->projections,
           sizeof(header->projections));
    memcpy(header->qparams, layer->qparams, sizeof(header->qparams));
}

static int qbh_build_bias_words(
    struct qbh_block_header *header, uint8_t *shared,
    uint32_t projection_index) {
    struct qbh_block_projection_desc *desc =
        &header->projections[projection_index];
    const uint8_t *packed = shared + desc->weight_offset;
    const float *scales = (const float *)(shared + desc->scale_offset);
    uint32_t *bias = (uint32_t *)(shared + desc->bias_offset);
    const struct qbh_block_qparam *input_qparam =
        &header->qparams[qbh_projection_input_qparam[projection_index]];
    const struct qbh_block_qparam *output_qparam =
        &header->qparams[qbh_projection_output_qparam[projection_index]];
    uint32_t k_tiles = desc->k / 32U;
    uint32_t n_tiles = desc->n / 32U;

    for (uint32_t n_tile = 0; n_tile < n_tiles; ++n_tile) {
        for (uint32_t output = 0; output < 32U; ++output) {
            int32_t sum = 0;
            uint32_t global_output = n_tile * 32U + output;
            float ratio = input_qparam->scale * scales[global_output] /
                          output_qparam->scale;
            int64_t offset;
            if (!(ratio > 0.0f) || !isfinite(ratio)) {
                return -1;
            }
            for (uint32_t k_tile = 0; k_tile < k_tiles; ++k_tile) {
                const uint8_t *tile = packed +
                    ((size_t)n_tile * k_tiles + k_tile) * 512U;
                for (uint32_t input = 0; input < 32U; ++input) {
                    uint32_t physical =
                        ((input / 4U) * 32U + output) * 4U +
                        input % 4U;
                    sum += qbh_decode_w4(tile, physical);
                }
            }
            bias[(size_t)n_tile * 64U + output] =
                qbh_float_to_half_bits(512.0f * ratio);
            offset = llround(
                -(double)input_qparam->zero_point * (double)sum +
                (double)output_qparam->zero_point / (double)ratio);
            if (offset < INT32_MIN || offset > INT32_MAX) {
                return -1;
            }
            bias[(size_t)n_tile * 64U + 32U + output] =
                (uint32_t)(int32_t)offset;
        }
    }
    return 0;
}

static int qbh_build_w4u8_streaming_bundles(
    struct qbh_block_header *header, uint8_t *shared,
    const struct qbh_projection_layout *gate_up_layout,
    const struct qbh_projection_layout *down_layout) {
    const struct qbh_block_projection_desc *gate;
    const struct qbh_block_projection_desc *up;
    const struct qbh_block_projection_desc *down;
    uint8_t *gate_up_destination;
    uint8_t *down_destination;
    uint32_t gate_up_source_tiles;

    if (header == NULL || shared == NULL || gate_up_layout == NULL ||
        down_layout == NULL ||
        header->variant != QBH_BLOCK_W4U8 ||
        !qbh_block_mlp_is_w4u8_streaming(header->mlp_mode) ||
        header->w4u8_gate_up_bundle_bytes !=
            gate_up_layout->stored_weight_bytes ||
        header->w4u8_down_bundle_bytes !=
            down_layout->stored_weight_bytes) {
        return -1;
    }
    gate = &header->projections[QBH_BLOCK_PROJ_GATE];
    up = &header->projections[QBH_BLOCK_PROJ_UP];
    down = &header->projections[QBH_BLOCK_PROJ_DOWN];
    gate_up_source_tiles = gate->n / QBH_HMX_OUTPUT_CHANNELS;
    if (gate->k != up->k || gate->n != up->n ||
        gate_up_layout->n_tiles != 2U * gate_up_source_tiles ||
        gate_up_layout->w4_packed_chunk_bytes !=
            gate->k / QBH_HMX_INPUT_CHANNELS * QBH_W4_PACKED_TILE_BYTES ||
        down_layout->w4_packed_chunk_bytes !=
            down->k / QBH_HMX_INPUT_CHANNELS * QBH_W4_PACKED_TILE_BYTES) {
        return -1;
    }

    gate_up_destination = shared + header->w4u8_gate_up_bundle_offset;
    down_destination = shared + header->w4u8_down_bundle_offset;
    memset(gate_up_destination, 0, header->w4u8_gate_up_bundle_bytes);
    memset(down_destination, 0, header->w4u8_down_bundle_bytes);
    for (uint32_t tile = 0U; tile < gate_up_source_tiles; ++tile) {
        const struct qbh_block_projection_desc *sources[2] = {gate, up};
        for (uint32_t lane = 0U; lane < 2U; ++lane) {
            uint32_t output_tile = 2U * tile + lane;
            uint8_t *bundle = gate_up_destination +
                (size_t)output_tile *
                    gate_up_layout->stored_weight_bundle_bytes;
            memcpy(bundle,
                   shared + sources[lane]->weight_offset +
                       (size_t)tile * gate_up_layout->w4_packed_chunk_bytes,
                   gate_up_layout->w4_packed_chunk_bytes);
            memcpy(bundle + gate_up_layout->w4_bias_offset,
                   shared + sources[lane]->bias_offset +
                       (size_t)tile * QBH_HMX_BIAS_BYTES,
                   QBH_HMX_BIAS_BYTES);
        }
    }
    for (uint32_t tile = 0U; tile < down_layout->n_tiles; ++tile) {
        uint8_t *bundle = down_destination +
            (size_t)tile * down_layout->stored_weight_bundle_bytes;
        memcpy(bundle,
               shared + down->weight_offset +
                   (size_t)tile * down_layout->w4_packed_chunk_bytes,
               down_layout->w4_packed_chunk_bytes);
        memcpy(bundle + down_layout->w4_bias_offset,
               shared + down->bias_offset +
                   (size_t)tile * QBH_HMX_BIAS_BYTES,
               QBH_HMX_BIAS_BYTES);
    }
    return 0;
}

static struct qbh_error_metrics qbh_compare_f16(
    const uint16_t *actual, const uint16_t *reference,
    uint32_t elements) {
    struct qbh_error_metrics metrics;
    double absolute_sum = 0.0;
    double squared_sum = 0.0;
    double dot = 0.0;
    double actual_norm = 0.0;
    double reference_norm = 0.0;
    memset(&metrics, 0, sizeof(metrics));
    for (uint32_t index = 0; index < elements; ++index) {
        double a = qbh_half_bits_to_float(actual[index]);
        double b = qbh_half_bits_to_float(reference[index]);
        double difference = fabs(a - b);
        if (difference > metrics.max_abs) {
            metrics.max_abs = difference;
        }
        absolute_sum += difference;
        squared_sum += difference * difference;
        dot += a * b;
        actual_norm += a * a;
        reference_norm += b * b;
    }
    metrics.mean_abs = absolute_sum / elements;
    metrics.rmse = sqrt(squared_sum / elements);
    metrics.nrmse = reference_norm > 0.0
                        ? sqrt(squared_sum / reference_norm)
                        : (squared_sum == 0.0 ? 0.0 : DBL_MAX);
    metrics.cosine = actual_norm > 0.0 && reference_norm > 0.0
                         ? dot / sqrt(actual_norm * reference_norm)
                         : 0.0;
    metrics.elements = elements;
    return metrics;
}

static struct qbh_error_metrics qbh_compare_u8(
    const uint8_t *actual, const uint8_t *reference,
    uint32_t elements,
    const struct qbh_block_qparam *qparam) {
    struct qbh_error_metrics metrics;
    double absolute_sum = 0.0;
    double squared_sum = 0.0;
    double dot = 0.0;
    double actual_norm = 0.0;
    double reference_norm = 0.0;
    memset(&metrics, 0, sizeof(metrics));
    for (uint32_t index = 0; index < elements; ++index) {
        uint32_t lsb = actual[index] > reference[index]
                           ? actual[index] - reference[index]
                           : reference[index] - actual[index];
        double a = ((double)actual[index] - qparam->zero_point) *
                   qparam->scale;
        double b = ((double)reference[index] - qparam->zero_point) *
                   qparam->scale;
        double difference = fabs(a - b);
        if (lsb != 0U) {
            ++metrics.mismatches;
        }
        if (lsb > metrics.max_lsb) {
            metrics.max_lsb = lsb;
        }
        if (difference > metrics.max_abs) {
            metrics.max_abs = difference;
        }
        absolute_sum += difference;
        squared_sum += difference * difference;
        dot += a * b;
        actual_norm += a * a;
        reference_norm += b * b;
    }
    metrics.mean_abs = absolute_sum / elements;
    metrics.rmse = sqrt(squared_sum / elements);
    metrics.nrmse = reference_norm > 0.0
                        ? sqrt(squared_sum / reference_norm)
                        : (squared_sum == 0.0 ? 0.0 : DBL_MAX);
    metrics.cosine = actual_norm > 0.0 && reference_norm > 0.0
                         ? dot / sqrt(actual_norm * reference_norm)
                         : 0.0;
    metrics.elements = elements;
    return metrics;
}

static struct qbh_error_metrics qbh_compare_scan_u8(
    const uint8_t *actual, const uint8_t *reference,
    uint32_t logical_m, const struct qbh_block_qparam *qparam) {
    struct qbh_error_metrics metrics;
    double absolute_sum = 0.0;
    double squared_sum = 0.0;
    double dot = 0.0;
    double actual_norm = 0.0;
    double reference_norm = 0.0;
    uint64_t elements = 0U;

    memset(&metrics, 0, sizeof(metrics));
    for (uint32_t logical_row = 0U;
         logical_row < logical_m; ++logical_row) {
        const uint32_t chunk = logical_row / QBH_BLOCK_M;
        const uint32_t row = logical_row % QBH_BLOCK_M;
        const size_t base =
            ((size_t)chunk * QBH_BLOCK_M + row) * QBH_BLOCK_HIDDEN;
        for (uint32_t channel = 0U;
             channel < QBH_BLOCK_HIDDEN; ++channel) {
            const uint8_t a_code = actual[base + channel];
            const uint8_t b_code = reference[base + channel];
            const uint32_t lsb = a_code > b_code
                ? a_code - b_code : b_code - a_code;
            const double a =
                ((double)a_code - qparam->zero_point) * qparam->scale;
            const double b =
                ((double)b_code - qparam->zero_point) * qparam->scale;
            const double difference = fabs(a - b);
            metrics.mismatches += lsb != 0U;
            if (lsb > metrics.max_lsb) {
                metrics.max_lsb = lsb;
            }
            if (difference > metrics.max_abs) {
                metrics.max_abs = difference;
            }
            absolute_sum += difference;
            squared_sum += difference * difference;
            dot += a * b;
            actual_norm += a * a;
            reference_norm += b * b;
            ++elements;
        }
    }
    metrics.mean_abs = absolute_sum / (double)elements;
    metrics.rmse = sqrt(squared_sum / (double)elements);
    metrics.nrmse = reference_norm > 0.0
        ? sqrt(squared_sum / reference_norm)
        : (squared_sum == 0.0 ? 0.0 : DBL_MAX);
    metrics.cosine = actual_norm > 0.0 && reference_norm > 0.0
        ? dot / sqrt(actual_norm * reference_norm) : 0.0;
    metrics.elements = elements;
    return metrics;
}

static struct qbh_error_metrics qbh_compare_scan_f16(
    const uint16_t *actual, const uint16_t *reference,
    uint32_t logical_m) {
    struct qbh_error_metrics metrics;
    double absolute_sum = 0.0;
    double squared_sum = 0.0;
    double dot = 0.0;
    double actual_norm = 0.0;
    double reference_norm = 0.0;
    uint64_t elements = 0U;

    memset(&metrics, 0, sizeof(metrics));
    for (uint32_t logical_row = 0U;
         logical_row < logical_m; ++logical_row) {
        const uint32_t chunk = logical_row / QBH_BLOCK_M;
        const uint32_t row = logical_row % QBH_BLOCK_M;
        const size_t base =
            ((size_t)chunk * QBH_BLOCK_M + row) * QBH_BLOCK_HIDDEN;
        for (uint32_t channel = 0U;
             channel < QBH_BLOCK_HIDDEN; ++channel) {
            const uint16_t a_bits = actual[base + channel];
            const uint16_t b_bits = reference[base + channel];
            float a_f32;
            float b_f32;
            double difference;
            __fp16 a_f16;
            __fp16 b_f16;

            memcpy(&a_f16, &a_bits, sizeof(a_f16));
            memcpy(&b_f16, &b_bits, sizeof(b_f16));
            a_f32 = (float)a_f16;
            b_f32 = (float)b_f16;
            metrics.mismatches += a_bits != b_bits;
            if (!isfinite(a_f32) || !isfinite(b_f32)) {
                ++metrics.nonfinite_count;
                ++metrics.mixed_tolerance_violations;
                metrics.max_abs = DBL_MAX;
                metrics.max_required_rtol_after_atol = DBL_MAX;
                ++elements;
                continue;
            }
            difference = fabs((double)a_f32 - (double)b_f32);
            if (difference > metrics.max_abs) {
                metrics.max_abs = difference;
            }
            if (difference > QBH_REPLAY_FP16_ATOL +
                                 QBH_REPLAY_FP16_RTOL * fabs((double)b_f32)) {
                ++metrics.mixed_tolerance_violations;
            }
            if (difference > QBH_REPLAY_FP16_ATOL) {
                const double required_rtol = b_f32 != 0.0f
                    ? (difference - QBH_REPLAY_FP16_ATOL) /
                          fabs((double)b_f32)
                    : DBL_MAX;
                if (required_rtol >
                        metrics.max_required_rtol_after_atol) {
                    metrics.max_required_rtol_after_atol = required_rtol;
                }
            }
            absolute_sum += difference;
            squared_sum += difference * difference;
            dot += (double)a_f32 * (double)b_f32;
            actual_norm += (double)a_f32 * (double)a_f32;
            reference_norm += (double)b_f32 * (double)b_f32;
            ++elements;
        }
    }
    metrics.mean_abs = absolute_sum / (double)elements;
    metrics.rmse = sqrt(squared_sum / (double)elements);
    metrics.nrmse = reference_norm > 0.0
        ? sqrt(squared_sum / reference_norm)
        : (squared_sum == 0.0 ? 0.0 : DBL_MAX);
    metrics.cosine = actual_norm > 0.0 && reference_norm > 0.0
        ? dot / sqrt(actual_norm * reference_norm) : 0.0;
    metrics.elements = elements;
    return metrics;
}

static uint64_t qbh_count_byte_mismatches(
    const uint8_t *actual, const uint8_t *reference,
    uint32_t bytes) {
    uint64_t mismatches = 0U;
    for (uint32_t index = 0U; index < bytes; ++index) {
        mismatches += actual[index] != reference[index];
    }
    return mismatches;
}

static int qbh_read_named_tensor(
    const char *root, const char *name, uint8_t *destination,
    uint32_t bytes) {
    char path[QBH_HOST_PATH_BYTES];
    FILE *stream;
    size_t read_bytes;
    if (qbh_make_path(path, sizeof(path), root, name) != 0 ||
        qbh_file_size(path, bytes) != 0) {
        return -1;
    }
    stream = fopen(path, "rb");
    if (stream == NULL) {
        return -1;
    }
    read_bytes = fread(destination, 1U, bytes, stream);
    return fclose(stream) == 0 && read_bytes == bytes ? 0 : -1;
}

static uint64_t qbh_cache_prefix_mismatches(
    const uint8_t *actual, const uint8_t *reference,
    uint32_t capacity, uint32_t valid_length,
    uint32_t element_bytes) {
    const uint32_t head_stride =
        capacity * QBH_BLOCK_HEAD_DIM * element_bytes;
    const uint32_t valid_bytes =
        valid_length * QBH_BLOCK_HEAD_DIM * element_bytes;
    uint64_t mismatches = 0U;
    for (uint32_t head = 0U; head < QBH_BLOCK_KV_HEADS; ++head) {
        mismatches += qbh_count_byte_mismatches(
            actual + (size_t)head * head_stride,
            reference + (size_t)head * head_stride,
            valid_bytes);
    }
    return mismatches;
}

static int qbh_hmx_cache_byte_is_appended_token(
    const struct qbh_decode_layer_state *layer, uint32_t kind,
    uint32_t head_relative, uint32_t token) {
    const uint32_t weight_bytes = layer->k_weight_bytes_per_head;

    if (qbh_hmx_native_u8_segmented_cache_formats(
            layer->k_format, layer->v_format)) {
        const uint32_t max_segments =
            QBH_KV_CACHE_HMX_U8_SEGMENT_COUNT(layer->capacity);
        const uint32_t tail_row =
            token % QBH_KV_CACHE_HMX_U8_SEGMENT_TOKENS;
        const int seals_segment =
            tail_row + 1U == QBH_KV_CACHE_HMX_U8_SEGMENT_TOKENS;
        const uint32_t segment =
            token / QBH_KV_CACHE_HMX_U8_SEGMENT_TOKENS;
        uint32_t tail_offset;

        if (kind == 0U) {
            tail_offset =
                max_segments * QBH_KV_CACHE_HMX_U8_SEGMENT_K_BYTES;
            if (head_relative >=
                    tail_offset + tail_row * QBH_BLOCK_HEAD_DIM &&
                head_relative <
                    tail_offset +
                    (tail_row + 1U) * QBH_BLOCK_HEAD_DIM) {
                return 1;
            }
            return seals_segment && segment < max_segments &&
                   head_relative >=
                       segment * QBH_KV_CACHE_HMX_U8_SEGMENT_K_BYTES &&
                   head_relative <
                       (segment + 1U) *
                           QBH_KV_CACHE_HMX_U8_SEGMENT_K_BYTES;
        }

        tail_offset =
            max_segments * QBH_KV_CACHE_HMX_U8_SEGMENT_V_BYTES +
            QBH_KV_CACHE_HMX_V_BIAS_BYTES_PER_HEAD;
        if (layer->v_format ==
                QBH_KV_CACHE_FORMAT_HMX_U8_V_QUARTET_TAIL_V5 ||
            layer->v_format ==
                QBH_KV_CACHE_FORMAT_HMX_U8_V_ATTENTION_PUBLISH_V6) {
            const uint32_t group = tail_row / 4U;
            const uint32_t lane = tail_row % 4U;
            for (uint32_t output_tile = 0U;
                 output_tile < QBH_ATTENTION_HEAD_DIM_TILES;
                 ++output_tile) {
                const uint32_t group_first = tail_offset +
                    output_tile * QBH_HMX_WEIGHT_BYTES +
                    group * 128U;
                const uint32_t first = lane + 1U == 4U
                    ? group_first
                    : group_first +
                        lane * QBH_HMX_OUTPUT_CHANNELS;
                const uint32_t bytes = lane + 1U == 4U
                    ? 128U : QBH_HMX_OUTPUT_CHANNELS;
                if (head_relative >= first &&
                    head_relative < first + bytes) {
                    return 1;
                }
            }
        } else if (head_relative >=
                       tail_offset + tail_row * QBH_BLOCK_HEAD_DIM &&
                   head_relative <
                       tail_offset +
                           (tail_row + 1U) * QBH_BLOCK_HEAD_DIM) {
            return 1;
        }
        if (seals_segment && segment < max_segments) {
            const uint32_t block_first =
                (segment /
                 QBH_KV_CACHE_HMX_U8_V_SEGMENT_BLOCK_SEGMENTS) *
                QBH_KV_CACHE_HMX_U8_V_SEGMENT_BLOCK_SEGMENTS;
            const uint32_t block_count =
                max_segments - block_first <
                        QBH_KV_CACHE_HMX_U8_V_SEGMENT_BLOCK_SEGMENTS
                    ? max_segments - block_first
                    : QBH_KV_CACHE_HMX_U8_V_SEGMENT_BLOCK_SEGMENTS;
            for (uint32_t output_tile = 0U;
                 output_tile < QBH_ATTENTION_HEAD_DIM_TILES;
                 ++output_tile) {
                const uint32_t first =
                    block_first *
                        QBH_KV_CACHE_HMX_U8_SEGMENT_WEIGHT_BYTES +
                    (output_tile * block_count +
                     (segment - block_first)) * QBH_HMX_WEIGHT_BYTES;
                if (head_relative >= first &&
                    head_relative < first + QBH_HMX_WEIGHT_BYTES) {
                    return 1;
                }
            }
        }
        return 0;
    }

    if (qbh_hmx_native_f16_cache_formats(
            layer->k_format, layer->v_format)) {
        if (head_relative >= weight_bytes) {
            if (token < QBH_BLOCK_M) {
                return 0;
            }
            const uint32_t delta_relative =
                head_relative - weight_bytes;
            const uint32_t token_relative = token - QBH_BLOCK_M;
            return delta_relative /
                       (QBH_BLOCK_HEAD_DIM * sizeof(uint16_t)) ==
                       token_relative;
        }
        const uint32_t element_relative = head_relative / 2U;
        const uint32_t byte_in_element = head_relative & 1U;
        const uint32_t tile =
            element_relative / QBH_HMX_FP16_TILE_ELEMENTS;
        const uint32_t within =
            element_relative % QBH_HMX_FP16_TILE_ELEMENTS;
        (void)byte_in_element;
        if (kind == 0U) {
            const uint32_t token_tile =
                token / QBH_HMX_FP16_COLS;
            const uint32_t token_lane =
                token % QBH_HMX_FP16_COLS;
            return tile / QBH_ATTENTION_HEAD_DIM_TILES == token_tile &&
                   ((within / 2U) % QBH_HMX_FP16_COLS) == token_lane;
        }
        {
            const uint32_t capacity_k_tiles =
                layer->padded_capacity / QBH_HMX_FP16_ROWS;
            const uint32_t token_tile =
                token / QBH_HMX_FP16_ROWS;
            const uint32_t token_lane =
                token % QBH_HMX_FP16_ROWS;
            const uint32_t physical_row =
                ((within / (2U * QBH_HMX_FP16_COLS)) * 2U) +
                (within & 1U);
            return tile % capacity_k_tiles == token_tile &&
                   physical_row == token_lane;
        }
    }

    if (qbh_hmx_native_u8_delta_cache_formats(
            layer->k_format, layer->v_format)) {
        const uint32_t bias_bytes = kind == 0U
            ? layer->k_bias_bytes_per_head
            : layer->v_bias_bytes_per_head;
        const uint32_t delta_offset = weight_bytes + bias_bytes;
        if (head_relative < delta_offset || token < QBH_BLOCK_M) {
            return 0;
        }
        return (head_relative - delta_offset) /
                   QBH_BLOCK_HEAD_DIM ==
               token - QBH_BLOCK_M;
    }

    if (kind == 0U) {
        const uint32_t n_tile = token / QBH_HMX_OUTPUT_CHANNELS;
        const uint32_t output = token % QBH_HMX_OUTPUT_CHANNELS;
        if (head_relative < weight_bytes) {
            const uint32_t tile =
                head_relative / QBH_HMX_WEIGHT_BYTES;
            const uint32_t within =
                head_relative % QBH_HMX_WEIGHT_BYTES;
            const uint32_t tile_n =
                tile / QBH_ATTENTION_HEAD_DIM_TILES;
            const uint32_t word_byte = within % 128U;
            return tile_n == n_tile &&
                   word_byte / sizeof(uint32_t) == output;
        }
        {
            const uint32_t bias_relative =
                head_relative - weight_bytes;
            const uint32_t tile_n =
                bias_relative / QBH_HMX_BIAS_BYTES;
            const uint32_t word =
                (bias_relative % QBH_HMX_BIAS_BYTES) /
                sizeof(uint32_t);
            return tile_n == n_tile &&
                   (word == output ||
                    word == QBH_HMX_OUTPUT_CHANNELS + output);
        }
    }
    if (head_relative < weight_bytes) {
        const uint32_t capacity_k_tiles =
            layer->padded_capacity / QBH_HMX_INPUT_CHANNELS;
        const uint32_t tile =
            head_relative / QBH_HMX_WEIGHT_BYTES;
        const uint32_t within =
            head_relative % QBH_HMX_WEIGHT_BYTES;
        const uint32_t tile_k = tile % capacity_k_tiles;
        const uint32_t input_group = within / 128U;
        const uint32_t word_byte = within % 128U;
        return tile_k == token / QBH_HMX_INPUT_CHANNELS &&
               input_group ==
                   (token % QBH_HMX_INPUT_CHANNELS) / 4U &&
               word_byte % sizeof(uint32_t) == token % 4U;
    }
    return 0;
}

static int32_t qbh_host_round_div_signed(
    int32_t numerator, int32_t denominator) {
    if (numerator >= 0) {
        return (numerator + denominator / 2) / denominator;
    }
    return -((-numerator + denominator / 2) / denominator);
}

/* Convert the canonical segmented-v4 reference tail to the EXP-0180/0181
 * physical state.  The immutable segment and bias regions are already
 * identical.  The mutable tail starts as zero after M64 prefill, retains the
 * most recently sealed decode segment thereafter, publishes complete groups
 * of four rows as final HMX carrier bytes, and keeps only the active partial
 * group in tile-major raw-U8 form. */
static int qbh_prepare_quartet_v_reference(
    uint8_t *reference,
    const struct qbh_decode_layer_state *layer,
    const struct qbh_attention_config *configs,
    uint32_t valid_length) {
    const uint32_t max_segments =
        QBH_KV_CACHE_HMX_U8_SEGMENT_COUNT(layer->capacity);
    const uint32_t sealed_segments =
        valid_length / QBH_KV_CACHE_HMX_U8_SEGMENT_TOKENS;
    const uint32_t tail_rows =
        valid_length % QBH_KV_CACHE_HMX_U8_SEGMENT_TOKENS;
    const uint32_t decoded_rows = valid_length > QBH_BLOCK_M
        ? valid_length - QBH_BLOCK_M : 0U;
    const uint32_t completed_decode_segments =
        decoded_rows / QBH_KV_CACHE_HMX_U8_SEGMENT_TOKENS;
    const uint32_t tail_offset =
        max_segments * QBH_KV_CACHE_HMX_U8_SEGMENT_V_BYTES +
        QBH_KV_CACHE_HMX_V_BIAS_BYTES_PER_HEAD;
    uint8_t logical_tail[QBH_KV_CACHE_HMX_U8_SEGMENT_TAIL_BYTES];
    uint8_t physical_tail[QBH_KV_CACHE_HMX_U8_SEGMENT_TAIL_BYTES];

    if (reference == NULL || layer == NULL || configs == NULL ||
        (layer->v_format !=
             QBH_KV_CACHE_FORMAT_HMX_U8_V_QUARTET_TAIL_V5 &&
         layer->v_format !=
             QBH_KV_CACHE_FORMAT_HMX_U8_V_ATTENTION_PUBLISH_V6) ||
        valid_length > layer->capacity ||
        sealed_segments > max_segments ||
        layer->v_head_stride_bytes <
            tail_offset + QBH_KV_CACHE_HMX_U8_SEGMENT_TAIL_BYTES) {
        return -1;
    }
    for (uint32_t head = 0U; head < layer->head_count; ++head) {
        uint8_t *head_reference = reference +
            (size_t)head * layer->v_head_stride_bytes;
        uint8_t *tail = head_reference + tail_offset;
        const struct qbh_attention_config *config = &configs[head];

        if (config->v_recenter_denominator == 0U) {
            return -1;
        }
        memcpy(logical_tail, tail, sizeof(logical_tail));
        memset(physical_tail, 0, sizeof(physical_tail));
        if (completed_decode_segments != 0U) {
            const uint32_t segment = sealed_segments - 1U;
            const uint32_t block_first =
                (segment /
                 QBH_KV_CACHE_HMX_U8_V_SEGMENT_BLOCK_SEGMENTS) *
                QBH_KV_CACHE_HMX_U8_V_SEGMENT_BLOCK_SEGMENTS;
            const uint32_t block_count =
                max_segments - block_first <
                        QBH_KV_CACHE_HMX_U8_V_SEGMENT_BLOCK_SEGMENTS
                    ? max_segments - block_first
                    : QBH_KV_CACHE_HMX_U8_V_SEGMENT_BLOCK_SEGMENTS;
            for (uint32_t output_tile = 0U;
                 output_tile < QBH_ATTENTION_HEAD_DIM_TILES;
                 ++output_tile) {
                const uint32_t segment_offset =
                    block_first *
                        QBH_KV_CACHE_HMX_U8_SEGMENT_WEIGHT_BYTES +
                    (output_tile * block_count +
                     (segment - block_first)) * QBH_HMX_WEIGHT_BYTES;
                memcpy(physical_tail +
                           output_tile * QBH_HMX_WEIGHT_BYTES,
                       head_reference + segment_offset,
                       QBH_HMX_WEIGHT_BYTES);
            }
        }
        for (uint32_t group = 0U; group < tail_rows / 4U; ++group) {
            for (uint32_t output_tile = 0U;
                 output_tile < QBH_ATTENTION_HEAD_DIM_TILES;
                 ++output_tile) {
                uint8_t *destination = physical_tail +
                    output_tile * QBH_HMX_WEIGHT_BYTES +
                    group * 128U;
                for (uint32_t output = 0U;
                     output < QBH_HMX_OUTPUT_CHANNELS; ++output) {
                    for (uint32_t lane = 0U; lane < 4U; ++lane) {
                        const uint8_t code = logical_tail[
                            (group * 4U + lane) * QBH_BLOCK_HEAD_DIM +
                            output_tile * QBH_HMX_OUTPUT_CHANNELS +
                            output];
                        int32_t value = qbh_host_round_div_signed(
                            ((int32_t)code - config->v_zero_point) *
                                (int32_t)config->v_recenter_numerator,
                            (int32_t)config->v_recenter_denominator);
                        if (value < INT8_MIN) {
                            value = INT8_MIN;
                        } else if (value > INT8_MAX) {
                            value = INT8_MAX;
                        }
                        destination[output * 4U + lane] =
                            (uint8_t)(int8_t)value;
                    }
                }
            }
        }
        if (tail_rows % 4U != 0U) {
            const uint32_t group = tail_rows / 4U;
            const uint32_t partial_rows = tail_rows % 4U;
            for (uint32_t output_tile = 0U;
                 output_tile < QBH_ATTENTION_HEAD_DIM_TILES;
                 ++output_tile) {
                for (uint32_t lane = 0U; lane < partial_rows; ++lane) {
                    memcpy(
                        physical_tail +
                            output_tile * QBH_HMX_WEIGHT_BYTES +
                            group * 128U +
                            lane * QBH_HMX_OUTPUT_CHANNELS,
                        logical_tail +
                            (group * 4U + lane) * QBH_BLOCK_HEAD_DIM +
                            output_tile * QBH_HMX_OUTPUT_CHANNELS,
                        QBH_HMX_OUTPUT_CHANNELS);
                }
            }
        }
        memcpy(tail, physical_tail, sizeof(physical_tail));
    }
    return 0;
}

static uint64_t qbh_hmx_cache_prefix_stability_mismatches(
    const uint8_t *actual, const uint8_t *snapshot,
    const struct qbh_decode_layer_state *layer,
    uint32_t kind, uint32_t appended_token) {
    const uint32_t head_stride = kind == 0U
        ? layer->k_head_stride_bytes : layer->v_head_stride_bytes;
    uint64_t mismatches = 0U;

    for (uint32_t head = 0U; head < layer->head_count; ++head) {
        for (uint32_t offset = 0U; offset < head_stride; ++offset) {
            if (!qbh_hmx_cache_byte_is_appended_token(
                    layer, kind, offset, appended_token)) {
                const size_t index =
                    (size_t)head * head_stride + offset;
                mismatches += actual[index] != snapshot[index];
            }
        }
    }
    return mismatches;
}

static struct qbh_error_metrics qbh_compare_cache_prefix_f16(
    const uint16_t *actual, const uint16_t *reference,
    uint32_t capacity, uint32_t valid_length) {
    struct qbh_error_metrics metrics;
    const size_t head_stride =
        (size_t)capacity * QBH_BLOCK_HEAD_DIM;
    double absolute_sum = 0.0;
    double squared_sum = 0.0;
    double dot = 0.0;
    double actual_norm = 0.0;
    double reference_norm = 0.0;

    memset(&metrics, 0, sizeof(metrics));
    for (uint32_t head = 0U; head < QBH_BLOCK_KV_HEADS; ++head) {
        const size_t head_base = (size_t)head * head_stride;
        for (uint32_t token = 0U; token < valid_length; ++token) {
            const size_t token_base =
                head_base + (size_t)token * QBH_BLOCK_HEAD_DIM;
            for (uint32_t channel = 0U;
                 channel < QBH_BLOCK_HEAD_DIM; ++channel) {
                const size_t index = token_base + channel;
                const uint16_t a_bits = actual[index];
                const uint16_t b_bits = reference[index];
                const double a = qbh_half_bits_to_float(a_bits);
                const double b = qbh_half_bits_to_float(b_bits);
                double difference;

                ++metrics.elements;
                metrics.mismatches += a_bits != b_bits;
                if (!isfinite(a) || !isfinite(b)) {
                    ++metrics.nonfinite_count;
                    ++metrics.mixed_tolerance_violations;
                    metrics.max_abs = DBL_MAX;
                    metrics.max_required_rtol_after_atol = DBL_MAX;
                    continue;
                }
                difference = fabs(a - b);
                if (difference > metrics.max_abs) {
                    metrics.max_abs = difference;
                }
                if (difference > QBH_REPLAY_FP16_ATOL +
                                     QBH_REPLAY_FP16_RTOL * fabs(b)) {
                    ++metrics.mixed_tolerance_violations;
                }
                if (difference > QBH_REPLAY_FP16_ATOL) {
                    const double required_rtol = b != 0.0
                        ? (difference - QBH_REPLAY_FP16_ATOL) / fabs(b)
                        : DBL_MAX;
                    if (required_rtol >
                            metrics.max_required_rtol_after_atol) {
                        metrics.max_required_rtol_after_atol = required_rtol;
                    }
                }
                absolute_sum += difference;
                squared_sum += difference * difference;
                dot += a * b;
                actual_norm += a * a;
                reference_norm += b * b;
            }
        }
    }
    if (metrics.elements == 0U || metrics.nonfinite_count != 0U) {
        metrics.mean_abs = metrics.elements == 0U ? 0.0 : DBL_MAX;
        metrics.rmse = metrics.elements == 0U ? 0.0 : DBL_MAX;
        metrics.nrmse = metrics.elements == 0U ? 0.0 : DBL_MAX;
        metrics.cosine = metrics.elements == 0U ? 1.0 : 0.0;
        return metrics;
    }
    metrics.mean_abs = absolute_sum / (double)metrics.elements;
    metrics.rmse = sqrt(squared_sum / (double)metrics.elements);
    metrics.nrmse = reference_norm > 0.0
        ? sqrt(squared_sum / reference_norm)
        : (squared_sum == 0.0 ? 0.0 : DBL_MAX);
    metrics.cosine = actual_norm > 0.0 && reference_norm > 0.0
        ? dot / sqrt(actual_norm * reference_norm)
        : (actual_norm == 0.0 && reference_norm == 0.0 ? 1.0 : 0.0);
    return metrics;
}

static uint16_t qbh_hmx_f16_cache_value(
    const uint16_t *cache, const struct qbh_decode_layer_state *layer,
    uint32_t kind, uint32_t head, uint32_t token, uint32_t channel) {
    const uint32_t head_stride_elements =
        (kind == 0U ? layer->k_head_stride_bytes
                    : layer->v_head_stride_bytes) /
        sizeof(uint16_t);
    if (token >= QBH_BLOCK_M) {
        const uint32_t weight_elements =
            (kind == 0U ? layer->k_weight_bytes_per_head
                        : layer->v_weight_bytes_per_head) /
            sizeof(uint16_t);
        return cache[(size_t)head * head_stride_elements +
                     weight_elements +
                     (size_t)(token - QBH_BLOCK_M) *
                         QBH_BLOCK_HEAD_DIM +
                     channel];
    }
    const uint32_t k_tiles = kind == 0U
        ? QBH_ATTENTION_HEAD_DIM_TILES
        : QBH_BLOCK_M / QBH_HMX_FP16_ROWS;
    const uint32_t input = kind == 0U ? channel : token;
    const uint32_t output = kind == 0U ? token : channel;
    const uint32_t k_tile = input / QBH_HMX_FP16_ROWS;
    const uint32_t n_tile = output / QBH_HMX_FP16_COLS;
    const size_t tile = ((size_t)n_tile * k_tiles + k_tile) *
        QBH_HMX_FP16_TILE_ELEMENTS;
    return cache[(size_t)head * head_stride_elements + tile +
                 qbh_hmx_fp16_tile_offset(
                     input % QBH_HMX_FP16_ROWS,
                     output % QBH_HMX_FP16_COLS)];
}

static struct qbh_error_metrics qbh_compare_hmx_f16_cache_prefix(
    const uint16_t *actual, const uint16_t *reference,
    const struct qbh_decode_layer_state *layer, uint32_t kind,
    uint32_t valid_length) {
    struct qbh_error_metrics metrics;
    double absolute_sum = 0.0;
    double squared_sum = 0.0;
    double dot = 0.0;
    double actual_norm = 0.0;
    double reference_norm = 0.0;

    memset(&metrics, 0, sizeof(metrics));
    for (uint32_t head = 0U; head < layer->head_count; ++head) {
        for (uint32_t token = 0U; token < valid_length; ++token) {
            for (uint32_t channel = 0U;
                 channel < layer->head_dim; ++channel) {
                const uint16_t a_bits = qbh_hmx_f16_cache_value(
                    actual, layer, kind, head, token, channel);
                const uint16_t b_bits = qbh_hmx_f16_cache_value(
                    reference, layer, kind, head, token, channel);
                const double a = qbh_half_bits_to_float(a_bits);
                const double b = qbh_half_bits_to_float(b_bits);
                double difference;
                ++metrics.elements;
                metrics.mismatches += a_bits != b_bits;
                if (!isfinite(a) || !isfinite(b)) {
                    ++metrics.nonfinite_count;
                    ++metrics.mixed_tolerance_violations;
                    metrics.max_abs = DBL_MAX;
                    metrics.max_required_rtol_after_atol = DBL_MAX;
                    continue;
                }
                difference = fabs(a - b);
                if (difference > metrics.max_abs) metrics.max_abs = difference;
                if (difference > QBH_REPLAY_FP16_ATOL +
                                     QBH_REPLAY_FP16_RTOL * fabs(b)) {
                    ++metrics.mixed_tolerance_violations;
                }
                if (difference > QBH_REPLAY_FP16_ATOL) {
                    const double required_rtol = b != 0.0
                        ? (difference - QBH_REPLAY_FP16_ATOL) / fabs(b)
                        : DBL_MAX;
                    if (required_rtol > metrics.max_required_rtol_after_atol)
                        metrics.max_required_rtol_after_atol = required_rtol;
                }
                absolute_sum += difference;
                squared_sum += difference * difference;
                dot += a * b;
                actual_norm += a * a;
                reference_norm += b * b;
            }
        }
    }
    if (metrics.elements == 0U || metrics.nonfinite_count != 0U) {
        metrics.mean_abs = metrics.elements == 0U ? 0.0 : DBL_MAX;
        metrics.rmse = metrics.elements == 0U ? 0.0 : DBL_MAX;
        metrics.nrmse = metrics.elements == 0U ? 0.0 : DBL_MAX;
        metrics.cosine = metrics.elements == 0U ? 1.0 : 0.0;
        return metrics;
    }
    metrics.mean_abs = absolute_sum / (double)metrics.elements;
    metrics.rmse = sqrt(squared_sum / (double)metrics.elements);
    metrics.nrmse = reference_norm > 0.0
        ? sqrt(squared_sum / reference_norm)
        : (squared_sum == 0.0 ? 0.0 : DBL_MAX);
    metrics.cosine = actual_norm > 0.0 && reference_norm > 0.0
        ? dot / sqrt(actual_norm * reference_norm)
        : (actual_norm == 0.0 && reference_norm == 0.0 ? 1.0 : 0.0);
    return metrics;
}

static int qbh_replay_step_pass(
    uint32_t variant, const struct qbh_replay_step_result *result) {
    const int output_pass = variant == QBH_BLOCK_W4U8
        ? result->output.mismatches == 0U
        : result->output.nonfinite_count == 0U &&
              isfinite(result->output.cosine) &&
              result->output.cosine >= QBH_REPLAY_FP16_MIN_COSINE &&
              isfinite(result->output.nrmse) &&
              result->output.nrmse <=
                  QBH_REPLAY_FP16_MAX_COMPOSED_NRMSE;
    const int cache_pass = variant == QBH_BLOCK_W4U8
        ? result->cache_mismatches == 0U
        : result->cache_tensor_count ==
              QBH_VERTICAL_SLICE_LAYER_COUNT * 2U &&
              result->cache_nonfinite_count == 0U;
    return output_pass && cache_pass &&
           result->cache_prefix_mismatches == 0U &&
           result->cache_structure_mismatches == 0U &&
           result->dsp_status == QBH_BLOCK_STATUS_OK &&
           result->numerical_status == QBH_BLOCK_NUMERICAL_OK &&
           result->vtcm_requested_bytes ==
               QBH_EXPECTED_FULL_VTCM_BYTES &&
           result->vtcm_acquired_bytes ==
               QBH_EXPECTED_FULL_VTCM_BYTES &&
           result->intermediate_ddr_read_bytes == 0U &&
           result->intermediate_ddr_write_bytes == 0U &&
           result->intermediate_spill_fill_count == 0U;
}

static void qbh_print_replay_profile(
    uint32_t experiment, const char *record,
    const char *step_key, uint32_t variant, uint32_t step,
    const struct qbh_block_header *header,
    const struct qbh_replay_step_result *result,
    const uint8_t *output, uint32_t output_bytes) {
#define QBH_REPLAY_PROFILE_U32(field) \
    printf(",\"" #field "\":%" PRIu32, header->field)
#define QBH_REPLAY_PROFILE_I32(field) \
    printf(",\"" #field "\":%" PRId32, header->field)
#define QBH_REPLAY_PROFILE_U64(field) \
    printf(",\"" #field "\":%" PRIu64, header->field)

    printf(
        "{\"experiment\":%" PRIu32 ",\"record\":\"%s\","
        "\"variant\":\"%s\",\"%s\":%" PRIu32 ","
        "\"mode\":\"%s\",\"logical_m\":%" PRIu32 ","
        "\"first_position\":%" PRIu32 ","
        "\"valid_length\":%" PRIu32 ","
        "\"host_wall_ns\":%" PRIu64 ","
        "\"output_mismatches\":%" PRIu64 ","
        "\"output_max_abs\":%.9g,\"output_cosine\":%.9g,"
        "\"output_nrmse\":%.9g,"
        "\"output_mixed_tolerance_violations\":%" PRIu64 ","
        "\"output_nonfinite_count\":%" PRIu64 ","
        "\"output_max_required_rtol_after_atol\":%.9g,"
        "\"output_fp16_atol\":%.9g,\"output_fp16_rtol\":%.9g,"
        "\"output_fp16_max_composed_nrmse\":%.9g,"
        "\"output_max_lsb\":%" PRIu32 ","
        "\"output_hash\":\"%016" PRIx64 "\","
        "\"cache_prefix_mismatches\":%" PRIu64 ","
        "\"cache_mismatches\":%" PRIu64 ","
        "\"cache_structure_mismatches\":%" PRIu64 ","
        "\"cache_min_cosine\":%.9g,"
        "\"cache_max_mixed_tolerance_violation_fraction\":%.9g,"
        "\"cache_max_nrmse\":%.9g,"
        "\"cache_compared_elements\":%" PRIu64 ","
        "\"cache_mixed_tolerance_violations\":%" PRIu64 ","
        "\"cache_nonfinite_count\":%" PRIu64 ","
        "\"cache_tensor_count\":%" PRIu32 ","
        "\"cache_composed_cosine_diagnostic_failure_count\":%" PRIu32 ","
        "\"cache_legacy_mixed_bound_failure_count\":%" PRIu32 ","
        "\"cache_fp16_min_cosine\":%.9g,"
        "\"cache_fp16_max_violation_fraction\":%.9g,"
        "\"fp16_gate_version\":\"composition_v2\","
        "\"backend\":\"standalone_fastrpc_dsp\","
        "\"qnn\":\"none\",\"intermediate_residency\":\"VTCM\"",
        experiment, record, qbh_variant_name(variant), step_key, step,
        step == 0U ? "prefill" : "decode", header->logical_m,
        result->first_position, result->valid_length,
        result->host_wall_ns, result->output.mismatches,
        result->output.max_abs, result->output.cosine,
        result->output.nrmse,
        result->output.mixed_tolerance_violations,
        result->output.nonfinite_count,
        result->output.max_required_rtol_after_atol,
        QBH_REPLAY_FP16_ATOL, QBH_REPLAY_FP16_RTOL,
        QBH_REPLAY_FP16_MAX_COMPOSED_NRMSE,
        result->output.max_lsb, qbh_fnv1a64(output, output_bytes),
        result->cache_prefix_mismatches, result->cache_mismatches,
        result->cache_structure_mismatches,
        result->cache_min_cosine,
        result->cache_max_mixed_tolerance_violation_fraction,
        result->cache_max_nrmse,
        result->cache_compared_elements,
        result->cache_mixed_tolerance_violations,
        result->cache_nonfinite_count,
        result->cache_tensor_count,
        result->cache_composed_cosine_diagnostic_failure_count,
        result->cache_legacy_mixed_bound_failure_count,
        QBH_REPLAY_FP16_MIN_COSINE,
        QBH_REPLAY_FP16_MAX_CACHE_VIOLATION_FRACTION);

    QBH_REPLAY_PROFILE_U32(repeat_count);
    QBH_REPLAY_PROFILE_U32(prepared_session_run_index);
    QBH_REPLAY_PROFILE_U32(numerical_audit_enabled);
    QBH_REPLAY_PROFILE_U32(kv_cache_k_format);
    QBH_REPLAY_PROFILE_U32(kv_cache_v_format);
    QBH_REPLAY_PROFILE_U32(w4u8_prefill_cache_mode);
    QBH_REPLAY_PROFILE_U32(w4u8_delta_reconstruction_mode);
    QBH_REPLAY_PROFILE_U32(w4u8_decode_softmax_mode);
    QBH_REPLAY_PROFILE_U32(w4u8_decode_lm_head_group_tiles);
    QBH_REPLAY_PROFILE_U32(w4u8_decode_o_batch_n_tiles);
    QBH_REPLAY_PROFILE_U32(w4u8_o_batch_n_tiles_observed);
    QBH_REPLAY_PROFILE_U32(w4u8_o_batch_count);
    QBH_REPLAY_PROFILE_U32(w4u8_decode_av_requant_rows);
    QBH_REPLAY_PROFILE_U32(w4u8_decode_av_padding_poison);
    QBH_REPLAY_PROFILE_U32(w4u8_av_requant_rows_observed);
    QBH_REPLAY_PROFILE_U32(w4u8_av_requant_call_count);
    QBH_REPLAY_PROFILE_U32(w4u8_av_requant_vector_count);
    QBH_REPLAY_PROFILE_U32(w4u8_av_padding_poison_count);
    QBH_REPLAY_PROFILE_U32(w4u8_decode_common_op_rows);
    QBH_REPLAY_PROFILE_U32(w4u8_decode_common_padding_poison);
    QBH_REPLAY_PROFILE_U32(w4u8_common_op_rows_observed);
    QBH_REPLAY_PROFILE_U32(w4u8_input_norm_direct_row4_call_count);
    QBH_REPLAY_PROFILE_U32(w4u8_post_residual_direct_row4_call_count);
    QBH_REPLAY_PROFILE_U32(w4u8_final_residual_direct_row4_call_count);
    QBH_REPLAY_PROFILE_U32(w4u8_common_padding_poison_count);
    QBH_REPLAY_PROFILE_U32(w4u8_decode_qk_norm_rope_rows);
    QBH_REPLAY_PROFILE_U32(w4u8_decode_qk_padding_poison);
    QBH_REPLAY_PROFILE_U32(w4u8_qk_norm_rope_rows_observed);
    QBH_REPLAY_PROFILE_U32(w4u8_decode_q_pair_row4_call_count);
    QBH_REPLAY_PROFILE_U32(w4u8_decode_k_pair_row4_call_count);
    QBH_REPLAY_PROFILE_U32(w4u8_decode_qk_rows_processed);
    QBH_REPLAY_PROFILE_U32(w4u8_decode_k_temp_carrier_skipped_count);
    QBH_REPLAY_PROFILE_U32(w4u8_qk_padding_poison_pair_count);
    QBH_REPLAY_PROFILE_U64(w4u8_decode_q_valid_row_hash);
    QBH_REPLAY_PROFILE_U64(w4u8_decode_k_valid_row_hash);
    QBH_REPLAY_PROFILE_U32(w4u8_input_norm_task_count);
    QBH_REPLAY_PROFILE_U64(w4u8_input_norm_main_work_ticks);
    QBH_REPLAY_PROFILE_U64(w4u8_input_norm_worker_work_ticks);
    QBH_REPLAY_PROFILE_U64(w4u8_input_norm_pool_wait_ticks);
    QBH_REPLAY_PROFILE_U32(w4u8_residual_active_contexts);
    QBH_REPLAY_PROFILE_U32(w4u8_post_residual_task_count);
    QBH_REPLAY_PROFILE_U32(w4u8_final_residual_task_count);
    QBH_REPLAY_PROFILE_U64(w4u8_post_residual_main_work_ticks);
    QBH_REPLAY_PROFILE_U64(w4u8_post_residual_worker_work_ticks);
    QBH_REPLAY_PROFILE_U64(w4u8_post_residual_pool_wait_ticks);
    QBH_REPLAY_PROFILE_U64(w4u8_final_residual_main_work_ticks);
    QBH_REPLAY_PROFILE_U64(w4u8_final_residual_worker_work_ticks);
    QBH_REPLAY_PROFILE_U64(w4u8_final_residual_pool_wait_ticks);
    QBH_REPLAY_PROFILE_U32(w4u8_decode_softmax_hvx_tile4_call_count);
    QBH_REPLAY_PROFILE_U32(w4u8_decode_softmax_hvx_tile4_mismatch_count);
    QBH_REPLAY_PROFILE_I32(dsp_status);
    QBH_REPLAY_PROFILE_I32(numerical_status);
    QBH_REPLAY_PROFILE_U32(scan_logical_m_observed);
    QBH_REPLAY_PROFILE_U32(scan_total_kv_length);
    QBH_REPLAY_PROFILE_U32(scan_padded_kv_length);
    QBH_REPLAY_PROFILE_U32(scan_attention_overlay_capacity_bytes);
    QBH_REPLAY_PROFILE_U32(scan_attention_overlay_required_bytes);
    QBH_REPLAY_PROFILE_U32(scan_cache_dma_descriptor_count);
    QBH_REPLAY_PROFILE_U32(scan_cache_append_mismatch_count);
    QBH_REPLAY_PROFILE_U64(scan_cache_ddr_read_bytes);
    QBH_REPLAY_PROFILE_U64(scan_cache_ddr_write_bytes);
    QBH_REPLAY_PROFILE_U64(scan_cache_stage_ticks);
    QBH_REPLAY_PROFILE_U64(scan_cache_append_ticks);
    QBH_REPLAY_PROFILE_U64(scan_cache_pack_ticks);
    QBH_REPLAY_PROFILE_U64(block_orchestration_ticks);
    QBH_REPLAY_PROFILE_U64(layer_bookkeeping_ticks);
    QBH_REPLAY_PROFILE_U64(scan_dynamic_attention_ticks);

    QBH_REPLAY_PROFILE_U64(total_ticks);
    QBH_REPLAY_PROFILE_U64(invocation_ticks);
    QBH_REPLAY_PROFILE_U64(runtime_setup_ticks);
    QBH_REPLAY_PROFILE_U64(runtime_teardown_ticks);
    QBH_REPLAY_PROFILE_U64(stage_boundary_ticks);
    QBH_REPLAY_PROFILE_U64(ledger_named_ticks);
    QBH_REPLAY_PROFILE_U64(ledger_unattributed_ticks);
    QBH_REPLAY_PROFILE_U64(input_stage_ticks);
    QBH_REPLAY_PROFILE_U64(metadata_stage_ticks);
    QBH_REPLAY_PROFILE_U64(input_norm_ticks);
    QBH_REPLAY_PROFILE_U64(qkv_projection_ticks);
    QBH_REPLAY_PROFILE_U64(qk_norm_rope_ticks);
    QBH_REPLAY_PROFILE_U64(attention_ticks);
    QBH_REPLAY_PROFILE_U64(o_projection_ticks);
    QBH_REPLAY_PROFILE_U64(post_attention_residual_ticks);
    QBH_REPLAY_PROFILE_U64(post_attention_norm_ticks);
    QBH_REPLAY_PROFILE_U64(gate_up_ticks);
    QBH_REPLAY_PROFILE_U64(activation_ticks);
    QBH_REPLAY_PROFILE_U64(down_ticks);
    QBH_REPLAY_PROFILE_U64(final_residual_ticks);
    QBH_REPLAY_PROFILE_U64(output_stage_ticks);
    QBH_REPLAY_PROFILE_U64(generation_embedding_ticks);
    QBH_REPLAY_PROFILE_U64(generation_final_norm_ticks);
    QBH_REPLAY_PROFILE_U64(generation_lm_head_ticks);
    QBH_REPLAY_PROFILE_U64(generation_lm_head_weight_dma_ticks);
    QBH_REPLAY_PROFILE_U64(generation_lm_head_scale_dma_ticks);
    QBH_REPLAY_PROFILE_U64(generation_lm_head_expand_ticks);
    QBH_REPLAY_PROFILE_U64(generation_lm_head_hmx_ticks);
    QBH_REPLAY_PROFILE_U64(generation_lm_head_argmax_ticks);
    QBH_REPLAY_PROFILE_U64(generation_lm_head_weight_dma_wait_ticks);
    QBH_REPLAY_PROFILE_U64(generation_lm_head_scale_init_ticks);
    QBH_REPLAY_PROFILE_U64(generation_lm_head_hmx_tail_wait_ticks);
    QBH_REPLAY_PROFILE_U32(generation_lm_head_batch_n_tiles);
    QBH_REPLAY_PROFILE_U32(generation_lm_head_command_count);
    QBH_REPLAY_PROFILE_U32(generation_lm_head_n_tiles);
    QBH_REPLAY_PROFILE_U32(generation_lm_head_prefetch_count);
    QBH_REPLAY_PROFILE_U32(generation_lm_head_scale_resident_bytes);
    QBH_REPLAY_PROFILE_U64(generation_embedding_ddr_read_bytes);
    QBH_REPLAY_PROFILE_U64(generation_lm_head_ddr_read_bytes);

    QBH_REPLAY_PROFILE_U64(weight_dma_ticks);
    QBH_REPLAY_PROFILE_U64(hmx_compute_ticks);
    QBH_REPLAY_PROFILE_U64(projection_pack_ticks);
    QBH_REPLAY_PROFILE_U64(projection_hmx_wait_ticks);
    QBH_REPLAY_PROFILE_U64(projection_unpack_ticks);
    QBH_REPLAY_PROFILE_U64(hmx_ready_wait_ticks);
    QBH_REPLAY_PROFILE_U64(w4f16_expand_ticks);
    QBH_REPLAY_PROFILE_U64(w4f16_expand_work_ticks);
    QBH_REPLAY_PROFILE_U64(w4f16_expand_pool_wait_ticks);
    QBH_REPLAY_PROFILE_U64(w4f16_prefetch_wait_ticks);
    QBH_REPLAY_PROFILE_U64(w4f16_hmx_tail_wait_ticks);
    QBH_REPLAY_PROFILE_U64(w4f16_cross_prefetch_wait_ticks);
    QBH_REPLAY_PROFILE_U64(w4f16_cross_prefetch_lifetime_ticks);

    QBH_REPLAY_PROFILE_U64(attention_setup_ticks);
    QBH_REPLAY_PROFILE_U64(attention_qk_pack_ticks);
    QBH_REPLAY_PROFILE_U64(attention_qk_hmx_ticks);
    QBH_REPLAY_PROFILE_U64(attention_qk_unpack_ticks);
    QBH_REPLAY_PROFILE_U64(attention_softmax_ticks);
    QBH_REPLAY_PROFILE_U64(attention_av_pack_ticks);
    QBH_REPLAY_PROFILE_U64(attention_av_hmx_ticks);
    QBH_REPLAY_PROFILE_U64(attention_av_unpack_ticks);
    QBH_REPLAY_PROFILE_U64(attention_gqa_pipeline_ticks);
    QBH_REPLAY_PROFILE_U64(attention_unattributed_ticks);
    QBH_REPLAY_PROFILE_U64(u8_attention_qk_norm_rope_ticks);
    QBH_REPLAY_PROFILE_U64(u8_attention_k_pack_ticks);
    QBH_REPLAY_PROFILE_U64(u8_attention_v_pack_ticks);
    QBH_REPLAY_PROFILE_U64(u8_cache_native_append_update_ticks);
    QBH_REPLAY_PROFILE_U32(u8_cache_native_prefill_build_count);
    QBH_REPLAY_PROFILE_U32(u8_cache_native_prefill_reuse_count);
    QBH_REPLAY_PROFILE_U64(
        u8_cache_native_prefill_reused_carrier_bytes);
    QBH_REPLAY_PROFILE_U32(u8_cache_native_incremental_append_count);
    QBH_REPLAY_PROFILE_U32(u8_cache_full_prefix_pack_count);
    QBH_REPLAY_PROFILE_U32(u8_cache_segment_tail_append_count);
    QBH_REPLAY_PROFILE_U32(u8_cache_segment_seal_count);
    QBH_REPLAY_PROFILE_U64(u8_cache_segment_sealed_bytes);
    QBH_REPLAY_PROFILE_U32(u8_cache_v_quartet_append_count);
    QBH_REPLAY_PROFILE_U32(u8_cache_v_quartet_publish_count);
    QBH_REPLAY_PROFILE_U32(
        u8_cache_v_quartet_attention_publish_count);
    QBH_REPLAY_PROFILE_U32(u8_cache_v_quartet_partial_pack_rows);
    QBH_REPLAY_PROFILE_U32(u8_cache_v_quartet_full_tile_rmw_count);
    QBH_REPLAY_PROFILE_U64(u8_cache_v_quartet_native_load_bytes);
    QBH_REPLAY_PROFILE_U32(u8_cache_v_vtcm_tail_init_count);
    QBH_REPLAY_PROFILE_U32(u8_cache_v_vtcm_tail_row_update_count);
    QBH_REPLAY_PROFILE_U32(u8_cache_v_vtcm_tail_publish_count);
    QBH_REPLAY_PROFILE_U32(u8_cache_v_vtcm_tail_seal_count);
    QBH_REPLAY_PROFILE_U32(u8_cache_v_vtcm_tail_partial_pack_rows);
    QBH_REPLAY_PROFILE_U64(u8_cache_v_vtcm_tail_init_bytes);
    QBH_REPLAY_PROFILE_U64(u8_cache_v_vtcm_tail_native_load_bytes);
    QBH_REPLAY_PROFILE_U32(f16_cache_native_prefill_reuse_count);
    QBH_REPLAY_PROFILE_U64(
        f16_cache_native_prefill_reused_carrier_bytes);
    QBH_REPLAY_PROFILE_U32(f16_cache_native_incremental_append_count);
    QBH_REPLAY_PROFILE_U32(f16_cache_full_prefix_pack_count);
    QBH_REPLAY_PROFILE_U64(f16_cache_native_append_update_ticks);
    QBH_REPLAY_PROFILE_U64(u8_attention_qk_hmx_ticks);
    QBH_REPLAY_PROFILE_U64(u8_attention_qk_requant_ticks);
    QBH_REPLAY_PROFILE_U64(u8_attention_softmax_ticks);
    QBH_REPLAY_PROFILE_U64(u8_attention_av_hmx_ticks);
    QBH_REPLAY_PROFILE_U64(u8_attention_av_requant_ticks);
    QBH_REPLAY_PROFILE_U64(u8_attention_pipeline_wait_ticks);
    QBH_REPLAY_PROFILE_U64(w4u8_qkvo_weight_expand_ticks);
    QBH_REPLAY_PROFILE_U64(w4u8_qkvo_prefetch_wait_ticks);
    QBH_REPLAY_PROFILE_U64(w4u8_qkvo_hmx_lifetime_ticks);

    QBH_REPLAY_PROFILE_U64(w4f16_gate_up_weight_dma_ticks);
    QBH_REPLAY_PROFILE_U64(w4f16_gate_up_expand_ticks);
    QBH_REPLAY_PROFILE_U64(w4f16_gate_up_expand_work_ticks);
    QBH_REPLAY_PROFILE_U64(w4f16_gate_up_expand_pool_wait_ticks);
    QBH_REPLAY_PROFILE_U64(w4f16_gate_up_hmx_wait_ticks);
    QBH_REPLAY_PROFILE_U64(w4f16_gate_up_hmx_tail_wait_ticks);
    QBH_REPLAY_PROFILE_U64(w4f16_gate_up_stream_work_ticks);
    QBH_REPLAY_PROFILE_U64(w4f16_gate_up_stream_ready_wait_ticks);
    QBH_REPLAY_PROFILE_U64(w4f16_gate_up_stream_join_wait_ticks);
    QBH_REPLAY_PROFILE_U64(w4u8_mlp_gate_up_pipeline_ticks);
    QBH_REPLAY_PROFILE_U64(w4u8_mlp_down_pipeline_ticks);
    QBH_REPLAY_PROFILE_U64(w4u8_mlp_activation_work_ticks);
    QBH_REPLAY_PROFILE_U64(w4u8_mlp_weight_stage_ticks);
    QBH_REPLAY_PROFILE_U64(w4u8_mlp_weight_expand_ticks);
    QBH_REPLAY_PROFILE_U64(w4u8_mlp_hmx_compute_ticks);
    QBH_REPLAY_PROFILE_U64(w4u8_mlp_hmx_ready_wait_ticks);
    QBH_REPLAY_PROFILE_U64(w4u8_mlp_producer_slot_wait_ticks);
    QBH_REPLAY_PROFILE_U64(w4u8_mlp_expanded_slot_wait_ticks);

    QBH_REPLAY_PROFILE_U32(vtcm_requested_bytes);
    QBH_REPLAY_PROFILE_U32(vtcm_acquired_bytes);
    QBH_REPLAY_PROFILE_U32(vtcm_peak_plan_bytes);
    QBH_REPLAY_PROFILE_U32(block_invocation_count);
    QBH_REPLAY_PROFILE_U32(hmx_command_count);
    QBH_REPLAY_PROFILE_U32(hmx_fp16_tile_pair_count);
    QBH_REPLAY_PROFILE_U32(hmx_u8s8_tile_pair_count);
    QBH_REPLAY_PROFILE_U32(weight_dma_descriptor_count);
    QBH_REPLAY_PROFILE_U32(boundary_dma_descriptor_count);
    QBH_REPLAY_PROFILE_U32(intermediate_dma_descriptor_count);
    QBH_REPLAY_PROFILE_U32(intermediate_spill_fill_count);
    QBH_REPLAY_PROFILE_U64(weight_ddr_read_bytes);
    QBH_REPLAY_PROFILE_U64(boundary_ddr_read_bytes);
    QBH_REPLAY_PROFILE_U64(boundary_ddr_write_bytes);
    QBH_REPLAY_PROFILE_U32(intermediate_ddr_read_bytes);
    QBH_REPLAY_PROFILE_U32(intermediate_ddr_write_bytes);
    QBH_REPLAY_PROFILE_U32(u8_attention_audit_ddr_write_bytes);
    QBH_REPLAY_PROFILE_U32(u8_attention_probability_mask_violation_count);
    QBH_REPLAY_PROFILE_U32(u8_attention_fused_k_operand_mismatch_count);
    QBH_REPLAY_PROFILE_U32(w4f16_expand_mismatch_count);
    for (uint32_t slice_index = 0U;
         slice_index < QBH_VERTICAL_SLICE_LAYER_COUNT; ++slice_index) {
        const struct qbh_block_slice_layer_profile *profile =
            &header->slice_profiles[slice_index];
        printf(
            ",\"slice_layer_%" PRIu32 "\":{"
            "\"layer_index\":%" PRIu32 ","
            "\"status\":%" PRId32 ","
            "\"cache_valid_before\":%" PRIu32 ","
            "\"cache_valid_after\":%" PRIu32 ","
            "\"hidden_ddr_read_bytes\":%" PRIu32 ","
            "\"hidden_ddr_write_bytes\":%" PRIu32 ","
            "\"layer_ticks\":%" PRIu64 ","
            "\"metadata_stage_ticks\":%" PRIu64 ","
            "\"input_stage_ticks\":%" PRIu64 ","
            "\"input_norm_ticks\":%" PRIu64 ","
            "\"qkv_projection_ticks\":%" PRIu64 ","
            "\"qk_norm_rope_ticks\":%" PRIu64 ","
            "\"attention_ticks\":%" PRIu64 ","
            "\"o_projection_ticks\":%" PRIu64 ","
            "\"post_attention_residual_ticks\":%" PRIu64 ","
            "\"post_attention_norm_ticks\":%" PRIu64 ","
            "\"gate_up_ticks\":%" PRIu64 ","
            "\"activation_ticks\":%" PRIu64 ","
            "\"down_ticks\":%" PRIu64 ","
            "\"final_residual_ticks\":%" PRIu64 ","
            "\"cache_append_pack_ticks\":%" PRIu64 ","
            "\"cache_append_dma_ticks\":%" PRIu64 ","
            "\"block_orchestration_ticks\":%" PRIu64 ","
            "\"layer_bookkeeping_ticks\":%" PRIu64 ","
            "\"layer_unattributed_ticks\":%" PRIu64 ","
            "\"weight_ddr_read_bytes\":%" PRIu64 ","
            "\"cache_ddr_read_bytes\":%" PRIu64 ","
            "\"cache_ddr_write_bytes\":%" PRIu64 "}",
            slice_index, profile->layer_index, profile->status,
            profile->cache_valid_before, profile->cache_valid_after,
            profile->hidden_ddr_read_bytes,
            profile->hidden_ddr_write_bytes, profile->layer_ticks,
            profile->metadata_stage_ticks, profile->input_stage_ticks,
            profile->input_norm_ticks, profile->qkv_projection_ticks,
            profile->qk_norm_rope_ticks, profile->attention_ticks,
            profile->o_projection_ticks,
            profile->post_attention_residual_ticks,
            profile->post_attention_norm_ticks,
            profile->gate_up_ticks, profile->activation_ticks,
            profile->down_ticks, profile->final_residual_ticks,
            profile->cache_append_pack_ticks,
            profile->cache_append_dma_ticks,
            profile->block_orchestration_ticks,
            profile->layer_bookkeeping_ticks,
            profile->layer_unattributed_ticks,
            profile->weight_ddr_read_bytes,
            profile->cache_ddr_read_bytes,
            profile->cache_ddr_write_bytes);
    }
    printf("}\n");

#undef QBH_REPLAY_PROFILE_U32
#undef QBH_REPLAY_PROFILE_I32
#undef QBH_REPLAY_PROFILE_U64
}

static int qbh_run_replay_sequence(
    struct qbh_session *session, int shared_fd, uint8_t *shared,
    uint32_t total_bytes, const char *package_root,
    struct qbh_block_header *header,
    const struct qbh_file_slot *input_slot,
    const struct qbh_file_slot *reference_slot,
    const struct qbh_file_slot rope_slots[2],
    const struct qbh_vertical_layer_slots
        vertical_slots[QBH_VERTICAL_SLICE_LAYER_COUNT],
    uint32_t variant) {
    struct qbh_decode_session_state *state =
        (struct qbh_decode_session_state *)(
            shared + header->replay_session_offset);
    struct qbh_replay_step_result step_storage;
    uint32_t decode_steps =
        header->kv_cache_capacity - QBH_BLOCK_M;
    const char *decode_steps_env = getenv("QBH_REPLAY_DECODE_STEPS");
    uint32_t total_steps;
    const uint32_t element_bytes =
        variant == QBH_BLOCK_W4U8 ? 1U : 2U;
    const char *tensor_suffix =
        variant == QBH_BLOCK_W4U8 ? "u8" : "f16";
    const char *dump_root = getenv("QBH_REPLAY_DUMP_DIR");
    size_t snapshot_offsets[QBH_VERTICAL_SLICE_LAYER_COUNT][2];
    size_t snapshot_bytes = 0U;
    uint8_t *cache_snapshots;
    const uint32_t first_layer = QBH_VERTICAL_SLICE_FIRST_LAYER;
    const uint32_t middle_layer =
        first_layer + QBH_VERTICAL_SLICE_LAYER_COUNT / 2U;
    const uint32_t last_layer =
        first_layer + QBH_VERTICAL_SLICE_LAYER_COUNT - 1U;
    int all_pass = 1;

    if (decode_steps_env != NULL && decode_steps_env[0] != '\0' &&
        (qbh_parse_u32(decode_steps_env, &decode_steps) != 0 ||
         decode_steps == 0U ||
         decode_steps > header->kv_cache_capacity - QBH_BLOCK_M)) {
        return -1;
    }
    total_steps = 1U + decode_steps;

    for (uint32_t slice_index = 0U;
         slice_index < QBH_VERTICAL_SLICE_LAYER_COUNT; ++slice_index) {
        const struct qbh_decode_layer_state *layer =
            &state->layers[QBH_VERTICAL_SLICE_FIRST_LAYER + slice_index];
        for (uint32_t kind = 0U; kind < 2U; ++kind) {
            const uint32_t bytes =
                kind == 0U ? layer->k_bytes : layer->v_bytes;
            snapshot_offsets[slice_index][kind] = snapshot_bytes;
            snapshot_bytes += bytes;
        }
    }
    cache_snapshots = malloc(snapshot_bytes);
    if (cache_snapshots == NULL) {
        return -1;
    }
    for (uint32_t slice_index = 0U;
         slice_index < QBH_VERTICAL_SLICE_LAYER_COUNT; ++slice_index) {
        const struct qbh_decode_layer_state *layer =
            &state->layers[QBH_VERTICAL_SLICE_FIRST_LAYER + slice_index];
        for (uint32_t kind = 0U; kind < 2U; ++kind) {
            const uint32_t bytes =
                kind == 0U ? layer->k_bytes : layer->v_bytes;
            memcpy(
                cache_snapshots + snapshot_offsets[slice_index][kind],
                shared + vertical_slots[slice_index].caches[kind].offset,
                bytes);
        }
    }
    for (uint32_t step = 0U; step < total_steps; ++step) {
        struct qbh_replay_step_result *step_result = &step_storage;
        uint32_t initial_length =
            state->layers[QBH_VERTICAL_SLICE_FIRST_LAYER].valid_length;
        uint64_t start;
        uint64_t end;
        int rpc_result;

        memset(step_result, 0, sizeof(*step_result));
        step_result->cache_min_cosine = 1.0;

        for (uint32_t slice_index = 1U;
             slice_index < QBH_VERTICAL_SLICE_LAYER_COUNT;
             ++slice_index) {
            if (state->layers[
                    QBH_VERTICAL_SLICE_FIRST_LAYER + slice_index]
                    .valid_length != initial_length) {
                fprintf(stderr,
                        "vertical cache lengths diverged before step "
                        "%" PRIu32 "\n",
                        step);
                free(cache_snapshots);
                return -1;
            }
        }

        if (step == 0U) {
            header->scan_mode = QBH_BLOCK_SCAN_PREFILL;
            header->logical_m = QBH_BLOCK_M;
        } else {
            char name[128];
            const uint32_t decode_index = step - 1U;
            header->scan_mode = QBH_BLOCK_SCAN_DECODE;
            header->logical_m = 1U;
            if (snprintf(name, sizeof(name),
                         "replay_decode_input_%02" PRIu32 "_%s.bin",
                         decode_index, tensor_suffix) < 0 ||
                qbh_read_named_tensor(
                    package_root, name,
                    shared + input_slot->offset,
                    input_slot->expected_bytes) != 0 ||
                snprintf(name, sizeof(name),
                         "replay_decode_reference_%02" PRIu32 "_%s.bin",
                         decode_index, tensor_suffix) < 0 ||
                qbh_read_named_tensor(
                    package_root, name,
                    shared + reference_slot->offset,
                    reference_slot->expected_bytes) != 0 ||
                snprintf(name, sizeof(name),
                         "replay_decode_rope_cos_%02" PRIu32 "_f16.bin",
                         decode_index) < 0 ||
                qbh_read_named_tensor(
                    package_root, name,
                    shared + rope_slots[0].offset,
                    rope_slots[0].expected_bytes) != 0 ||
                snprintf(name, sizeof(name),
                         "replay_decode_rope_sin_%02" PRIu32 "_f16.bin",
                         decode_index) < 0 ||
                qbh_read_named_tensor(
                    package_root, name,
                    shared + rope_slots[1].offset,
                    rope_slots[1].expected_bytes) != 0) {
                fprintf(stderr, "failed to stage replay decode step %" PRIu32 "\n",
                        decode_index);
                free(cache_snapshots);
                return -1;
            }
        }

        qbh_bind_host_slice_layer(header, 0U);
        header->initial_kv_length = initial_length;
        header->replay_expected_step = state->completed_step_count;
        header->replay_first_position = state->next_position;
        header->repeat_count = 1U;
        header->dsp_status = QBH_BLOCK_STATUS_HOST_READY;
        memset(shared + header->output_offset, 0xa5, header->output_bytes);
        start = qbh_monotonic_ns();
        rpc_result = qwen3_probe_run_block(
            session->handle, shared_fd, total_bytes);
        end = qbh_monotonic_ns();
        if (rpc_result != AEE_SUCCESS) {
            fprintf(stderr, "replay step %" PRIu32 " RPC failed: 0x%08x\n",
                    step, (unsigned int)rpc_result);
            free(cache_snapshots);
            return -1;
        }

        step_result->host_wall_ns = end - start;
        step_result->step_index = step;
        step_result->first_position = header->replay_first_position;
        step_result->valid_length =
            state->layers[QBH_VERTICAL_SLICE_FIRST_LAYER]
                .valid_length;
        step_result->dsp_status = (uint32_t)header->dsp_status;
        step_result->numerical_status = (uint32_t)header->numerical_status;
        step_result->vtcm_requested_bytes = header->vtcm_requested_bytes;
        step_result->vtcm_acquired_bytes = header->vtcm_acquired_bytes;
        step_result->intermediate_ddr_read_bytes =
            header->intermediate_ddr_read_bytes;
        step_result->intermediate_ddr_write_bytes =
            header->intermediate_ddr_write_bytes;
        step_result->intermediate_spill_fill_count =
            header->intermediate_spill_fill_count;
        step_result->scan_cache_ddr_read_bytes =
            header->scan_cache_ddr_read_bytes;
        step_result->scan_cache_ddr_write_bytes =
            header->scan_cache_ddr_write_bytes;
        step_result->scan_dynamic_attention_ticks =
            header->scan_dynamic_attention_ticks;
        step_result->output = variant == QBH_BLOCK_W4U8
            ? qbh_compare_scan_u8(
                  shared + header->output_offset,
                  shared + header->reference_offset,
                  header->logical_m,
                  &header->slice_layers[
                       QBH_VERTICAL_SLICE_LAYER_COUNT - 1U]
                       .qparams[QBH_BLOCK_QP_BLOCK_OUTPUT])
            : qbh_compare_scan_f16(
                  (const uint16_t *)(shared + header->output_offset),
                  (const uint16_t *)(shared + header->reference_offset),
                  header->logical_m);
        for (uint32_t slice_index = 0U;
             slice_index < QBH_VERTICAL_SLICE_LAYER_COUNT;
             ++slice_index) {
            const struct qbh_decode_layer_state *layer =
                &state->layers[
                    QBH_VERTICAL_SLICE_FIRST_LAYER + slice_index];
            if (layer->valid_length != step_result->valid_length ||
                header->slice_profiles[slice_index].cache_valid_before !=
                    initial_length ||
                header->slice_profiles[slice_index].cache_valid_after !=
                    layer->valid_length) {
                ++step_result->cache_structure_mismatches;
                ++step_result->cache_mismatches;
            }
            for (uint32_t kind = 0U; kind < 2U; ++kind) {
                const uint32_t cache_bytes =
                    kind == 0U ? layer->k_bytes : layer->v_bytes;
                uint8_t *snapshot =
                    cache_snapshots +
                    snapshot_offsets[slice_index][kind];
                const uint8_t *actual =
                    shared + vertical_slots[slice_index]
                                 .caches[kind].offset;
                uint8_t *reference;
                const int hmx_native_u8 =
                    qbh_hmx_native_u8_cache_formats(
                        layer->k_format, layer->v_format);
                const int hmx_native_u8_delta =
                    qbh_hmx_native_u8_delta_cache_formats(
                        layer->k_format, layer->v_format);
                const int hmx_native_u8_segmented =
                    qbh_hmx_native_u8_segmented_cache_formats(
                        layer->k_format, layer->v_format);
                const int hmx_native_f16 =
                    qbh_hmx_native_f16_cache_formats(
                        layer->k_format, layer->v_format);
                const int hmx_native = hmx_native_u8 || hmx_native_f16;
                if (hmx_native) {
                    char cache_reference_name[160];
                    if (snprintf(
                            cache_reference_name,
                            sizeof(cache_reference_name),
                            "layer%" PRIu32
                            "/reference_kv_cache_%c_hmx_%s_step%02"
                            PRIu32 ".bin",
                            layer->layer_index,
                            kind == 0U ? 'k' : 'v',
                            hmx_native_u8_segmented
                                ? "u8_segmented"
                            : hmx_native_u8_delta
                                ? "u8_delta"
                                : (hmx_native_u8 ? "u8" : "f16"),
                            step) < 0 ||
                        qbh_read_named_tensor(
                            package_root, cache_reference_name,
                            shared + vertical_slots[slice_index]
                                         .cache_references[kind].offset,
                            cache_bytes) != 0) {
                        free(cache_snapshots);
                        return -1;
                    }
                }
                reference =
                    shared + vertical_slots[slice_index]
                                 .cache_references[kind].offset;
                if (kind == 1U &&
                    (layer->v_format ==
                         QBH_KV_CACHE_FORMAT_HMX_U8_V_QUARTET_TAIL_V5 ||
                     layer->v_format ==
                         QBH_KV_CACHE_FORMAT_HMX_U8_V_ATTENTION_PUBLISH_V6) &&
                    qbh_prepare_quartet_v_reference(
                        reference, layer,
                        (const struct qbh_attention_config *)(
                            shared + vertical_slots[slice_index]
                                         .attention_config.offset),
                        layer->valid_length) != 0) {
                    free(cache_snapshots);
                    return -1;
                }
                if (hmx_native) {
                    if (step != 0U) {
                        step_result->cache_prefix_mismatches +=
                            qbh_hmx_cache_prefix_stability_mismatches(
                                actual, snapshot, layer, kind,
                                initial_length);
                    }
                    step_result->cache_mismatches +=
                        qbh_count_byte_mismatches(
                            actual, reference, cache_bytes);
                } else {
                    step_result->cache_prefix_mismatches +=
                        qbh_cache_prefix_mismatches(
                            actual, snapshot, layer->capacity,
                            initial_length, element_bytes);
                    step_result->cache_mismatches +=
                        qbh_cache_prefix_mismatches(
                            actual, reference, layer->capacity,
                            layer->valid_length, element_bytes);
                }
                if (variant != QBH_BLOCK_W4U8) {
                    const struct qbh_error_metrics cache_metrics =
                        hmx_native_f16
                            ? qbh_compare_hmx_f16_cache_prefix(
                                  (const uint16_t *)actual,
                                  (const uint16_t *)reference,
                                  layer, kind, layer->valid_length)
                            : qbh_compare_cache_prefix_f16(
                                  (const uint16_t *)actual,
                                  (const uint16_t *)reference,
                                  layer->capacity,
                                  layer->valid_length);
                    const double violation_fraction =
                        cache_metrics.elements != 0U
                            ? (double)cache_metrics
                                  .mixed_tolerance_violations /
                                  (double)cache_metrics.elements
                            : 0.0;
                    const int composed_cosine_diagnostic_pass =
                        cache_metrics.nonfinite_count == 0U &&
                        isfinite(cache_metrics.cosine) &&
                        cache_metrics.cosine >=
                            QBH_REPLAY_FP16_MIN_COSINE;
                    const int legacy_local_bound_pass =
                        violation_fraction <=
                            QBH_REPLAY_FP16_MAX_CACHE_VIOLATION_FRACTION;
                    if (cache_metrics.cosine <
                            step_result->cache_min_cosine) {
                        step_result->cache_min_cosine =
                            cache_metrics.cosine;
                    }
                    if (violation_fraction >
                            step_result
                                ->cache_max_mixed_tolerance_violation_fraction) {
                        step_result
                            ->cache_max_mixed_tolerance_violation_fraction =
                            violation_fraction;
                    }
                    if (cache_metrics.nrmse >
                            step_result->cache_max_nrmse) {
                        step_result->cache_max_nrmse = cache_metrics.nrmse;
                    }
                    step_result->cache_compared_elements +=
                        cache_metrics.elements;
                    step_result->cache_mixed_tolerance_violations +=
                        cache_metrics.mixed_tolerance_violations;
                    step_result->cache_nonfinite_count +=
                        cache_metrics.nonfinite_count;
                    ++step_result->cache_tensor_count;
                    step_result
                        ->cache_composed_cosine_diagnostic_failure_count +=
                        !composed_cosine_diagnostic_pass;
                    step_result->cache_legacy_mixed_bound_failure_count +=
                        !legacy_local_bound_pass;
                }
                memcpy(snapshot, actual, cache_bytes);
            }
        }
        if (dump_root != NULL && dump_root[0] != '\0') {
            char name[128];
            if (snprintf(
                    name, sizeof(name),
                    "actual_replay_output_%02" PRIu32 "_%s.bin",
                    step, tensor_suffix) < 0 ||
                qbh_write_named_tensor(
                    dump_root, name, shared + header->output_offset,
                    header->output_bytes) != 0) {
                free(cache_snapshots);
                return -1;
            }
            if (header->u8_attention_audit_output_bytes != 0U &&
                (snprintf(
                     name, sizeof(name),
                     "actual_replay_attention_audit_%02" PRIu32 ".bin",
                     step) < 0 ||
                 qbh_write_named_tensor(
                     dump_root, name,
                     shared + header->u8_attention_audit_output_offset,
                     header->u8_attention_audit_output_bytes) != 0)) {
                free(cache_snapshots);
                return -1;
            }
        }
        all_pass &= qbh_replay_step_pass(variant, step_result);
        printf(
            "{\"experiment\":163,\"variant\":\"%s\","
            "\"replay_step\":%" PRIu32 ",\"mode\":\"%s\","
            "\"first_position\":%" PRIu32 ","
            "\"valid_length\":%" PRIu32 ","
            "\"host_wall_ns\":%" PRIu64 ","
            "\"output_mismatches\":%" PRIu64 ","
            "\"output_max_abs\":%.9g,\"output_cosine\":%.9g,"
            "\"output_nrmse\":%.9g,"
            "\"output_mixed_tolerance_violations\":%" PRIu64 ","
            "\"output_nonfinite_count\":%" PRIu64 ","
            "\"output_max_required_rtol_after_atol\":%.9g,"
            "\"output_fp16_atol\":%.9g,\"output_fp16_rtol\":%.9g,"
            "\"output_fp16_max_composed_nrmse\":%.9g,"
            "\"cache_prefix_mismatches\":%" PRIu64 ","
            "\"cache_mismatches\":%" PRIu64 ","
            "\"cache_structure_mismatches\":%" PRIu64 ","
            "\"cache_min_cosine\":%.9g,"
            "\"cache_max_mixed_tolerance_violation_fraction\":%.9g,"
            "\"cache_max_nrmse\":%.9g,"
            "\"cache_compared_elements\":%" PRIu64 ","
            "\"cache_mixed_tolerance_violations\":%" PRIu64 ","
            "\"cache_nonfinite_count\":%" PRIu64 ","
            "\"cache_tensor_count\":%" PRIu32 ","
            "\"cache_composed_cosine_diagnostic_failure_count\":%" PRIu32 ","
            "\"cache_legacy_mixed_bound_failure_count\":%" PRIu32 ","
            "\"cache_fp16_min_cosine\":%.9g,"
            "\"cache_fp16_max_violation_fraction\":%.9g,"
            "\"fp16_gate_version\":\"composition_v2\","
            "\"cache_ddr_read_bytes\":%" PRIu64 ","
            "\"cache_ddr_write_bytes\":%" PRIu64 ","
            "\"dynamic_attention_ticks\":%" PRIu64 ","
            "\"vtcm_requested_bytes\":%" PRIu32 ","
            "\"vtcm_acquired_bytes\":%" PRIu32 ","
            "\"intermediate_ddr_read_bytes\":%" PRIu64 ","
            "\"intermediate_ddr_write_bytes\":%" PRIu64 ","
            "\"intermediate_spill_fill_count\":%" PRIu32 ","
            "\"first_layer\":%" PRIu32 ","
            "\"first_layer_valid_length\":%" PRIu32 ","
            "\"middle_layer\":%" PRIu32 ","
            "\"middle_layer_valid_length\":%" PRIu32 ","
            "\"last_layer\":%" PRIu32 ","
            "\"last_layer_valid_length\":%" PRIu32 ","
            "\"pass\":%s}\n",
            qbh_variant_name(variant), step,
            step == 0U ? "prefill" : "decode",
            step_result->first_position, step_result->valid_length,
            step_result->host_wall_ns, step_result->output.mismatches,
            step_result->output.max_abs, step_result->output.cosine,
            step_result->output.nrmse,
            step_result->output.mixed_tolerance_violations,
            step_result->output.nonfinite_count,
            step_result->output.max_required_rtol_after_atol,
            QBH_REPLAY_FP16_ATOL, QBH_REPLAY_FP16_RTOL,
            QBH_REPLAY_FP16_MAX_COMPOSED_NRMSE,
            step_result->cache_prefix_mismatches,
            step_result->cache_mismatches,
            step_result->cache_structure_mismatches,
            step_result->cache_min_cosine,
            step_result->cache_max_mixed_tolerance_violation_fraction,
            step_result->cache_max_nrmse,
            step_result->cache_compared_elements,
            step_result->cache_mixed_tolerance_violations,
            step_result->cache_nonfinite_count,
            step_result->cache_tensor_count,
            step_result->cache_composed_cosine_diagnostic_failure_count,
            step_result->cache_legacy_mixed_bound_failure_count,
            QBH_REPLAY_FP16_MIN_COSINE,
            QBH_REPLAY_FP16_MAX_CACHE_VIOLATION_FRACTION,
            step_result->scan_cache_ddr_read_bytes,
            step_result->scan_cache_ddr_write_bytes,
            step_result->scan_dynamic_attention_ticks,
            step_result->vtcm_requested_bytes,
            step_result->vtcm_acquired_bytes,
            step_result->intermediate_ddr_read_bytes,
            step_result->intermediate_ddr_write_bytes,
            step_result->intermediate_spill_fill_count,
            first_layer, state->layers[first_layer].valid_length,
            middle_layer, state->layers[middle_layer].valid_length,
            last_layer, state->layers[last_layer].valid_length,
            qbh_replay_step_pass(variant, step_result) ? "true" : "false");
        qbh_print_replay_profile(
            163U, "replay_profile", "replay_step",
            variant, step, header, step_result,
            shared + header->output_offset, header->output_bytes);
    }
    if (dump_root != NULL && dump_root[0] != '\0') {
        for (uint32_t slice_index = 0U;
             slice_index < QBH_VERTICAL_SLICE_LAYER_COUNT;
             ++slice_index) {
            for (uint32_t kind = 0U; kind < 2U; ++kind) {
                char name[128];
                if (snprintf(
                        name, sizeof(name),
                        "actual_layer%" PRIu32 "_replay_%c_cache.bin",
                        QBH_VERTICAL_SLICE_FIRST_LAYER + slice_index,
                        kind == 0U ? 'k' : 'v') < 0 ||
                    qbh_write_named_tensor(
                        dump_root, name,
                        shared + vertical_slots[slice_index]
                                     .caches[kind].offset,
                        kind == 0U
                            ? state->layers[
                                  QBH_VERTICAL_SLICE_FIRST_LAYER +
                                  slice_index].k_bytes
                            : state->layers[
                                  QBH_VERTICAL_SLICE_FIRST_LAYER +
                                  slice_index].v_bytes) != 0) {
                    free(cache_snapshots);
                    return -1;
                }
            }
        }
    }
    for (uint32_t slice_index = 0U;
         slice_index < QBH_VERTICAL_SLICE_LAYER_COUNT; ++slice_index) {
        if (state->layers[first_layer + slice_index].valid_length !=
                QBH_BLOCK_M + decode_steps) {
            all_pass = 0;
        }
    }
    printf(
        "{\"experiment\":163,\"variant\":\"%s\","
        "\"replay_sequence_complete\":true,"
        "\"completed_steps\":%" PRIu32 ","
        "\"first_layer\":%" PRIu32 ","
        "\"first_layer_final_valid_length\":%" PRIu32 ","
        "\"middle_layer\":%" PRIu32 ","
        "\"middle_layer_final_valid_length\":%" PRIu32 ","
        "\"last_layer\":%" PRIu32 ","
        "\"last_layer_final_valid_length\":%" PRIu32 ","
        "\"all_steps_pass\":%s}\n",
        qbh_variant_name(variant), state->completed_step_count,
        first_layer, state->layers[first_layer].valid_length,
        middle_layer, state->layers[middle_layer].valid_length,
        last_layer, state->layers[last_layer].valid_length,
        all_pass ? "true" : "false");
    free(cache_snapshots);
    return all_pass && state->completed_step_count == total_steps
               ? 0 : -1;
}

static int qbh_run_generation_sequence(
    struct qbh_session *session, int shared_fd, uint8_t *shared,
    uint32_t total_bytes, const char *package_root,
    struct qbh_block_header *header,
    const struct qbh_file_slot *token_slot,
    const struct qbh_file_slot rope_slots[2]) {
    struct qbh_decode_session_state *state =
        (struct qbh_decode_session_state *)(
            shared + header->replay_session_offset);
    uint32_t generated[QBH_GENERATION_MAX_TOKENS];
    uint32_t generation_steps = QBH_GENERATION_DEFAULT_TOKENS;
    const char *generation_steps_env = getenv("QBH_GENERATION_STEPS");
    uint64_t total_wall_ns = 0U;
    const uint32_t w4u8 =
        qbh_generation_w4u8_enabled(header->generation_mode);
    uint32_t experiment;
    const uint32_t generation_variant =
        w4u8 != 0U ? QBH_BLOCK_W4U8 : QBH_BLOCK_W4F16;
    const char *audit_root = getenv("QBH_GENERATION_AUDIT_DIR");
    int all_pass = 1;

    if (generation_steps_env != NULL && generation_steps_env[0] != '\0' &&
        (qbh_parse_u32(generation_steps_env, &generation_steps) != 0 ||
         generation_steps == 0U ||
         generation_steps > QBH_GENERATION_MAX_TOKENS ||
         header->kv_cache_capacity < QBH_BLOCK_M ||
         generation_steps - 1U >
             header->kv_cache_capacity - QBH_BLOCK_M ||
         (w4u8 == 0U &&
          generation_steps > header->generation_expected_token_count))) {
        return -1;
    }
    experiment = QBH_BLOCK_EXPERIMENT;

    memset(generated, 0, sizeof(generated));
    for (uint32_t step = 0U; step < generation_steps;
         ++step) {
        const uint32_t initial_length =
            state->layers[QBH_VERTICAL_SLICE_FIRST_LAYER].valid_length;
        uint64_t start;
        uint64_t end;
        int rpc_result;
        int step_pass;
        struct qbh_replay_step_result profile_result;

        for (uint32_t slice_index = 1U;
             slice_index < QBH_VERTICAL_SLICE_LAYER_COUNT;
             ++slice_index) {
            if (state->layers[
                    QBH_VERTICAL_SLICE_FIRST_LAYER + slice_index]
                    .valid_length != initial_length) {
                fprintf(stderr,
                        "generation cache lengths diverged before step "
                        "%" PRIu32 "\n", step);
                return -1;
            }
        }
        if (step == 0U) {
            header->scan_mode = QBH_BLOCK_SCAN_PREFILL;
            header->logical_m = QBH_BLOCK_M;
            header->generation_token_count = QBH_BLOCK_M;
        } else {
            char name[128];
            uint32_t *token_ids =
                (uint32_t *)(shared + token_slot->offset);
            token_ids[0] = generated[step - 1U];
            memset(token_ids + 1, 0,
                   token_slot->expected_bytes - sizeof(*token_ids));
            header->scan_mode = QBH_BLOCK_SCAN_DECODE;
            header->logical_m = 1U;
            header->generation_token_count = 1U;
            if (snprintf(
                    name, sizeof(name),
                    "generation_decode_rope_cos_%02" PRIu32
                    "_f16.bin", step - 1U) < 0 ||
                qbh_read_named_tensor(
                    package_root, name,
                    shared + rope_slots[0].offset,
                    rope_slots[0].expected_bytes) != 0 ||
                snprintf(
                    name, sizeof(name),
                    "generation_decode_rope_sin_%02" PRIu32
                    "_f16.bin", step - 1U) < 0 ||
                qbh_read_named_tensor(
                    package_root, name,
                    shared + rope_slots[1].offset,
                    rope_slots[1].expected_bytes) != 0) {
                fprintf(stderr,
                        "failed to stage generation RoPE step %" PRIu32
                        "\n", step);
                return -1;
            }
        }

        qbh_bind_host_slice_layer(header, 0U);
        header->initial_kv_length = initial_length;
        header->replay_expected_step = state->completed_step_count;
        header->replay_first_position = state->next_position;
        header->repeat_count = 1U;
        header->dsp_status = QBH_BLOCK_STATUS_HOST_READY;
        start = qbh_monotonic_ns();
        rpc_result = qwen3_probe_run_block(
            session->handle, shared_fd, total_bytes);
        end = qbh_monotonic_ns();
        total_wall_ns += end - start;
        generated[step] = header->generation_selected_token_id;

        step_pass =
            rpc_result == AEE_SUCCESS &&
            header->dsp_status == QBH_BLOCK_STATUS_OK &&
            (w4u8 != 0U || header->generation_token_match != 0U) &&
            header->generation_input_token_count_observed ==
                header->generation_token_count &&
            header->vtcm_requested_bytes == QBH_EXPECTED_FULL_VTCM_BYTES &&
            header->vtcm_acquired_bytes == QBH_EXPECTED_FULL_VTCM_BYTES &&
            header->intermediate_ddr_read_bytes == 0U &&
            header->intermediate_ddr_write_bytes == 0U &&
            header->intermediate_spill_fill_count == 0U &&
            header->boundary_ddr_write_bytes ==
                (header->generation_boundary_audit_enabled != 0U
                     ? QBH_BLOCK_HIDDEN : 0U) &&
            state->completed_step_count == step + 1U;
        for (uint32_t slice_index = 0U;
             slice_index < QBH_VERTICAL_SLICE_LAYER_COUNT;
             ++slice_index) {
            const struct qbh_decode_layer_state *layer =
                &state->layers[
                    QBH_VERTICAL_SLICE_FIRST_LAYER + slice_index];
            if (layer->valid_length !=
                    initial_length + header->logical_m ||
                header->slice_profiles[slice_index].cache_valid_before !=
                    initial_length ||
                header->slice_profiles[slice_index].cache_valid_after !=
                    layer->valid_length ||
                header->slice_profiles[slice_index]
                        .hidden_ddr_write_bytes != 0U) {
                step_pass = 0;
            }
        }
        if (w4u8 != 0U &&
            header->generation_boundary_audit_enabled != 0U &&
            audit_root != NULL && audit_root[0] != '\0') {
            char audit_name[96];
            if (snprintf(
                    audit_name, sizeof(audit_name),
                    "generation_hidden_step%02" PRIu32 "_u8.bin",
                    step) < 0 ||
                qbh_write_named_tensor(
                    audit_root, audit_name,
                    shared + header->output_offset,
                    QBH_BLOCK_HIDDEN) != 0) {
                step_pass = 0;
            }
        }
        all_pass &= step_pass;
        printf(
            "{\"experiment\":%" PRIu32
            ",\"generation_step\":%" PRIu32
            ",\"generation_mode\":%" PRIu32
            ",\"mode\":\"%s\",\"first_position\":%" PRIu32
            ",\"valid_length\":%" PRIu32
            ",\"host_wall_ns\":%" PRIu64
            ",\"selected_token_id\":%" PRIu32
            ",\"expected_token_id\":%" PRIu32
            ",\"token_match\":%s"
            ",\"selected_logit_half_bits\":%" PRIu32
            ",\"selected_logit_encoding\":\"%s\""
            ",\"embedding_ticks\":%" PRIu64
            ",\"final_norm_ticks\":%" PRIu64
            ",\"lm_head_ticks\":%" PRIu64
            ",\"lm_head_weight_dma_ticks\":%" PRIu64
            ",\"lm_head_scale_dma_ticks\":%" PRIu64
            ",\"lm_head_expand_ticks\":%" PRIu64
            ",\"lm_head_hmx_ticks\":%" PRIu64
            ",\"lm_head_argmax_ticks\":%" PRIu64
            ",\"lm_head_weight_dma_wait_ticks\":%" PRIu64
            ",\"lm_head_scale_init_ticks\":%" PRIu64
            ",\"lm_head_hmx_tail_wait_ticks\":%" PRIu64
            ",\"lm_head_commands\":%" PRIu32
            ",\"lm_head_n_tiles\":%" PRIu32
            ",\"lm_head_prefetch_count\":%" PRIu32
            ",\"lm_head_scale_resident_bytes\":%" PRIu32
            ",\"embedding_ddr_read_bytes\":%" PRIu64
            ",\"lm_head_ddr_read_bytes\":%" PRIu64
            ",\"boundary_ddr_write_bytes\":%" PRIu64
            ",\"intermediate_ddr_read_bytes\":%" PRIu32
            ",\"intermediate_ddr_write_bytes\":%" PRIu32
            ",\"intermediate_spill_fill_count\":%" PRIu32
            ",\"vtcm_acquired_bytes\":%" PRIu32
            ",\"rpc_result\":%d,\"dsp_status\":%d"
            ",\"pass\":%s}\n",
            experiment, step, header->generation_mode,
            step == 0U ? "prefill" : "decode",
            header->replay_first_position,
            state->layers[QBH_VERTICAL_SLICE_FIRST_LAYER].valid_length,
            end - start, header->generation_selected_token_id,
            header->generation_expected_token_id,
            header->generation_token_match != 0U ? "true" : "false",
            header->generation_selected_logit_half_bits,
            w4u8 != 0U ? "u8_code" : "fp16_bits",
            header->generation_embedding_ticks,
            header->generation_final_norm_ticks,
            header->generation_lm_head_ticks,
            header->generation_lm_head_weight_dma_ticks,
            header->generation_lm_head_scale_dma_ticks,
            header->generation_lm_head_expand_ticks,
            header->generation_lm_head_hmx_ticks,
            header->generation_lm_head_argmax_ticks,
            header->generation_lm_head_weight_dma_wait_ticks,
            header->generation_lm_head_scale_init_ticks,
            header->generation_lm_head_hmx_tail_wait_ticks,
            header->generation_lm_head_command_count,
            header->generation_lm_head_n_tiles,
            header->generation_lm_head_prefetch_count,
            header->generation_lm_head_scale_resident_bytes,
            header->generation_embedding_ddr_read_bytes,
            header->generation_lm_head_ddr_read_bytes,
            header->boundary_ddr_write_bytes,
            header->intermediate_ddr_read_bytes,
            header->intermediate_ddr_write_bytes,
            header->intermediate_spill_fill_count,
            header->vtcm_acquired_bytes, rpc_result,
            header->dsp_status, step_pass ? "true" : "false");
        memset(&profile_result, 0, sizeof(profile_result));
        profile_result.host_wall_ns = end - start;
        profile_result.output.elements = 1U;
        profile_result.output.mismatches =
            header->generation_token_match != 0U ? 0U : 1U;
        profile_result.output.cosine =
            header->generation_token_match != 0U ? 1.0 : 0.0;
        profile_result.cache_min_cosine = 1.0;
        profile_result.step_index = step;
        profile_result.first_position = header->replay_first_position;
        profile_result.valid_length =
            state->layers[QBH_VERTICAL_SLICE_FIRST_LAYER].valid_length;
        profile_result.dsp_status = (uint32_t)header->dsp_status;
        profile_result.numerical_status =
            (uint32_t)header->numerical_status;
        profile_result.vtcm_requested_bytes =
            header->vtcm_requested_bytes;
        profile_result.vtcm_acquired_bytes =
            header->vtcm_acquired_bytes;
        profile_result.intermediate_ddr_read_bytes =
            header->intermediate_ddr_read_bytes;
        profile_result.intermediate_ddr_write_bytes =
            header->intermediate_ddr_write_bytes;
        profile_result.intermediate_spill_fill_count =
            header->intermediate_spill_fill_count;
        profile_result.scan_cache_ddr_read_bytes =
            header->scan_cache_ddr_read_bytes;
        profile_result.scan_cache_ddr_write_bytes =
            header->scan_cache_ddr_write_bytes;
        profile_result.scan_dynamic_attention_ticks =
            header->scan_dynamic_attention_ticks;
        qbh_print_replay_profile(
            experiment, "generation_profile", "generation_step",
            generation_variant, step, header, &profile_result,
            (const uint8_t *)&generated[step], sizeof(generated[step]));
        if (rpc_result != AEE_SUCCESS ||
            header->dsp_status != QBH_BLOCK_STATUS_OK) {
            return -1;
        }
    }

    printf(
        "{\"experiment\":%" PRIu32
        ",\"generation_sequence_complete\":true,"
        "\"generation_mode\":%" PRIu32 ","
        "\"variant\":\"%s\",\"requested_steps\":%" PRIu32
        ",\"completed_steps\":%" PRIu32
        ",\"total_host_wall_ns\":%" PRIu64
        ",\"token_ids\":[",
        experiment, header->generation_mode,
        qbh_variant_name(generation_variant),
        generation_steps, state->completed_step_count, total_wall_ns);
    for (uint32_t index = 0U; index < generation_steps;
         ++index) {
        printf("%s%" PRIu32, index == 0U ? "" : ",",
               generated[index]);
    }
    printf("],\"all_steps_pass\":%s}\n",
           all_pass ? "true" : "false");
    return all_pass &&
                   state->completed_step_count ==
                       generation_steps
               ? 0 : -1;
}

static int qbh_run_full_stack_hidden_capture(
    struct qbh_session *session, int shared_fd, uint8_t *shared,
    uint32_t total_bytes, struct qbh_block_header *header,
    uint32_t variant) {
    struct qbh_decode_session_state *state =
        (struct qbh_decode_session_state *)(
            shared + header->replay_session_offset);
    const char *dump_root = getenv("QBH_HIDDEN_CAPTURE_DIR");
    const char *suffix = variant == QBH_BLOCK_W4U8 ? "u8" : "f16";
    const uint32_t layer_bytes =
        header->full_stack_hidden_capture_layer_bytes;
    const uint8_t *last_layer =
        shared + header->full_stack_hidden_capture_offset +
        (QBH_VERTICAL_SLICE_LAYER_COUNT - 1U) * layer_bytes;
    uint64_t host_start;
    uint64_t host_end;
    uint64_t final_output_mismatches;
    char name[128];
    int rpc_result;
    int gate_pass;

    if (dump_root == NULL || dump_root[0] == '\0' ||
        header->full_stack_stage_mode !=
            QBH_BLOCK_FULL_STACK_HIDDEN_CAPTURE ||
        header->full_stack_hidden_capture_bytes == 0U ||
        layer_bytes == 0U) {
        return -1;
    }
    header->scan_mode = QBH_BLOCK_SCAN_PREFILL;
    header->logical_m = QBH_BLOCK_M;
    header->initial_kv_length = 0U;
    header->replay_expected_step = state->completed_step_count;
    header->replay_first_position = state->next_position;
    header->repeat_count = 1U;
    qbh_bind_host_slice_layer(header, 0U);
    header->dsp_status = QBH_BLOCK_STATUS_HOST_READY;
    memset(shared + header->output_offset, 0xa5, header->output_bytes);
    memset(shared + header->full_stack_hidden_capture_offset, 0xa5,
           header->full_stack_hidden_capture_bytes);
    host_start = qbh_monotonic_ns();
    rpc_result = qwen3_probe_run_block(
        session->handle, shared_fd, total_bytes);
    host_end = qbh_monotonic_ns();

    final_output_mismatches = qbh_count_byte_mismatches(
        last_layer, shared + header->output_offset, layer_bytes);
    gate_pass = rpc_result == AEE_SUCCESS &&
        header->dsp_status == QBH_BLOCK_STATUS_OK &&
        header->full_stack_hidden_capture_layer_count ==
            QBH_VERTICAL_SLICE_LAYER_COUNT &&
        header->full_stack_hidden_capture_dma_descriptor_count ==
            QBH_VERTICAL_SLICE_LAYER_COUNT &&
        header->full_stack_hidden_capture_ddr_write_bytes ==
            header->full_stack_hidden_capture_bytes &&
        header->intermediate_ddr_read_bytes == 0U &&
        header->intermediate_ddr_write_bytes == 0U &&
        header->intermediate_spill_fill_count == 0U &&
        state->completed_step_count == 1U &&
        final_output_mismatches == 0U;

    if (snprintf(name, sizeof(name),
                 "actual_hidden_stack_%s.bin", suffix) < 0 ||
        qbh_write_named_tensor(
            dump_root, name,
            shared + header->full_stack_hidden_capture_offset,
            header->full_stack_hidden_capture_bytes) != 0 ||
        snprintf(name, sizeof(name),
                 "actual_full_stack_output_%s.bin", suffix) < 0 ||
        qbh_write_named_tensor(
            dump_root, name, shared + header->output_offset,
            header->output_bytes) != 0) {
        gate_pass = 0;
    }
    printf(
        "{\"experiment\":163,\"hidden_capture\":true,"
        "\"variant\":\"%s\",\"host_wall_ns\":%" PRIu64 ","
        "\"rpc_result\":%d,\"dsp_status\":%d,"
        "\"captured_layers\":%" PRIu32 ","
        "\"layer_bytes\":%" PRIu32 ","
        "\"capture_bytes\":%" PRIu32 ","
        "\"diagnostic_ddr_write_bytes\":%" PRIu64 ","
        "\"diagnostic_dma_descriptors\":%" PRIu32 ","
        "\"diagnostic_capture_ticks\":%" PRIu64 ","
        "\"formal_intermediate_ddr_read_bytes\":%" PRIu32 ","
        "\"formal_intermediate_ddr_write_bytes\":%" PRIu32 ","
        "\"formal_intermediate_spill_fill_count\":%" PRIu32 ","
        "\"final_output_capture_mismatches\":%" PRIu64 ","
        "\"formal_physical_evidence\":false,\"gate_pass\":%s}\n",
        qbh_variant_name(variant), host_end - host_start, rpc_result,
        header->dsp_status,
        header->full_stack_hidden_capture_layer_count, layer_bytes,
        header->full_stack_hidden_capture_bytes,
        header->full_stack_hidden_capture_ddr_write_bytes,
        header->full_stack_hidden_capture_dma_descriptor_count,
        header->full_stack_hidden_capture_ticks,
        header->intermediate_ddr_read_bytes,
        header->intermediate_ddr_write_bytes,
        header->intermediate_spill_fill_count,
        final_output_mismatches, gate_pass ? "true" : "false");
    return gate_pass ? 0 : -1;
}

int main(int argc, char **argv) {
    struct qbh_session session = {(remote_handle64)-1, 0};
    struct qbh_file_slot input_slot;
    struct qbh_file_slot reference_slot;
    struct qbh_file_slot qparam_slot;
    struct qbh_file_slot attention_config_slot;
    struct qbh_file_slot attention_audit_slots[3];
    struct qbh_file_slot norm_slots[4];
    struct qbh_file_slot rope_slots[2];
    struct qbh_file_slot kv_cache_slots[2];
    struct qbh_file_slot kv_reference_slots[2];
    struct qbh_file_slot w4u8_lut_slot;
    struct qbh_file_slot weight_slots[QBH_BLOCK_PROJECTION_COUNT];
    struct qbh_file_slot scale_slots[QBH_BLOCK_PROJECTION_COUNT];
    struct qbh_file_slot generation_token_slot;
    struct qbh_file_slot generation_embedding_slot;
    struct qbh_file_slot generation_final_norm_slot;
    struct qbh_file_slot generation_lm_head_weight_slot;
    struct qbh_file_slot generation_lm_head_scale_slot;
    struct qbh_file_slot generation_lm_head_bias_slot;
    struct qbh_file_slot generation_qparam_slot;
    struct qbh_file_slot generation_expected_token_slot;
    struct qbh_vertical_layer_slots
        vertical_slots[QBH_VERTICAL_SLICE_LAYER_COUNT];
    struct qbh_projection_layout w4u8_gate_up_layout;
    struct qbh_projection_layout w4u8_down_layout;
    struct qbh_block_header *header = NULL;
    uint8_t *shared = NULL;
    uint32_t variant;
    uint32_t repeats = 1U;
    uint32_t w4f16_hvx_workers = 2U;
    uint32_t w4f16_region_tiles = 16U;
    uint32_t common_ops_mask = QBH_BLOCK_COMMON_OPS_HVX_FP16;
    uint32_t attribution_enabled = 0U;
    uint32_t numerical_audit_enabled = 1U;
    uint32_t residual_mode = QBH_BLOCK_RESIDUAL_SCALAR;
    uint32_t f16f16_projection_mode =
        QBH_BLOCK_F16F16_PROJECTION_SERIAL;
    uint32_t w4f16_pipeline_mode = QBH_BLOCK_W4F16_PIPELINE_CONTROL;
    uint32_t attention_pack_mode = QBH_BLOCK_ATTENTION_PACK_CONTROL;
    uint32_t attention_pipeline_mode =
        QBH_BLOCK_ATTENTION_PIPELINE_CONTROL;
    uint32_t attention_hvx_contexts = 1U;
    uint32_t mlp_mode = QBH_BLOCK_MLP_CONTROL;
    uint32_t mlp_hvx_contexts = 1U;
    uint32_t mlp_chunk_vectors = 64U;
    uint32_t crouton_boundary_mode =
        QBH_BLOCK_CROUTON_BOUNDARY_CONTROL;
    uint32_t w4u8_qkvo_pipeline_mode =
        QBH_BLOCK_W4U8_QKVO_SERIAL;
    uint32_t u8_norm_reduction_mode =
        QBH_BLOCK_U8_NORM_REDUCTION_SCALAR;
    uint32_t fp16_common_schedule_mode =
        QBH_BLOCK_FP16_COMMON_SCHEDULE_CONTROL;
    uint32_t fp16_norm_rows_per_task = 4U;
    uint32_t fp16_norm_contexts = 4U;
    uint32_t w4u8_down_hmx_batch_outputs = 1U;
    uint32_t w4u8_qk_pair_kernel_mode =
        QBH_BLOCK_W4U8_QK_PAIR_SERIAL_INNER;
    uint32_t qkv_schedule_mode = QBH_BLOCK_QKV_SCHEDULE_CONTROL;
    uint32_t w4f16_group_fence_mode =
        QBH_BLOCK_W4F16_GROUP_FENCE_CONTROL;
    uint32_t w4f16_expand_claim_regions = 1U;
    uint32_t w4f16_gate_up_extra_expand_worker = 0U;
    uint32_t w4f16_gate_up_extra_stream_worker = 0U;
    uint32_t w4f16_gate_up_stream_group_tiles = 8U;
    uint32_t w4u8_stream_fence_mode =
        QBH_BLOCK_W4U8_STREAM_FENCE_CONTROL;
    uint32_t w4u8_gate_up_ring_slots = 8U;
    uint32_t w4u8_qkv_ring_expand_workers = 0U;
    uint32_t w4u8_prefill_cache_mode =
        QBH_BLOCK_W4U8_PREFILL_CACHE_DUPLICATE_BUILD;
    uint32_t w4u8_delta_reconstruction_mode =
        QBH_BLOCK_W4U8_DELTA_RECONSTRUCTION_SERIAL;
    uint32_t w4u8_decode_softmax_mode =
        QBH_BLOCK_W4U8_DECODE_SOFTMAX_SCALAR;
    uint32_t w4u8_decode_lm_head_group_tiles = 8U;
    uint32_t w4u8_decode_o_batch_n_tiles = 4U;
    uint32_t w4u8_decode_av_requant_rows =
        QBH_BLOCK_W4U8_AV_REQUANT_FULL_ROWS;
    uint32_t w4u8_decode_av_padding_poison = 0U;
    uint32_t w4u8_decode_common_op_rows =
        QBH_BLOCK_W4U8_COMMON_OP_FULL_ROWS;
    uint32_t w4u8_decode_common_padding_poison = 0U;
    uint32_t w4u8_decode_qk_norm_rope_rows =
        QBH_BLOCK_W4U8_QK_PREP_FULL_ROWS;
    uint32_t w4u8_decode_qk_padding_poison = 0U;
    uint32_t scan_mode = QBH_BLOCK_SCAN_DISABLED;
    uint32_t logical_m = QBH_BLOCK_M;
    uint32_t initial_kv_length = 0U;
    uint32_t kv_cache_capacity = 0U;
    uint32_t kv_cache_k_format =
        QBH_KV_CACHE_FORMAT_HEAD_MAJOR_ROW_V1;
    uint32_t kv_cache_v_format =
        QBH_KV_CACHE_FORMAT_HEAD_MAJOR_ROW_V1;
    uint32_t physical_chunks = 1U;
    uint32_t replay_mode = QBH_BLOCK_REPLAY_DISABLED;
    uint32_t vertical_slice_mode = QBH_BLOCK_SLICE_DISABLED;
    uint32_t generation_mode = QBH_BLOCK_GENERATION_DISABLED;
    uint32_t generation_boundary_audit_enabled = 0U;
    uint32_t full_stack_stage_mode = QBH_BLOCK_FULL_STACK_RUN;
    uint32_t w4u8_boundary_audit_enabled = 0U;
    uint32_t replay_session_offset = 0U;
    uint32_t element_bytes;
    uint32_t output_bytes;
    size_t w4u8_gate_up_bundle_offset = 0U;
    size_t w4u8_down_bundle_offset = 0U;
    size_t vertical_bias_offsets[QBH_VERTICAL_SLICE_LAYER_COUNT]
                                [QBH_BLOCK_PROJECTION_COUNT];
    size_t single_gate_up_scale_cache_offset = 0U;
    size_t attention_audit_output_offset = 0U;
    size_t scan_attention_audit_output_offset = 0U;
    size_t w4u8_boundary_audit_output_offset = 0U;
    size_t full_stack_hidden_capture_offset = 0U;
    size_t full_stack_hidden_capture_bytes = 0U;
    size_t cursor = qbh_align_up_size(sizeof(*header), QBH_HOST_ALIGNMENT);
    size_t total_bytes;
    int shared_fd = -1;
    int mapped = 0;
    int open_result;
    int prepare_result = AEE_EFAILED;
    int warmup_result = AEE_EFAILED;
    int measured_result = AEE_EFAILED;
    int release_result = AEE_EFAILED;
    int close_result = AEE_EFAILED;
    uint64_t warmup_start;
    uint64_t warmup_end;
    uint64_t measured_start;
    uint64_t measured_end;
    struct qbh_error_metrics warmup_metrics;
    struct qbh_error_metrics measured_metrics;
    uint32_t warmup_run_index = 0U;
    uint64_t output_hash = 0U;
    uint64_t kv_cache_mismatches = 0U;
    uint64_t kv_cache_k_hash = 0U;
    uint64_t kv_cache_v_hash = 0U;
    int exit_code = 1;
    char file_name[128];

    memset(&warmup_metrics, 0, sizeof(warmup_metrics));
    memset(&measured_metrics, 0, sizeof(measured_metrics));
    memset(&w4u8_lut_slot, 0, sizeof(w4u8_lut_slot));
    memset(&attention_config_slot, 0, sizeof(attention_config_slot));
    memset(attention_audit_slots, 0, sizeof(attention_audit_slots));
    memset(kv_cache_slots, 0, sizeof(kv_cache_slots));
    memset(kv_reference_slots, 0, sizeof(kv_reference_slots));
    memset(&generation_token_slot, 0, sizeof(generation_token_slot));
    memset(&generation_embedding_slot, 0,
           sizeof(generation_embedding_slot));
    memset(&generation_final_norm_slot, 0,
           sizeof(generation_final_norm_slot));
    memset(&generation_lm_head_weight_slot, 0,
           sizeof(generation_lm_head_weight_slot));
    memset(&generation_lm_head_scale_slot, 0,
           sizeof(generation_lm_head_scale_slot));
    memset(&generation_lm_head_bias_slot, 0,
           sizeof(generation_lm_head_bias_slot));
    memset(&generation_qparam_slot, 0,
           sizeof(generation_qparam_slot));
    memset(&generation_expected_token_slot, 0,
           sizeof(generation_expected_token_slot));
    memset(vertical_slots, 0, sizeof(vertical_slots));
    memset(vertical_bias_offsets, 0, sizeof(vertical_bias_offsets));
    memset(&w4u8_gate_up_layout, 0, sizeof(w4u8_gate_up_layout));
    memset(&w4u8_down_layout, 0, sizeof(w4u8_down_layout));
    {
        const char *schedule = getenv("QBH_QKV_SCHEDULE");
        if (schedule != NULL && schedule[0] != '\0') {
            if (strcmp(schedule, "control") == 0) {
                qkv_schedule_mode = QBH_BLOCK_QKV_SCHEDULE_CONTROL;
            } else if (strcmp(schedule, "q_prefix4_k_all") == 0) {
                qkv_schedule_mode =
                    QBH_BLOCK_QKV_SCHEDULE_Q_PREFIX4_K_ALL;
            } else if (strcmp(schedule,
                              "head_aligned_batch4") == 0) {
                qkv_schedule_mode =
                    QBH_BLOCK_QKV_SCHEDULE_HEAD_ALIGNED_BATCH4;
            } else if (strcmp(schedule, "v_batch4") == 0) {
                qkv_schedule_mode = QBH_BLOCK_QKV_SCHEDULE_V_BATCH4;
            } else if (strcmp(schedule, "kv_batch4") == 0) {
                qkv_schedule_mode = QBH_BLOCK_QKV_SCHEDULE_KV_BATCH4;
            } else {
                qkv_schedule_mode = UINT32_MAX;
            }
        }
    }
    {
        const char *group_fence = getenv("QBH_W4F16_GROUP_FENCE");
        if (group_fence != NULL && group_fence[0] != '\0') {
            if (strcmp(group_fence, "control") == 0) {
                w4f16_group_fence_mode =
                    QBH_BLOCK_W4F16_GROUP_FENCE_CONTROL;
            } else if (strcmp(group_fence, "join_only") == 0) {
                w4f16_group_fence_mode =
                    QBH_BLOCK_W4F16_GROUP_FENCE_JOIN_ONLY;
            } else if (strcmp(group_fence, "join_only_down") == 0) {
                w4f16_group_fence_mode =
                    QBH_BLOCK_W4F16_GROUP_FENCE_JOIN_ONLY_DOWN;
            } else {
                w4f16_group_fence_mode = UINT32_MAX;
            }
        }
    }
    {
        const char *claim_regions =
            getenv("QBH_W4F16_EXPAND_CLAIM_REGIONS");
        if (claim_regions != NULL && claim_regions[0] != '\0' &&
            qbh_parse_u32(
                claim_regions, &w4f16_expand_claim_regions) != 0) {
            w4f16_expand_claim_regions = UINT32_MAX;
        }
    }
    {
        const char *extra_worker =
            getenv("QBH_W4F16_GATE_UP_EXTRA_EXPAND_WORKER");
        if (extra_worker != NULL && extra_worker[0] != '\0' &&
            qbh_parse_u32(
                extra_worker,
                &w4f16_gate_up_extra_expand_worker) != 0) {
            w4f16_gate_up_extra_expand_worker = UINT32_MAX;
        }
    }
    {
        const char *extra_worker =
            getenv("QBH_W4F16_GATE_UP_EXTRA_STREAM_WORKER");
        if (extra_worker != NULL && extra_worker[0] != '\0' &&
            qbh_parse_u32(
                extra_worker,
                &w4f16_gate_up_extra_stream_worker) != 0) {
            w4f16_gate_up_extra_stream_worker = UINT32_MAX;
        }
    }
    {
        const char *group_tiles =
            getenv("QBH_W4F16_GATE_UP_STREAM_GROUP_TILES");
        if (group_tiles != NULL && group_tiles[0] != '\0' &&
            qbh_parse_u32(
                group_tiles,
                &w4f16_gate_up_stream_group_tiles) != 0) {
            w4f16_gate_up_stream_group_tiles = UINT32_MAX;
        }
    }
    {
        const char *stream_fence = getenv("QBH_W4U8_STREAM_FENCE");
        if (stream_fence != NULL && stream_fence[0] != '\0') {
            if (strcmp(stream_fence, "control") == 0) {
                w4u8_stream_fence_mode =
                    QBH_BLOCK_W4U8_STREAM_FENCE_CONTROL;
            } else if (strcmp(stream_fence, "single_fence") == 0) {
                w4u8_stream_fence_mode =
                    QBH_BLOCK_W4U8_STREAM_FENCE_SINGLE;
            } else if (strcmp(stream_fence, "release_only") == 0) {
                w4u8_stream_fence_mode =
                    QBH_BLOCK_W4U8_STREAM_FENCE_RELEASE_ONLY;
            } else {
                w4u8_stream_fence_mode = UINT32_MAX;
            }
        }
    }
    {
        const char *ring_slots = getenv("QBH_W4U8_GATE_UP_RING_SLOTS");
        if (ring_slots != NULL && ring_slots[0] != '\0' &&
            qbh_parse_u32(ring_slots, &w4u8_gate_up_ring_slots) != 0) {
            w4u8_gate_up_ring_slots = UINT32_MAX;
        }
    }
    {
        const char *expand_workers =
            getenv("QBH_W4U8_QKV_RING_EXPAND_WORKERS");
        if (expand_workers != NULL && expand_workers[0] != '\0' &&
            qbh_parse_u32(
                expand_workers,
                &w4u8_qkv_ring_expand_workers) != 0) {
            w4u8_qkv_ring_expand_workers = UINT32_MAX;
        }
    }
    {
        const char *mode = getenv("QBH_W4U8_PREFILL_CACHE_MODE");
        if (mode != NULL && mode[0] != '\0') {
            if (strcmp(mode, "duplicate") == 0) {
                w4u8_prefill_cache_mode =
                    QBH_BLOCK_W4U8_PREFILL_CACHE_DUPLICATE_BUILD;
            } else if (strcmp(mode, "reuse") == 0) {
                w4u8_prefill_cache_mode =
                    QBH_BLOCK_W4U8_PREFILL_CACHE_REUSE_ATTENTION_CARRIERS;
            } else {
                w4u8_prefill_cache_mode = UINT32_MAX;
            }
        }
    }
    {
        const char *mode =
            getenv("QBH_W4U8_DELTA_RECONSTRUCTION");
        if (mode != NULL && mode[0] != '\0') {
            if (strcmp(mode, "serial") == 0) {
                w4u8_delta_reconstruction_mode =
                    QBH_BLOCK_W4U8_DELTA_RECONSTRUCTION_SERIAL;
            } else if (strcmp(mode, "direct") == 0) {
                w4u8_delta_reconstruction_mode =
                    QBH_BLOCK_W4U8_DELTA_RECONSTRUCTION_DIRECT;
            } else if (strcmp(mode, "pipeline") == 0) {
                w4u8_delta_reconstruction_mode =
                    QBH_BLOCK_W4U8_DELTA_RECONSTRUCTION_PIPELINE;
            } else {
                w4u8_delta_reconstruction_mode = UINT32_MAX;
            }
        }
    }
    {
        const char *mode = getenv("QBH_W4U8_DECODE_SOFTMAX");
        if (mode != NULL && mode[0] != '\0') {
            if (strcmp(mode, "scalar") == 0) {
                w4u8_decode_softmax_mode =
                    QBH_BLOCK_W4U8_DECODE_SOFTMAX_SCALAR;
            } else if (strcmp(mode, "hvx_tile4") == 0) {
                w4u8_decode_softmax_mode =
                    QBH_BLOCK_W4U8_DECODE_SOFTMAX_HVX_TILE4;
            } else {
                w4u8_decode_softmax_mode = UINT32_MAX;
            }
        }
    }
    {
        const char *value = getenv("QBH_W4U8_DECODE_LM_HEAD_GROUP_TILES");
        if (value != NULL && value[0] != '\0' &&
            qbh_parse_u32(value, &w4u8_decode_lm_head_group_tiles) != 0) {
            w4u8_decode_lm_head_group_tiles = UINT32_MAX;
        }
    }
    {
        const char *value = getenv("QBH_W4U8_DECODE_O_BATCH_N_TILES");
        if (value != NULL && value[0] != '\0' &&
            qbh_parse_u32(value, &w4u8_decode_o_batch_n_tiles) != 0) {
            w4u8_decode_o_batch_n_tiles = UINT32_MAX;
        }
    }
    {
        const char *value = getenv("QBH_W4U8_DECODE_AV_REQUANT_ROWS");
        if (value != NULL && value[0] != '\0' &&
            qbh_parse_u32(value, &w4u8_decode_av_requant_rows) != 0) {
            w4u8_decode_av_requant_rows = UINT32_MAX;
        }
    }
    {
        const char *value = getenv("QBH_W4U8_DECODE_AV_PADDING_POISON");
        if (value != NULL && value[0] != '\0' &&
            qbh_parse_u32(value, &w4u8_decode_av_padding_poison) != 0) {
            w4u8_decode_av_padding_poison = UINT32_MAX;
        }
    }
    {
        const char *value = getenv("QBH_W4U8_DECODE_COMMON_OP_ROWS");
        if (value != NULL && value[0] != '\0' &&
            qbh_parse_u32(value, &w4u8_decode_common_op_rows) != 0) {
            w4u8_decode_common_op_rows = UINT32_MAX;
        }
    }
    {
        const char *value = getenv("QBH_W4U8_DECODE_COMMON_PADDING_POISON");
        if (value != NULL && value[0] != '\0' &&
            qbh_parse_u32(
                value, &w4u8_decode_common_padding_poison) != 0) {
            w4u8_decode_common_padding_poison = UINT32_MAX;
        }
    }
    {
        const char *value = getenv("QBH_W4U8_DECODE_QK_NORM_ROPE_ROWS");
        if (value != NULL && value[0] != '\0' &&
            qbh_parse_u32(
                value, &w4u8_decode_qk_norm_rope_rows) != 0) {
            w4u8_decode_qk_norm_rope_rows = UINT32_MAX;
        }
    }
    {
        const char *value = getenv("QBH_W4U8_DECODE_QK_PADDING_POISON");
        if (value != NULL && value[0] != '\0' &&
            qbh_parse_u32(
                value, &w4u8_decode_qk_padding_poison) != 0) {
            w4u8_decode_qk_padding_poison = UINT32_MAX;
        }
    }
    {
        const char *mode = getenv("QBH_SCAN_MODE");
        const char *logical = getenv("QBH_LOGICAL_M");
        const char *past = getenv("QBH_KV_CACHE_LENGTH");
        const char *capacity = getenv("QBH_KV_CACHE_CAPACITY");
        if (mode != NULL && mode[0] != '\0') {
            if (strcmp(mode, "disabled") == 0) {
                scan_mode = QBH_BLOCK_SCAN_DISABLED;
            } else if (strcmp(mode, "prefill") == 0) {
                scan_mode = QBH_BLOCK_SCAN_PREFILL;
            } else if (strcmp(mode, "decode") == 0) {
                scan_mode = QBH_BLOCK_SCAN_DECODE;
            } else {
                scan_mode = UINT32_MAX;
            }
        }
        if (logical != NULL && logical[0] != '\0' &&
            qbh_parse_u32(logical, &logical_m) != 0) {
            logical_m = UINT32_MAX;
        }
        if (past != NULL && past[0] != '\0' &&
            qbh_parse_u32(past, &initial_kv_length) != 0) {
            initial_kv_length = UINT32_MAX;
        }
        if (scan_mode != QBH_BLOCK_SCAN_DISABLED) {
            kv_cache_capacity = initial_kv_length + logical_m;
            if (capacity != NULL && capacity[0] != '\0' &&
                qbh_parse_u32(capacity, &kv_cache_capacity) != 0) {
                kv_cache_capacity = UINT32_MAX;
            }
            physical_chunks =
                (logical_m + QBH_BLOCK_M - 1U) / QBH_BLOCK_M;
        }
    }
    {
        const char *replay = getenv("QBH_REPLAY_SEQUENCE");
        if (replay != NULL && replay[0] != '\0') {
            if (strcmp(replay, "0") == 0) {
                replay_mode = QBH_BLOCK_REPLAY_DISABLED;
            } else if (strcmp(replay, "1") == 0) {
                replay_mode = QBH_BLOCK_REPLAY_CONTINUOUS;
            } else {
                replay_mode = UINT32_MAX;
            }
        }
    }
    {
        const char *generation = getenv("QBH_GENERATION_SEQUENCE");
        if (generation != NULL && generation[0] != '\0') {
            if (strcmp(generation, "0") == 0) {
                generation_mode = QBH_BLOCK_GENERATION_DISABLED;
            } else if (strcmp(generation, "1") == 0) {
                generation_mode =
                    QBH_BLOCK_GENERATION_GREEDY_W4F16;
            } else if (strcmp(generation, "2") == 0) {
                generation_mode =
                    QBH_BLOCK_GENERATION_GREEDY_W4F16_HVX_ARGMAX;
            } else if (strcmp(generation, "3") == 0) {
                generation_mode =
                    QBH_BLOCK_GENERATION_GREEDY_W4F16_HVX_ARGMAX_BATCH4;
            } else if (strcmp(generation, "4") == 0) {
                generation_mode =
                    QBH_BLOCK_GENERATION_GREEDY_W4F16_HVX_ARGMAX_BATCH8;
            } else if (strcmp(generation, "5") == 0) {
                generation_mode =
                    QBH_BLOCK_GENERATION_GREEDY_W4F16_LM_HEAD_OVERLAP;
            } else if (strcmp(generation, "6") == 0) {
                generation_mode =
                    QBH_BLOCK_GENERATION_GREEDY_W4F16_DMA_HVX_OVERLAP;
            } else if (strcmp(generation, "7") == 0) {
                generation_mode =
                    QBH_BLOCK_GENERATION_GREEDY_W4F16_COARSE_PIPELINE;
            } else if (strcmp(generation, "8") == 0) {
                generation_mode =
                    QBH_BLOCK_GENERATION_GREEDY_W4U8_COARSE_PIPELINE;
            } else if (strcmp(generation, "9") == 0) {
                generation_mode =
                    QBH_BLOCK_GENERATION_GREEDY_W4U8_BATCH8_RESIDENT_BIAS;
            } else {
                generation_mode = UINT32_MAX;
            }
        }
    }
    {
        const char *generation_audit =
            getenv("QBH_GENERATION_BOUNDARY_AUDIT");
        if (generation_audit != NULL && generation_audit[0] != '\0') {
            if (strcmp(generation_audit, "0") == 0) {
                generation_boundary_audit_enabled = 0U;
            } else if (strcmp(generation_audit, "1") == 0) {
                generation_boundary_audit_enabled = 1U;
            } else {
                generation_boundary_audit_enabled = UINT32_MAX;
            }
        }
    }
    {
        const char *layout = getenv("QBH_KV_CACHE_LAYOUT");
        if (layout != NULL && layout[0] != '\0') {
            if (strcmp(layout, "row_major") == 0) {
                kv_cache_k_format =
                    QBH_KV_CACHE_FORMAT_HEAD_MAJOR_ROW_V1;
                kv_cache_v_format =
                    QBH_KV_CACHE_FORMAT_HEAD_MAJOR_ROW_V1;
            } else if (strcmp(layout, "hmx_native_u8") == 0) {
                kv_cache_k_format =
                    QBH_KV_CACHE_FORMAT_HMX_U8_K_WEIGHT_V1;
                kv_cache_v_format =
                    QBH_KV_CACHE_FORMAT_HMX_U8_V_WEIGHT_V1;
            } else if (strcmp(layout, "hmx_native_u8_delta") == 0) {
                kv_cache_k_format =
                    QBH_KV_CACHE_FORMAT_HMX_U8_K_WEIGHT_DELTA_V2;
                kv_cache_v_format =
                    QBH_KV_CACHE_FORMAT_HMX_U8_V_WEIGHT_DELTA_V2;
            } else if (strcmp(layout, "hmx_native_u8_segmented_v4") == 0) {
                kv_cache_k_format =
                    QBH_KV_CACHE_FORMAT_HMX_U8_K_SEGMENTED_V4;
                kv_cache_v_format =
                    QBH_KV_CACHE_FORMAT_HMX_U8_V_SEGMENTED_V4;
            } else if (strcmp(
                           layout,
                           "hmx_native_u8_segmented_quartet_v5") == 0) {
                kv_cache_k_format =
                    QBH_KV_CACHE_FORMAT_HMX_U8_K_SEGMENTED_V4;
                kv_cache_v_format =
                    QBH_KV_CACHE_FORMAT_HMX_U8_V_QUARTET_TAIL_V5;
            } else if (strcmp(
                           layout,
                           "hmx_native_u8_segmented_attention_publish_v6") ==
                       0) {
                kv_cache_k_format =
                    QBH_KV_CACHE_FORMAT_HMX_U8_K_SEGMENTED_V4;
                kv_cache_v_format =
                    QBH_KV_CACHE_FORMAT_HMX_U8_V_ATTENTION_PUBLISH_V6;
            } else if (strcmp(
                           layout,
                           "hmx_native_u8_segmented_vtcm_tail_v7") == 0) {
                kv_cache_k_format =
                    QBH_KV_CACHE_FORMAT_HMX_U8_K_SEGMENTED_V4;
                kv_cache_v_format =
                    QBH_KV_CACHE_FORMAT_HMX_U8_V_VTCM_TAIL_V7;
            } else if (strcmp(layout, "hmx_native_f16") == 0) {
                kv_cache_k_format =
                    QBH_KV_CACHE_FORMAT_HMX_F16_K_WEIGHT_V1;
                kv_cache_v_format =
                    QBH_KV_CACHE_FORMAT_HMX_F16_V_WEIGHT_V1;
            } else {
                kv_cache_k_format = UINT32_MAX;
                kv_cache_v_format = UINT32_MAX;
            }
        }
    }
    {
        const char *slice = getenv("QBH_VERTICAL_SLICE");
        if (slice != NULL && slice[0] != '\0') {
            if (strcmp(slice, "0") == 0) {
                vertical_slice_mode = QBH_BLOCK_SLICE_DISABLED;
            } else if (strcmp(slice, "1") == 0) {
                vertical_slice_mode = QBH_BLOCK_SLICE_ACTIVE_RANGE;
            } else {
                vertical_slice_mode = UINT32_MAX;
            }
        }
    }
    {
        const char *map_only = getenv("QBH_MAP_ONLY");
        const char *hidden_capture = getenv("QBH_HIDDEN_CAPTURE");
        const int map_enabled =
            map_only != NULL && strcmp(map_only, "1") == 0;
        const int capture_enabled =
            hidden_capture != NULL && strcmp(hidden_capture, "1") == 0;
        if (map_enabled && capture_enabled) {
            full_stack_stage_mode = UINT32_MAX;
        } else if (map_enabled) {
            full_stack_stage_mode = QBH_BLOCK_FULL_STACK_MAP_GATE;
        } else if (capture_enabled) {
            full_stack_stage_mode =
                QBH_BLOCK_FULL_STACK_HIDDEN_CAPTURE;
        }
    }
    {
        const char *boundary_audit = getenv("QBH_W4U8_BOUNDARY_AUDIT");
        if (boundary_audit != NULL && boundary_audit[0] != '\0') {
            if (strcmp(boundary_audit, "0") == 0) {
                w4u8_boundary_audit_enabled = 0U;
            } else if (strcmp(boundary_audit, "1") == 0) {
                w4u8_boundary_audit_enabled = 1U;
            } else {
                w4u8_boundary_audit_enabled = UINT32_MAX;
            }
        }
    }
    if (argc < 3 || argc > 26 ||
        qbh_parse_variant(argv[2], &variant) != 0 ||
        (argc >= 4 && qbh_parse_u32(argv[3], &repeats) != 0) ||
        (argc >= 5 && qbh_parse_u32(
                          argv[4], &w4f16_hvx_workers) != 0) ||
        (argc >= 6 && qbh_parse_u32(
                          argv[5], &w4f16_region_tiles) != 0) ||
        (argc >= 7 && qbh_parse_common_ops_mode(
                          argv[6], &common_ops_mask) != 0) ||
        (argc >= 8 && qbh_parse_attribution_mode(
                          argv[7], &attribution_enabled) != 0) ||
        (argc >= 9 && qbh_parse_attribution_mode(
                          argv[8], &numerical_audit_enabled) != 0) ||
        (argc >= 10 && qbh_parse_residual_mode(
                           argv[9], &residual_mode) != 0) ||
        (argc >= 11 && qbh_parse_f16f16_projection_mode(
                           argv[10], &f16f16_projection_mode) != 0) ||
        (argc >= 12 && qbh_parse_w4f16_pipeline_mode(
                           argv[11], &w4f16_pipeline_mode) != 0) ||
        (argc >= 13 && qbh_parse_attention_pack_mode(
                           argv[12], &attention_pack_mode) != 0) ||
        (argc >= 14 && qbh_parse_mlp_mode(
                           argv[13], &mlp_mode) != 0) ||
        (argc >= 15 && qbh_parse_u32(
                           argv[14], &mlp_hvx_contexts) != 0) ||
        (argc >= 16 && qbh_parse_u32(
                           argv[15], &mlp_chunk_vectors) != 0) ||
        (argc >= 17 && qbh_parse_attention_pipeline_mode(
                           argv[16], &attention_pipeline_mode) != 0) ||
        (argc >= 18 && qbh_parse_u32(
                           argv[17], &attention_hvx_contexts) != 0) ||
        (argc >= 19 && qbh_parse_crouton_boundary_mode(
                           argv[18], &crouton_boundary_mode) != 0) ||
        (argc >= 20 && qbh_parse_w4u8_qkvo_pipeline_mode(
                           argv[19], &w4u8_qkvo_pipeline_mode) != 0) ||
        (argc >= 21 && qbh_parse_u8_norm_reduction_mode(
                           argv[20], &u8_norm_reduction_mode) != 0) ||
        (argc >= 22 && qbh_parse_fp16_common_schedule_mode(
                           argv[21], &fp16_common_schedule_mode) != 0) ||
        (argc >= 23 && qbh_parse_u32(
                           argv[22], &fp16_norm_rows_per_task) != 0) ||
        (argc >= 24 && qbh_parse_u32(
                           argv[23], &fp16_norm_contexts) != 0) ||
        (argc >= 25 && qbh_parse_u32(
                           argv[24],
                           &w4u8_down_hmx_batch_outputs) != 0) ||
        (argc >= 26 && qbh_parse_u32(
                           argv[25],
                           &w4u8_qk_pair_kernel_mode) != 0) ||
        repeats == 0U || repeats > 100U ||
        w4u8_prefill_cache_mode >
            QBH_BLOCK_W4U8_PREFILL_CACHE_REUSE_ATTENTION_CARRIERS ||
        w4u8_delta_reconstruction_mode >
            QBH_BLOCK_W4U8_DELTA_RECONSTRUCTION_PIPELINE ||
        w4u8_decode_softmax_mode >
            QBH_BLOCK_W4U8_DECODE_SOFTMAX_HVX_TILE4 ||
        (w4u8_decode_o_batch_n_tiles != 4U &&
         w4u8_decode_o_batch_n_tiles != 8U) ||
        (w4u8_decode_o_batch_n_tiles != 4U &&
         variant != QBH_BLOCK_W4U8) ||
        (w4u8_decode_av_requant_rows !=
             QBH_BLOCK_W4U8_AV_REQUANT_FULL_ROWS &&
         w4u8_decode_av_requant_rows !=
             QBH_BLOCK_W4U8_AV_REQUANT_DECODE_ROWS) ||
        (w4u8_decode_av_requant_rows !=
             QBH_BLOCK_W4U8_AV_REQUANT_FULL_ROWS &&
         variant != QBH_BLOCK_W4U8) ||
        w4u8_decode_av_padding_poison > 1U ||
        (w4u8_decode_av_padding_poison != 0U &&
         (variant != QBH_BLOCK_W4U8 ||
          w4u8_decode_av_requant_rows !=
              QBH_BLOCK_W4U8_AV_REQUANT_DECODE_ROWS)) ||
        (w4u8_decode_common_op_rows !=
             QBH_BLOCK_W4U8_COMMON_OP_FULL_ROWS &&
         w4u8_decode_common_op_rows !=
             QBH_BLOCK_W4U8_COMMON_OP_DECODE_ROWS) ||
        (w4u8_decode_common_op_rows !=
             QBH_BLOCK_W4U8_COMMON_OP_FULL_ROWS &&
         variant != QBH_BLOCK_W4U8) ||
        w4u8_decode_common_padding_poison > 1U ||
        (w4u8_decode_common_padding_poison != 0U &&
         (variant != QBH_BLOCK_W4U8 ||
          w4u8_decode_common_op_rows !=
              QBH_BLOCK_W4U8_COMMON_OP_DECODE_ROWS)) ||
        (w4u8_decode_qk_norm_rope_rows !=
             QBH_BLOCK_W4U8_QK_PREP_FULL_ROWS &&
         w4u8_decode_qk_norm_rope_rows !=
             QBH_BLOCK_W4U8_QK_PREP_DECODE_ROWS) ||
        (w4u8_decode_qk_norm_rope_rows !=
             QBH_BLOCK_W4U8_QK_PREP_FULL_ROWS &&
         variant != QBH_BLOCK_W4U8) ||
        w4u8_decode_qk_padding_poison > 1U ||
        (w4u8_decode_qk_padding_poison != 0U &&
         (variant != QBH_BLOCK_W4U8 ||
          w4u8_decode_qk_norm_rope_rows !=
              QBH_BLOCK_W4U8_QK_PREP_DECODE_ROWS)) ||
        (w4u8_decode_lm_head_group_tiles != 8U &&
         w4u8_decode_lm_head_group_tiles != 16U) ||
        (w4u8_decode_lm_head_group_tiles != 8U &&
         variant != QBH_BLOCK_W4U8) ||
        (w4u8_decode_softmax_mode !=
             QBH_BLOCK_W4U8_DECODE_SOFTMAX_SCALAR &&
         variant != QBH_BLOCK_W4U8) ||
        scan_mode > QBH_BLOCK_SCAN_DECODE ||
        replay_mode > QBH_BLOCK_REPLAY_CONTINUOUS ||
        (kv_cache_k_format !=
             QBH_KV_CACHE_FORMAT_HEAD_MAJOR_ROW_V1 &&
         kv_cache_k_format !=
             QBH_KV_CACHE_FORMAT_HMX_U8_K_WEIGHT_V1 &&
         kv_cache_k_format !=
             QBH_KV_CACHE_FORMAT_HMX_U8_K_WEIGHT_DELTA_V2 &&
         kv_cache_k_format !=
             QBH_KV_CACHE_FORMAT_HMX_U8_K_SEGMENTED_V4 &&
         kv_cache_k_format !=
             QBH_KV_CACHE_FORMAT_HMX_F16_K_WEIGHT_V1) ||
        (kv_cache_v_format !=
             QBH_KV_CACHE_FORMAT_HEAD_MAJOR_ROW_V1 &&
         kv_cache_v_format !=
             QBH_KV_CACHE_FORMAT_HMX_U8_V_WEIGHT_V1 &&
         kv_cache_v_format !=
             QBH_KV_CACHE_FORMAT_HMX_U8_V_WEIGHT_DELTA_V2 &&
         kv_cache_v_format !=
             QBH_KV_CACHE_FORMAT_HMX_U8_V_SEGMENTED_V4 &&
         kv_cache_v_format !=
             QBH_KV_CACHE_FORMAT_HMX_U8_V_QUARTET_TAIL_V5 &&
         kv_cache_v_format !=
             QBH_KV_CACHE_FORMAT_HMX_U8_V_ATTENTION_PUBLISH_V6 &&
         kv_cache_v_format !=
             QBH_KV_CACHE_FORMAT_HMX_U8_V_VTCM_TAIL_V7 &&
         kv_cache_v_format !=
             QBH_KV_CACHE_FORMAT_HMX_F16_V_WEIGHT_V1) ||
        (qbh_hmx_native_u8_cache_formats(
             kv_cache_k_format, kv_cache_v_format) &&
         (variant != QBH_BLOCK_W4U8 ||
          (replay_mode != QBH_BLOCK_REPLAY_CONTINUOUS &&
           !(vertical_slice_mode == QBH_BLOCK_SLICE_DISABLED &&
             scan_mode == QBH_BLOCK_SCAN_DECODE)))) ||
        (qbh_hmx_native_f16_cache_formats(
             kv_cache_k_format, kv_cache_v_format) &&
         (variant == QBH_BLOCK_W4U8 ||
          replay_mode != QBH_BLOCK_REPLAY_CONTINUOUS)) ||
        (w4u8_prefill_cache_mode ==
             QBH_BLOCK_W4U8_PREFILL_CACHE_REUSE_ATTENTION_CARRIERS &&
         (!qbh_hmx_native_u8_cache_formats(
              kv_cache_k_format, kv_cache_v_format) ||
          variant != QBH_BLOCK_W4U8 ||
          replay_mode != QBH_BLOCK_REPLAY_CONTINUOUS)) ||
        vertical_slice_mode > QBH_BLOCK_SLICE_ACTIVE_RANGE ||
        (vertical_slice_mode == QBH_BLOCK_SLICE_ACTIVE_RANGE &&
         replay_mode != QBH_BLOCK_REPLAY_CONTINUOUS) ||
        (vertical_slice_mode == QBH_BLOCK_SLICE_ACTIVE_RANGE &&
         numerical_audit_enabled != 0U) ||
        (replay_mode == QBH_BLOCK_REPLAY_CONTINUOUS &&
         vertical_slice_mode != QBH_BLOCK_SLICE_ACTIVE_RANGE) ||
        generation_mode >
            QBH_BLOCK_GENERATION_GREEDY_W4U8_BATCH8_RESIDENT_BIAS ||
        generation_boundary_audit_enabled > 1U ||
        (generation_boundary_audit_enabled != 0U &&
         !qbh_generation_w4u8_enabled(generation_mode)) ||
        (generation_mode != QBH_BLOCK_GENERATION_DISABLED &&
         ((!qbh_generation_w4f16_enabled(generation_mode) &&
           !qbh_generation_w4u8_enabled(generation_mode)) ||
          (qbh_generation_w4f16_enabled(generation_mode)
               ? (variant != QBH_BLOCK_W4F16 ||
                  !qbh_hmx_native_f16_cache_formats(
                      kv_cache_k_format, kv_cache_v_format))
               : (variant != QBH_BLOCK_W4U8 ||
                  !qbh_hmx_native_u8_segmented_cache_formats(
                      kv_cache_k_format, kv_cache_v_format))) ||
          replay_mode != QBH_BLOCK_REPLAY_CONTINUOUS ||
          vertical_slice_mode != QBH_BLOCK_SLICE_ACTIVE_RANGE ||
          full_stack_stage_mode != QBH_BLOCK_FULL_STACK_RUN ||
          scan_mode != QBH_BLOCK_SCAN_PREFILL ||
          logical_m != QBH_BLOCK_M ||
          initial_kv_length != 0U ||
          (kv_cache_capacity != 80U &&
           kv_cache_capacity != 257U))) ||
        full_stack_stage_mode >
            QBH_BLOCK_FULL_STACK_HIDDEN_CAPTURE ||
        w4u8_boundary_audit_enabled > 1U ||
        (w4u8_boundary_audit_enabled != 0U &&
         (variant != QBH_BLOCK_W4U8 ||
          vertical_slice_mode != QBH_BLOCK_SLICE_DISABLED ||
          numerical_audit_enabled == 0U)) ||
        (full_stack_stage_mode != QBH_BLOCK_FULL_STACK_RUN &&
         (replay_mode != QBH_BLOCK_REPLAY_CONTINUOUS ||
          vertical_slice_mode != QBH_BLOCK_SLICE_ACTIVE_RANGE)) ||
        (replay_mode == QBH_BLOCK_REPLAY_CONTINUOUS &&
         (scan_mode != QBH_BLOCK_SCAN_PREFILL ||
          logical_m != QBH_BLOCK_M || initial_kv_length != 0U ||
          kv_cache_capacity <= QBH_BLOCK_M ||
          kv_cache_capacity > QBH_BLOCK_SCAN_MAX_KV ||
          physical_chunks != 1U)) ||
        (scan_mode == QBH_BLOCK_SCAN_DISABLED &&
         (logical_m != QBH_BLOCK_M || initial_kv_length != 0U ||
          kv_cache_capacity != 0U || physical_chunks != 1U)) ||
        (scan_mode == QBH_BLOCK_SCAN_PREFILL &&
         ((logical_m != 16U && logical_m != 32U &&
           logical_m != 64U && logical_m != 128U) ||
          initial_kv_length != 0U)) ||
        (scan_mode == QBH_BLOCK_SCAN_DECODE &&
         (logical_m != 1U ||
          (initial_kv_length != 64U &&
           initial_kv_length != 256U &&
           initial_kv_length != 1024U &&
           initial_kv_length != 4096U))) ||
        (scan_mode != QBH_BLOCK_SCAN_DISABLED &&
         (kv_cache_capacity < initial_kv_length + logical_m ||
          kv_cache_capacity > QBH_BLOCK_SCAN_MAX_KV)) ||
        w4f16_hvx_workers == 0U || w4f16_hvx_workers > 4U ||
        (variant == QBH_BLOCK_W4U8 &&
         ((!qbh_attention_u8_enabled(attention_pipeline_mode) &&
           (attention_pack_mode !=
                QBH_BLOCK_ATTENTION_PACK_CONTROL ||
            attention_pipeline_mode !=
                QBH_BLOCK_ATTENTION_PIPELINE_CONTROL)) ||
          (qbh_attention_u8_enabled(attention_pipeline_mode) &&
           (attention_pack_mode != QBH_BLOCK_ATTENTION_PACK_HVX ||
            (common_ops_mask &
                 (QBH_BLOCK_COMMON_OP_RMS_NORM |
                  QBH_BLOCK_COMMON_OP_ROPE |
                  QBH_BLOCK_COMMON_OP_SOFTMAX)) !=
                (QBH_BLOCK_COMMON_OP_RMS_NORM |
                 QBH_BLOCK_COMMON_OP_ROPE |
                 QBH_BLOCK_COMMON_OP_SOFTMAX))) ||
          (mlp_mode != QBH_BLOCK_MLP_CONTROL &&
           !qbh_block_mlp_is_w4u8_streaming(mlp_mode)))) ||
        attention_pipeline_mode >
            QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER_VDEAL_FUSED_QK_REQUANT_HMX_BATCH_LUT_TEMPLATES_GQA_BATCH_DEPENDENCY_STREAM_SOFTMAX_SHUFFLE4 ||
        attention_hvx_contexts == 0U ||
        attention_hvx_contexts > 6U ||
        (attention_pipeline_mode ==
             QBH_BLOCK_ATTENTION_PIPELINE_CONTROL &&
         attention_hvx_contexts != 1U) ||
        (attention_pipeline_mode !=
             QBH_BLOCK_ATTENTION_PIPELINE_CONTROL &&
         (qbh_attention_u8_enabled(attention_pipeline_mode)
              ? (variant != QBH_BLOCK_W4U8 ||
                 attention_hvx_contexts < 4U ||
                 attention_hvx_contexts > 6U)
              : (variant == QBH_BLOCK_W4U8 ||
                 attention_hvx_contexts != 4U))) ||
        ((attention_pipeline_mode ==
              QBH_BLOCK_ATTENTION_PIPELINE_PARALLEL_QK_NORM_ROPE ||
          attention_pipeline_mode ==
              QBH_BLOCK_ATTENTION_PIPELINE_PARALLEL_HVX) &&
         (common_ops_mask & QBH_BLOCK_COMMON_OP_ROPE) == 0U) ||
        ((attention_pipeline_mode ==
              QBH_BLOCK_ATTENTION_PIPELINE_PARALLEL_SOFTMAX ||
          attention_pipeline_mode ==
              QBH_BLOCK_ATTENTION_PIPELINE_PARALLEL_HVX) &&
         (common_ops_mask & QBH_BLOCK_COMMON_OP_SOFTMAX) == 0U) ||
        ((attention_pipeline_mode ==
              QBH_BLOCK_ATTENTION_PIPELINE_GQA ||
          attention_pipeline_mode ==
              QBH_BLOCK_ATTENTION_PIPELINE_GQA_QKV_OVERLAP) &&
         (((common_ops_mask &
            (QBH_BLOCK_COMMON_OP_ROPE |
             QBH_BLOCK_COMMON_OP_SOFTMAX)) !=
           (QBH_BLOCK_COMMON_OP_ROPE |
            QBH_BLOCK_COMMON_OP_SOFTMAX)) ||
          attention_pack_mode != QBH_BLOCK_ATTENTION_PACK_HVX)) ||
        (attention_pipeline_mode ==
             QBH_BLOCK_ATTENTION_PIPELINE_GQA_QKV_OVERLAP &&
         variant == QBH_BLOCK_W4F16 &&
         w4f16_hvx_workers != 3U &&
         w4f16_hvx_workers != 4U) ||
        (variant == QBH_BLOCK_W4U8 &&
         ((crouton_boundary_mode &
           ~((uint32_t)(QBH_BLOCK_CROUTON_BOUNDARY_W4U8_MLP_INPUT |
                        QBH_BLOCK_CROUTON_BOUNDARY_W4U8_MLP_OUTPUT |
                        QBH_BLOCK_CROUTON_BOUNDARY_W4U8_QKV_INPUT |
                        QBH_BLOCK_CROUTON_BOUNDARY_W4U8_O_OUTPUT))) != 0U ||
          ((crouton_boundary_mode &
            QBH_BLOCK_CROUTON_BOUNDARY_W4U8_MLP_OUTPUT) != 0U &&
           (crouton_boundary_mode &
            QBH_BLOCK_CROUTON_BOUNDARY_W4U8_MLP_INPUT) == 0U) ||
          ((crouton_boundary_mode &
            QBH_BLOCK_CROUTON_BOUNDARY_W4U8_O_OUTPUT) != 0U &&
           (crouton_boundary_mode &
            (QBH_BLOCK_CROUTON_BOUNDARY_W4U8_MLP_INPUT |
             QBH_BLOCK_CROUTON_BOUNDARY_W4U8_MLP_OUTPUT |
             QBH_BLOCK_CROUTON_BOUNDARY_W4U8_QKV_INPUT)) !=
            (QBH_BLOCK_CROUTON_BOUNDARY_W4U8_MLP_INPUT |
             QBH_BLOCK_CROUTON_BOUNDARY_W4U8_MLP_OUTPUT |
             QBH_BLOCK_CROUTON_BOUNDARY_W4U8_QKV_INPUT)) ||
          (crouton_boundary_mode != QBH_BLOCK_CROUTON_BOUNDARY_CONTROL &&
           !qbh_block_mlp_is_w4u8_streaming(mlp_mode)))) ||
        (variant != QBH_BLOCK_W4U8 &&
         (crouton_boundary_mode &
          (QBH_BLOCK_CROUTON_BOUNDARY_W4U8_MLP_INPUT |
           QBH_BLOCK_CROUTON_BOUNDARY_W4U8_MLP_OUTPUT |
           QBH_BLOCK_CROUTON_BOUNDARY_W4U8_QKV_INPUT |
           QBH_BLOCK_CROUTON_BOUNDARY_W4U8_O_OUTPUT)) != 0U) ||
        w4u8_qkvo_pipeline_mode >
            QBH_BLOCK_W4U8_QKVO_BATCH4_QK_HEAD_PAIRS ||
        u8_norm_reduction_mode >
            QBH_BLOCK_U8_NORM_REDUCTION_HVX_TREE_QK_BATCHED_RSQRT_SHARED_ROPE_PARALLEL_INPUT ||
        w4u8_qk_pair_kernel_mode >
            QBH_BLOCK_W4U8_QK_PAIR_QUARTER_TILED_SIMD_IO ||
        (variant != QBH_BLOCK_W4U8 &&
         u8_norm_reduction_mode !=
             QBH_BLOCK_U8_NORM_REDUCTION_SCALAR) ||
        (variant != QBH_BLOCK_W4U8 &&
         w4u8_qk_pair_kernel_mode !=
             QBH_BLOCK_W4U8_QK_PAIR_SERIAL_INNER) ||
        (w4u8_qk_pair_kernel_mode >=
             QBH_BLOCK_W4U8_QK_PAIR_QUARTER_TILED &&
         u8_norm_reduction_mode <
             QBH_BLOCK_U8_NORM_REDUCTION_HVX_TREE_QK_BATCHED_RSQRT) ||
        fp16_common_schedule_mode >
            QBH_BLOCK_FP16_COMMON_SCHEDULE_ALL ||
        (fp16_norm_rows_per_task != 2U &&
         fp16_norm_rows_per_task != 4U &&
         fp16_norm_rows_per_task != 8U) ||
        fp16_norm_contexts < 2U || fp16_norm_contexts > 4U ||
        qkv_schedule_mode >
            QBH_BLOCK_QKV_SCHEDULE_KV_BATCH4 ||
        (qkv_schedule_mode != QBH_BLOCK_QKV_SCHEDULE_CONTROL &&
         (variant != QBH_BLOCK_W4F16 ||
          attention_pipeline_mode !=
              QBH_BLOCK_ATTENTION_PIPELINE_GQA_QKV_OVERLAP ||
          (crouton_boundary_mode &
           QBH_BLOCK_CROUTON_BOUNDARY_INPUT_NORM) == 0U)) ||
        (w4u8_down_hmx_batch_outputs != 1U &&
         w4u8_down_hmx_batch_outputs != 4U) ||
        (variant != QBH_BLOCK_W4U8 &&
         w4u8_down_hmx_batch_outputs != 1U) ||
        (w4u8_gate_up_ring_slots != 8U &&
         w4u8_gate_up_ring_slots != 16U) ||
        (variant != QBH_BLOCK_W4U8 &&
         w4u8_gate_up_ring_slots != 8U) ||
        w4u8_qkv_ring_expand_workers > 3U ||
        (variant != QBH_BLOCK_W4U8 &&
         w4u8_qkv_ring_expand_workers != 0U) ||
        (w4u8_qkv_ring_expand_workers != 0U &&
         (w4u8_qkvo_pipeline_mode !=
              QBH_BLOCK_W4U8_QKVO_BATCH4_QK_HEAD_PAIRS ||
          !qbh_attention_u8_qkv_overlap_enabled(
              attention_pipeline_mode) ||
          attention_hvx_contexts != 6U ||
          (crouton_boundary_mode &
           QBH_BLOCK_CROUTON_BOUNDARY_W4U8_QKV_INPUT) == 0U)) ||
        (variant == QBH_BLOCK_W4U8 &&
         fp16_common_schedule_mode !=
             QBH_BLOCK_FP16_COMMON_SCHEDULE_CONTROL) ||
        (variant != QBH_BLOCK_W4U8 &&
         w4u8_qkvo_pipeline_mode != QBH_BLOCK_W4U8_QKVO_SERIAL) ||
        ((crouton_boundary_mode & QBH_BLOCK_CROUTON_BOUNDARY_QKV) != 0U &&
         attention_pipeline_mode !=
             QBH_BLOCK_ATTENTION_PIPELINE_GQA_QKV_OVERLAP) ||
        ((crouton_boundary_mode & QBH_BLOCK_CROUTON_BOUNDARY_AV_TO_O) != 0U &&
         attention_pipeline_mode !=
             QBH_BLOCK_ATTENTION_PIPELINE_GQA_QKV_OVERLAP) ||
        ((crouton_boundary_mode &
          QBH_BLOCK_CROUTON_BOUNDARY_W4U8_QKV_INPUT) != 0U &&
         (!qbh_attention_u8_qkv_overlap_enabled(
              attention_pipeline_mode) ||
          w4u8_qkvo_pipeline_mode <
              QBH_BLOCK_W4U8_QKVO_BATCH4)) ||
        ((crouton_boundary_mode &
          QBH_BLOCK_CROUTON_BOUNDARY_W4U8_O_OUTPUT) != 0U &&
         (!qbh_attention_u8_qkv_overlap_enabled(
              attention_pipeline_mode) ||
          w4u8_qkvo_pipeline_mode <
              QBH_BLOCK_W4U8_QKVO_BATCH4 ||
          (residual_mode != QBH_BLOCK_RESIDUAL_HVX_FUSED_POST_NORM &&
           residual_mode !=
               QBH_BLOCK_RESIDUAL_HVX_FUSED_POST_NORM_POOL4 &&
           residual_mode !=
               QBH_BLOCK_RESIDUAL_HVX_FUSED_POST_NORM_POOL6 &&
           residual_mode !=
               QBH_BLOCK_RESIDUAL_HVX_FUSED_POST_NORM_POOL6_SHUFFLE4))) ||
        ((residual_mode ==
              QBH_BLOCK_RESIDUAL_HVX_FUSED_POST_NORM_POOL4 ||
          residual_mode ==
              QBH_BLOCK_RESIDUAL_HVX_FUSED_POST_NORM_POOL6 ||
          residual_mode ==
              QBH_BLOCK_RESIDUAL_HVX_FUSED_POST_NORM_POOL6_SHUFFLE4) &&
         (variant != QBH_BLOCK_W4U8 ||
          attention_hvx_contexts < 4U ||
          attention_hvx_contexts > 6U ||
          (crouton_boundary_mode &
           (QBH_BLOCK_CROUTON_BOUNDARY_W4U8_MLP_INPUT |
            QBH_BLOCK_CROUTON_BOUNDARY_W4U8_MLP_OUTPUT |
            QBH_BLOCK_CROUTON_BOUNDARY_W4U8_QKV_INPUT |
            QBH_BLOCK_CROUTON_BOUNDARY_W4U8_O_OUTPUT)) !=
           (QBH_BLOCK_CROUTON_BOUNDARY_W4U8_MLP_INPUT |
            QBH_BLOCK_CROUTON_BOUNDARY_W4U8_MLP_OUTPUT |
            QBH_BLOCK_CROUTON_BOUNDARY_W4U8_QKV_INPUT |
            QBH_BLOCK_CROUTON_BOUNDARY_W4U8_O_OUTPUT))) ||
        ((residual_mode ==
              QBH_BLOCK_RESIDUAL_HVX_FUSED_POST_NORM_POOL6 ||
          residual_mode ==
              QBH_BLOCK_RESIDUAL_HVX_FUSED_POST_NORM_POOL6_SHUFFLE4) &&
         attention_hvx_contexts != 6U) ||
        mlp_mode >
            QBH_BLOCK_MLP_W4U8_STREAMING_PERSISTENT_MLP_HVX ||
        mlp_hvx_contexts == 0U || mlp_hvx_contexts > 4U ||
        (mlp_mode == QBH_BLOCK_MLP_CONTROL && mlp_hvx_contexts != 1U) ||
        (mlp_mode != QBH_BLOCK_MLP_CONTROL &&
         !qbh_block_mlp_is_w4u8_streaming(mlp_mode) &&
         (variant == QBH_BLOCK_W4U8 ||
          (common_ops_mask & QBH_BLOCK_COMMON_OP_SILU) == 0U)) ||
        (qbh_block_mlp_is_w4u8_streaming(mlp_mode) &&
         (variant != QBH_BLOCK_W4U8 || mlp_hvx_contexts != 3U)) ||
        (mlp_mode == QBH_BLOCK_MLP_STREAMING &&
         (mlp_hvx_contexts != 4U ||
          (variant == QBH_BLOCK_F16F16 &&
           f16f16_projection_mode !=
               QBH_BLOCK_F16F16_PROJECTION_BATCH2 &&
           f16f16_projection_mode !=
               QBH_BLOCK_F16F16_PROJECTION_GATE4) ||
          (variant == QBH_BLOCK_W4F16 &&
           w4f16_pipeline_mode !=
               QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_CROSS_PREFETCH &&
           w4f16_pipeline_mode !=
               QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_GATE16_CROSS_PREFETCH &&
           w4f16_pipeline_mode !=
               QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_GATE8_CROSS_PREFETCH &&
           w4f16_pipeline_mode !=
               QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_GATE4_CROSS_PREFETCH &&
           w4f16_pipeline_mode !=
               QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_GATE4_DMA8_CROSS_PREFETCH))) ||
        (mlp_mode == QBH_BLOCK_MLP_CROUTON_NATIVE &&
         (variant != QBH_BLOCK_W4F16 ||
          mlp_hvx_contexts != 4U ||
          w4f16_pipeline_mode !=
              QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_GATE4_DMA8_CROSS_PREFETCH)) ||
        (mlp_mode == QBH_BLOCK_MLP_CROUTON_NATIVE_BATCH8 &&
         (mlp_hvx_contexts != 4U ||
          (variant == QBH_BLOCK_W4F16 &&
           w4f16_pipeline_mode !=
               QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_GATE4_DMA8_CROSS_PREFETCH) ||
          (variant == QBH_BLOCK_F16F16 &&
           f16f16_projection_mode !=
               QBH_BLOCK_F16F16_PROJECTION_GATE8 &&
           f16f16_projection_mode !=
               QBH_BLOCK_F16F16_PROJECTION_GATE8_INTERLEAVED))) ||
        (mlp_chunk_vectors != 16U && mlp_chunk_vectors != 32U &&
         mlp_chunk_vectors != 64U && mlp_chunk_vectors != 128U &&
         mlp_chunk_vectors != 256U) ||
        (variant != QBH_BLOCK_F16F16 &&
         f16f16_projection_mode !=
             QBH_BLOCK_F16F16_PROJECTION_SERIAL) ||
        (variant != QBH_BLOCK_W4F16 &&
         w4f16_pipeline_mode != QBH_BLOCK_W4F16_PIPELINE_CONTROL) ||
        ((w4f16_pipeline_mode ==
              QBH_BLOCK_W4F16_PIPELINE_HYBRID_WORKERS ||
          w4f16_pipeline_mode ==
              QBH_BLOCK_W4F16_PIPELINE_HYBRID_CROSS_PREFETCH ||
          w4f16_pipeline_mode ==
              QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN64_CROSS_PREFETCH ||
          w4f16_pipeline_mode ==
              QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN48_CROSS_PREFETCH ||
          w4f16_pipeline_mode ==
              QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_CROSS_PREFETCH ||
          w4f16_pipeline_mode ==
              QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_GATE16_CROSS_PREFETCH ||
          w4f16_pipeline_mode ==
              QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_GATE8_CROSS_PREFETCH ||
          w4f16_pipeline_mode ==
              QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_GATE4_CROSS_PREFETCH ||
          w4f16_pipeline_mode ==
              QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_GATE4_DMA8_CROSS_PREFETCH) &&
         w4f16_hvx_workers != 3U &&
         w4f16_hvx_workers != 4U) ||
        ((w4f16_pipeline_mode ==
              QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN64_CROSS_PREFETCH ||
          w4f16_pipeline_mode ==
              QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN48_CROSS_PREFETCH ||
          w4f16_pipeline_mode ==
              QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_CROSS_PREFETCH ||
          w4f16_pipeline_mode ==
              QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_GATE16_CROSS_PREFETCH ||
          w4f16_pipeline_mode ==
              QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_GATE8_CROSS_PREFETCH ||
          w4f16_pipeline_mode ==
              QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_GATE4_CROSS_PREFETCH ||
          w4f16_pipeline_mode ==
              QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_GATE4_DMA8_CROSS_PREFETCH) &&
         w4f16_region_tiles != 32U) ||
        (w4f16_region_tiles != 8U && w4f16_region_tiles != 16U &&
         w4f16_region_tiles != 32U && w4f16_region_tiles != 64U) ||
        (w4f16_expand_claim_regions != 1U &&
         w4f16_expand_claim_regions != 2U &&
         w4f16_expand_claim_regions != 3U) ||
        (w4f16_expand_claim_regions != 1U &&
         (variant != QBH_BLOCK_W4F16 ||
          (w4f16_group_fence_mode !=
               QBH_BLOCK_W4F16_GROUP_FENCE_JOIN_ONLY &&
           w4f16_group_fence_mode !=
               QBH_BLOCK_W4F16_GROUP_FENCE_JOIN_ONLY_DOWN))) ||
        w4f16_gate_up_extra_expand_worker > 1U ||
        (w4f16_gate_up_extra_expand_worker != 0U &&
         (variant != QBH_BLOCK_W4F16 ||
          (w4f16_group_fence_mode !=
               QBH_BLOCK_W4F16_GROUP_FENCE_JOIN_ONLY &&
           w4f16_group_fence_mode !=
               QBH_BLOCK_W4F16_GROUP_FENCE_JOIN_ONLY_DOWN) ||
          w4f16_hvx_workers != 4U)) ||
        w4f16_gate_up_extra_stream_worker > 1U ||
        (w4f16_gate_up_extra_stream_worker != 0U &&
         (w4f16_gate_up_extra_expand_worker == 0U ||
          variant != QBH_BLOCK_W4F16 ||
          w4f16_hvx_workers != 4U)) ||
        (w4f16_gate_up_stream_group_tiles != 4U &&
         w4f16_gate_up_stream_group_tiles != 8U) ||
        (w4f16_gate_up_stream_group_tiles != 8U &&
         (w4f16_gate_up_extra_expand_worker == 0U ||
          w4f16_gate_up_extra_stream_worker == 0U ||
          variant != QBH_BLOCK_W4F16 ||
          w4f16_hvx_workers != 4U ||
          mlp_mode != QBH_BLOCK_MLP_CROUTON_NATIVE_BATCH8)) ||
        (w4f16_pipeline_mode == QBH_BLOCK_W4F16_PIPELINE_EARLY_REGION &&
         w4f16_region_tiles > 32U)) {
        fprintf(stderr, "usage: %s PACKAGE_DIR VARIANT [repeat_count] "
                        "[w4f16_hvx_workers] [w4f16_region_tiles] "
                        "[scalar|rms|rope|softmax|silu|rms_silu|"
                        "rms_silu_rope|hvx] [attribution:off|on] "
                        "[audit:off|on] [residual:scalar|hvx|fused] "
                        "[f16_projection:serial|async|batch2|gate4|gate8] "
                        "[w4_pipeline:control|early|hybrid|main_half|"
                        "main_two_thirds|cross|hybrid_cross|"
                        "adaptive_down48_cross|adaptive_down64_cross|"
                        "adaptive_down96_cross|adaptive_down96_gate16_cross|"
                        "adaptive_down96_gate8_cross|"
                        "adaptive_down96_gate4_cross|"
                        "adaptive_down96_gate4_dma8_cross] "
                        "[attention_pack:control|qk_hvx|av_hvx|hvx] "
                        "[mlp:control|parallel_silu|streaming|"
                        "crouton_native|crouton_native_batch8|"
                        "w4u8_streaming|"
                        "w4u8_streaming_persistent_gate_up_hvx|"
                        "w4u8_streaming_persistent_mlp_hvx] "
                        "[mlp_hvx_contexts:1..4] "
                        "[mlp_chunk_vectors:16|32|64|128|256] "
                        "[attention_pipeline:control|parallel_qk_norm_rope|"
                        "parallel_softmax|parallel_hvx|gqa_pipeline|"
                        "gqa_qkv_overlap|u8_log2_gqa|"
                        "u8_log2_gqa_qkv_overlap|"
                        "u8_log2_gqa_qkv_overlap_vgather|"
                        "u8_log2_gqa_qkv_overlap_vgather_vdeal|"
                        "u8_log2_gqa_qkv_overlap_vgather_vdeal_fused_qk_requant] "
                        "[attention_hvx_contexts:1..6] "
                        "[crouton_boundary:control|qkv|av_to_o|"
                        "input_norm|post_norm|norms|qkv_norms|all|"
                        "w4u8_mlp_input|w4u8_mlp_io] "
                        "[w4u8_qkvo_pipeline:serial|qkv_batch2|"
                        "qkv_batch4|qkvo_batch4|"
                        "qkvo_batch4_qk_head_tasks|"
                        "qkvo_batch4_qk_head_pairs] "
                        "[u8_norm_reduction:scalar|hvx_tree|"
                        "hvx_tree_qk_batched_rsqrt|"
                        "hvx_tree_qk_batched_rsqrt_shared_rope] "
                        "[fp16_common_schedule:control|qk_head_pairs|"
                        "input_norm_pool|post_norm_pool|"
                        "input_norm_pool_post_norm_pool|"
                        "qk_head_pairs_input_norm_pool|all] "
                        "[fp16_norm_rows_per_task:2|4|8] "
                        "[fp16_norm_contexts:2|3|4] "
                        "[w4u8_down_hmx_batch_outputs:1|4] "
                        "[w4u8_qk_pair_kernel:0|1]\n",
                argv[0]);
        return 2;
    }
    if (argc < 7 && variant == QBH_BLOCK_W4U8) {
        common_ops_mask = QBH_BLOCK_COMMON_OPS_SCALAR;
    }
    element_bytes = variant == QBH_BLOCK_W4U8 ? 1U : 2U;
    output_bytes = physical_chunks * QBH_BLOCK_M *
                   QBH_BLOCK_HIDDEN * element_bytes;
    if (replay_mode == QBH_BLOCK_REPLAY_CONTINUOUS) {
        cursor = qbh_align_up_size(cursor, QBH_HOST_ALIGNMENT);
        replay_session_offset = (uint32_t)cursor;
        if (cursor > UINT32_MAX ||
            sizeof(struct qbh_decode_session_state) >
                UINT32_MAX - cursor) {
            return 2;
        }
        cursor += sizeof(struct qbh_decode_session_state);
    }
    if (qbh_block_mlp_is_w4u8_streaming(mlp_mode) &&
        (qbh_projection_layout_init(
             QBH_PROJECTION_GATE_UP_PAIR,
             QBH_WEIGHT_PACKED_W4_HMX_SCALE,
             QBH_PHYSICAL_PLAN_STREAMING_E7_DMA_CHAIN4, 8U,
             QBH_W4_COARSE_CHUNK_TILES,
             &w4u8_gate_up_layout) != 0 ||
         qbh_projection_layout_init(
             QBH_PROJECTION_DOWN,
             QBH_WEIGHT_PACKED_W4_HMX_SCALE,
             QBH_PHYSICAL_PLAN_CHUNKED_DMA_BATCH2, 4U,
             QBH_W4_WIDE_CHUNK_TILES,
             &w4u8_down_layout) != 0)) {
        fprintf(stderr, "W4U8 streaming layout initialization failed\n");
        return 2;
    }

    if (qbh_prepare_slot(
            &input_slot, argv[1],
            variant == QBH_BLOCK_W4U8
                ? "reference_w4u8_block_input_u8.bin"
                : "block_input_f16.bin",
            output_bytes, &cursor) != 0 ||
        qbh_prepare_slot(
            &reference_slot, argv[1],
            variant == QBH_BLOCK_F16F16
                ? "reference_f16f16_block_output_f16.bin"
                : (variant == QBH_BLOCK_W4F16
                       ? "reference_w4f16_block_output_f16.bin"
                       : (qbh_attention_u8_enabled(
                              attention_pipeline_mode)
                              ? "reference_w4u8_integer_attention_block_output_u8.bin"
                              : "reference_w4u8_block_output_u8.bin")),
            output_bytes, &cursor) != 0 ||
        qbh_prepare_slot(&rope_slots[0], argv[1], "rope_cos_f16.bin",
                         physical_chunks * QBH_BLOCK_M *
                             QBH_BLOCK_HEAD_DIM *
                             sizeof(uint16_t),
                         &cursor) != 0 ||
        qbh_prepare_slot(&rope_slots[1], argv[1], "rope_sin_f16.bin",
                         physical_chunks * QBH_BLOCK_M *
                             QBH_BLOCK_HEAD_DIM *
                             sizeof(uint16_t),
                         &cursor) != 0) {
        fprintf(stderr, "package common tensor audit failed\n");
        return 2;
    }
    if (generation_mode != QBH_BLOCK_GENERATION_DISABLED &&
        (qbh_prepare_slot(
             &generation_token_slot, argv[1],
             "generation_prompt_token_ids_u32.bin",
             QBH_BLOCK_M * (uint32_t)sizeof(uint32_t), &cursor) != 0 ||
         qbh_prepare_slot(
             &generation_embedding_slot, argv[1],
             qbh_generation_w4u8_enabled(generation_mode)
                 ? "generation_embedding_weight_u8.bin"
                 : "generation_embedding_weight_f16.bin",
             QBH_QWEN3_VOCAB_SIZE * QBH_BLOCK_HIDDEN *
                 (qbh_generation_w4u8_enabled(generation_mode)
                      ? (uint32_t)sizeof(uint8_t)
                      : (uint32_t)sizeof(uint16_t)),
             &cursor) != 0 ||
         qbh_prepare_slot(
             &generation_final_norm_slot, argv[1],
             "generation_final_norm_weight_f16.bin",
             QBH_BLOCK_HIDDEN * (uint32_t)sizeof(uint16_t),
             &cursor) != 0 ||
         qbh_prepare_slot(
             &generation_lm_head_weight_slot, argv[1],
             "generation_lm_head_weight_w4_hmx.bin",
             QBH_QWEN3_VOCAB_SIZE * QBH_BLOCK_HIDDEN / 2U,
             &cursor) != 0 ||
         qbh_prepare_slot(
             &generation_lm_head_scale_slot, argv[1],
             "generation_lm_head_weight_w4_scale_f32.bin",
             QBH_QWEN3_VOCAB_SIZE * (uint32_t)sizeof(float),
             &cursor) != 0 ||
         (qbh_generation_w4u8_enabled(generation_mode) &&
          (qbh_prepare_slot(
               &generation_lm_head_bias_slot, argv[1],
               "generation_lm_head_bias_u32.bin",
               (QBH_QWEN3_VOCAB_SIZE /
                QBH_HMX_OUTPUT_CHANNELS) * QBH_HMX_BIAS_BYTES,
               &cursor) != 0 ||
           qbh_prepare_slot(
               &generation_qparam_slot, argv[1],
               "generation_qparams_u8.bin",
               QBH_GENERATION_QPARAM_COUNT *
                   QBH_BLOCK_QPARAM_RECORD_BYTES,
               &cursor) != 0)) ||
         qbh_prepare_slot(
             &generation_expected_token_slot, argv[1],
             "generation_expected_token_ids_u32.bin",
             QBH_GENERATION_DEFAULT_TOKENS *
                 (uint32_t)sizeof(uint32_t),
             &cursor) != 0)) {
        fprintf(stderr, "generation boundary package audit failed\n");
        return 2;
    }
    if (vertical_slice_mode == QBH_BLOCK_SLICE_ACTIVE_RANGE) {
        for (uint32_t slice_index = 0U;
             slice_index < QBH_VERTICAL_SLICE_LAYER_COUNT;
             ++slice_index) {
            if (qbh_prepare_vertical_layer_slots(
                    &vertical_slots[slice_index], argv[1],
                    QBH_VERTICAL_SLICE_FIRST_LAYER + slice_index,
                    variant, attention_pipeline_mode, mlp_mode,
                    kv_cache_capacity, element_bytes,
                    kv_cache_k_format, kv_cache_v_format,
                    &cursor) != 0) {
                fprintf(stderr,
                        "vertical layer package audit failed: %" PRIu32 "\n",
                        QBH_VERTICAL_SLICE_FIRST_LAYER + slice_index);
                return 2;
            }
        }
    } else if (qbh_prepare_slot(
                   &qparam_slot, argv[1], "qparams_u8.bin",
                   QBH_BLOCK_QPARAM_COUNT *
                       QBH_BLOCK_QPARAM_RECORD_BYTES,
                   &cursor) != 0 ||
               qbh_prepare_slot(&norm_slots[0], argv[1],
                                "input_norm_weight_f16.bin",
                                QBH_BLOCK_HIDDEN * sizeof(uint16_t),
                                &cursor) != 0 ||
               qbh_prepare_slot(&norm_slots[1], argv[1],
                                "post_norm_weight_f16.bin",
                                QBH_BLOCK_HIDDEN * sizeof(uint16_t),
                                &cursor) != 0 ||
               qbh_prepare_slot(&norm_slots[2], argv[1],
                                "q_norm_weight_f16.bin",
                                QBH_BLOCK_HEAD_DIM * sizeof(uint16_t),
                                &cursor) != 0 ||
               qbh_prepare_slot(&norm_slots[3], argv[1],
                                "k_norm_weight_f16.bin",
                                QBH_BLOCK_HEAD_DIM * sizeof(uint16_t),
                                &cursor) != 0) {
        fprintf(stderr, "single-layer metadata audit failed\n");
        return 2;
    }
    if (vertical_slice_mode == QBH_BLOCK_SLICE_DISABLED &&
        qbh_attention_u8_enabled(attention_pipeline_mode) &&
        qbh_prepare_slot(
            &attention_config_slot, argv[1],
            "attention_config_all_groups.bin",
            QBH_BLOCK_ATTENTION_CONFIG_BYTES, &cursor) != 0) {
        fprintf(stderr, "integer Attention config audit failed\n");
        return 2;
    }
    if (qbh_attention_u8_enabled(attention_pipeline_mode) &&
        scan_mode == QBH_BLOCK_SCAN_DISABLED &&
        (qbh_prepare_slot(
             &attention_audit_slots[0], argv[1],
             "reference_w4u8_integer_attention_score_tiles_u8.bin",
             QBH_BLOCK_HEADS * QBH_BLOCK_M * QBH_BLOCK_M,
             &cursor) != 0 ||
         qbh_prepare_slot(
             &attention_audit_slots[1], argv[1],
             "reference_w4u8_integer_attention_probability_tiles_u8.bin",
             QBH_BLOCK_HEADS * QBH_BLOCK_M * QBH_BLOCK_M,
             &cursor) != 0 ||
         qbh_prepare_slot(
             &attention_audit_slots[2], argv[1],
             "reference_w4u8_integer_attention_av_tiles_u8.bin",
             QBH_BLOCK_M * QBH_BLOCK_HIDDEN, &cursor) != 0)) {
        fprintf(stderr, "integer Attention stage-reference audit failed\n");
        return 2;
    }
    if (vertical_slice_mode == QBH_BLOCK_SLICE_DISABLED &&
        scan_mode != QBH_BLOCK_SCAN_DISABLED) {
        const uint32_t cache_bytes[2] = {
            qbh_host_k_cache_bytes(
                variant, kv_cache_capacity, element_bytes,
                kv_cache_k_format),
            qbh_host_v_cache_bytes(
                variant, kv_cache_capacity, element_bytes,
                kv_cache_v_format),
        };
        const char *suffix = variant == QBH_BLOCK_W4U8 ? "u8" : "f16";
        const int hmx_native_u8 = qbh_hmx_native_u8_cache_formats(
            kv_cache_k_format, kv_cache_v_format);
        const int hmx_native_u8_delta =
            qbh_hmx_native_u8_delta_cache_formats(
                kv_cache_k_format, kv_cache_v_format);
        const int hmx_native_u8_segmented =
            qbh_hmx_native_u8_segmented_cache_formats(
                kv_cache_k_format, kv_cache_v_format);
        const int hmx_native_f16 = qbh_hmx_native_f16_cache_formats(
            kv_cache_k_format, kv_cache_v_format);
        const int hmx_native = hmx_native_u8 || hmx_native_f16;
        const char *hmx_suffix = hmx_native_u8_segmented
            ? "u8_segmented"
            : (hmx_native_u8_delta
                   ? "u8_delta" : (hmx_native_u8 ? "u8" : "f16"));
        int status = hmx_native
            ? snprintf(file_name, sizeof(file_name),
                       "kv_cache_k_hmx_%s.bin", hmx_suffix)
            : snprintf(file_name, sizeof(file_name),
                       "kv_cache_k_%s.bin", suffix);
        if (status < 0 || (size_t)status >= sizeof(file_name) ||
            qbh_prepare_slot(
                &kv_cache_slots[0], argv[1], file_name,
                cache_bytes[0], &cursor) != 0) {
            fprintf(stderr, "K cache package audit failed\n");
            return 2;
        }
        status = hmx_native
            ? snprintf(file_name, sizeof(file_name),
                       "kv_cache_v_hmx_%s.bin", hmx_suffix)
            : snprintf(file_name, sizeof(file_name),
                       "kv_cache_v_%s.bin", suffix);
        if (status < 0 || (size_t)status >= sizeof(file_name) ||
            qbh_prepare_slot(
                &kv_cache_slots[1], argv[1], file_name,
                cache_bytes[1], &cursor) != 0) {
            fprintf(stderr, "V cache package audit failed\n");
            return 2;
        }
        status = hmx_native
            ? snprintf(file_name, sizeof(file_name),
                       "reference_kv_cache_k_hmx_%s.bin", hmx_suffix)
            : snprintf(file_name, sizeof(file_name),
                       "reference_kv_cache_k_%s.bin", suffix);
        if (status < 0 || (size_t)status >= sizeof(file_name) ||
            qbh_prepare_slot(
                &kv_reference_slots[0], argv[1], file_name,
                cache_bytes[0], &cursor) != 0) {
            fprintf(stderr, "K cache reference audit failed\n");
            return 2;
        }
        status = hmx_native
            ? snprintf(file_name, sizeof(file_name),
                       "reference_kv_cache_v_hmx_%s.bin", hmx_suffix)
            : snprintf(file_name, sizeof(file_name),
                       "reference_kv_cache_v_%s.bin", suffix);
        if (status < 0 || (size_t)status >= sizeof(file_name) ||
            qbh_prepare_slot(
                &kv_reference_slots[1], argv[1], file_name,
                cache_bytes[1], &cursor) != 0) {
            fprintf(stderr, "V cache reference audit failed\n");
            return 2;
        }
    }
    if (vertical_slice_mode == QBH_BLOCK_SLICE_DISABLED &&
        qbh_block_mlp_is_w4u8_streaming(mlp_mode) &&
        qbh_prepare_slot(&w4u8_lut_slot, argv[1],
                         "silu_up_lut_u16.bin", QBH_MLP_LUT_BYTES,
                         &cursor) != 0) {
        fprintf(stderr, "W4U8 streaming LUT audit failed\n");
        return 2;
    }

    for (uint32_t projection = 0;
         vertical_slice_mode == QBH_BLOCK_SLICE_DISABLED &&
         projection < QBH_BLOCK_PROJECTION_COUNT; ++projection) {
        uint32_t weight_bytes = variant == QBH_BLOCK_F16F16
            ? qbh_projection_k[projection] *
                  qbh_projection_n[projection] * sizeof(uint16_t)
            : qbh_projection_k[projection] *
                  qbh_projection_n[projection] / 2U;
        int name_status = snprintf(
            file_name, sizeof(file_name),
            variant == QBH_BLOCK_F16F16
                ? "%s_weight_f16_hmx.bin"
                : "%s_weight_w4_hmx.bin",
            qbh_projection_names[projection]);
        if (name_status < 0 || (size_t)name_status >= sizeof(file_name) ||
            qbh_prepare_slot(&weight_slots[projection], argv[1],
                             file_name, weight_bytes, &cursor) != 0) {
            fprintf(stderr, "projection weight audit failed: %s\n",
                    qbh_projection_names[projection]);
            return 2;
        }
        if (variant != QBH_BLOCK_F16F16) {
            name_status = snprintf(
                file_name, sizeof(file_name),
                "%s_weight_w4_scale_f32.bin",
                qbh_projection_names[projection]);
            if (name_status < 0 ||
                (size_t)name_status >= sizeof(file_name) ||
                qbh_prepare_slot(
                    &scale_slots[projection], argv[1], file_name,
                    qbh_projection_n[projection] * sizeof(float),
                    &cursor) != 0) {
                fprintf(stderr, "projection scale audit failed: %s\n",
                        qbh_projection_names[projection]);
                return 2;
            }
        } else {
            memset(&scale_slots[projection], 0,
                   sizeof(scale_slots[projection]));
        }
    }
    cursor = qbh_align_up_size(cursor, QBH_HOST_ALIGNMENT);
    if (cursor > UINT32_MAX || output_bytes > UINT32_MAX - cursor) {
        return 2;
    }
    {
        size_t output_offset = cursor;
        cursor += output_bytes;
        if (vertical_slice_mode == QBH_BLOCK_SLICE_DISABLED &&
            variant == QBH_BLOCK_W4U8) {
            for (uint32_t projection = 0;
                 projection < QBH_BLOCK_PROJECTION_COUNT; ++projection) {
                size_t bias_bytes =
                    qbh_projection_n[projection] / 32U *
                    QBH_HMX_BIAS_BYTES;
                cursor = qbh_align_up_size(cursor, QBH_HOST_ALIGNMENT);
                if (cursor > UINT32_MAX ||
                    bias_bytes > UINT32_MAX - cursor) {
                    return 2;
                }
                cursor += bias_bytes;
            }
        }
        if (vertical_slice_mode == QBH_BLOCK_SLICE_DISABLED &&
            qbh_block_mlp_is_w4u8_streaming(mlp_mode)) {
            cursor = qbh_align_up_size(cursor, QBH_HOST_ALIGNMENT);
            w4u8_gate_up_bundle_offset = cursor;
            if (w4u8_gate_up_layout.stored_weight_bytes >
                UINT32_MAX - cursor) {
                return 2;
            }
            cursor += w4u8_gate_up_layout.stored_weight_bytes;
            cursor = qbh_align_up_size(cursor, QBH_HOST_ALIGNMENT);
            w4u8_down_bundle_offset = cursor;
            if (w4u8_down_layout.stored_weight_bytes >
                UINT32_MAX - cursor) {
                return 2;
            }
            cursor += w4u8_down_layout.stored_weight_bytes;
        }
        if (vertical_slice_mode == QBH_BLOCK_SLICE_DISABLED &&
            w4f16_pipeline_mode ==
                QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_GATE4_DMA8_CROSS_PREFETCH) {
            const size_t cache_bytes =
                2U * (QBH_BLOCK_INTERMEDIATE / QBH_HMX_FP16_COLS) *
                QBH_HMX_FP16_SCALE_BYTES;
            cursor = qbh_align_up_size(cursor, QBH_HOST_ALIGNMENT);
            single_gate_up_scale_cache_offset = cursor;
            if (cache_bytes > UINT32_MAX - cursor) {
                return 2;
            }
            cursor += cache_bytes;
        }
        if (vertical_slice_mode == QBH_BLOCK_SLICE_ACTIVE_RANGE) {
            for (uint32_t slice_index = 0U;
                 slice_index < QBH_VERTICAL_SLICE_LAYER_COUNT;
                 ++slice_index) {
                if (variant == QBH_BLOCK_W4U8) {
                    for (uint32_t projection = 0U;
                         projection < QBH_BLOCK_PROJECTION_COUNT;
                         ++projection) {
                        const size_t bias_bytes =
                            qbh_projection_n[projection] / 32U *
                            QBH_HMX_BIAS_BYTES;
                        cursor = qbh_align_up_size(
                            cursor, QBH_HOST_ALIGNMENT);
                        vertical_bias_offsets[slice_index][projection] =
                            cursor;
                        if (bias_bytes > UINT32_MAX - cursor) {
                            return 2;
                        }
                        cursor += bias_bytes;
                    }
                }
                if (qbh_block_mlp_is_w4u8_streaming(mlp_mode)) {
                    cursor = qbh_align_up_size(
                        cursor, QBH_HOST_ALIGNMENT);
                    vertical_slots[slice_index].gate_up_bundle_offset =
                        cursor;
                    if (w4u8_gate_up_layout.stored_weight_bytes >
                        UINT32_MAX - cursor) {
                        return 2;
                    }
                    cursor += w4u8_gate_up_layout.stored_weight_bytes;
                    cursor = qbh_align_up_size(
                        cursor, QBH_HOST_ALIGNMENT);
                    vertical_slots[slice_index].down_bundle_offset = cursor;
                    if (w4u8_down_layout.stored_weight_bytes >
                        UINT32_MAX - cursor) {
                        return 2;
                    }
                    cursor += w4u8_down_layout.stored_weight_bytes;
                }
                if (w4f16_pipeline_mode ==
                    QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_GATE4_DMA8_CROSS_PREFETCH) {
                    const size_t cache_bytes =
                        2U *
                        (QBH_BLOCK_INTERMEDIATE /
                         QBH_HMX_FP16_COLS) *
                        QBH_HMX_FP16_SCALE_BYTES;
                    cursor = qbh_align_up_size(
                        cursor, QBH_HOST_ALIGNMENT);
                    vertical_slots[slice_index]
                        .gate_up_scale_cache_offset = cursor;
                    if (cache_bytes > UINT32_MAX - cursor) {
                        return 2;
                    }
                    cursor += cache_bytes;
                }
            }
        }
        if (vertical_slice_mode == QBH_BLOCK_SLICE_DISABLED &&
            qbh_attention_u8_enabled(attention_pipeline_mode) &&
            numerical_audit_enabled != 0U) {
            cursor = qbh_align_up_size(cursor, QBH_HOST_ALIGNMENT);
            attention_audit_output_offset = cursor;
            if (QBH_BLOCK_U8_ATTENTION_AUDIT_BYTES >
                UINT32_MAX - cursor) {
                return 2;
            }
            cursor += QBH_BLOCK_U8_ATTENTION_AUDIT_BYTES;
        }
        if (w4u8_boundary_audit_enabled != 0U) {
            const size_t boundary_bytes =
                (size_t)QBH_BLOCK_M * QBH_BLOCK_HIDDEN;
            cursor = qbh_align_up_size(cursor, QBH_HOST_ALIGNMENT);
            w4u8_boundary_audit_output_offset = cursor;
            if (boundary_bytes > UINT32_MAX - cursor) {
                return 2;
            }
            cursor += boundary_bytes;
        }
        if (vertical_slice_mode == QBH_BLOCK_SLICE_DISABLED &&
            scan_mode != QBH_BLOCK_SCAN_DISABLED &&
            variant != QBH_BLOCK_W4U8 &&
            numerical_audit_enabled != 0U) {
            cursor = qbh_align_up_size(cursor, QBH_HOST_ALIGNMENT);
            scan_attention_audit_output_offset = cursor;
            if (QBH_BLOCK_SCAN_F16_AUDIT_BYTES > UINT32_MAX - cursor) {
                return 2;
            }
            cursor += QBH_BLOCK_SCAN_F16_AUDIT_BYTES;
        }
        if (full_stack_stage_mode ==
            QBH_BLOCK_FULL_STACK_HIDDEN_CAPTURE) {
            cursor = qbh_align_up_size(cursor, QBH_HOST_ALIGNMENT);
            full_stack_hidden_capture_offset = cursor;
            full_stack_hidden_capture_bytes =
                (size_t)QBH_VERTICAL_SLICE_LAYER_COUNT * output_bytes;
            if (cursor > UINT32_MAX ||
                full_stack_hidden_capture_bytes >
                    UINT32_MAX - cursor) {
                return 2;
            }
            cursor += full_stack_hidden_capture_bytes;
        }
        total_bytes = cursor;
        if (total_bytes > UINT32_MAX) {
            fprintf(stderr, "32-bit rpcmem package too large: %zu\n",
                    total_bytes);
            return 2;
        }
        {
            const char *layout_only = getenv("QBH_LAYOUT_ONLY");
            if (layout_only != NULL && strcmp(layout_only, "1") == 0) {
                printf(
                    "{\"experiment\":163,\"layout_only\":true,"
                    "\"variant\":\"%s\",\"layer_first\":%" PRIu32
                    ",\"layer_count\":%" PRIu32
                    ",\"shared_bytes\":%zu,"
                    "\"hidden_capture_bytes\":%zu,"
                    "\"uint32_fit\":true}\n",
                    qbh_variant_name(variant),
                    QBH_VERTICAL_SLICE_FIRST_LAYER,
                    QBH_VERTICAL_SLICE_LAYER_COUNT, total_bytes,
                    full_stack_hidden_capture_bytes);
                return 0;
            }
        }
        if (rpcmem_alloc2 == NULL) {
            fprintf(stderr, "rpcmem_alloc2 symbol is unavailable\n");
            return 2;
        }
        shared = rpcmem_alloc2(RPCMEM_HEAP_ID_SYSTEM,
                               RPCMEM_FLAG_UNCACHED, total_bytes);
        if (shared == NULL) {
            fprintf(stderr, "rpcmem_alloc2 failed for %zu bytes\n",
                    total_bytes);
            return 2;
        }
        memset(shared, 0, total_bytes);
        header = (struct qbh_block_header *)shared;
        header->output_offset = (uint32_t)output_offset;
    }

    if (qbh_read_slot(shared, &input_slot) != 0 ||
        qbh_read_slot(shared, &reference_slot) != 0) {
        fprintf(stderr, "package boundary tensor read failed\n");
        goto cleanup;
    }
    if (generation_mode != QBH_BLOCK_GENERATION_DISABLED &&
        (qbh_read_slot(shared, &generation_token_slot) != 0 ||
         qbh_read_slot(shared, &generation_embedding_slot) != 0 ||
         qbh_read_slot(shared, &generation_final_norm_slot) != 0 ||
         qbh_read_slot(shared, &generation_lm_head_weight_slot) != 0 ||
         qbh_read_slot(shared, &generation_lm_head_scale_slot) != 0 ||
         (qbh_generation_w4u8_enabled(generation_mode) &&
          (qbh_read_slot(shared, &generation_lm_head_bias_slot) != 0 ||
           qbh_read_slot(shared, &generation_qparam_slot) != 0)) ||
         qbh_read_slot(shared, &generation_expected_token_slot) != 0)) {
        fprintf(stderr, "generation boundary tensor read failed\n");
        goto cleanup;
    }
    for (uint32_t index = 0; index < 2U; ++index) {
        if (qbh_read_slot(shared, &rope_slots[index]) != 0) {
            goto cleanup;
        }
    }
    if (vertical_slice_mode == QBH_BLOCK_SLICE_ACTIVE_RANGE) {
        for (uint32_t slice_index = 0U;
             slice_index < QBH_VERTICAL_SLICE_LAYER_COUNT;
             ++slice_index) {
            if (qbh_read_vertical_layer_slots(
                    shared, &vertical_slots[slice_index], variant,
                    attention_pipeline_mode, mlp_mode) != 0) {
                fprintf(stderr,
                        "vertical layer package read failed: %" PRIu32 "\n",
                        QBH_VERTICAL_SLICE_FIRST_LAYER + slice_index);
                goto cleanup;
            }
        }
    } else {
        if (qbh_read_slot(shared, &qparam_slot) != 0) {
            goto cleanup;
        }
        for (uint32_t index = 0; index < 4U; ++index) {
            if (qbh_read_slot(shared, &norm_slots[index]) != 0) {
                goto cleanup;
            }
        }
        if (qbh_block_mlp_is_w4u8_streaming(mlp_mode) &&
            qbh_read_slot(shared, &w4u8_lut_slot) != 0) {
            goto cleanup;
        }
        if (qbh_attention_u8_enabled(attention_pipeline_mode) &&
            qbh_read_slot(shared, &attention_config_slot) != 0) {
            goto cleanup;
        }
    }
    if (qbh_attention_u8_enabled(attention_pipeline_mode) &&
        scan_mode == QBH_BLOCK_SCAN_DISABLED &&
        (qbh_read_slot(shared, &attention_audit_slots[0]) != 0 ||
         qbh_read_slot(shared, &attention_audit_slots[1]) != 0 ||
         qbh_read_slot(shared, &attention_audit_slots[2]) != 0)) {
        goto cleanup;
    }
    if (vertical_slice_mode == QBH_BLOCK_SLICE_DISABLED &&
        scan_mode != QBH_BLOCK_SCAN_DISABLED &&
        (qbh_read_slot(shared, &kv_cache_slots[0]) != 0 ||
         qbh_read_slot(shared, &kv_cache_slots[1]) != 0 ||
         qbh_read_slot(shared, &kv_reference_slots[0]) != 0 ||
         qbh_read_slot(shared, &kv_reference_slots[1]) != 0)) {
        goto cleanup;
    }
    for (uint32_t projection = 0;
         vertical_slice_mode == QBH_BLOCK_SLICE_DISABLED &&
         projection < QBH_BLOCK_PROJECTION_COUNT; ++projection) {
        if (qbh_read_slot(shared, &weight_slots[projection]) != 0 ||
            (variant != QBH_BLOCK_F16F16 &&
             qbh_read_slot(shared, &scale_slots[projection]) != 0)) {
            goto cleanup;
        }
    }

    header->magic = QBH_BLOCK_MAGIC;
    header->abi_version = QBH_BLOCK_ABI_VERSION;
    header->experiment = QBH_BLOCK_EXPERIMENT;
    header->header_bytes = (uint32_t)sizeof(*header);
    header->shared_bytes = (uint32_t)total_bytes;
    header->variant = variant;
    header->repeat_count = repeats;
    header->w4f16_requested_hvx_workers = w4f16_hvx_workers;
    header->w4f16_region_tiles = w4f16_region_tiles;
    header->common_ops_mask = common_ops_mask;
    header->attribution_enabled = attribution_enabled;
    header->numerical_audit_enabled = numerical_audit_enabled;
    header->residual_mode = residual_mode;
    header->f16f16_projection_mode = f16f16_projection_mode;
    header->w4f16_pipeline_mode = w4f16_pipeline_mode;
    header->attention_pack_mode = attention_pack_mode;
    header->attention_pipeline_mode = attention_pipeline_mode;
    header->attention_hvx_contexts = attention_hvx_contexts;
    header->mlp_mode = mlp_mode;
    header->mlp_hvx_contexts = mlp_hvx_contexts;
    header->mlp_chunk_vectors = mlp_chunk_vectors;
    header->crouton_boundary_mode = crouton_boundary_mode;
    header->w4u8_qkvo_pipeline_mode = w4u8_qkvo_pipeline_mode;
    header->u8_norm_reduction_mode = u8_norm_reduction_mode;
    header->fp16_common_schedule_mode = fp16_common_schedule_mode;
    header->fp16_norm_rows_per_task = fp16_norm_rows_per_task;
    header->fp16_norm_contexts = fp16_norm_contexts;
    header->w4u8_down_hmx_batch_outputs =
        w4u8_down_hmx_batch_outputs;
    header->qkv_schedule_mode = qkv_schedule_mode;
    header->w4f16_group_fence_mode = w4f16_group_fence_mode;
    header->w4f16_expand_claim_regions =
        w4f16_expand_claim_regions;
    header->w4f16_gate_up_extra_expand_worker =
        w4f16_gate_up_extra_expand_worker;
    header->w4f16_gate_up_extra_stream_worker =
        w4f16_gate_up_extra_stream_worker;
    header->w4f16_gate_up_stream_group_tiles =
        w4f16_gate_up_stream_group_tiles;
    header->w4u8_stream_fence_mode = w4u8_stream_fence_mode;
    header->w4u8_gate_up_ring_slots = w4u8_gate_up_ring_slots;
    header->w4u8_qkv_ring_expand_workers =
        w4u8_qkv_ring_expand_workers;
    header->w4u8_prefill_cache_mode =
        w4u8_prefill_cache_mode;
    header->w4u8_delta_reconstruction_mode =
        w4u8_delta_reconstruction_mode;
    header->w4u8_decode_softmax_mode =
        w4u8_decode_softmax_mode;
    header->w4u8_decode_lm_head_group_tiles =
        w4u8_decode_lm_head_group_tiles;
    header->w4u8_decode_o_batch_n_tiles =
        w4u8_decode_o_batch_n_tiles;
    header->w4u8_decode_av_requant_rows =
        w4u8_decode_av_requant_rows;
    header->w4u8_decode_av_padding_poison =
        w4u8_decode_av_padding_poison;
    header->w4u8_decode_common_op_rows =
        w4u8_decode_common_op_rows;
    header->w4u8_decode_common_padding_poison =
        w4u8_decode_common_padding_poison;
    header->w4u8_decode_qk_norm_rope_rows =
        w4u8_decode_qk_norm_rope_rows;
    header->w4u8_decode_qk_padding_poison =
        w4u8_decode_qk_padding_poison;
    header->w4u8_qk_pair_kernel_mode =
        w4u8_qk_pair_kernel_mode;
    header->scan_mode = scan_mode;
    header->logical_m = logical_m;
    header->initial_kv_length = initial_kv_length;
    header->kv_cache_capacity = kv_cache_capacity;
    header->kv_cache_k_format = scan_mode == QBH_BLOCK_SCAN_DISABLED
        ? QBH_KV_CACHE_FORMAT_NONE : kv_cache_k_format;
    header->kv_cache_v_format = scan_mode == QBH_BLOCK_SCAN_DISABLED
        ? QBH_KV_CACHE_FORMAT_NONE : kv_cache_v_format;
    header->kv_cache_padded_capacity =
        scan_mode == QBH_BLOCK_SCAN_DISABLED
            ? 0U
            : (qbh_hmx_native_cache_formats(
                   kv_cache_k_format, kv_cache_v_format)
                   ? QBH_KV_CACHE_HMX_PADDED_CAPACITY(
                         kv_cache_capacity)
                   : kv_cache_capacity);
    header->replay_mode = replay_mode;
    header->replay_session_offset = replay_session_offset;
    header->replay_session_bytes =
        replay_mode == QBH_BLOCK_REPLAY_CONTINUOUS
            ? (uint32_t)sizeof(struct qbh_decode_session_state) : 0U;
    header->replay_expected_step = 0U;
    header->replay_first_position = 0U;
    header->slice_mode = vertical_slice_mode;
    header->slice_first_layer =
        vertical_slice_mode == QBH_BLOCK_SLICE_ACTIVE_RANGE
            ? QBH_VERTICAL_SLICE_FIRST_LAYER : 0U;
    header->slice_layer_count =
        vertical_slice_mode == QBH_BLOCK_SLICE_ACTIVE_RANGE
            ? QBH_VERTICAL_SLICE_LAYER_COUNT : 0U;
    header->full_stack_stage_mode = full_stack_stage_mode;
    header->full_stack_hidden_capture_offset =
        (uint32_t)full_stack_hidden_capture_offset;
    header->full_stack_hidden_capture_bytes =
        (uint32_t)full_stack_hidden_capture_bytes;
    header->full_stack_hidden_capture_layer_bytes =
        full_stack_stage_mode == QBH_BLOCK_FULL_STACK_HIDDEN_CAPTURE
            ? output_bytes : 0U;
    header->w4f16_gate_up_scale_cache_offset =
        (uint32_t)single_gate_up_scale_cache_offset;
    header->w4f16_gate_up_scale_cache_bytes =
        w4f16_pipeline_mode ==
                QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_GATE4_DMA8_CROSS_PREFETCH
            ? 2U *
                  (QBH_BLOCK_INTERMEDIATE / QBH_HMX_FP16_COLS) *
                  QBH_HMX_FP16_SCALE_BYTES
            : 0U;
    header->input_offset = input_slot.offset;
    header->input_bytes = input_slot.expected_bytes;
    header->output_bytes = output_bytes;
    header->reference_offset = reference_slot.offset;
    header->reference_bytes = reference_slot.expected_bytes;
    header->generation_mode = generation_mode;
    if (generation_mode != QBH_BLOCK_GENERATION_DISABLED) {
        header->generation_token_ids_offset =
            generation_token_slot.offset;
        header->generation_token_ids_bytes =
            generation_token_slot.expected_bytes;
        header->generation_token_count = QBH_BLOCK_M;
        header->generation_embedding_offset =
            generation_embedding_slot.offset;
        header->generation_embedding_bytes =
            generation_embedding_slot.expected_bytes;
        header->generation_final_norm_offset =
            generation_final_norm_slot.offset;
        header->generation_final_norm_bytes =
            generation_final_norm_slot.expected_bytes;
        header->generation_lm_head.k = QBH_BLOCK_HIDDEN;
        header->generation_lm_head.n = QBH_QWEN3_VOCAB_SIZE;
        header->generation_lm_head.weight_offset =
            generation_lm_head_weight_slot.offset;
        header->generation_lm_head.weight_bytes =
            generation_lm_head_weight_slot.expected_bytes;
        header->generation_lm_head.scale_offset =
            generation_lm_head_scale_slot.offset;
        header->generation_lm_head.scale_bytes =
            generation_lm_head_scale_slot.expected_bytes;
        if (qbh_generation_w4u8_enabled(generation_mode)) {
            header->generation_lm_head.bias_offset =
                generation_lm_head_bias_slot.offset;
            header->generation_lm_head.bias_bytes =
                generation_lm_head_bias_slot.expected_bytes;
        }
        header->generation_boundary_audit_enabled =
            generation_boundary_audit_enabled;
        header->generation_expected_token_ids_offset =
            generation_expected_token_slot.offset;
        header->generation_expected_token_ids_bytes =
            generation_expected_token_slot.expected_bytes;
        header->generation_expected_token_count =
            QBH_GENERATION_DEFAULT_TOKENS;
    }
    if (vertical_slice_mode == QBH_BLOCK_SLICE_DISABLED) {
        header->qparam_offset = qparam_slot.offset;
        header->qparam_bytes = qparam_slot.expected_bytes;
        header->input_norm_weight_offset = norm_slots[0].offset;
        header->input_norm_weight_bytes = norm_slots[0].expected_bytes;
        header->post_norm_weight_offset = norm_slots[1].offset;
        header->post_norm_weight_bytes = norm_slots[1].expected_bytes;
        header->q_norm_weight_offset = norm_slots[2].offset;
        header->q_norm_weight_bytes = norm_slots[2].expected_bytes;
        header->k_norm_weight_offset = norm_slots[3].offset;
        header->k_norm_weight_bytes = norm_slots[3].expected_bytes;
    }
    header->rope_cos_offset = rope_slots[0].offset;
    header->rope_cos_bytes = rope_slots[0].expected_bytes;
    header->rope_sin_offset = rope_slots[1].offset;
    header->rope_sin_bytes = rope_slots[1].expected_bytes;
    if (vertical_slice_mode == QBH_BLOCK_SLICE_DISABLED &&
        scan_mode != QBH_BLOCK_SCAN_DISABLED) {
        header->kv_cache_k_offset = kv_cache_slots[0].offset;
        header->kv_cache_k_bytes = kv_cache_slots[0].expected_bytes;
        header->kv_cache_v_offset = kv_cache_slots[1].offset;
        header->kv_cache_v_bytes = kv_cache_slots[1].expected_bytes;
    }
    if (replay_mode == QBH_BLOCK_REPLAY_CONTINUOUS) {
        struct qbh_decode_session_state *state =
            (struct qbh_decode_session_state *)(
                shared + replay_session_offset);
        memset(state, 0, sizeof(*state));
        state->magic = QBH_DECODE_SESSION_MAGIC;
        state->abi_version = QBH_DECODE_SESSION_ABI_VERSION;
        state->state_bytes = (uint32_t)sizeof(*state);
        state->declared_layer_count = QBH_QWEN3_TRANSFORMER_LAYERS;
        state->active_layer = QBH_VERTICAL_SLICE_FIRST_LAYER;
        state->active_layer_count = QBH_VERTICAL_SLICE_LAYER_COUNT;
        for (uint32_t index = 0U;
             index < QBH_QWEN3_TRANSFORMER_LAYERS; ++index) {
            state->layers[index].layer_index = index;
        }
        for (uint32_t slice_index = 0U;
             slice_index < QBH_VERTICAL_SLICE_LAYER_COUNT;
             ++slice_index) {
            const uint32_t layer_index =
                QBH_VERTICAL_SLICE_FIRST_LAYER + slice_index;
            struct qbh_decode_layer_state *layer =
                &state->layers[layer_index];
            const struct qbh_vertical_layer_slots *slots =
                &vertical_slots[slice_index];
            layer->element_type = variant == QBH_BLOCK_W4U8
                ? QBH_KV_CACHE_ELEMENT_U8 : QBH_KV_CACHE_ELEMENT_F16;
            layer->k_format = kv_cache_k_format;
            layer->v_format = kv_cache_v_format;
            layer->capacity = kv_cache_capacity;
            layer->k_offset = slots->caches[0].offset;
            layer->k_bytes = slots->caches[0].expected_bytes;
            layer->v_offset = slots->caches[1].offset;
            layer->v_bytes = slots->caches[1].expected_bytes;
            layer->head_count = QBH_BLOCK_KV_HEADS;
            layer->head_dim = QBH_BLOCK_HEAD_DIM;
            if (qbh_hmx_native_cache_formats(
                    kv_cache_k_format, kv_cache_v_format)) {
                const int native_u8 = qbh_hmx_native_u8_cache_formats(
                    kv_cache_k_format, kv_cache_v_format);
                const int native_u8_delta =
                    qbh_hmx_native_u8_delta_cache_formats(
                        kv_cache_k_format, kv_cache_v_format);
                const int native_u8_segmented =
                    qbh_hmx_native_u8_segmented_cache_formats(
                        kv_cache_k_format, kv_cache_v_format);
                layer->padded_capacity =
                    QBH_KV_CACHE_HMX_PADDED_CAPACITY(
                        kv_cache_capacity);
                layer->k_head_stride_bytes = native_u8
                    ? (native_u8_segmented
                           ? QBH_KV_CACHE_HMX_U8_K_SEGMENTED_HEAD_BYTES(
                                 kv_cache_capacity)
                           : (native_u8_delta
                           ? QBH_KV_CACHE_HMX_U8_K_DELTA_HEAD_BYTES(
                                 kv_cache_capacity)
                           : QBH_KV_CACHE_HMX_K_HEAD_BYTES(
                                 kv_cache_capacity)))
                    : QBH_KV_CACHE_HMX_F16_K_HEAD_BYTES(kv_cache_capacity);
                layer->v_head_stride_bytes = native_u8
                    ? (native_u8_segmented
                           ? QBH_KV_CACHE_HMX_U8_V_SEGMENTED_HEAD_BYTES(
                                 kv_cache_capacity)
                           : (native_u8_delta
                           ? QBH_KV_CACHE_HMX_U8_V_DELTA_HEAD_BYTES(
                                 kv_cache_capacity)
                           : QBH_KV_CACHE_HMX_V_HEAD_BYTES(
                                 kv_cache_capacity)))
                    : QBH_KV_CACHE_HMX_F16_V_HEAD_BYTES(kv_cache_capacity);
                layer->head_stride_bytes =
                    layer->k_head_stride_bytes;
                layer->token_stride_bytes = 0U;
                layer->k_weight_bytes_per_head = native_u8
                    ? (native_u8_segmented
                           ? QBH_KV_CACHE_HMX_U8_SEGMENT_COUNT(
                                 kv_cache_capacity) *
                                 QBH_KV_CACHE_HMX_U8_SEGMENT_K_BYTES
                           : (native_u8_delta
                           ? QBH_KV_CACHE_HMX_U8_BASE_WEIGHT_BYTES_PER_HEAD
                           : QBH_KV_CACHE_HMX_WEIGHT_BYTES_PER_HEAD(
                                 kv_cache_capacity)))
                    : QBH_KV_CACHE_HMX_F16_WEIGHT_BYTES_PER_HEAD(
                          kv_cache_capacity);
                layer->v_weight_bytes_per_head =
                    layer->k_weight_bytes_per_head;
                layer->k_bias_bytes_per_head = native_u8
                    ? (native_u8_segmented
                           ? 0U
                           : (native_u8_delta
                           ? QBH_KV_CACHE_HMX_U8_K_BASE_BIAS_BYTES_PER_HEAD
                           : QBH_KV_CACHE_HMX_K_BIAS_BYTES_PER_HEAD(
                                 kv_cache_capacity)))
                    : 0U;
                layer->v_bias_bytes_per_head = native_u8
                    ? QBH_KV_CACHE_HMX_V_BIAS_BYTES_PER_HEAD
                    : 0U;
            } else {
                layer->padded_capacity = kv_cache_capacity;
                layer->k_head_stride_bytes =
                    kv_cache_capacity * QBH_BLOCK_HEAD_DIM *
                    element_bytes;
                layer->v_head_stride_bytes =
                    layer->k_head_stride_bytes;
                layer->head_stride_bytes =
                    layer->k_head_stride_bytes;
                layer->token_stride_bytes =
                    QBH_BLOCK_HEAD_DIM * element_bytes;
            }
        }
    }
    if (vertical_slice_mode == QBH_BLOCK_SLICE_DISABLED &&
        qbh_attention_u8_enabled(attention_pipeline_mode)) {
        header->attention_config_offset =
            attention_config_slot.offset;
        header->attention_config_bytes =
            attention_config_slot.expected_bytes;
        /* The package stage tensors originate from the offline W4U8
         * projection/Norm path, while this candidate deliberately uses the
         * device's native integer projection and HVX Norm/RoPE outputs.
         * Comparing their hashes here conflates upstream quantization error
         * with Attention-core correctness.  Audit mode exports the actual
         * Q/K/V boundary and all three stages; the independent host verifier
         * recomputes QK, Softmax, and AV from those captured inputs. */
        header->u8_attention_expected_score_hash = 0U;
        header->u8_attention_expected_probability_hash = 0U;
        header->u8_attention_expected_av_hash = 0U;
        if (numerical_audit_enabled != 0U) {
            header->u8_attention_audit_output_offset =
                (uint32_t)attention_audit_output_offset;
            header->u8_attention_audit_output_bytes =
                QBH_BLOCK_U8_ATTENTION_AUDIT_BYTES;
        }
    }
    if (vertical_slice_mode == QBH_BLOCK_SLICE_DISABLED &&
        scan_mode != QBH_BLOCK_SCAN_DISABLED &&
        variant != QBH_BLOCK_W4U8 &&
        numerical_audit_enabled != 0U) {
        header->scan_attention_audit_output_offset =
            (uint32_t)scan_attention_audit_output_offset;
        header->scan_attention_audit_output_bytes =
            QBH_BLOCK_SCAN_F16_AUDIT_BYTES;
    }
    header->w4u8_boundary_audit_enabled =
        w4u8_boundary_audit_enabled;
    if (w4u8_boundary_audit_enabled != 0U) {
        header->w4u8_boundary_audit_output_offset =
            (uint32_t)w4u8_boundary_audit_output_offset;
        header->w4u8_boundary_audit_output_bytes =
            QBH_BLOCK_M * QBH_BLOCK_HIDDEN;
    }
    if (vertical_slice_mode == QBH_BLOCK_SLICE_DISABLED &&
        qbh_block_mlp_is_w4u8_streaming(mlp_mode)) {
        header->w4u8_gate_up_bundle_offset =
            (uint32_t)w4u8_gate_up_bundle_offset;
        header->w4u8_gate_up_bundle_bytes =
            w4u8_gate_up_layout.stored_weight_bytes;
        header->w4u8_down_bundle_offset =
            (uint32_t)w4u8_down_bundle_offset;
        header->w4u8_down_bundle_bytes =
            w4u8_down_layout.stored_weight_bytes;
        header->w4u8_silu_lut_offset = w4u8_lut_slot.offset;
        header->w4u8_silu_lut_bytes = w4u8_lut_slot.expected_bytes;
    }
    if (vertical_slice_mode == QBH_BLOCK_SLICE_DISABLED &&
        qbh_load_qparams(header, shared + qparam_slot.offset) != 0) {
        fprintf(stderr, "qparam record audit failed\n");
        goto cleanup;
    }

    if (vertical_slice_mode == QBH_BLOCK_SLICE_ACTIVE_RANGE) {
        for (uint32_t slice_index = 0U;
             slice_index < QBH_VERTICAL_SLICE_LAYER_COUNT;
             ++slice_index) {
            struct qbh_block_layer_desc *layer =
                &header->slice_layers[slice_index];
            const struct qbh_vertical_layer_slots *slots =
                &vertical_slots[slice_index];
            layer->layer_index =
                QBH_VERTICAL_SLICE_FIRST_LAYER + slice_index;
            layer->qparam_offset = slots->qparam.offset;
            layer->qparam_bytes = slots->qparam.expected_bytes;
            layer->input_norm_weight_offset = slots->norms[0].offset;
            layer->input_norm_weight_bytes = slots->norms[0].expected_bytes;
            layer->post_norm_weight_offset = slots->norms[1].offset;
            layer->post_norm_weight_bytes = slots->norms[1].expected_bytes;
            layer->q_norm_weight_offset = slots->norms[2].offset;
            layer->q_norm_weight_bytes = slots->norms[2].expected_bytes;
            layer->k_norm_weight_offset = slots->norms[3].offset;
            layer->k_norm_weight_bytes = slots->norms[3].expected_bytes;
            layer->attention_config_offset =
                slots->attention_config.offset;
            layer->attention_config_bytes =
                slots->attention_config.expected_bytes;
            layer->kv_cache_k_format = kv_cache_k_format;
            layer->kv_cache_v_format = kv_cache_v_format;
            layer->kv_cache_padded_capacity =
                qbh_hmx_native_cache_formats(
                    kv_cache_k_format, kv_cache_v_format)
                    ? QBH_KV_CACHE_HMX_PADDED_CAPACITY(
                          kv_cache_capacity)
                    : kv_cache_capacity;
            layer->kv_cache_k_offset = slots->caches[0].offset;
            layer->kv_cache_k_bytes = slots->caches[0].expected_bytes;
            layer->kv_cache_v_offset = slots->caches[1].offset;
            layer->kv_cache_v_bytes = slots->caches[1].expected_bytes;
            layer->w4u8_silu_lut_offset = slots->silu_lut.offset;
            layer->w4u8_silu_lut_bytes = slots->silu_lut.expected_bytes;
            if (qbh_block_mlp_is_w4u8_streaming(mlp_mode)) {
                layer->w4u8_gate_up_bundle_offset =
                    (uint32_t)slots->gate_up_bundle_offset;
                layer->w4u8_gate_up_bundle_bytes =
                    w4u8_gate_up_layout.stored_weight_bytes;
                layer->w4u8_down_bundle_offset =
                    (uint32_t)slots->down_bundle_offset;
                layer->w4u8_down_bundle_bytes =
                    w4u8_down_layout.stored_weight_bytes;
            }
            if (w4f16_pipeline_mode ==
                QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_GATE4_DMA8_CROSS_PREFETCH) {
                layer->w4f16_gate_up_scale_cache_offset =
                    (uint32_t)slots->gate_up_scale_cache_offset;
                layer->w4f16_gate_up_scale_cache_bytes =
                    2U *
                    (QBH_BLOCK_INTERMEDIATE /
                     QBH_HMX_FP16_COLS) *
                    QBH_HMX_FP16_SCALE_BYTES;
            }
            if (qbh_load_qparams_array(
                    layer->qparams,
                    shared + slots->qparam.offset) != 0) {
                fprintf(stderr,
                        "layer %" PRIu32 " qparam audit failed\n",
                        layer->layer_index);
                goto cleanup;
            }
            for (uint32_t projection = 0U;
                 projection < QBH_BLOCK_PROJECTION_COUNT;
                 ++projection) {
                struct qbh_block_projection_desc *desc =
                    &layer->projections[projection];
                desc->k = qbh_projection_k[projection];
                desc->n = qbh_projection_n[projection];
                desc->weight_offset = slots->weights[projection].offset;
                desc->weight_bytes =
                    slots->weights[projection].expected_bytes;
                if (variant != QBH_BLOCK_F16F16) {
                    desc->scale_offset = slots->scales[projection].offset;
                    desc->scale_bytes =
                        slots->scales[projection].expected_bytes;
                }
                if (variant == QBH_BLOCK_W4U8) {
                    desc->bias_offset = (uint32_t)
                        vertical_bias_offsets[slice_index][projection];
                    desc->bias_bytes =
                        desc->n / 32U * QBH_HMX_BIAS_BYTES;
                }
            }
            qbh_bind_host_slice_layer(header, slice_index);
            if (variant == QBH_BLOCK_W4U8) {
                for (uint32_t projection = 0U;
                     projection < QBH_BLOCK_PROJECTION_COUNT;
                     ++projection) {
                    if (qbh_build_bias_words(
                            header, shared, projection) != 0) {
                        fprintf(stderr,
                                "layer %" PRIu32
                                " bias generation failed: %s\n",
                                layer->layer_index,
                                qbh_projection_names[projection]);
                        goto cleanup;
                    }
                }
            }
            if (qbh_block_mlp_is_w4u8_streaming(mlp_mode) &&
                qbh_build_w4u8_streaming_bundles(
                    header, shared, &w4u8_gate_up_layout,
                    &w4u8_down_layout) != 0) {
                fprintf(stderr,
                        "layer %" PRIu32
                        " W4U8 bundle construction failed\n",
                        layer->layer_index);
                goto cleanup;
            }
            if (qbh_prepare_gate_up_scale_cache(header, shared) != 0) {
                fprintf(stderr,
                        "layer %" PRIu32
                        " Gate/Up scale-cache preparation failed\n",
                        layer->layer_index);
                goto cleanup;
            }
        }
        if (variant == QBH_BLOCK_W4U8) {
            for (uint32_t slice_index = 0U;
                 slice_index + 1U < QBH_VERTICAL_SLICE_LAYER_COUNT;
                 ++slice_index) {
                const struct qbh_block_qparam *output_qparam =
                    &header->slice_layers[slice_index]
                         .qparams[QBH_BLOCK_QP_BLOCK_OUTPUT];
                const struct qbh_block_qparam *input_qparam =
                    &header->slice_layers[slice_index + 1U]
                         .qparams[QBH_BLOCK_QP_BLOCK_INPUT];
                if (output_qparam->scale != input_qparam->scale ||
                    output_qparam->zero_point !=
                        input_qparam->zero_point) {
                    fprintf(stderr,
                            "U8 hidden handoff qparam mismatch between "
                            "layers %" PRIu32 " and %" PRIu32 "\n",
                            QBH_VERTICAL_SLICE_FIRST_LAYER + slice_index,
                            QBH_VERTICAL_SLICE_FIRST_LAYER +
                                slice_index + 1U);
                    goto cleanup;
                }
            }
        }
        qbh_bind_host_slice_layer(header, 0U);
    } else {
        size_t bias_cursor = qbh_align_up_size(
            (size_t)header->output_offset + output_bytes,
            QBH_HOST_ALIGNMENT);
        for (uint32_t projection = 0;
             projection < QBH_BLOCK_PROJECTION_COUNT; ++projection) {
            struct qbh_block_projection_desc *desc =
                &header->projections[projection];
            desc->k = qbh_projection_k[projection];
            desc->n = qbh_projection_n[projection];
            desc->weight_offset = weight_slots[projection].offset;
            desc->weight_bytes = weight_slots[projection].expected_bytes;
            if (variant != QBH_BLOCK_F16F16) {
                desc->scale_offset = scale_slots[projection].offset;
                desc->scale_bytes = scale_slots[projection].expected_bytes;
            }
            if (variant == QBH_BLOCK_W4U8) {
                desc->bias_offset = (uint32_t)bias_cursor;
                desc->bias_bytes = desc->n / 32U * QBH_HMX_BIAS_BYTES;
                bias_cursor = qbh_align_up_size(
                    bias_cursor + desc->bias_bytes,
                    QBH_HOST_ALIGNMENT);
                if (qbh_build_bias_words(header, shared, projection) != 0) {
                    fprintf(stderr, "bias generation failed: %s\n",
                            qbh_projection_names[projection]);
                    goto cleanup;
                }
            }
        }
    }
    if (vertical_slice_mode == QBH_BLOCK_SLICE_DISABLED &&
        qbh_block_mlp_is_w4u8_streaming(mlp_mode) &&
        qbh_build_w4u8_streaming_bundles(
            header, shared, &w4u8_gate_up_layout,
            &w4u8_down_layout) != 0) {
        fprintf(stderr, "W4U8 streaming bundle construction failed\n");
        goto cleanup;
    }
    if (qbh_generation_w4u8_enabled(generation_mode) &&
        qbh_load_generation_qparams(
            header, shared + generation_qparam_slot.offset) != 0) {
        fprintf(stderr, "generation qparam record audit failed\n");
        goto cleanup;
    }
    header->dsp_status = QBH_BLOCK_STATUS_HOST_READY;
    shared_fd = rpcmem_to_fd(shared);
    if (shared_fd < 0) {
        goto cleanup;
    }
    open_result = qbh_session_open(&session);
    if (open_result != AEE_SUCCESS) {
        goto cleanup;
    }
    open_result = fastrpc_mmap(CDSP_DOMAIN_ID, shared_fd, shared, 0,
                               total_bytes, FASTRPC_MAP_FD);
    if (open_result != AEE_SUCCESS) {
        goto cleanup;
    }
    mapped = 1;
    prepare_result = qbh_session_prepare(&session);
    if (prepare_result != AEE_SUCCESS) {
        goto cleanup;
    }

    if (header->full_stack_stage_mode ==
        QBH_BLOCK_FULL_STACK_MAP_GATE) {
        int map_gate_result = qwen3_probe_run_block(
            session.handle, shared_fd, (uint32_t)total_bytes);
        printf(
            "{\"experiment\":163,\"map_gate\":true,"
            "\"variant\":\"%s\",\"shared_bytes\":%zu,"
            "\"rpc_result\":%d,\"dsp_status\":%d,"
            "\"layer_count\":%" PRIu32 ","
            "\"all_layers_hash\":\"%016" PRIx64 "\","
            "\"first_layer_hash\":\"%016" PRIx64 "\","
            "\"middle_layer_hash\":\"%016" PRIx64 "\","
            "\"last_layer_hash\":\"%016" PRIx64 "\","
            "\"gate_pass\":%s}\n",
            qbh_variant_name(variant), total_bytes, map_gate_result,
            header->dsp_status,
            header->full_stack_map_gate_layer_count,
            header->full_stack_map_gate_hash,
            header->full_stack_map_gate_first_layer_hash,
            header->full_stack_map_gate_middle_layer_hash,
            header->full_stack_map_gate_last_layer_hash,
            map_gate_result == AEE_SUCCESS &&
                    header->dsp_status == QBH_BLOCK_STATUS_OK &&
                    header->full_stack_map_gate_layer_count ==
                        QBH_VERTICAL_SLICE_LAYER_COUNT
                ? "true" : "false");
        exit_code = map_gate_result == AEE_SUCCESS &&
                            header->dsp_status == QBH_BLOCK_STATUS_OK &&
                            header->full_stack_map_gate_layer_count ==
                                QBH_VERTICAL_SLICE_LAYER_COUNT
                        ? 0 : 1;
        goto cleanup;
    }

    if (header->full_stack_stage_mode ==
        QBH_BLOCK_FULL_STACK_HIDDEN_CAPTURE) {
        exit_code = qbh_run_full_stack_hidden_capture(
            &session, shared_fd, shared, (uint32_t)total_bytes,
            header, variant) == 0
            ? 0 : 1;
        goto cleanup;
    }

    if (generation_mode != QBH_BLOCK_GENERATION_DISABLED) {
        exit_code = qbh_run_generation_sequence(
            &session, shared_fd, shared, (uint32_t)total_bytes,
            argv[1], header, &generation_token_slot, rope_slots) == 0
            ? 0 : 1;
        goto cleanup;
    }

    if (replay_mode == QBH_BLOCK_REPLAY_CONTINUOUS) {
        exit_code = qbh_run_replay_sequence(
            &session, shared_fd, shared, (uint32_t)total_bytes,
            argv[1], header, &input_slot, &reference_slot, rope_slots,
            vertical_slots, variant) == 0
            ? 0 : 1;
        goto cleanup;
    }

    header->repeat_count = 1U;
    header->dsp_status = QBH_BLOCK_STATUS_HOST_READY;
    memset(shared + header->output_offset, 0xa5, output_bytes);
    if (qbh_prepare_gate_up_scale_cache(header, shared) != 0) {
        fprintf(stderr, "Gate/Up scale-cache preparation failed\n");
        goto cleanup;
    }
    warmup_start = qbh_monotonic_ns();
    warmup_result = qwen3_probe_run_block(
        session.handle, shared_fd, (uint32_t)total_bytes);
    warmup_end = qbh_monotonic_ns();
    warmup_run_index = header->prepared_session_run_index;
    if (warmup_result != AEE_SUCCESS ||
        header->dsp_status != QBH_BLOCK_STATUS_OK) {
        fprintf(stderr,
                "warmup failed: rpc=0x%08x dsp=%d projection=%" PRIu32
                " n_tile=%" PRIu32 " step=%" PRIu32 " result=%d"
                " expand_mismatches=%" PRIu32
                " first_index=%" PRIu32 " expected=0x%04" PRIx32
                " actual=0x%04" PRIx32 "\n",
                (unsigned int)warmup_result, header->dsp_status,
                header->projection_failure_index,
                header->projection_failure_n_tile,
                header->projection_failure_step,
                header->projection_failure_result,
                header->w4f16_expand_mismatch_count,
                header->w4f16_expand_first_logical_index,
                header->w4f16_expand_expected_half_bits,
                header->w4f16_expand_actual_half_bits);
        goto cleanup;
    }
    warmup_metrics = variant == QBH_BLOCK_W4U8
        ? (scan_mode != QBH_BLOCK_SCAN_DISABLED
               ? qbh_compare_scan_u8(
                     shared + header->output_offset,
                     shared + header->reference_offset,
                     logical_m,
                     &header->qparams[QBH_BLOCK_QP_BLOCK_OUTPUT])
               : qbh_compare_u8(
              shared + header->output_offset,
              shared + header->reference_offset,
              QBH_BLOCK_M * QBH_BLOCK_HIDDEN,
              &header->qparams[QBH_BLOCK_QP_BLOCK_OUTPUT]))
        : (scan_mode != QBH_BLOCK_SCAN_DISABLED
               ? qbh_compare_scan_f16(
                     (const uint16_t *)(
                         shared + header->output_offset),
                     (const uint16_t *)(
                         shared + header->reference_offset),
                     logical_m)
               : qbh_compare_f16(
                     (const uint16_t *)(
                         shared + header->output_offset),
                     (const uint16_t *)(
                         shared + header->reference_offset),
                     QBH_BLOCK_M * QBH_BLOCK_HIDDEN));

    header->repeat_count = repeats;
    header->dsp_status = QBH_BLOCK_STATUS_HOST_READY;
    memset(shared + header->output_offset, 0xa5, output_bytes);
    if (qbh_prepare_gate_up_scale_cache(header, shared) != 0) {
        fprintf(stderr, "Gate/Up scale-cache preparation failed\n");
        goto cleanup;
    }
    measured_start = qbh_monotonic_ns();
    measured_result = qwen3_probe_run_block(
        session.handle, shared_fd, (uint32_t)total_bytes);
    measured_end = qbh_monotonic_ns();
    if (measured_result != AEE_SUCCESS ||
        header->dsp_status != QBH_BLOCK_STATUS_OK) {
        fprintf(stderr,
                "measured run failed: rpc=0x%08x dsp=%d projection=%" PRIu32
                " n_tile=%" PRIu32 " step=%" PRIu32 " result=%d"
                " expand_mismatches=%" PRIu32
                " first_index=%" PRIu32 " expected=0x%04" PRIx32
                " actual=0x%04" PRIx32 "\n",
                (unsigned int)measured_result, header->dsp_status,
                header->projection_failure_index,
                header->projection_failure_n_tile,
                header->projection_failure_step,
                header->projection_failure_result,
                header->w4f16_expand_mismatch_count,
                header->w4f16_expand_first_logical_index,
                header->w4f16_expand_expected_half_bits,
                header->w4f16_expand_actual_half_bits);
        goto cleanup;
    }
    measured_metrics = variant == QBH_BLOCK_W4U8
        ? (scan_mode != QBH_BLOCK_SCAN_DISABLED
               ? qbh_compare_scan_u8(
                     shared + header->output_offset,
                     shared + header->reference_offset,
                     logical_m,
                     &header->qparams[QBH_BLOCK_QP_BLOCK_OUTPUT])
               : qbh_compare_u8(
              shared + header->output_offset,
              shared + header->reference_offset,
              QBH_BLOCK_M * QBH_BLOCK_HIDDEN,
              &header->qparams[QBH_BLOCK_QP_BLOCK_OUTPUT]))
        : (scan_mode != QBH_BLOCK_SCAN_DISABLED
               ? qbh_compare_scan_f16(
                     (const uint16_t *)(
                         shared + header->output_offset),
                     (const uint16_t *)(
                         shared + header->reference_offset),
                     logical_m)
               : qbh_compare_f16(
                     (const uint16_t *)(
                         shared + header->output_offset),
                     (const uint16_t *)(
                         shared + header->reference_offset),
                     QBH_BLOCK_M * QBH_BLOCK_HIDDEN));
    output_hash = qbh_fnv1a64(
        shared + header->output_offset, header->output_bytes);
    if (scan_mode != QBH_BLOCK_SCAN_DISABLED) {
        kv_cache_mismatches = qbh_count_byte_mismatches(
            shared + kv_cache_slots[0].offset,
            shared + kv_reference_slots[0].offset,
            kv_cache_slots[0].expected_bytes) +
            qbh_count_byte_mismatches(
                shared + kv_cache_slots[1].offset,
            shared + kv_reference_slots[1].offset,
            kv_cache_slots[1].expected_bytes);
        kv_cache_k_hash = qbh_fnv1a64(
            shared + kv_cache_slots[0].offset,
            kv_cache_slots[0].expected_bytes);
        kv_cache_v_hash = qbh_fnv1a64(
            shared + kv_cache_slots[1].offset,
            kv_cache_slots[1].expected_bytes);
        header->scan_cache_append_mismatch_count =
            kv_cache_mismatches > UINT32_MAX
                ? UINT32_MAX : (uint32_t)kv_cache_mismatches;
    }
    if (scan_mode != QBH_BLOCK_SCAN_DISABLED) {
        const char *cache_dump_root = getenv("QBH_DUMP_CACHE_DIR");
        if (cache_dump_root != NULL && cache_dump_root[0] != '\0') {
            const char *suffix =
                variant == QBH_BLOCK_W4U8 ? "u8" : "f16";
            const struct qbh_file_slot *slots[2] = {
                &kv_cache_slots[0], &kv_cache_slots[1]};
            const char *kinds[2] = {"k", "v"};
            for (uint32_t index = 0U; index < 2U; ++index) {
                char name[64];
                char dump_path[512];
                FILE *dump;
                size_t written;
                int status = snprintf(
                    name, sizeof(name),
                    "actual_kv_cache_%s_%s.bin",
                    kinds[index], suffix);
                if (status < 0 || (size_t)status >= sizeof(name) ||
                    qbh_make_path(
                        dump_path, sizeof(dump_path),
                        cache_dump_root, name) != 0) {
                    fprintf(stderr, "invalid cache dump path\n");
                    goto cleanup;
                }
                dump = fopen(dump_path, "wb");
                if (dump == NULL) {
                    fprintf(stderr, "failed to open cache dump: %s\n",
                            dump_path);
                    goto cleanup;
                }
                written = fwrite(
                    shared + slots[index]->offset, 1U,
                    slots[index]->expected_bytes, dump);
                if (fclose(dump) != 0 ||
                    written != slots[index]->expected_bytes) {
                    fprintf(stderr, "failed to write cache dump: %s\n",
                            dump_path);
                    goto cleanup;
                }
            }
        }
    }
    {
        const char *dump_path = getenv("QBH_DUMP_OUTPUT_PATH");
        if (dump_path != NULL && dump_path[0] != '\0') {
            FILE *dump = fopen(dump_path, "wb");
            if (dump == NULL) {
                fprintf(stderr, "failed to write output dump: %s\n",
                        dump_path);
                goto cleanup;
            }
            size_t written = fwrite(
                shared + header->output_offset, 1U,
                header->output_bytes, dump);
            int dump_close_result = fclose(dump);
            if (written != header->output_bytes ||
                dump_close_result != 0) {
                fprintf(stderr, "failed to write output dump: %s\n",
                        dump_path);
                goto cleanup;
            }
        }
    }
    if (qbh_attention_u8_enabled(attention_pipeline_mode) &&
        numerical_audit_enabled != 0U) {
        const char *dump_root = getenv("QBH_DUMP_ATTENTION_DIR");
        if (dump_root != NULL && dump_root[0] != '\0') {
            static const char *const names[6] = {
                "actual_q_tiles_u8.bin",
                "actual_k_tiles_u8.bin",
                "actual_v_tiles_u8.bin",
                "actual_score_tiles_u8.bin",
                "actual_probability_tiles_u8.bin",
                "actual_av_tiles_u8.bin",
            };
            static const uint32_t bytes[6] = {
                QBH_BLOCK_U8_ATTENTION_Q_BYTES,
                QBH_BLOCK_U8_ATTENTION_KV_BYTES,
                QBH_BLOCK_U8_ATTENTION_KV_BYTES,
                QBH_BLOCK_U8_ATTENTION_SCORE_BYTES,
                QBH_BLOCK_U8_ATTENTION_SCORE_BYTES,
                QBH_BLOCK_U8_ATTENTION_AV_BYTES,
            };
            uint32_t audit_offset =
                header->u8_attention_audit_output_offset;
            for (uint32_t index = 0U; index < 6U; ++index) {
                char dump_path[512];
                FILE *dump;
                size_t written;
                if (qbh_make_path(
                        dump_path, sizeof(dump_path),
                        dump_root, names[index]) != 0) {
                    fprintf(stderr, "invalid Attention dump path\n");
                    goto cleanup;
                }
                dump = fopen(dump_path, "wb");
                if (dump == NULL) {
                    fprintf(stderr, "failed to open Attention dump: %s\n",
                            dump_path);
                    goto cleanup;
                }
                written = fwrite(
                    shared + audit_offset, 1U, bytes[index], dump);
                if (fclose(dump) != 0 || written != bytes[index]) {
                    fprintf(stderr, "failed to write Attention dump: %s\n",
                            dump_path);
                    goto cleanup;
                }
                audit_offset += bytes[index];
            }
            {
                static const char *const tail_names[6] = {
                    "actual_o_tiles_u8.bin",
                    "actual_post_residual_u8.bin",
                    "actual_post_norm_tiles_u8.bin",
                    "actual_middle_tiles_u8.bin",
                    "actual_down_tiles_u8.bin",
                    "actual_final_u8.bin",
                };
                static const uint32_t tail_offsets[6] = {
                    QBH_BLOCK_U8_TAIL_O_OFFSET,
                    QBH_BLOCK_U8_TAIL_POST_RESIDUAL_OFFSET,
                    QBH_BLOCK_U8_TAIL_POST_NORM_OFFSET,
                    QBH_BLOCK_U8_TAIL_MIDDLE_OFFSET,
                    QBH_BLOCK_U8_TAIL_DOWN_OFFSET,
                    QBH_BLOCK_U8_TAIL_FINAL_OFFSET,
                };
                static const uint32_t tail_bytes[6] = {
                    QBH_BLOCK_M * QBH_BLOCK_HIDDEN,
                    QBH_BLOCK_M * QBH_BLOCK_HIDDEN,
                    QBH_BLOCK_M * QBH_BLOCK_HIDDEN,
                    QBH_BLOCK_M * QBH_BLOCK_INTERMEDIATE,
                    QBH_BLOCK_M * QBH_BLOCK_HIDDEN,
                    QBH_BLOCK_M * QBH_BLOCK_HIDDEN,
                };

                for (uint32_t index = 0U; index < 6U; ++index) {
                    if (qbh_write_named_tensor(
                            dump_root, tail_names[index],
                            shared +
                                header->u8_attention_audit_output_offset +
                                tail_offsets[index],
                            tail_bytes[index]) != 0) {
                        fprintf(stderr,
                                "failed to write W4U8 tail dump: %s\n",
                                tail_names[index]);
                        goto cleanup;
                    }
                }
            }
        }
    }
    if (w4u8_boundary_audit_enabled != 0U) {
        const char *dump_root = getenv("QBH_DUMP_BOUNDARY_DIR");
        if (dump_root != NULL && dump_root[0] != '\0' &&
            qbh_write_named_tensor(
                dump_root, "actual_input_norm_tiles_u8.bin",
                shared + header->w4u8_boundary_audit_output_offset,
                header->w4u8_boundary_audit_output_bytes) != 0) {
            fprintf(stderr, "failed to write W4U8 boundary dump\n");
            goto cleanup;
        }
    }
    if (scan_mode != QBH_BLOCK_SCAN_DISABLED &&
        variant != QBH_BLOCK_W4U8 &&
        numerical_audit_enabled != 0U) {
        const char *dump_root = getenv("QBH_DUMP_ATTENTION_DIR");
        if (dump_root != NULL && dump_root[0] != '\0') {
            static const char *const names[3] = {
                "actual_scan_q_f16.bin",
                "actual_scan_attention_f16.bin",
                "actual_scan_o_projection_f16.bin",
            };
            const uint32_t bytes =
                QBH_BLOCK_M * QBH_BLOCK_HIDDEN * sizeof(uint16_t);
            for (uint32_t index = 0U; index < 3U; ++index) {
                char dump_path[512];
                FILE *dump;
                size_t written;
                if (qbh_make_path(
                        dump_path, sizeof(dump_path),
                        dump_root, names[index]) != 0) {
                    fprintf(stderr, "invalid FP16 Attention dump path\n");
                    goto cleanup;
                }
                dump = fopen(dump_path, "wb");
                if (dump == NULL) {
                    fprintf(stderr, "failed to open FP16 Attention dump: %s\n",
                            dump_path);
                    goto cleanup;
                }
                written = fwrite(
                    shared + header->scan_attention_audit_output_offset +
                        (size_t)index * bytes,
                    1U, bytes, dump);
                if (fclose(dump) != 0 || written != bytes) {
                    fprintf(stderr, "failed to write FP16 Attention dump: %s\n",
                            dump_path);
                    goto cleanup;
                }
            }
        }
    }

    release_result = qbh_session_release(&session);
    close_result = qbh_session_close(&session);
    printf(
        "{\"experiment\":\"EXP-0163\","
        "\"execution_unit\":\"qwen3_layer14_shape_kv_scan\","
        "\"scan_mode\":\"%s\","
        "\"logical_m\":%" PRIu32 ","
        "\"initial_kv_length\":%" PRIu32 ","
        "\"kv_cache_capacity\":%" PRIu32 ","
        "\"scan_physical_chunk_count\":%" PRIu32 ","
        "\"scan_total_kv_length\":%" PRIu32 ","
        "\"scan_padded_kv_length\":%" PRIu32 ","
        "\"scan_useful_query_rows\":%" PRIu32 ","
        "\"scan_physical_query_rows\":%" PRIu32 ","
        "\"scan_attention_overlay_capacity_bytes\":%" PRIu32 ","
        "\"scan_attention_overlay_required_bytes\":%" PRIu32 ","
        "\"scan_cache_dma_descriptor_count\":%" PRIu32 ","
        "\"scan_cache_ddr_read_bytes\":%" PRIu64 ","
        "\"scan_cache_ddr_write_bytes\":%" PRIu64 ","
        "\"scan_cache_stage_ticks\":%" PRIu64 ","
        "\"scan_cache_append_ticks\":%" PRIu64 ","
        "\"scan_dynamic_attention_ticks\":%" PRIu64 ","
        "\"scan_cache_mismatches\":%" PRIu64 ","
        "\"scan_cache_k_hash\":\"%016" PRIx64 "\","
        "\"scan_cache_v_hash\":\"%016" PRIx64 "\","
        "\"variant\":\"%s\",\"attention_compute\":\"%s\","
        "\"projection_compute\":\"%s\","
        "\"common_ops_mode\":\"%s\","
        "\"u8_norm_reduction_mode\":\"%s\","
        "\"w4u8_qk_pair_kernel_mode\":%" PRIu32 ","
        "\"fp16_common_schedule_mode\":\"%s\","
        "\"qkv_schedule_mode\":\"%s\","
        "\"w4f16_group_fence_mode\":\"%s\","
        "\"w4f16_expand_claim_regions\":%" PRIu32 ","
        "\"w4f16_gate_up_extra_expand_worker\":%" PRIu32 ","
        "\"w4f16_gate_up_extra_stream_worker\":%" PRIu32 ","
        "\"w4f16_gate_up_stream_group_tiles\":%" PRIu32 ","
        "\"w4u8_stream_fence_mode\":\"%s\","
        "\"w4u8_decode_softmax_mode\":\"%s\","
        "\"w4u8_decode_lm_head_group_tiles\":%" PRIu32 ","
        "\"w4u8_qkv_ring_expand_workers\":%" PRIu32 ","
        "\"fp16_norm_rows_per_task\":%" PRIu32 ","
        "\"fp16_norm_contexts\":%" PRIu32 ","
        "\"w4u8_down_hmx_batch_outputs\":%" PRIu32 ","
        "\"attribution_mode\":\"%s\","
        "\"numerical_audit_mode\":\"%s\","
        "\"residual_mode\":\"%s\","
        "\"f16f16_projection_mode\":\"%s\","
        "\"w4f16_pipeline_mode\":\"%s\","
        "\"attention_pack_mode\":\"%s\","
        "\"attention_pipeline_mode\":\"%s\","
        "\"attention_hvx_contexts\":%" PRIu32 ","
        "\"mlp_mode\":\"%s\","
        "\"mlp_hvx_contexts\":%" PRIu32 ","
        "\"mlp_chunk_vectors\":%" PRIu32 ","
        "\"crouton_boundary_mode\":\"%s\","
        "\"w4u8_qkvo_pipeline_mode\":\"%s\","
        "\"w4f16_scale_placement\":\"%s\","
        "\"intermediate_residency\":\"VTCM\","
        "\"warmup_rpc_result\":%d,"
        "\"warmup_prepared_session_run_index\":%" PRIu32 ","
        "\"warmup_host_wall_ns\":%" PRIu64 ","
        "\"warmup_max_abs\":%.9g,\"warmup_mean_abs\":%.9g,"
        "\"warmup_rmse\":%.9g,\"warmup_cosine\":%.9g,"
        "\"warmup_mismatches\":%" PRIu64 ","
        "\"warmup_max_lsb\":%" PRIu32 ","
        "\"repeat_count\":%" PRIu32 ","
        "\"prepared_session_run_index\":%" PRIu32 ","
        "\"rpc_result\":%d,\"dsp_status\":%d,"
        "\"numerical_status\":%d,"
        "\"attention_qk_max_abs\":%.9g,"
        "\"attention_probability_max_abs\":%.9g,"
        "\"attention_av_max_abs\":%.9g,"
        "\"common_op_rms_max_abs\":%.9g,"
        "\"common_op_rms_cosine\":%.9g,"
        "\"common_op_rope_max_abs\":%.9g,"
        "\"common_op_rope_cosine\":%.9g,"
        "\"common_op_softmax_max_abs\":%.9g,"
        "\"common_op_softmax_cosine\":%.9g,"
        "\"common_op_silu_max_abs\":%.9g,"
        "\"common_op_silu_cosine\":%.9g,"
        "\"common_op_nonfinite_count\":%" PRIu32 ","
        "\"common_op_softmax_mask_violation_count\":%" PRIu32 ","
        "\"projection_failure_result\":%d,"
        "\"projection_failure_index\":%" PRIu32 ","
        "\"projection_failure_n_tile\":%" PRIu32 ","
        "\"projection_failure_step\":%" PRIu32 ","
        "\"w4f16_expand_mismatch_count\":%" PRIu32 ","
        "\"w4f16_expand_first_logical_index\":%" PRIu32 ","
        "\"w4f16_expand_expected_half_bits\":%" PRIu32 ","
        "\"w4f16_expand_actual_half_bits\":%" PRIu32 ","
        "\"w4f16_hvx_workers_created\":%" PRIu32 ","
        "\"w4f16_hvx_workers_locked\":%" PRIu32 ","
        "\"w4f16_requested_hvx_workers\":%" PRIu32 ","
        "\"w4f16_region_tiles\":%" PRIu32 ","
        "\"w4f16_pool_status\":%d,"
        "\"f16f16_weight_batch_n_tiles\":%" PRIu32 ","
        "\"w4f16_active_worker_min\":%" PRIu32 ","
        "\"w4f16_active_worker_max\":%" PRIu32 ","
        "\"w4f16_effective_region_min\":%" PRIu32 ","
        "\"w4f16_effective_region_max\":%" PRIu32 ","
        "\"mlp_hvx_workers_created\":%" PRIu32 ","
        "\"mlp_hvx_workers_locked\":%" PRIu32 ","
        "\"mlp_pool_status\":%d,"
        "\"mlp_silu_chunk_count\":%" PRIu32 ","
        "\"mlp_stream_group_count\":%" PRIu32 ","
        "\"mlp_down_pack_skipped\":%" PRIu32 ","
        "\"mlp_down_input_hash\":\"%016" PRIx64 "\","
        "\"attention_hvx_workers_created\":%" PRIu32 ","
        "\"attention_hvx_workers_locked\":%" PRIu32 ","
        "\"attention_pool_status\":%d,"
        "\"attention_qk_norm_task_count\":%" PRIu32 ","
        "\"fp16_qk_norm_pair_task_count\":%" PRIu32 ","
        "\"attention_softmax_task_count\":%" PRIu32 ","
        "\"u8_attention_softmax_shuffle4_row_group_count\":%" PRIu32 ","
        "\"attention_gqa_group_count\":%" PRIu32 ","
        "\"u8_attention_group_count\":%" PRIu32 ","
        "\"u8_attention_qk_execution_count\":%" PRIu32 ","
        "\"u8_attention_av_execution_count\":%" PRIu32 ","
        "\"u8_attention_score_saturation_count\":%" PRIu32 ","
        "\"u8_attention_v_recenter_saturation_count\":%" PRIu32 ","
        "\"u8_attention_probability_mask_violation_count\":%" PRIu32 ","
        "\"u8_attention_probability_row_sum_min\":%" PRIu32 ","
        "\"u8_attention_probability_row_sum_max\":%" PRIu32 ","
        "\"u8_attention_direct_o_tile_count\":%" PRIu32 ","
        "\"u8_attention_qkv_unpack_skipped\":%" PRIu32 ","
        "\"u8_attention_fused_k_operand_mismatch_count\":%" PRIu32 ","
        "\"w4u8_qkv_batch_n_tiles\":%" PRIu32 ","
        "\"w4u8_qkv_batch_count\":%" PRIu32 ","
        "\"w4u8_qkvo_prefetch_count\":%" PRIu32 ","
        "\"w4u8_qkvo_overlap_schedule_count\":%" PRIu32 ","
        "\"w4u8_qk_pair_kernel_mode_observed\":%" PRIu32 ","
        "\"w4u8_qk_quarter_pair_count\":%" PRIu32 ","
        "\"w4u8_decode_softmax_hvx_tile4_call_count\":%" PRIu32 ","
        "\"w4u8_decode_softmax_hvx_tile4_mismatch_count\":%" PRIu32 ","
        "\"u8_attention_expected_score_hash\":\"%016" PRIx64 "\","
        "\"u8_attention_actual_score_hash\":\"%016" PRIx64 "\","
        "\"u8_attention_expected_probability_hash\":\"%016" PRIx64 "\","
        "\"u8_attention_actual_probability_hash\":\"%016" PRIx64 "\","
        "\"u8_attention_expected_av_hash\":\"%016" PRIx64 "\","
        "\"u8_attention_actual_av_hash\":\"%016" PRIx64 "\","
        "\"u8_input_norm_actual_hash\":\"%016" PRIx64 "\","
        "\"u8_attention_audit_ddr_write_bytes\":%" PRIu32 ","
        "\"w4u8_boundary_audit_enabled\":%" PRIu32 ","
        "\"w4u8_boundary_audit_ddr_write_bytes\":%" PRIu32 ","
        "\"crouton_qkv_projection_count\":%" PRIu32 ","
        "\"crouton_qkv_unpack_skipped\":%" PRIu32 ","
        "\"crouton_qk_operand_count\":%" PRIu32 ","
        "\"crouton_av_weight_count\":%" PRIu32 ","
        "\"crouton_av_o_head_count\":%" PRIu32 ","
        "\"crouton_av_unpack_skipped\":%" PRIu32 ","
        "\"crouton_norm_projection_count\":%" PRIu32 ","
        "\"crouton_q_operand_mismatch_count\":%" PRIu32 ","
        "\"crouton_k_operand_mismatch_count\":%" PRIu32 ","
        "\"crouton_v_operand_mismatch_count\":%" PRIu32 ","
        "\"qkv_operand_audit_tensor_count\":%" PRIu32 ","
        "\"qkv_schedule_command_count\":%" PRIu32 ","
        "\"qkv_schedule_trace_hash\":\"%016" PRIx64 "\","
        "\"host_wall_ns\":%" PRIu64 ","
        "\"host_wall_ns_per_block\":%.3f,"
        "\"max_abs\":%.9g,\"mean_abs\":%.9g,\"rmse\":%.9g,"
        "\"cosine\":%.9g,\"mismatches\":%" PRIu64 ","
        "\"max_lsb\":%" PRIu32 ","
        "\"output_hash\":\"%016" PRIx64 "\","
        "\"vtcm_requested_bytes\":%" PRIu32 ","
        "\"vtcm_acquired_bytes\":%" PRIu32 ","
        "\"vtcm_peak_plan_bytes\":%" PRIu32 ","
        "\"block_invocation_count\":%" PRIu32 ","
        "\"weight_ddr_read_bytes\":%" PRIu64 ","
        "\"weight_dma_descriptor_count\":%" PRIu32 ","
        "\"boundary_ddr_read_bytes\":%" PRIu64 ","
        "\"boundary_ddr_write_bytes\":%" PRIu64 ","
        "\"intermediate_ddr_read_bytes\":%" PRIu32 ","
        "\"intermediate_ddr_write_bytes\":%" PRIu32 ","
        "\"intermediate_dma_descriptor_count\":%" PRIu32 ","
        "\"intermediate_spill_fill_count\":%" PRIu32 ","
        "\"hmx_command_count\":%" PRIu32 ","
        "\"hmx_fp16_tile_pair_count\":%" PRIu32 ","
        "\"hmx_u8s8_tile_pair_count\":%" PRIu32 ","
        "\"input_stage_ticks\":%" PRIu64 ","
        "\"metadata_stage_ticks\":%" PRIu64 ","
        "\"input_norm_ticks\":%" PRIu64 ","
        "\"qkv_projection_ticks\":%" PRIu64 ","
        "\"qk_norm_rope_ticks\":%" PRIu64 ","
        "\"attention_ticks\":%" PRIu64 ","
        "\"o_projection_ticks\":%" PRIu64 ","
        "\"post_attention_residual_ticks\":%" PRIu64 ","
        "\"post_attention_norm_ticks\":%" PRIu64 ","
        "\"gate_up_ticks\":%" PRIu64 ","
        "\"activation_ticks\":%" PRIu64 ","
        "\"down_ticks\":%" PRIu64 ","
        "\"final_residual_ticks\":%" PRIu64 ","
        "\"output_stage_ticks\":%" PRIu64 ","
        "\"total_ticks\":%" PRIu64 ","
        "\"invocation_ticks\":%" PRIu64 ","
        "\"runtime_setup_ticks\":%" PRIu64 ","
        "\"runtime_teardown_ticks\":%" PRIu64 ","
        "\"stage_boundary_ticks\":%" PRIu64 ","
        "\"ledger_named_ticks\":%" PRIu64 ","
        "\"ledger_unattributed_ticks\":%" PRIu64 ","
        "\"input_norm_audit_ticks\":%" PRIu64 ","
        "\"qkv_audit_ticks\":%" PRIu64 ","
        "\"qk_norm_rope_audit_ticks\":%" PRIu64 ","
        "\"o_projection_audit_ticks\":%" PRIu64 ","
        "\"post_attention_residual_audit_ticks\":%" PRIu64 ","
        "\"post_attention_norm_audit_ticks\":%" PRIu64 ","
        "\"gate_up_audit_ticks\":%" PRIu64 ","
        "\"activation_audit_ticks\":%" PRIu64 ","
        "\"down_audit_ticks\":%" PRIu64 ","
        "\"final_residual_audit_ticks\":%" PRIu64 ","
        "\"attention_setup_ticks\":%" PRIu64 ","
        "\"attention_qk_pack_ticks\":%" PRIu64 ","
        "\"attention_qk_hmx_ticks\":%" PRIu64 ","
        "\"attention_qk_unpack_ticks\":%" PRIu64 ","
        "\"attention_qk_audit_ticks\":%" PRIu64 ","
        "\"attention_softmax_ticks\":%" PRIu64 ","
        "\"attention_softmax_audit_ticks\":%" PRIu64 ","
        "\"attention_av_pack_ticks\":%" PRIu64 ","
        "\"attention_av_hmx_ticks\":%" PRIu64 ","
        "\"attention_av_unpack_ticks\":%" PRIu64 ","
        "\"attention_av_audit_ticks\":%" PRIu64 ","
        "\"attention_gqa_pipeline_ticks\":%" PRIu64 ","
        "\"attention_unattributed_ticks\":%" PRIu64 ","
        "\"u8_attention_qk_norm_rope_ticks\":%" PRIu64 ","
        "\"u8_attention_k_pack_ticks\":%" PRIu64 ","
        "\"u8_attention_v_pack_ticks\":%" PRIu64 ","
        "\"u8_attention_qk_hmx_ticks\":%" PRIu64 ","
        "\"u8_attention_qk_av_hmx_ticks\":%" PRIu64 ","
        "\"u8_attention_qk_requant_ticks\":%" PRIu64 ","
        "\"u8_attention_softmax_ticks\":%" PRIu64 ","
        "\"u8_attention_qk_requant_softmax_ticks\":%" PRIu64 ","
        "\"u8_attention_av_hmx_ticks\":%" PRIu64 ","
        "\"u8_attention_av_requant_ticks\":%" PRIu64 ","
        "\"u8_attention_pipeline_wait_ticks\":%" PRIu64 ","
        "\"w4u8_qkvo_weight_expand_ticks\":%" PRIu64 ","
        "\"w4u8_qkvo_prefetch_wait_ticks\":%" PRIu64 ","
        "\"w4u8_qkvo_hmx_lifetime_ticks\":%" PRIu64 ","
        "\"w4u8_qkv_ring_slot_count\":%" PRIu32 ","
        "\"w4u8_qkv_ring_expand_worker_count\":%" PRIu32 ","
        "\"w4u8_qkv_ring_prep_worker_count\":%" PRIu32 ","
        "\"w4u8_qkv_ring_dispatch_count\":%" PRIu32 ","
        "\"w4u8_qkv_ring_batch_count\":%" PRIu32 ","
        "\"w4u8_qkv_ring_expand_task_count\":%" PRIu32 ","
        "\"w4u8_qkv_ring_hmx_dispatch_count\":%" PRIu32 ","
        "\"w4u8_qkv_ring_head_publish_count\":%" PRIu32 ","
        "\"w4u8_qkv_ring_pipeline_ticks\":%" PRIu64 ","
        "\"w4u8_qkv_ring_dma_wait_ticks\":%" PRIu64 ","
        "\"w4u8_qkv_ring_producer_slot_wait_ticks\":%" PRIu64 ","
        "\"w4u8_qkv_ring_expand_ticks\":%" PRIu64 ","
        "\"w4u8_qkv_ring_hmx_ready_wait_ticks\":%" PRIu64 ","
        "\"w4u8_qkv_ring_hmx_compute_ticks\":%" PRIu64 ","
        "\"w4u8_qkv_ring_pool_wait_ticks\":%" PRIu64 ","
        "\"w4u8_input_norm_task_count\":%" PRIu32 ","
        "\"w4u8_input_norm_main_work_ticks\":%" PRIu64 ","
        "\"w4u8_input_norm_worker_work_ticks\":%" PRIu64 ","
        "\"w4u8_input_norm_pool_wait_ticks\":%" PRIu64 ","
        "\"w4u8_residual_active_contexts\":%" PRIu32 ","
        "\"w4u8_post_residual_task_count\":%" PRIu32 ","
        "\"w4u8_final_residual_task_count\":%" PRIu32 ","
        "\"w4u8_post_residual_main_work_ticks\":%" PRIu64 ","
        "\"w4u8_post_residual_worker_work_ticks\":%" PRIu64 ","
        "\"w4u8_post_residual_pool_wait_ticks\":%" PRIu64 ","
        "\"w4u8_final_residual_main_work_ticks\":%" PRIu64 ","
        "\"w4u8_final_residual_worker_work_ticks\":%" PRIu64 ","
        "\"w4u8_final_residual_pool_wait_ticks\":%" PRIu64 ","
        "\"fp16_input_norm_task_count\":%" PRIu32 ","
        "\"fp16_input_norm_active_contexts\":%" PRIu32 ","
        "\"fp16_input_norm_main_work_ticks\":%" PRIu64 ","
        "\"fp16_input_norm_worker_work_ticks\":%" PRIu64 ","
        "\"fp16_input_norm_pool_wait_ticks\":%" PRIu64 ","
        "\"fp16_post_residual_norm_task_count\":%" PRIu32 ","
        "\"fp16_post_residual_norm_active_contexts\":%" PRIu32 ","
        "\"fp16_post_residual_norm_main_work_ticks\":%" PRIu64 ","
        "\"fp16_post_residual_norm_worker_work_ticks\":%" PRIu64 ","
        "\"fp16_post_residual_norm_pool_wait_ticks\":%" PRIu64 ","
        "\"w4f16_gate_up_effective_region_tiles\":%" PRIu32 ","
        "\"w4f16_gate_up_scale_cache_bytes\":%" PRIu32 ","
        "\"w4f16_gate_up_weight_dma_ticks\":%" PRIu64 ","
        "\"w4f16_gate_up_expand_ticks\":%" PRIu64 ","
        "\"w4f16_gate_up_expand_work_ticks\":%" PRIu64 ","
        "\"w4f16_gate_up_expand_pool_wait_ticks\":%" PRIu64 ","
        "\"w4f16_gate_up_prefetch_wait_ticks\":%" PRIu64 ","
        "\"w4f16_gate_up_hmx_wait_ticks\":%" PRIu64 ","
        "\"w4f16_gate_up_hmx_tail_wait_ticks\":%" PRIu64 ","
        "\"w4f16_gate_up_unpack_ticks\":%" PRIu64 ","
        "\"w4f16_gate_up_stream_work_ticks\":%" PRIu64 ","
        "\"w4f16_gate_up_stream_ready_wait_ticks\":%" PRIu64 ","
        "\"w4f16_gate_up_stream_join_wait_ticks\":%" PRIu64 ","
        "\"w4f16_gate_up_hmx_command_count\":%" PRIu64 ","
        "\"w4f16_gate_up_scale_init_ticks\":%" PRIu64 ","
        "\"w4u8_mlp_vtcm_base_offset\":%" PRIu32 ","
        "\"w4u8_mlp_vtcm_plan_bytes\":%" PRIu32 ","
        "\"w4u8_mlp_lut_vtcm_bytes\":%" PRIu32 ","
        "\"w4u8_mlp_gather_scratch_vtcm_bytes\":%" PRIu32 ","
        "\"w4u8_mlp_gate_up_hvx_workers\":%" PRIu32 ","
        "\"w4u8_mlp_down_hvx_workers\":%" PRIu32 ","
        "\"w4u8_mlp_gate_up_hmx_batch_n_tiles\":%" PRIu32 ","
        "\"w4u8_mlp_down_hmx_batch_n_tiles\":%" PRIu32 ","
        "\"w4u8_mlp_down_in_command_slot_release_count\":%" PRIu32 ","
        "\"w4u8_mlp_down_producer_progress_command_count\":%" PRIu32 ","
        "\"w4u8_mlp_gate_up_expanded_slot_count\":%" PRIu32 ","
        "\"w4u8_mlp_pair_publish_count\":%" PRIu32 ","
        "\"w4u8_mlp_pair_consume_count\":%" PRIu32 ","
        "\"w4u8_mlp_gate_up_hvx_hmx_overlap\":%" PRIu32 ","
        "\"w4u8_mlp_down_hvx_hmx_overlap\":%" PRIu32 ","
        "\"w4u8_mlp_gate_up_hvx_parallel_overlap\":%" PRIu32 ","
        "\"w4u8_mlp_down_hvx_parallel_overlap\":%" PRIu32 ","
        "\"w4u8_mlp_input_pack_skipped\":%" PRIu32 ","
        "\"w4u8_mlp_output_unpack_skipped\":%" PRIu32 ","
        "\"w4u8_mlp_input_pack_ticks\":%" PRIu64 ","
        "\"w4u8_mlp_output_unpack_ticks\":%" PRIu64 ","
        "\"w4u8_mlp_gate_up_pipeline_ticks\":%" PRIu64 ","
        "\"w4u8_mlp_gate_up_hmx_command_count\":%" PRIu64 ","
        "\"w4u8_mlp_down_pipeline_ticks\":%" PRIu64 ","
        "\"w4u8_mlp_down_hmx_command_count\":%" PRIu64 ","
        "\"w4u8_mlp_activation_work_ticks\":%" PRIu64 ","
        "\"w4u8_mlp_weight_stage_ticks\":%" PRIu64 ","
        "\"w4u8_mlp_weight_expand_ticks\":%" PRIu64 ","
        "\"w4u8_mlp_hmx_compute_ticks\":%" PRIu64 ","
        "\"w4u8_mlp_hmx_ready_wait_ticks\":%" PRIu64 ","
        "\"w4u8_mlp_producer_slot_wait_ticks\":%" PRIu64 ","
        "\"w4u8_mlp_expanded_slot_wait_ticks\":%" PRIu64 ","
        "\"w4u8_gate_up_persistent_hvx_dispatch_count\":%" PRIu32 ","
        "\"w4u8_gate_up_persistent_hvx_worker_count\":%" PRIu32 ","
        "\"w4u8_gate_up_transient_hvx_thread_count\":%" PRIu32 ","
        "\"w4u8_down_persistent_hvx_dispatch_count\":%" PRIu32 ","
        "\"w4u8_down_persistent_hvx_worker_count\":%" PRIu32 ","
        "\"w4u8_down_transient_hvx_thread_count\":%" PRIu32 ","
        "\"weight_dma_ticks\":%" PRIu64 ","
        "\"hmx_compute_ticks\":%" PRIu64 ","
        "\"projection_pack_ticks\":%" PRIu64 ","
        "\"w4f16_expand_ticks\":%" PRIu64 ","
        "\"projection_hmx_wait_ticks\":%" PRIu64 ","
        "\"projection_unpack_ticks\":%" PRIu64 ","
        "\"hmx_ready_wait_ticks\":%" PRIu64 ","
        "\"w4f16_streamed_command_count\":%" PRIu64 ","
        "\"w4f16_expand_work_ticks\":%" PRIu64 ","
        "\"w4f16_expand_region_count\":%" PRIu64 ","
        "\"w4f16_prefetch_count\":%" PRIu64 ","
        "\"w4f16_prefetch_wait_ticks\":%" PRIu64 ","
        "\"f16f16_prefetch_count\":%" PRIu64 ","
        "\"f16f16_prefetch_wait_ticks\":%" PRIu64 ","
        "\"w4f16_first_expand_ticks\":%" PRIu64 ","
        "\"w4f16_steady_expand_ticks\":%" PRIu64 ","
        "\"w4f16_expand_pool_wait_ticks\":%" PRIu64 ","
        "\"w4f16_hmx_tail_wait_ticks\":%" PRIu64 ","
        "\"w4f16_early_region_command_count\":%" PRIu64 ","
        "\"w4f16_cross_prefetch_count\":%" PRIu64 ","
        "\"w4f16_cross_prefetch_wait_ticks\":%" PRIu64 ","
        "\"w4f16_cross_prefetch_lifetime_ticks\":%" PRIu64 ","
        "\"mlp_silu_main_work_ticks\":%" PRIu64 ","
        "\"mlp_silu_worker_work_ticks\":%" PRIu64 ","
        "\"mlp_silu_pool_wait_ticks\":%" PRIu64 ","
        "\"mlp_stream_worker_work_ticks\":%" PRIu64 ","
        "\"mlp_stream_main_work_ticks\":%" PRIu64 ","
        "\"mlp_stream_ready_wait_ticks\":%" PRIu64 ","
        "\"mlp_stream_join_wait_ticks\":%" PRIu64 ","
        "\"attention_qk_norm_main_work_ticks\":%" PRIu64 ","
        "\"attention_qk_norm_worker_work_ticks\":%" PRIu64 ","
        "\"attention_qk_norm_pool_wait_ticks\":%" PRIu64 ","
        "\"attention_softmax_main_work_ticks\":%" PRIu64 ","
        "\"attention_softmax_worker_work_ticks\":%" PRIu64 ","
        "\"attention_softmax_pool_wait_ticks\":%" PRIu64 ","
        "\"attention_gqa_worker_work_ticks\":%" PRIu64 ","
        "\"attention_gqa_hmx_wait_ticks\":%" PRIu64 ","
        "\"attention_gqa_queue_wait_ticks\":%" PRIu64 ","
        "\"crouton_qkv_transform_ticks\":%" PRIu64 ","
        "\"crouton_av_o_copy_ticks\":%" PRIu64 ","
        "\"crouton_norm_store_ticks\":%" PRIu64 ","
        "\"release_result\":%d,\"close_result\":%d}\n",
        qbh_scan_mode_name(scan_mode),
        logical_m,
        initial_kv_length,
        kv_cache_capacity,
        header->scan_physical_chunk_count,
        header->scan_total_kv_length,
        header->scan_padded_kv_length,
        header->scan_useful_query_rows,
        header->scan_physical_query_rows,
        header->scan_attention_overlay_capacity_bytes,
        header->scan_attention_overlay_required_bytes,
        header->scan_cache_dma_descriptor_count,
        header->scan_cache_ddr_read_bytes,
        header->scan_cache_ddr_write_bytes,
        header->scan_cache_stage_ticks,
        header->scan_cache_append_ticks,
        header->scan_dynamic_attention_ticks,
        kv_cache_mismatches,
        kv_cache_k_hash,
        kv_cache_v_hash,
        qbh_variant_name(variant),
        qbh_attention_u8_enabled(attention_pipeline_mode)
            ? "U8xS8_HMX_log2_softmax"
            : "FP16_HMX",
        variant == QBH_BLOCK_W4U8 ? "U8xS8_integer_HMX"
                                  : "FP16_HMX",
        qbh_common_ops_mode_name(header->common_ops_mask),
        qbh_u8_norm_reduction_mode_name(
            header->u8_norm_reduction_mode),
        header->w4u8_qk_pair_kernel_mode,
        qbh_fp16_common_schedule_mode_name(
            header->fp16_common_schedule_mode),
        qbh_qkv_schedule_mode_name(header->qkv_schedule_mode),
        qbh_w4f16_group_fence_mode_name(
            header->w4f16_group_fence_mode),
        header->w4f16_expand_claim_regions,
        header->w4f16_gate_up_extra_expand_worker,
        header->w4f16_gate_up_extra_stream_worker,
        header->w4f16_gate_up_stream_group_tiles,
        qbh_w4u8_stream_fence_mode_name(
            header->w4u8_stream_fence_mode),
        qbh_w4u8_decode_softmax_mode_name(
            header->w4u8_decode_softmax_mode),
        header->w4u8_decode_lm_head_group_tiles,
        header->w4u8_qkv_ring_expand_workers,
        header->fp16_norm_rows_per_task,
        header->fp16_norm_contexts,
        header->w4u8_down_hmx_batch_outputs,
        header->attribution_enabled != 0U ? "on" : "off",
        header->numerical_audit_enabled != 0U ? "on" : "off",
        qbh_residual_mode_name(header->residual_mode),
        variant == QBH_BLOCK_F16F16
            ? qbh_f16f16_projection_mode_name(
                  header->f16f16_projection_mode)
            : "not_applicable",
        variant == QBH_BLOCK_W4F16
            ? qbh_w4f16_pipeline_mode_name(
                  header->w4f16_pipeline_mode)
            : "not_applicable",
        qbh_attention_pack_mode_name(header->attention_pack_mode),
        qbh_attention_pipeline_mode_name(
            header->attention_pipeline_mode),
        header->attention_hvx_contexts,
        qbh_mlp_mode_name(header->mlp_mode),
        header->mlp_hvx_contexts,
        header->mlp_chunk_vectors,
        qbh_crouton_boundary_mode_name(
            header->crouton_boundary_mode),
        qbh_w4u8_qkvo_pipeline_mode_name(
            header->w4u8_qkvo_pipeline_mode),
        variant == QBH_BLOCK_W4F16 ? "hmx_output_per_channel"
                                   : "not_applicable",
        warmup_result, warmup_run_index, warmup_end - warmup_start,
        warmup_metrics.max_abs, warmup_metrics.mean_abs,
        warmup_metrics.rmse, warmup_metrics.cosine,
        warmup_metrics.mismatches, warmup_metrics.max_lsb, repeats,
        header->prepared_session_run_index, measured_result,
        header->dsp_status, header->numerical_status,
        header->attention_qk_max_abs,
        header->attention_probability_max_abs,
        header->attention_av_max_abs,
        header->common_op_rms_max_abs,
        header->common_op_rms_cosine,
        header->common_op_rope_max_abs,
        header->common_op_rope_cosine,
        header->common_op_softmax_max_abs,
        header->common_op_softmax_cosine,
        header->common_op_silu_max_abs,
        header->common_op_silu_cosine,
        header->common_op_nonfinite_count,
        header->common_op_softmax_mask_violation_count,
        header->projection_failure_result,
        header->projection_failure_index,
        header->projection_failure_n_tile,
        header->projection_failure_step,
        header->w4f16_expand_mismatch_count,
        header->w4f16_expand_first_logical_index,
        header->w4f16_expand_expected_half_bits,
        header->w4f16_expand_actual_half_bits,
        header->w4f16_hvx_workers_created,
        header->w4f16_hvx_workers_locked,
        header->w4f16_requested_hvx_workers,
        header->w4f16_region_tiles,
        header->w4f16_pool_status,
        header->f16f16_weight_batch_n_tiles,
        header->w4f16_active_worker_min,
        header->w4f16_active_worker_max,
        header->w4f16_effective_region_min,
        header->w4f16_effective_region_max,
        header->mlp_hvx_workers_created,
        header->mlp_hvx_workers_locked,
        header->mlp_pool_status,
        header->mlp_silu_chunk_count,
        header->mlp_stream_group_count,
        header->mlp_down_pack_skipped,
        header->mlp_down_input_hash,
        header->attention_hvx_workers_created,
        header->attention_hvx_workers_locked,
        header->attention_pool_status,
        header->attention_qk_norm_task_count,
        header->fp16_qk_norm_pair_task_count,
        header->attention_softmax_task_count,
        header->u8_attention_softmax_shuffle4_row_group_count,
        header->attention_gqa_group_count,
        header->u8_attention_group_count,
        header->u8_attention_qk_execution_count,
        header->u8_attention_av_execution_count,
        header->u8_attention_score_saturation_count,
        header->u8_attention_v_recenter_saturation_count,
        header->u8_attention_probability_mask_violation_count,
        header->u8_attention_probability_row_sum_min,
        header->u8_attention_probability_row_sum_max,
        header->u8_attention_direct_o_tile_count,
        header->u8_attention_qkv_unpack_skipped,
        header->u8_attention_fused_k_operand_mismatch_count,
        header->w4u8_qkv_batch_n_tiles,
        header->w4u8_qkv_batch_count,
        header->w4u8_qkvo_prefetch_count,
        header->w4u8_qkvo_overlap_schedule_count,
        header->w4u8_qk_pair_kernel_mode_observed,
        header->w4u8_qk_quarter_pair_count,
        header->w4u8_decode_softmax_hvx_tile4_call_count,
        header->w4u8_decode_softmax_hvx_tile4_mismatch_count,
        header->u8_attention_expected_score_hash,
        header->u8_attention_actual_score_hash,
        header->u8_attention_expected_probability_hash,
        header->u8_attention_actual_probability_hash,
        header->u8_attention_expected_av_hash,
        header->u8_attention_actual_av_hash,
        header->u8_input_norm_actual_hash,
        header->u8_attention_audit_ddr_write_bytes,
        header->w4u8_boundary_audit_enabled,
        header->w4u8_boundary_audit_ddr_write_bytes,
        header->crouton_qkv_projection_count,
        header->crouton_qkv_unpack_skipped,
        header->crouton_qk_operand_count,
        header->crouton_av_weight_count,
        header->crouton_av_o_head_count,
        header->crouton_av_unpack_skipped,
        header->crouton_norm_projection_count,
        header->crouton_q_operand_mismatch_count,
        header->crouton_k_operand_mismatch_count,
        header->crouton_v_operand_mismatch_count,
        header->qkv_operand_audit_tensor_count,
        header->qkv_schedule_command_count,
        header->qkv_schedule_trace_hash,
        measured_end - measured_start,
        (double)(measured_end - measured_start) / repeats,
        measured_metrics.max_abs, measured_metrics.mean_abs,
        measured_metrics.rmse, measured_metrics.cosine,
        measured_metrics.mismatches, measured_metrics.max_lsb,
        output_hash,
        header->vtcm_requested_bytes, header->vtcm_acquired_bytes,
        header->vtcm_peak_plan_bytes, header->block_invocation_count,
        header->weight_ddr_read_bytes,
        header->weight_dma_descriptor_count,
        header->boundary_ddr_read_bytes,
        header->boundary_ddr_write_bytes,
        header->intermediate_ddr_read_bytes,
        header->intermediate_ddr_write_bytes,
        header->intermediate_dma_descriptor_count,
        header->intermediate_spill_fill_count,
        header->hmx_command_count, header->hmx_fp16_tile_pair_count,
        header->hmx_u8s8_tile_pair_count, header->input_stage_ticks,
        header->metadata_stage_ticks, header->input_norm_ticks,
        header->qkv_projection_ticks, header->qk_norm_rope_ticks,
        header->attention_ticks, header->o_projection_ticks,
        header->post_attention_residual_ticks,
        header->post_attention_norm_ticks, header->gate_up_ticks,
        header->activation_ticks, header->down_ticks,
        header->final_residual_ticks, header->output_stage_ticks,
        header->total_ticks, header->invocation_ticks,
        header->runtime_setup_ticks, header->runtime_teardown_ticks,
        header->stage_boundary_ticks,
        header->ledger_named_ticks,
        header->ledger_unattributed_ticks,
        header->input_norm_audit_ticks,
        header->qkv_audit_ticks,
        header->qk_norm_rope_audit_ticks,
        header->o_projection_audit_ticks,
        header->post_attention_residual_audit_ticks,
        header->post_attention_norm_audit_ticks,
        header->gate_up_audit_ticks,
        header->activation_audit_ticks,
        header->down_audit_ticks,
        header->final_residual_audit_ticks,
        header->attention_setup_ticks,
        header->attention_qk_pack_ticks,
        header->attention_qk_hmx_ticks,
        header->attention_qk_unpack_ticks,
        header->attention_qk_audit_ticks,
        header->attention_softmax_ticks,
        header->attention_softmax_audit_ticks,
        header->attention_av_pack_ticks,
        header->attention_av_hmx_ticks,
        header->attention_av_unpack_ticks,
        header->attention_av_audit_ticks,
        header->attention_gqa_pipeline_ticks,
        header->attention_unattributed_ticks,
        header->u8_attention_qk_norm_rope_ticks,
        header->u8_attention_k_pack_ticks,
        header->u8_attention_v_pack_ticks,
        header->u8_attention_qk_hmx_ticks,
        header->u8_attention_qk_hmx_ticks +
            header->u8_attention_av_hmx_ticks,
        header->u8_attention_qk_requant_ticks,
        header->u8_attention_softmax_ticks,
        header->u8_attention_qk_requant_ticks +
            header->u8_attention_softmax_ticks,
        header->u8_attention_av_hmx_ticks,
        header->u8_attention_av_requant_ticks,
        header->u8_attention_pipeline_wait_ticks,
        header->w4u8_qkvo_weight_expand_ticks,
        header->w4u8_qkvo_prefetch_wait_ticks,
        header->w4u8_qkvo_hmx_lifetime_ticks,
        header->w4u8_qkv_ring_slot_count,
        header->w4u8_qkv_ring_expand_worker_count,
        header->w4u8_qkv_ring_prep_worker_count,
        header->w4u8_qkv_ring_dispatch_count,
        header->w4u8_qkv_ring_batch_count,
        header->w4u8_qkv_ring_expand_task_count,
        header->w4u8_qkv_ring_hmx_dispatch_count,
        header->w4u8_qkv_ring_head_publish_count,
        header->w4u8_qkv_ring_pipeline_ticks,
        header->w4u8_qkv_ring_dma_wait_ticks,
        header->w4u8_qkv_ring_producer_slot_wait_ticks,
        header->w4u8_qkv_ring_expand_ticks,
        header->w4u8_qkv_ring_hmx_ready_wait_ticks,
        header->w4u8_qkv_ring_hmx_compute_ticks,
        header->w4u8_qkv_ring_pool_wait_ticks,
        header->w4u8_input_norm_task_count,
        header->w4u8_input_norm_main_work_ticks,
        header->w4u8_input_norm_worker_work_ticks,
        header->w4u8_input_norm_pool_wait_ticks,
        header->w4u8_residual_active_contexts,
        header->w4u8_post_residual_task_count,
        header->w4u8_final_residual_task_count,
        header->w4u8_post_residual_main_work_ticks,
        header->w4u8_post_residual_worker_work_ticks,
        header->w4u8_post_residual_pool_wait_ticks,
        header->w4u8_final_residual_main_work_ticks,
        header->w4u8_final_residual_worker_work_ticks,
        header->w4u8_final_residual_pool_wait_ticks,
        header->fp16_input_norm_task_count,
        header->fp16_input_norm_active_contexts,
        header->fp16_input_norm_main_work_ticks,
        header->fp16_input_norm_worker_work_ticks,
        header->fp16_input_norm_pool_wait_ticks,
        header->fp16_post_residual_norm_task_count,
        header->fp16_post_residual_norm_active_contexts,
        header->fp16_post_residual_norm_main_work_ticks,
        header->fp16_post_residual_norm_worker_work_ticks,
        header->fp16_post_residual_norm_pool_wait_ticks,
        header->w4f16_gate_up_effective_region_tiles,
        header->w4f16_gate_up_scale_cache_bytes,
        header->w4f16_gate_up_weight_dma_ticks,
        header->w4f16_gate_up_expand_ticks,
        header->w4f16_gate_up_expand_work_ticks,
        header->w4f16_gate_up_expand_pool_wait_ticks,
        header->w4f16_gate_up_prefetch_wait_ticks,
        header->w4f16_gate_up_hmx_wait_ticks,
        header->w4f16_gate_up_hmx_tail_wait_ticks,
        header->w4f16_gate_up_unpack_ticks,
        header->w4f16_gate_up_stream_work_ticks,
        header->w4f16_gate_up_stream_ready_wait_ticks,
        header->w4f16_gate_up_stream_join_wait_ticks,
        header->w4f16_gate_up_hmx_command_count,
        header->w4f16_gate_up_scale_init_ticks,
        header->w4u8_mlp_vtcm_base_offset,
        header->w4u8_mlp_vtcm_plan_bytes,
        header->w4u8_mlp_lut_vtcm_bytes,
        header->w4u8_mlp_gather_scratch_vtcm_bytes,
        header->w4u8_mlp_gate_up_hvx_workers,
        header->w4u8_mlp_down_hvx_workers,
        header->w4u8_mlp_gate_up_hmx_batch_n_tiles,
        header->w4u8_mlp_down_hmx_batch_n_tiles,
        header->w4u8_mlp_down_in_command_slot_release_count,
        header->w4u8_mlp_down_producer_progress_command_count,
        header->w4u8_mlp_gate_up_expanded_slot_count,
        header->w4u8_mlp_pair_publish_count,
        header->w4u8_mlp_pair_consume_count,
        header->w4u8_mlp_gate_up_hvx_hmx_overlap,
        header->w4u8_mlp_down_hvx_hmx_overlap,
        header->w4u8_mlp_gate_up_hvx_parallel_overlap,
        header->w4u8_mlp_down_hvx_parallel_overlap,
        header->w4u8_mlp_input_pack_skipped,
        header->w4u8_mlp_output_unpack_skipped,
        header->w4u8_mlp_input_pack_ticks,
        header->w4u8_mlp_output_unpack_ticks,
        header->w4u8_mlp_gate_up_pipeline_ticks,
        header->w4u8_mlp_gate_up_hmx_command_count,
        header->w4u8_mlp_down_pipeline_ticks,
        header->w4u8_mlp_down_hmx_command_count,
        header->w4u8_mlp_activation_work_ticks,
        header->w4u8_mlp_weight_stage_ticks,
        header->w4u8_mlp_weight_expand_ticks,
        header->w4u8_mlp_hmx_compute_ticks,
        header->w4u8_mlp_hmx_ready_wait_ticks,
        header->w4u8_mlp_producer_slot_wait_ticks,
        header->w4u8_mlp_expanded_slot_wait_ticks,
        header->w4u8_gate_up_persistent_hvx_dispatch_count,
        header->w4u8_gate_up_persistent_hvx_worker_count,
        header->w4u8_gate_up_transient_hvx_thread_count,
        header->w4u8_down_persistent_hvx_dispatch_count,
        header->w4u8_down_persistent_hvx_worker_count,
        header->w4u8_down_transient_hvx_thread_count,
        header->weight_dma_ticks,
        header->hmx_compute_ticks, header->projection_pack_ticks,
        header->w4f16_expand_ticks,
        header->projection_hmx_wait_ticks,
        header->projection_unpack_ticks,
        header->hmx_ready_wait_ticks,
        header->w4f16_streamed_command_count,
        header->w4f16_expand_work_ticks,
        header->w4f16_expand_region_count,
        header->w4f16_prefetch_count,
        header->w4f16_prefetch_wait_ticks,
        header->f16f16_prefetch_count,
        header->f16f16_prefetch_wait_ticks,
        header->w4f16_first_expand_ticks,
        header->w4f16_steady_expand_ticks,
        header->w4f16_expand_pool_wait_ticks,
        header->w4f16_hmx_tail_wait_ticks,
        header->w4f16_early_region_command_count,
        header->w4f16_cross_prefetch_count,
        header->w4f16_cross_prefetch_wait_ticks,
        header->w4f16_cross_prefetch_lifetime_ticks,
        header->mlp_silu_main_work_ticks,
        header->mlp_silu_worker_work_ticks,
        header->mlp_silu_pool_wait_ticks,
        header->mlp_stream_worker_work_ticks,
        header->mlp_stream_main_work_ticks,
        header->mlp_stream_ready_wait_ticks,
        header->mlp_stream_join_wait_ticks,
        header->attention_qk_norm_main_work_ticks,
        header->attention_qk_norm_worker_work_ticks,
        header->attention_qk_norm_pool_wait_ticks,
        header->attention_softmax_main_work_ticks,
        header->attention_softmax_worker_work_ticks,
        header->attention_softmax_pool_wait_ticks,
        header->attention_gqa_worker_work_ticks,
        header->attention_gqa_hmx_wait_ticks,
        header->attention_gqa_queue_wait_ticks,
        header->crouton_qkv_transform_ticks,
        header->crouton_av_o_copy_ticks,
        header->crouton_norm_store_ticks,
        release_result, close_result);

    exit_code = warmup_result == AEE_SUCCESS &&
                        measured_result == AEE_SUCCESS &&
                        release_result == AEE_SUCCESS &&
                        close_result == AEE_SUCCESS &&
                        header->dsp_status == QBH_BLOCK_STATUS_OK &&
                        header->numerical_status ==
                            QBH_BLOCK_NUMERICAL_OK &&
                        header->vtcm_requested_bytes ==
                            QBH_EXPECTED_FULL_VTCM_BYTES &&
                        header->vtcm_acquired_bytes ==
                            QBH_EXPECTED_FULL_VTCM_BYTES &&
                        header->block_invocation_count == repeats &&
                        header->intermediate_ddr_read_bytes == 0U &&
                        header->intermediate_ddr_write_bytes == 0U &&
                        header->intermediate_dma_descriptor_count == 0U &&
                        header->intermediate_spill_fill_count == 0U &&
                        (scan_mode == QBH_BLOCK_SCAN_DISABLED ||
                         (kv_cache_mismatches == 0U &&
                          measured_metrics.mismatches == 0U)) &&
                        (!qbh_block_mlp_is_w4u8_streaming(mlp_mode) ||
                         measured_metrics.mismatches == 0U) &&
                        isfinite(measured_metrics.cosine) &&
                        measured_metrics.cosine > 0.90
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
    return exit_code;
}
