#include <AEEStdErr.h>
#include <errno.h>
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

#define QBH_HOST_ALIGNMENT ((size_t)128)
#define QBH_HOST_PATH_BYTES ((size_t)1024)

struct qbh_file_slot {
    char path[QBH_HOST_PATH_BYTES];
    uint32_t expected_bytes;
    uint32_t offset;
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
    double cosine;
    uint64_t mismatches;
    uint32_t max_lsb;
};

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
    return mode == QBH_BLOCK_QKV_SCHEDULE_Q_PREFIX4_K_ALL
               ? "q_prefix4_k_all" : "control";
}

static const char *qbh_w4f16_group_fence_mode_name(uint32_t mode) {
    return mode == QBH_BLOCK_W4F16_GROUP_FENCE_JOIN_ONLY
               ? "join_only" : "control";
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
    destination = shared + header->output_offset;
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
            projection_bytes > header->output_bytes - cache_bytes) {
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

static int qbh_load_qparams(struct qbh_block_header *header,
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
                header->qparams[target].scale = source[record].scale;
                header->qparams[target].zero_point =
                    source[record].zero_point;
                header->qparams[target].minimum = source[record].minimum;
                header->qparams[target].maximum = source[record].maximum;
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
    metrics.cosine = actual_norm > 0.0 && reference_norm > 0.0
                         ? dot / sqrt(actual_norm * reference_norm)
                         : 0.0;
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
    metrics.cosine = actual_norm > 0.0 && reference_norm > 0.0
                         ? dot / sqrt(actual_norm * reference_norm)
                         : 0.0;
    return metrics;
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
    struct qbh_file_slot w4u8_lut_slot;
    struct qbh_file_slot weight_slots[QBH_BLOCK_PROJECTION_COUNT];
    struct qbh_file_slot scale_slots[QBH_BLOCK_PROJECTION_COUNT];
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
    uint32_t w4u8_stream_fence_mode =
        QBH_BLOCK_W4U8_STREAM_FENCE_CONTROL;
    uint32_t w4u8_gate_up_ring_slots = 8U;
    uint32_t w4u8_qkv_ring_expand_workers = 0U;
    uint32_t element_bytes;
    uint32_t output_bytes;
    size_t w4u8_gate_up_bundle_offset = 0U;
    size_t w4u8_down_bundle_offset = 0U;
    size_t attention_audit_output_offset = 0U;
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
    int exit_code = 1;
    char file_name[128];

    memset(&warmup_metrics, 0, sizeof(warmup_metrics));
    memset(&measured_metrics, 0, sizeof(measured_metrics));
    memset(&w4u8_lut_slot, 0, sizeof(w4u8_lut_slot));
    memset(&attention_config_slot, 0, sizeof(attention_config_slot));
    memset(attention_audit_slots, 0, sizeof(attention_audit_slots));
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
            } else {
                w4f16_group_fence_mode = UINT32_MAX;
            }
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
        w4f16_hvx_workers == 0U || w4f16_hvx_workers > 3U ||
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
         w4f16_hvx_workers != 3U) ||
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
            QBH_BLOCK_W4U8_QK_PAIR_QUARTER_TILED ||
        (variant != QBH_BLOCK_W4U8 &&
         u8_norm_reduction_mode !=
             QBH_BLOCK_U8_NORM_REDUCTION_SCALAR) ||
        (variant != QBH_BLOCK_W4U8 &&
         w4u8_qk_pair_kernel_mode !=
             QBH_BLOCK_W4U8_QK_PAIR_SERIAL_INNER) ||
        (w4u8_qk_pair_kernel_mode ==
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
            QBH_BLOCK_QKV_SCHEDULE_Q_PREFIX4_K_ALL ||
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
         w4f16_hvx_workers != 3U) ||
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
                        "input_norm|post_norm|norms|all|"
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
    output_bytes = QBH_BLOCK_M * QBH_BLOCK_HIDDEN * element_bytes;
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
        qbh_prepare_slot(&qparam_slot, argv[1], "qparams_u8.bin",
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
                         &cursor) != 0 ||
        qbh_prepare_slot(&rope_slots[0], argv[1], "rope_cos_f16.bin",
                         QBH_BLOCK_M * QBH_BLOCK_HEAD_DIM *
                             sizeof(uint16_t),
                         &cursor) != 0 ||
        qbh_prepare_slot(&rope_slots[1], argv[1], "rope_sin_f16.bin",
                         QBH_BLOCK_M * QBH_BLOCK_HEAD_DIM *
                             sizeof(uint16_t),
                         &cursor) != 0) {
        fprintf(stderr, "package common tensor audit failed\n");
        return 2;
    }
    if (qbh_attention_u8_enabled(attention_pipeline_mode) &&
        qbh_prepare_slot(
            &attention_config_slot, argv[1],
            "attention_config_all_groups.bin",
            QBH_BLOCK_ATTENTION_CONFIG_BYTES, &cursor) != 0) {
        fprintf(stderr, "integer Attention config audit failed\n");
        return 2;
    }
    if (qbh_attention_u8_enabled(attention_pipeline_mode) &&
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
    if (qbh_block_mlp_is_w4u8_streaming(mlp_mode) &&
        qbh_prepare_slot(&w4u8_lut_slot, argv[1],
                         "silu_up_lut_u16.bin", QBH_MLP_LUT_BYTES,
                         &cursor) != 0) {
        fprintf(stderr, "W4U8 streaming LUT audit failed\n");
        return 2;
    }

    for (uint32_t projection = 0;
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
        if (variant == QBH_BLOCK_W4U8) {
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
        if (qbh_block_mlp_is_w4u8_streaming(mlp_mode)) {
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
        if (qbh_attention_u8_enabled(attention_pipeline_mode) &&
            numerical_audit_enabled != 0U) {
            cursor = qbh_align_up_size(cursor, QBH_HOST_ALIGNMENT);
            attention_audit_output_offset = cursor;
            if (QBH_BLOCK_U8_ATTENTION_AUDIT_BYTES >
                UINT32_MAX - cursor) {
                return 2;
            }
            cursor += QBH_BLOCK_U8_ATTENTION_AUDIT_BYTES;
        }
        total_bytes = cursor;
        if (total_bytes > INT_MAX) {
            fprintf(stderr, "rpcmem package too large: %zu\n", total_bytes);
            return 2;
        }
        shared = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM,
                              RPCMEM_FLAG_UNCACHED, (int)total_bytes);
        if (shared == NULL) {
            fprintf(stderr, "rpcmem_alloc failed for %zu bytes\n",
                    total_bytes);
            return 2;
        }
        memset(shared, 0, total_bytes);
        header = (struct qbh_block_header *)shared;
        header->output_offset = (uint32_t)output_offset;
    }

    if (qbh_read_slot(shared, &input_slot) != 0 ||
        qbh_read_slot(shared, &reference_slot) != 0 ||
        qbh_read_slot(shared, &qparam_slot) != 0) {
        fprintf(stderr, "package boundary tensor read failed\n");
        goto cleanup;
    }
    for (uint32_t index = 0; index < 4U; ++index) {
        if (qbh_read_slot(shared, &norm_slots[index]) != 0) {
            goto cleanup;
        }
    }
    for (uint32_t index = 0; index < 2U; ++index) {
        if (qbh_read_slot(shared, &rope_slots[index]) != 0) {
            goto cleanup;
        }
    }
    if (qbh_block_mlp_is_w4u8_streaming(mlp_mode) &&
        qbh_read_slot(shared, &w4u8_lut_slot) != 0) {
        goto cleanup;
    }
    if (qbh_attention_u8_enabled(attention_pipeline_mode) &&
        (qbh_read_slot(shared, &attention_config_slot) != 0 ||
         qbh_read_slot(shared, &attention_audit_slots[0]) != 0 ||
         qbh_read_slot(shared, &attention_audit_slots[1]) != 0 ||
         qbh_read_slot(shared, &attention_audit_slots[2]) != 0)) {
        goto cleanup;
    }
    for (uint32_t projection = 0;
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
    header->w4u8_stream_fence_mode = w4u8_stream_fence_mode;
    header->w4u8_gate_up_ring_slots = w4u8_gate_up_ring_slots;
    header->w4u8_qkv_ring_expand_workers =
        w4u8_qkv_ring_expand_workers;
    header->w4u8_qk_pair_kernel_mode =
        w4u8_qk_pair_kernel_mode;
    header->input_offset = input_slot.offset;
    header->input_bytes = input_slot.expected_bytes;
    header->output_bytes = output_bytes;
    header->reference_offset = reference_slot.offset;
    header->reference_bytes = reference_slot.expected_bytes;
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
    header->rope_cos_offset = rope_slots[0].offset;
    header->rope_cos_bytes = rope_slots[0].expected_bytes;
    header->rope_sin_offset = rope_slots[1].offset;
    header->rope_sin_bytes = rope_slots[1].expected_bytes;
    if (qbh_attention_u8_enabled(attention_pipeline_mode)) {
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
    if (qbh_block_mlp_is_w4u8_streaming(mlp_mode)) {
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
    if (qbh_load_qparams(header, shared + qparam_slot.offset) != 0) {
        fprintf(stderr, "qparam record audit failed\n");
        goto cleanup;
    }

    {
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
    if (qbh_block_mlp_is_w4u8_streaming(mlp_mode) &&
        qbh_build_w4u8_streaming_bundles(
            header, shared, &w4u8_gate_up_layout,
            &w4u8_down_layout) != 0) {
        fprintf(stderr, "W4U8 streaming bundle construction failed\n");
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
        ? qbh_compare_u8(
              shared + header->output_offset,
              shared + header->reference_offset,
              QBH_BLOCK_M * QBH_BLOCK_HIDDEN,
              &header->qparams[QBH_BLOCK_QP_BLOCK_OUTPUT])
        : qbh_compare_f16(
              (const uint16_t *)(shared + header->output_offset),
              (const uint16_t *)(shared + header->reference_offset),
              QBH_BLOCK_M * QBH_BLOCK_HIDDEN);

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
        ? qbh_compare_u8(
              shared + header->output_offset,
              shared + header->reference_offset,
              QBH_BLOCK_M * QBH_BLOCK_HIDDEN,
              &header->qparams[QBH_BLOCK_QP_BLOCK_OUTPUT])
        : qbh_compare_f16(
              (const uint16_t *)(shared + header->output_offset),
              (const uint16_t *)(shared + header->reference_offset),
              QBH_BLOCK_M * QBH_BLOCK_HIDDEN);
    output_hash = qbh_fnv1a64(
        shared + header->output_offset, header->output_bytes);
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
        }
    }

    release_result = qbh_session_release(&session);
    close_result = qbh_session_close(&session);
    printf(
        "{\"experiment\":\"EXP-0124\","
        "\"execution_unit\":\"qwen3_layer14_complete_block_m64\","
        "\"variant\":\"%s\",\"attention_compute\":\"%s\","
        "\"projection_compute\":\"%s\","
        "\"common_ops_mode\":\"%s\","
        "\"u8_norm_reduction_mode\":\"%s\","
        "\"w4u8_qk_pair_kernel_mode\":%" PRIu32 ","
        "\"fp16_common_schedule_mode\":\"%s\","
        "\"qkv_schedule_mode\":\"%s\","
        "\"w4f16_group_fence_mode\":\"%s\","
        "\"w4u8_stream_fence_mode\":\"%s\","
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
        "\"u8_attention_expected_score_hash\":\"%016" PRIx64 "\","
        "\"u8_attention_actual_score_hash\":\"%016" PRIx64 "\","
        "\"u8_attention_expected_probability_hash\":\"%016" PRIx64 "\","
        "\"u8_attention_actual_probability_hash\":\"%016" PRIx64 "\","
        "\"u8_attention_expected_av_hash\":\"%016" PRIx64 "\","
        "\"u8_attention_actual_av_hash\":\"%016" PRIx64 "\","
        "\"u8_input_norm_actual_hash\":\"%016" PRIx64 "\","
        "\"u8_attention_audit_ddr_write_bytes\":%" PRIu32 ","
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
        qbh_w4u8_stream_fence_mode_name(
            header->w4u8_stream_fence_mode),
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
        header->u8_attention_expected_score_hash,
        header->u8_attention_actual_score_hash,
        header->u8_attention_expected_probability_hash,
        header->u8_attention_actual_probability_hash,
        header->u8_attention_expected_av_hash,
        header->u8_attention_actual_av_hash,
        header->u8_input_norm_actual_hash,
        header->u8_attention_audit_ddr_write_bytes,
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
