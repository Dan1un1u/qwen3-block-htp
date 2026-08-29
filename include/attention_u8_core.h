#ifndef QWEN3_BLOCK_HTP_ATTENTION_U8_CORE_H
#define QWEN3_BLOCK_HTP_ATTENTION_U8_CORE_H

#include <stdint.h>

#include "attention_protocol.h"

#define QBH_ATTN_U8_GROUP_SCRATCH_BYTES UINT32_C(18432)
#define QBH_ATTN_U8_K_WEIGHT_OFFSET UINT32_C(0)
#define QBH_ATTN_U8_K_WEIGHT_BYTES UINT32_C(8192)
#define QBH_ATTN_U8_V_WEIGHT_OFFSET UINT32_C(8192)
#define QBH_ATTN_U8_V_WEIGHT_BYTES UINT32_C(8192)
#define QBH_ATTN_U8_QK_BIAS_OFFSET UINT32_C(16384)
#define QBH_ATTN_U8_QK_BIAS_BYTES UINT32_C(512)
#define QBH_ATTN_U8_AV_BIAS_OFFSET UINT32_C(16896)
#define QBH_ATTN_U8_AV_BIAS_BYTES UINT32_C(1024)
#define QBH_ATTN_U8_SOFTMAX_SCRATCH_OFFSET UINT32_C(17920)
#define QBH_ATTN_U8_SOFTMAX_SCRATCH_BYTES UINT32_C(256)

struct qbh_attention_u8_telemetry {
    uint32_t score_saturation_count;
    uint32_t v_recenter_saturation_count;
    uint32_t probability_mask_violation_count;
    uint32_t probability_row_sum_min;
    uint32_t probability_row_sum_max;
};

void qbh_attention_u8_pack_k_native(
    const uint8_t *k_head_tiles,
    const struct qbh_attention_config *config,
    int8_t *weight_tiles, uint32_t *bias_words);

void qbh_attention_u8_pack_v_native(
    const uint8_t *v_head_tiles,
    const struct qbh_attention_config *config,
    int8_t *weight_tiles, uint32_t *bias_words,
    uint32_t *saturation_count);

void qbh_attention_u8_requant_qk(
    uint8_t *score_tiles,
    const struct qbh_attention_config *config,
    uint32_t *saturation_count);

void qbh_attention_u8_softmax_group(
    const uint8_t *score_tiles, uint8_t *probability_tiles,
    uint8_t *scratch,
    const struct qbh_attention_config *config,
    struct qbh_attention_u8_telemetry *telemetry);

void qbh_attention_u8_softmax_group_paired_rows(
    const uint8_t *score_tiles, uint8_t *probability_tiles,
    uint8_t *scratch,
    const struct qbh_attention_config *config,
    struct qbh_attention_u8_telemetry *telemetry);

void qbh_attention_u8_requant_av(
    uint8_t *output_tiles,
    const struct qbh_attention_config *config);

#endif
