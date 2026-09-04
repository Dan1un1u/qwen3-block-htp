#ifndef QWEN3_BLOCK_HTP_ATTENTION_U8_CORE_H
#define QWEN3_BLOCK_HTP_ATTENTION_U8_CORE_H

#include <stdint.h>

#include "attention_protocol.h"

#define QBH_ATTN_U8_GROUP_SCRATCH_BYTES UINT32_C(18944)
#define QBH_ATTN_U8_K_WEIGHT_OFFSET UINT32_C(0)
#define QBH_ATTN_U8_K_WEIGHT_BYTES UINT32_C(8192)
#define QBH_ATTN_U8_V_WEIGHT_OFFSET UINT32_C(8192)
#define QBH_ATTN_U8_V_WEIGHT_BYTES UINT32_C(8192)
#define QBH_ATTN_U8_QK_BIAS_OFFSET UINT32_C(16384)
#define QBH_ATTN_U8_QK_BIAS_BYTES UINT32_C(512)
#define QBH_ATTN_U8_AV_BIAS_OFFSET UINT32_C(16896)
#define QBH_ATTN_U8_AV_BIAS_BYTES UINT32_C(1024)
#define QBH_ATTN_U8_SOFTMAX_SCRATCH_OFFSET UINT32_C(17920)
/* Softmax runs after V packing and may reuse the vgather/V-deal workspace.
 * The template candidate needs 704 bytes, rounded to six HVX vectors. */
#define QBH_ATTN_U8_SOFTMAX_SCRATCH_BYTES UINT32_C(768)
#define QBH_ATTN_U8_SOFTMAX_TEMPLATE_OFFSET UINT32_C(256)
#define QBH_ATTN_U8_SOFTMAX_CARRIER_BYTES UINT32_C(1024)
#define QBH_ATTN_U8_VGATHER_SCRATCH_OFFSET \
    QBH_ATTN_U8_SOFTMAX_SCRATCH_OFFSET
#define QBH_ATTN_U8_VGATHER_SCRATCH_BYTES UINT32_C(256)
#define QBH_ATTN_U8_VGATHER_LUT_OFFSET UINT32_C(18432)
#define QBH_ATTN_U8_VGATHER_LUT_BYTES UINT32_C(512)

struct qbh_attention_u8_telemetry {
    uint32_t score_saturation_count;
    uint32_t v_recenter_saturation_count;
    uint32_t probability_mask_violation_count;
    uint32_t probability_row_sum_min;
    uint32_t probability_row_sum_max;
    uint32_t dynamic_hvx_tile4_call_count;
    uint32_t dynamic_hvx_tile4_mismatch_count;
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

void qbh_attention_u8_pack_v_native_vgather(
    const uint8_t *v_head_tiles,
    const struct qbh_attention_config *config,
    int8_t *weight_tiles, uint32_t *bias_words, uint8_t *scratch,
    uint32_t *saturation_count);

void qbh_attention_u8_pack_v_native_vgather_vdeal(
    const uint8_t *v_head_tiles,
    const struct qbh_attention_config *config,
    int8_t *weight_tiles, uint32_t *bias_words, uint8_t *scratch,
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

void qbh_attention_u8_requant_softmax_group(
    uint8_t *score_tiles, uint8_t *probability_tiles,
    uint8_t *scratch,
    const struct qbh_attention_config *config,
    struct qbh_attention_u8_telemetry *telemetry);

void qbh_attention_u8_requant_softmax_group_lut_templates(
    uint8_t *score_tiles, uint8_t *probability_tiles,
    uint8_t *scratch,
    const struct qbh_attention_config *config,
    struct qbh_attention_u8_telemetry *telemetry);

void qbh_attention_u8_build_sole_lut_template_bank(
    uint8_t *templates);

void qbh_attention_u8_requant_softmax_group_rows_prebuilt_templates(
    uint8_t *score_tiles, uint8_t *probability_tiles,
    uint8_t *scratch,
    const struct qbh_attention_config *config,
    struct qbh_attention_u8_telemetry *telemetry,
    uint32_t first_row, uint32_t row_count);

void qbh_attention_u8_requant_softmax_group_rows_prebuilt_templates_shuffle4(
    uint8_t *score_tiles, uint8_t *probability_tiles,
    uint8_t *scratch, uint8_t *carrier_scratch,
    const struct qbh_attention_config *config,
    struct qbh_attention_u8_telemetry *telemetry,
    uint32_t first_row, uint32_t row_count);

void qbh_attention_u8_requant_av(
    uint8_t *output_tiles,
    const struct qbh_attention_config *config);

void qbh_attention_u8_requant_av_rows(
    uint8_t *output_tiles,
    const struct qbh_attention_config *config,
    uint32_t physical_rows);

void qbh_attention_u8_poison_av_padding(
    uint8_t *output_tiles, uint32_t first_padding_row);

/* EXP-0147 generalized-cache helpers.  These deliberately coexist with the
 * immutable 64x64 fast path above. */
void qbh_attention_u8_native_head_to_row_major(
    const uint8_t *head_tiles, uint8_t *rows,
    uint32_t valid_rows);

void qbh_attention_u8_pack_k_row_major(
    const uint8_t *rows, uint32_t valid_tokens,
    uint32_t padded_tokens,
    const struct qbh_attention_config *config,
    int8_t *weight_tiles, uint32_t *bias_words);

void qbh_attention_u8_pack_v_row_major(
    const uint8_t *rows, uint32_t valid_tokens,
    uint32_t padded_tokens,
    const struct qbh_attention_config *config,
    int8_t *weight_tiles, uint32_t *bias_words,
    uint32_t *saturation_count);

/* EXP-0155/0156 fixed-carrier append helpers.  The K helper updates one output
 * lane of one [N=32,K=128] carrier tile.  The V helper updates one input
 * lane across the four [N=32,K=32] head-dimension tiles. */
void qbh_attention_u8_update_k_native_token(
    const uint8_t *k_head_tiles, uint32_t source_row,
    uint32_t output_lane,
    const struct qbh_attention_config *config,
    int8_t *n_tile_weight, uint32_t *n_tile_bias);

void qbh_attention_u8_update_v_native_token(
    const uint8_t *v_head_tiles, uint32_t source_row,
    uint32_t input_lane,
    const struct qbh_attention_config *config,
    int8_t *k_tile_weights, uint32_t *saturation_count);

/* EXP-0159 delta-journal helpers.  Rows are contiguous logical U8 vectors;
 * only journaled lanes are patched into the immutable prefill carrier. */
void qbh_attention_u8_update_k_native_row(
    const uint8_t *row, uint32_t output_lane,
    const struct qbh_attention_config *config,
    int8_t *n_tile_weight, uint32_t *n_tile_bias);

void qbh_attention_u8_update_v_native_row(
    const uint8_t *row, uint32_t input_lane,
    const struct qbh_attention_config *config,
    int8_t *k_tile_weights, uint32_t k_tile_stride_bytes,
    uint32_t *saturation_count);

void qbh_attention_u8_patch_k_delta_rows_hvx(
    const uint8_t *rows, uint32_t row_count,
    const struct qbh_attention_config *config,
    int8_t *n_tile_weight, uint32_t *n_tile_bias);

void qbh_attention_u8_prepare_v_delta_lut(
    const struct qbh_attention_config *config, uint8_t *scratch);

void qbh_attention_u8_patch_v_delta_rows_hvx(
    const uint8_t *rows, uint32_t row_count,
    const struct qbh_attention_config *config,
    int8_t *k_tile_weights, uint32_t k_tile_stride_bytes,
    uint8_t *scratch, uint32_t *saturation_count);

void qbh_attention_u8_requant_softmax_dynamic(
    uint8_t *score_tiles, uint8_t *probability_tiles,
    uint32_t query_rows, uint32_t past_tokens,
    uint32_t valid_tokens, uint32_t padded_tokens,
    const struct qbh_attention_config *config,
    struct qbh_attention_u8_telemetry *telemetry,
    uint32_t use_hvx_tile4, uint32_t verify_hvx_tile4);

/* EXP-0161 segmented decode helper.  It converts a histogram of raw HMX QK
 * bytes into the exact per-raw-byte probability map used by the existing
 * log2 Softmax, without retaining a sequence-length probability tensor. */
void qbh_attention_u8_probability_map_from_raw_histogram(
    const uint32_t histogram[256], uint32_t valid_count,
    const struct qbh_attention_config *config,
    uint8_t probability_by_raw[256],
    uint32_t *probability_sum, uint32_t *score_saturation_count);

void qbh_attention_u8_probability_map_from_active_histogram(
    const uint32_t histogram[256], const uint8_t *active_scores,
    uint32_t active_count, uint32_t valid_count,
    const struct qbh_attention_config *config,
    uint8_t probability_by_raw[256],
    uint32_t *probability_sum, uint32_t *score_saturation_count);

#endif
