#ifndef QWEN3_BLOCK_HTP_HVX_U8_OPS_H
#define QWEN3_BLOCK_HTP_HVX_U8_OPS_H

#include <stdint.h>

#include "block_protocol.h"

#define QBH_QK_PAIR_RSQRT_ROWS UINT32_C(16)
#define QBH_QK_PAIR_RSQRT_SCRATCH_BYTES \
    (2U * QBH_QK_PAIR_RSQRT_ROWS * QBH_BLOCK_HEAD_DIM)
#define QBH_QK_PAIR_RSQRT_SCRATCH_OFFSET UINT32_C(40960)
#define QBH_QK_ROPE_SF32_CACHE_OFFSET UINT32_C(65536)
#define QBH_QK_ROPE_SF32_CACHE_BYTES \
    (QBH_BLOCK_M * 8U * QBH_BLOCK_HEAD_DIM)

void qbh_hvx_u8_set_norm_reduction_mode(uint32_t mode);

void qbh_hvx_qk_rope_preconvert_sf32(
    const __fp16 *cosine, const __fp16 *sine,
    uint8_t *rope_sf32_cache);

void qbh_hvx_rms_norm_u8(
    const uint8_t *input,
    const struct qbh_block_qparam *input_qparam,
    const __fp16 *gamma, uint8_t *output,
    const struct qbh_block_qparam *output_qparam,
    uint32_t rows, uint32_t width);

void qbh_hvx_rms_norm_u8_native_activation(
    const uint8_t *input,
    const struct qbh_block_qparam *input_qparam,
    const __fp16 *gamma, uint8_t *output_tiles,
    const struct qbh_block_qparam *output_qparam,
    uint32_t rows, uint32_t width);

void qbh_hvx_rms_norm_u8_native_activation_rows(
    const uint8_t *input,
    const struct qbh_block_qparam *input_qparam,
    const __fp16 *gamma, uint8_t *output_tiles,
    const struct qbh_block_qparam *output_qparam,
    uint32_t first_row, uint32_t row_count, uint32_t width);

void qbh_hvx_residual_add_u8(
    const uint8_t *left,
    const struct qbh_block_qparam *left_qparam,
    const uint8_t *right,
    const struct qbh_block_qparam *right_qparam,
    uint8_t *output,
    const struct qbh_block_qparam *output_qparam,
    uint32_t elements);

void qbh_hvx_residual_add_u8_native_output(
    uint8_t *residual,
    const struct qbh_block_qparam *residual_qparam,
    const uint8_t *addition_tiles,
    const struct qbh_block_qparam *addition_qparam,
    const struct qbh_block_qparam *output_qparam,
    uint32_t rows, uint32_t width);

void qbh_hvx_residual_add_u8_native_output_rows(
    uint8_t *residual,
    const struct qbh_block_qparam *residual_qparam,
    const uint8_t *addition_tiles,
    const struct qbh_block_qparam *addition_qparam,
    const struct qbh_block_qparam *output_qparam,
    uint32_t first_row, uint32_t row_count, uint32_t width);

void qbh_hvx_residual_rms_norm_u8(
    uint8_t *residual,
    const struct qbh_block_qparam *residual_qparam,
    const uint8_t *addition,
    const struct qbh_block_qparam *addition_qparam,
    const struct qbh_block_qparam *sum_qparam,
    const __fp16 *gamma, uint8_t *normalized,
    const struct qbh_block_qparam *normalized_qparam,
    uint32_t rows, uint32_t width);

void qbh_hvx_residual_rms_norm_u8_native_activation(
    uint8_t *residual,
    const struct qbh_block_qparam *residual_qparam,
    const uint8_t *addition,
    const struct qbh_block_qparam *addition_qparam,
    const struct qbh_block_qparam *sum_qparam,
    const __fp16 *gamma, uint8_t *normalized_tiles,
    const struct qbh_block_qparam *normalized_qparam,
    uint32_t rows, uint32_t width);

void qbh_hvx_residual_rms_norm_u8_native_io(
    uint8_t *residual,
    const struct qbh_block_qparam *residual_qparam,
    const uint8_t *addition_tiles,
    const struct qbh_block_qparam *addition_qparam,
    const struct qbh_block_qparam *sum_qparam,
    const __fp16 *gamma, uint8_t *normalized_tiles,
    const struct qbh_block_qparam *normalized_qparam,
    uint32_t rows, uint32_t width);

void qbh_hvx_residual_rms_norm_u8_native_io_rows(
    uint8_t *residual,
    const struct qbh_block_qparam *residual_qparam,
    const uint8_t *addition_tiles,
    const struct qbh_block_qparam *addition_qparam,
    const struct qbh_block_qparam *sum_qparam,
    const __fp16 *gamma, uint8_t *normalized_tiles,
    const struct qbh_block_qparam *normalized_qparam,
    uint32_t first_row, uint32_t row_count, uint32_t width);

void qbh_hvx_qk_norm_rope_u8(
    uint8_t *tensor, uint32_t rows, uint32_t heads,
    uint32_t row_stride, uint32_t head_dim,
    const struct qbh_block_qparam *input_qparam,
    const struct qbh_block_qparam *output_qparam,
    const __fp16 *gamma, const __fp16 *cosine,
    const __fp16 *sine);

void qbh_hvx_qk_norm_rope_u8_native_head(
    uint8_t *head_tiles,
    const struct qbh_block_qparam *input_qparam,
    const struct qbh_block_qparam *output_qparam,
    const __fp16 *gamma, const __fp16 *cosine,
    const __fp16 *sine);

void qbh_hvx_qk_norm_rope_u8_native_head_pair(
    uint8_t *first_head_tiles, uint8_t *second_head_tiles,
    const struct qbh_block_qparam *input_qparam,
    const struct qbh_block_qparam *output_qparam,
    const __fp16 *gamma, const __fp16 *cosine,
    const __fp16 *sine, uint8_t *rsqrt_scratch,
    const uint8_t *rope_sf32_cache);

void qbh_hvx_qk_norm_rope_u8_native_k_head(
    uint8_t *head_tiles,
    const struct qbh_block_qparam *input_qparam,
    const struct qbh_block_qparam *output_qparam,
    const __fp16 *gamma, const __fp16 *cosine,
    const __fp16 *sine,
    const struct qbh_attention_config *config,
    int8_t *weight_tiles, uint32_t *bias_words);

void qbh_hvx_qk_norm_rope_u8_native_k_head_pair(
    uint8_t *first_head_tiles, uint8_t *second_head_tiles,
    const struct qbh_block_qparam *input_qparam,
    const struct qbh_block_qparam *output_qparam,
    const __fp16 *gamma, const __fp16 *cosine,
    const __fp16 *sine,
    const struct qbh_attention_config *first_config,
    const struct qbh_attention_config *second_config,
    int8_t *first_weight_tiles, int8_t *second_weight_tiles,
    uint32_t *first_bias_words, uint32_t *second_bias_words,
    uint8_t *rsqrt_scratch, const uint8_t *rope_sf32_cache);

void qbh_hvx_expand_u8_to_f16_in_place(
    uint8_t *buffer, uint32_t elements,
    const struct qbh_block_qparam *qparam);

void qbh_hvx_quantize_f16_to_u8(
    const __fp16 *input, uint8_t *output, uint32_t elements,
    const struct qbh_block_qparam *qparam);

#endif
