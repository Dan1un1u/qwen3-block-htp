#ifndef QWEN3_BLOCK_HTP_W4_U8_EXPAND_H
#define QWEN3_BLOCK_HTP_W4_U8_EXPAND_H

#include <stdint.h>

void qbh_expand_w4_to_s8_hvx(const uint8_t *packed_w4,
                             const uint8_t *channel_scales,
                             int8_t *expanded_s8, uint32_t k_tiles);

void qbh_unpack_w4_to_s8_hvx(const uint8_t *packed_w4,
                             int8_t *expanded_s8,
                             uint32_t k_tiles);

void qbh_unpack_w4_to_s8_hvx_interleaved2(
    const uint8_t *packed_w4, int8_t *expanded_s8,
    uint32_t k_tiles);

void qbh_copy_s8_hmx_tiles_hvx(const int8_t *source,
                               int8_t *destination,
                               uint32_t k_tiles);

void qbh_expand_w4_to_f16_hvx(const uint8_t *packed_w4,
                              const float *channel_scales,
                              void *expanded_f16,
                              uint32_t k_tiles);

void qbh_unpack_w4_to_f16_hvx(const uint8_t *packed_w4,
                              void *expanded_f16,
                              uint32_t k_tiles);

uint32_t qbh_audit_w4_to_f16_tile(
    const uint8_t *packed_w4, const float *channel_scales,
    const void *expanded_f16, uint32_t *first_logical_index,
    uint32_t *expected_half_bits, uint32_t *actual_half_bits);

uint32_t qbh_audit_unscaled_w4_to_f16_tile(
    const uint8_t *packed_w4, const void *expanded_f16,
    uint32_t *first_logical_index, uint32_t *expected_half_bits,
    uint32_t *actual_half_bits);

void qbh_copy_hmx_bias_hvx(const uint8_t *source, uint8_t *destination);

#endif
