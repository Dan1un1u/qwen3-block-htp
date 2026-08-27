#ifndef QWEN3_BLOCK_HTP_W4_U8_EXPAND_H
#define QWEN3_BLOCK_HTP_W4_U8_EXPAND_H

#include <stdint.h>

void qbh_expand_w4_to_s8_hvx(const uint8_t *packed_w4,
                             const uint8_t *channel_scales,
                             int8_t *expanded_s8, uint32_t k_tiles);

void qbh_unpack_w4_to_s8_hvx(const uint8_t *packed_w4,
                             int8_t *expanded_s8,
                             uint32_t k_tiles);

void qbh_expand_w4_to_f16_hvx(const uint8_t *packed_w4,
                              const float *channel_scales,
                              void *expanded_f16,
                              uint32_t k_tiles);

void qbh_copy_hmx_bias_hvx(const uint8_t *source, uint8_t *destination);

#endif
