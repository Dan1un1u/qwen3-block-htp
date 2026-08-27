#ifndef QWEN3_BLOCK_HTP_HMX_FP16_H
#define QWEN3_BLOCK_HTP_HMX_FP16_H

#include <stddef.h>
#include <stdint.h>

#define QBH_HMX_FP16_ROWS UINT32_C(32)
#define QBH_HMX_FP16_COLS UINT32_C(32)
#define QBH_HMX_FP16_TILE_ELEMENTS UINT32_C(1024)
#define QBH_HMX_FP16_TILE_BYTES UINT32_C(2048)
#define QBH_HMX_FP16_SCALE_BYTES UINT32_C(256)

void qbh_hmx_fp16_init_unity_scale(void *scale_block);
void qbh_hmx_fp16_init_channel_scales(void *scale_block,
                                      const float *channel_scales);
void qbh_hmx_fp16_matmul_tiles(const __fp16 *activation_tiles,
                               const __fp16 *weight_tiles,
                               const void *scale_block,
                               __fp16 *output_tiles,
                               uint32_t m_tiles, uint32_t k_tiles,
                               uint32_t n_tiles);

int qbh_hmx_fp16_matmul_streaming(
    const __fp16 *activation_tiles, const __fp16 *weight_tiles,
    const void *scale_block, __fp16 *output_tiles,
    uint32_t m_tiles, uint32_t k_tiles,
    uint32_t region_tiles, const volatile uint32_t *ready_generations,
    uint32_t expected_generation, uint64_t *ready_wait_ticks);

static inline size_t qbh_hmx_fp16_matrix_tile_offset(
    uint32_t tile_row, uint32_t tile_column, uint32_t column_tiles) {
    return ((size_t)tile_row * column_tiles + tile_column) *
           QBH_HMX_FP16_TILE_ELEMENTS;
}

static inline size_t qbh_hmx_fp16_tile_offset(uint32_t row,
                                               uint32_t column) {
    return ((size_t)(row / 2U) * QBH_HMX_FP16_COLS + column) * 2U +
           row % 2U;
}

#endif
