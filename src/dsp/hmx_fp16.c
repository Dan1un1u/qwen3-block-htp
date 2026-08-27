#include <hexagon_types.h>
#include <stdint.h>

#include "hmx_fp16.h"

static inline void qbh_hmx_fp16_load_tiles(
    const __fp16 *activation_tiles, const __fp16 *weight_tiles,
    uint32_t tile_count) {
    uint32_t limit = tile_count * QBH_HMX_FP16_TILE_BYTES - 1U;
    asm volatile("{ activation.hf = mxmem(%0, %1):deep\n"
                 "  weight.hf = mxmem(%2, %3) }\n"
                 :
                 : "r"(activation_tiles), "r"(limit),
                   "r"(weight_tiles), "r"(limit)
                 : "memory");
}

static inline void qbh_hmx_fp16_store_tile(__fp16 *output_tile) {
    asm volatile("cvt.hf = acc(%0)\n"
                 "mxmem(%1, %2) = cvt\n"
                 :
                 : "r"(2), "r"(output_tile), "r"(0)
                 : "memory");
}

void qbh_hmx_fp16_init_unity_scale(void *scale_block) {
    uint16_t *values = (uint16_t *)scale_block;
    for (uint32_t index = 0; index < 32U; ++index) {
        values[index * 2U] = UINT16_C(0x3c00);
        values[index * 2U + 1U] = 0U;
    }
    for (uint32_t index = 64U; index < 128U; ++index) {
        values[index] = 0U;
    }
}

void qbh_hmx_fp16_matmul_tiles(const __fp16 *activation_tiles,
                               const __fp16 *weight_tiles,
                               const void *scale_block,
                               __fp16 *output_tiles,
                               uint32_t m_tiles, uint32_t k_tiles,
                               uint32_t n_tiles) {
    Q6_bias_mxmem2_A((void *)scale_block);
    asm volatile("mxclracc.hf" ::: "memory");

    for (uint32_t row_tile = 0; row_tile < m_tiles; ++row_tile) {
        for (uint32_t column_tile = 0; column_tile < n_tiles;
             ++column_tile) {
            const __fp16 *activation = activation_tiles +
                (size_t)row_tile * k_tiles *
                    QBH_HMX_FP16_TILE_ELEMENTS;
            const __fp16 *weight = weight_tiles +
                (size_t)column_tile * k_tiles *
                    QBH_HMX_FP16_TILE_ELEMENTS;
            uint32_t remaining = k_tiles;
            while (remaining != 0U) {
                uint32_t stream = remaining > 32U ? 32U : remaining;
                qbh_hmx_fp16_load_tiles(activation, weight, stream);
                activation += (size_t)stream *
                              QBH_HMX_FP16_TILE_ELEMENTS;
                weight += (size_t)stream *
                          QBH_HMX_FP16_TILE_ELEMENTS;
                remaining -= stream;
            }
            qbh_hmx_fp16_store_tile(
                output_tiles +
                qbh_hmx_fp16_matrix_tile_offset(
                    row_tile, column_tile, n_tiles));
        }
    }
    asm volatile("barrier" ::: "memory");
}
