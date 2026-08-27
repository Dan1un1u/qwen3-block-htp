#include <hexagon_types.h>
#include <HAP_perf.h>
#include <hvx_hexagon_protos.h>
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

static int qbh_hmx_fp16_wait_region(
    const volatile uint32_t *ready_generation,
    uint32_t expected_generation, uint64_t *wait_ticks) {
    uint64_t start = HAP_perf_get_qtimer_count();
    while (*ready_generation != expected_generation) {
        asm volatile("pause(#8)" : : : "memory");
    }
    asm volatile("barrier" : : : "memory");
    *wait_ticks += HAP_perf_get_qtimer_count() - start;
    return 0;
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

void qbh_hmx_fp16_init_channel_scales(void *scale_block,
                                      const float *channel_scales) {
    const HVX_Vector scale_f32 =
        *(const HVX_Vector *)channel_scales;
    const HVX_Vector zero = Q6_V_vzero();
    const HVX_VectorPair scale_bias_f32 = Q6_W_vshuff_VVR(
        zero, scale_f32, -4);
    HVX_Vector *values = (HVX_Vector *)scale_block;

    values[0] = Q6_Vhf_vcvt_VsfVsf(
        Q6_V_hi_W(scale_bias_f32), Q6_V_lo_W(scale_bias_f32));
    values[1] = zero;
    asm volatile("barrier" ::: "memory");
}

void qbh_hmx_fp16_matmul_tiles(const __fp16 *activation_tiles,
                               const __fp16 *weight_tiles,
                               const void *scale_block,
                               __fp16 *output_tiles,
                               uint32_t m_tiles, uint32_t k_tiles,
                               uint32_t n_tiles) {
    asm volatile("mxclracc.hf" ::: "memory");
    Q6_bias_mxmem2_A((void *)scale_block);

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

int qbh_hmx_fp16_matmul_streaming(
    const __fp16 *activation_tiles, const __fp16 *weight_tiles,
    const void *scale_block, __fp16 *output_tiles,
    uint32_t m_tiles, uint32_t k_tiles,
    uint32_t region_tiles, const volatile uint32_t *ready_generations,
    uint32_t expected_generation, uint64_t *ready_wait_ticks) {
    uint32_t region_count;

    if (activation_tiles == NULL || weight_tiles == NULL ||
        scale_block == NULL || output_tiles == NULL ||
        ready_generations == NULL || ready_wait_ticks == NULL ||
        m_tiles == 0U || k_tiles == 0U || region_tiles == 0U ||
        region_tiles > 32U || k_tiles % region_tiles != 0U) {
        return -1;
    }
    region_count = k_tiles / region_tiles;
    asm volatile("mxclracc.hf" ::: "memory");
    Q6_bias_mxmem2_A((void *)scale_block);

    for (uint32_t row_tile = 0; row_tile < m_tiles; ++row_tile) {
        const __fp16 *activation = activation_tiles +
            (size_t)row_tile * k_tiles * QBH_HMX_FP16_TILE_ELEMENTS;
        const __fp16 *weight = weight_tiles;
        for (uint32_t region = 0; region < region_count; ++region) {
            if (qbh_hmx_fp16_wait_region(
                    &ready_generations[region], expected_generation,
                    ready_wait_ticks) != 0) {
                return -1;
            }
            qbh_hmx_fp16_load_tiles(
                activation + (size_t)region * region_tiles *
                                 QBH_HMX_FP16_TILE_ELEMENTS,
                weight + (size_t)region * region_tiles *
                             QBH_HMX_FP16_TILE_ELEMENTS,
                region_tiles);
        }
        qbh_hmx_fp16_store_tile(
            output_tiles + (size_t)row_tile *
                               QBH_HMX_FP16_TILE_ELEMENTS);
    }
    asm volatile("barrier" ::: "memory");
    return 0;
}
