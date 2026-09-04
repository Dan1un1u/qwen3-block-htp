#include <HAP_perf.h>
#include <hmx_hexagon_protos.h>
#include <stdint.h>

#include "hmx_u8s8_projection.h"

#define QBH_HMX_SPATIAL_MASK UINT32_C(0x38)
#define QBH_HMX_ACTIVATION_RT_BASE \
    (((QBH_HMX_SPATIAL_MASK >> 2U) << 7U) | \
     ((QBH_HMX_INPUT_CHANNELS - 1U) << 2U) | \
     (QBH_HMX_SPATIAL_MASK & 0x3U))
#define QBH_HMX_WRITE_RT \
    (((QBH_HMX_SPATIAL_MASK >> 2U) << 7U) | \
     (QBH_HMX_SPATIAL_MASK & 0x3U))

static inline void qbh_hmx_issue_u8s8_stream(
    const uint8_t *activation_tiles, const int8_t *packed_weight_tiles,
    uint32_t stream_tiles) {
    uint32_t activation_rt =
        QBH_HMX_ACTIVATION_RT_BASE +
        (stream_tiles - 1U) * QBH_HMX_ACTIVATION_BYTES;
    uint32_t weight_rt =
        stream_tiles * QBH_HMX_WEIGHT_BYTES - 1U;

    asm volatile("{ activation.ub = mxmem(%0, %1):deep:cm\n"
                 "  weight.b = mxmem(%2, %3) }\n"
                 :
                 : "r"(activation_tiles), "r"(activation_rt),
                   "r"(packed_weight_tiles), "r"(weight_rt)
                 : "memory");
}

static inline void qbh_hmx_issue_u8n4_stream(
    const uint8_t *activation_tiles, const uint8_t *packed_weight_tiles,
    uint32_t stream_tiles) {
    uint32_t activation_rt =
        QBH_HMX_ACTIVATION_RT_BASE +
        (stream_tiles - 1U) * QBH_HMX_ACTIVATION_BYTES;
    uint32_t weight_rt =
        stream_tiles * QBH_W4_PACKED_TILE_BYTES - 1U;

    asm volatile("{ activation.ub = mxmem(%0, %1):deep:cm\n"
                 "  weight.n = mxmem(%2, %3) }\n"
                 :
                 : "r"(activation_tiles), "r"(activation_rt),
                   "r"(packed_weight_tiles), "r"(weight_rt)
                 : "memory");
}

__attribute__((noinline)) void qbh_hmx_begin_u8s8_output(
    const uint32_t *bias_words) {
    Q6_bias_mxmem2_A((void *)bias_words);
    Q6_mxclracc();
}

__attribute__((noinline)) uint32_t qbh_hmx_accumulate_u8s8_projection(
    const uint8_t *activation_tiles, const int8_t *packed_weight_tiles,
    uint32_t k_tiles) {
    uint32_t stream_count = 0;

    while (k_tiles != 0U) {
        uint32_t stream_tiles =
            k_tiles > QBH_HMX_MAX_STREAM_TILES
                ? QBH_HMX_MAX_STREAM_TILES
                : k_tiles;
        qbh_hmx_issue_u8s8_stream(
            activation_tiles, packed_weight_tiles, stream_tiles);

        activation_tiles +=
            (size_t)stream_tiles * QBH_HMX_ACTIVATION_BYTES;
        packed_weight_tiles +=
            (size_t)stream_tiles * QBH_HMX_WEIGHT_BYTES;
        k_tiles -= stream_tiles;
        ++stream_count;
    }
    return stream_count;
}

__attribute__((noinline)) uint32_t qbh_hmx_accumulate_u8n4_projection(
    const uint8_t *activation_tiles, const uint8_t *packed_weight_tiles,
    uint32_t k_tiles) {
    uint32_t stream_count = 0;

    while (k_tiles != 0U) {
        uint32_t stream_tiles =
            k_tiles > QBH_HMX_MAX_STREAM_TILES
                ? QBH_HMX_MAX_STREAM_TILES
                : k_tiles;
        qbh_hmx_issue_u8n4_stream(
            activation_tiles, packed_weight_tiles, stream_tiles);

        activation_tiles +=
            (size_t)stream_tiles * QBH_HMX_ACTIVATION_BYTES;
        packed_weight_tiles +=
            (size_t)stream_tiles * QBH_W4_PACKED_TILE_BYTES;
        k_tiles -= stream_tiles;
        ++stream_count;
    }
    return stream_count;
}

__attribute__((noinline)) int32_t qbh_hmx_accumulate_u8s8_streaming(
    const uint8_t *activation_tiles, const int8_t *expanded_weight_tiles,
    const uint32_t *bias_words, uint32_t begin_output,
    const volatile uint32_t *ready_generations,
    uint32_t expected_generation, uint32_t stream_count,
    volatile int32_t *abort_status, uint64_t timeout_ticks,
    uint64_t *ready_wait_ticks,
    volatile uint32_t *hmx_consumption_started) {
    uint64_t accumulated_wait = 0U;

    if (activation_tiles == NULL || expanded_weight_tiles == NULL ||
        ready_generations == NULL || abort_status == NULL ||
        ready_wait_ticks == NULL || hmx_consumption_started == NULL ||
        expected_generation == 0U ||
        stream_count == 0U ||
        stream_count > QBH_W4_MAX_STREAM_REGIONS) {
        return -1;
    }

    for (uint32_t stream = 0; stream < stream_count; ++stream) {
        uint64_t wait_start = HAP_perf_get_qtimer_count();
        uint32_t spins = 0U;
        while (ready_generations[stream] != expected_generation) {
            if (*abort_status != 0) {
                *ready_wait_ticks += accumulated_wait +
                    HAP_perf_get_qtimer_count() - wait_start;
                return -1;
            }
            ++spins;
            if ((spins & UINT32_C(255)) == 0U &&
                HAP_perf_get_qtimer_count() - wait_start > timeout_ticks) {
                *ready_wait_ticks += accumulated_wait +
                    HAP_perf_get_qtimer_count() - wait_start;
                return -2;
            }
        }
        asm volatile("barrier" : : : "memory");
        accumulated_wait += HAP_perf_get_qtimer_count() - wait_start;

        if (*hmx_consumption_started == 0U) {
            *hmx_consumption_started = 1U;
            asm volatile("release(%0):at"
                         :
                         : "r"(hmx_consumption_started)
                         : "memory");
        }

        if (stream == 0U && begin_output != 0U) {
            qbh_hmx_begin_u8s8_output(bias_words);
        }
        qbh_hmx_issue_u8s8_stream(
            activation_tiles +
                (size_t)stream * QBH_W4_STREAM_REGION_TILES *
                    QBH_HMX_ACTIVATION_BYTES,
            expanded_weight_tiles +
                (size_t)stream * QBH_W4_STREAM_REGION_TILES *
                    QBH_HMX_WEIGHT_BYTES,
            QBH_W4_STREAM_REGION_TILES);
    }
    *ready_wait_ticks += accumulated_wait;
    return (int32_t)stream_count;
}

__attribute__((noinline)) void qbh_hmx_store_u8_output(uint8_t *output) {
    Q6_mxmem_AR_after_cm_sat_ub(output, QBH_HMX_WRITE_RT);
    asm volatile("barrier" : : : "memory");
}
