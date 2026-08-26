#include <hmx_hexagon_protos.h>
#include <stdint.h>

#include "hmx_u8s8_projection.h"

#define QBH_HMX_SPATIAL_MASK UINT32_C(0x38)
#define QBH_HMX_ACTIVATION_RT \
    (((QBH_HMX_SPATIAL_MASK >> 2U) << 7U) | \
     ((QBH_HMX_INPUT_CHANNELS - 1U) << 2U) | \
     (QBH_HMX_SPATIAL_MASK & 0x3U))
#define QBH_HMX_PROJECTION_ACTIVATION_RT \
    (QBH_HMX_ACTIVATION_RT + \
     (QBH_PROJ_K_TILES - 1U) * QBH_HMX_ACTIVATION_BYTES)
#define QBH_HMX_PROJECTION_WEIGHT_RT (QBH_PROJ_WEIGHT_CHUNK_BYTES - 1U)
#define QBH_HMX_WRITE_RT \
    (((QBH_HMX_SPATIAL_MASK >> 2U) << 7U) | \
     (QBH_HMX_SPATIAL_MASK & 0x3U))

__attribute__((noinline)) void qbh_hmx_begin_u8s8_output(
    const uint32_t *bias_words) {
    Q6_bias_mxmem2_A((void *)bias_words);
    Q6_mxclracc();
}

__attribute__((noinline)) void qbh_hmx_accumulate_u8s8_projection(
    const uint8_t *activation_tiles, const int8_t *packed_weight_tiles) {
    asm volatile("{ activation.ub = mxmem(%0, %1):deep:cm\n"
                 "  weight.b = mxmem(%2, %3) }\n"
                 :
                 : "r"(activation_tiles),
                   "r"(QBH_HMX_PROJECTION_ACTIVATION_RT),
                   "r"(packed_weight_tiles),
                   "r"(QBH_HMX_PROJECTION_WEIGHT_RT)
                 : "memory");
}

__attribute__((noinline)) void qbh_hmx_store_u8_output(uint8_t *output) {
    Q6_mxmem_AR_after_cm_sat_ub(output, QBH_HMX_WRITE_RT);
    asm volatile("barrier" : : : "memory");
}
