#ifndef QWEN3_BLOCK_HTP_HMX_CONVERT_PROTOCOL_H
#define QWEN3_BLOCK_HTP_HMX_CONVERT_PROTOCOL_H

#include <stdint.h>

#include "probe_protocol.h"

#define QBH_HMX_CONVERT_MAGIC UINT32_C(0x51484356)
#define QBH_HMX_CONVERT_ABI_VERSION UINT32_C(1)
#define QBH_HMX_CONVERT_ALIGNMENT UINT32_C(4096)

enum qbh_hmx_convert_status {
    QBH_HMX_CONVERT_STATUS_OK = 0,
    QBH_HMX_CONVERT_STATUS_HOST_INITIALIZED = 1,
    QBH_HMX_CONVERT_STATUS_BAD_HEADER = -1,
    QBH_HMX_CONVERT_STATUS_CACHE_FAILED = -2,
    QBH_HMX_CONVERT_STATUS_SESSION_NOT_PREPARED = -3,
    QBH_HMX_CONVERT_STATUS_HMX_LOCK_FAILED = -4,
    QBH_HMX_CONVERT_STATUS_HMX_UNLOCK_FAILED = -5,
};

struct qbh_hmx_convert_header {
    uint32_t magic;
    uint32_t abi_version;
    uint32_t header_bytes;
    uint32_t total_bytes;

    uint32_t activation_offset;
    uint32_t weight_offset;
    uint32_t bias_offset;
    uint32_t output_offset;

    int32_t dsp_status;
    int32_t cache_status;
    int32_t hmx_lock_status;
    int32_t hmx_unlock_status;
    uint32_t vtcm_address;
    uint32_t hmx_context_id;
    uint64_t hmx_ticks;
};

#endif
