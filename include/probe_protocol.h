#ifndef QWEN3_BLOCK_HTP_PROBE_PROTOCOL_H
#define QWEN3_BLOCK_HTP_PROBE_PROTOCOL_H

#include <stdint.h>

#define QBH_PROBE_MAGIC UINT32_C(0x51424850)
#define QBH_PROBE_ABI_VERSION UINT32_C(1)
#define QBH_PROBE_ALIGNMENT UINT32_C(128)
#define QBH_PROBE_DEFAULT_LENGTH UINT32_C(65536)

enum qbh_probe_status {
    QBH_PROBE_STATUS_HOST_INITIALIZED = 1,
    QBH_PROBE_STATUS_DSP_RUNNING = 2,
    QBH_PROBE_STATUS_OK = 0,
    QBH_PROBE_STATUS_BAD_HEADER = -1,
    QBH_PROBE_STATUS_CACHE_INVALIDATE_FAILED = -2,
    QBH_PROBE_STATUS_VTCM_CONFIG_FAILED = -3,
    QBH_PROBE_STATUS_VTCM_ACQUIRE_FAILED = -4,
    QBH_PROBE_STATUS_VTCM_POINTER_FAILED = -5,
    QBH_PROBE_STATUS_CACHE_FLUSH_FAILED = -6,
};

struct qbh_probe_header {
    uint32_t magic;
    uint32_t abi_version;
    uint32_t header_bytes;
    uint32_t total_bytes;

    uint32_t vector_length;
    uint32_t input_offset;
    uint32_t output_offset;
    uint32_t addend;

    int32_t dsp_status;
    uint32_t vtcm_requested_bytes;
    uint32_t vtcm_acquired_bytes;
    int32_t cache_status;

    uint64_t qtimer_start;
    uint64_t qtimer_end;
    uint64_t qtimer_elapsed;
    uint64_t pcycles_start;
    uint64_t pcycles_end;
};

_Static_assert(sizeof(struct qbh_probe_header) == 88,
               "probe header ABI changed");

#endif
