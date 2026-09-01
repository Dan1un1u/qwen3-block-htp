#ifndef QWEN3_BLOCK_HTP_RPCMEM2_PROTOCOL_H
#define QWEN3_BLOCK_HTP_RPCMEM2_PROTOCOL_H

#include <stdint.h>

#define QBH_RPCMEM2_MAGIC UINT32_C(0x5152324d)
#define QBH_RPCMEM2_ABI_VERSION UINT32_C(1)
#define QBH_RPCMEM2_EXPERIMENT UINT32_C(151)
#define QBH_RPCMEM2_REQUEST_BYTES UINT32_C(2900000000)
#define QBH_RPCMEM2_SENTINEL_COUNT UINT32_C(3)

#define QBH_RPCMEM2_BEGIN_OFFSET UINT32_C(4096)
#define QBH_RPCMEM2_MIDDLE_OFFSET UINT32_C(1449996288)
#define QBH_RPCMEM2_END_OFFSET UINT32_C(2899992576)

enum qbh_rpcmem2_status {
    QBH_RPCMEM2_STATUS_HOST_READY = 1,
    QBH_RPCMEM2_STATUS_OK = 2,
    QBH_RPCMEM2_STATUS_BAD_HEADER = -1,
    QBH_RPCMEM2_STATUS_BAD_RANGE = -2,
    QBH_RPCMEM2_STATUS_CACHE_FAILED = -3,
    QBH_RPCMEM2_STATUS_SENTINEL_MISMATCH = -4,
};

struct qbh_rpcmem2_header {
    uint32_t magic;
    uint32_t abi_version;
    uint32_t experiment;
    uint32_t header_bytes;
    uint32_t requested_bytes;
    uint32_t sentinel_count;
    uint32_t sentinel_offsets[QBH_RPCMEM2_SENTINEL_COUNT];
    uint32_t host_values[QBH_RPCMEM2_SENTINEL_COUNT];
    uint32_t dsp_write_values[QBH_RPCMEM2_SENTINEL_COUNT];
    uint32_t dsp_observed_values[QBH_RPCMEM2_SENTINEL_COUNT];
    int32_t dsp_status;
    int32_t mmap_get_result;
    int32_t mmap_put_result;
    int32_t cache_invalidate_result;
    int32_t cache_flush_result;
    uint32_t dsp_virtual_base;
    uint32_t dsp_virtual_end;
    uint64_t dsp_physical_base;
    uint32_t address_range_valid;
    uint32_t sentinel_mismatch_count;
};

#endif
