#ifndef QWEN3_BLOCK_HTP_RESOURCE_PROTOCOL_H
#define QWEN3_BLOCK_HTP_RESOURCE_PROTOCOL_H

#include <stdint.h>

#include "probe_protocol.h"

#define QBH_RESOURCE_MAGIC UINT32_C(0x51425253)
#define QBH_RESOURCE_ABI_VERSION UINT32_C(1)
#define QBH_RESOURCE_EXPERIMENT UINT32_C(22)

enum qbh_resource_status {
    QBH_RESOURCE_STATUS_HOST_READY = 1,
    QBH_RESOURCE_STATUS_OK = 2,
    QBH_RESOURCE_STATUS_BAD_HEADER = -1,
};

struct qbh_resource_header {
    uint32_t magic;
    uint32_t abi_version;
    uint32_t experiment;
    uint32_t header_bytes;
    int32_t dsp_status;
    int32_t prepare_result;
    int32_t query_status;
    int32_t configure_status;
    int32_t acquire_status;
    int32_t get_pointer_status;
    uint32_t expected_total_bytes;
    uint32_t queried_total_bytes;
    uint32_t available_before_bytes;
    uint32_t requested_bytes;
    uint32_t minimum_page_bytes;
    uint32_t minimum_required_bytes;
    uint32_t granted_bytes;
    uint32_t context_id;
    uint32_t vtcm_address;
    uint32_t exact_full_grant;
    uint64_t acquire_ticks;
};

#endif
