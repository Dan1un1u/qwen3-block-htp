#ifndef QWEN3_BLOCK_HTP_W4_PARALLEL_PIPELINE_H
#define QWEN3_BLOCK_HTP_W4_PARALLEL_PIPELINE_H

#include <stdint.h>

#include "hmx_u8s8_projection.h"
#include "probe_protocol.h"

int qbh_run_chunked_w4_pipeline(
    struct qbh_probe_header *header,
    const struct qbh_projection_layout *layout,
    const uint8_t *stored_weights, const uint8_t *activation_tiles,
    uint8_t *vtcm, uint32_t hmx_context_id);

#endif
