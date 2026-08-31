#ifndef QWEN3_BLOCK_HTP_W4_PARALLEL_PIPELINE_H
#define QWEN3_BLOCK_HTP_W4_PARALLEL_PIPELINE_H

#include <stdint.h>

#include "hmx_u8s8_projection.h"
#include "probe_protocol.h"

#define QBH_W4_HMX_MAX_BATCH_OUTPUTS UINT32_C(8)
#define QBH_W4_HMX_MAX_CONTINUATION_CHUNKS UINT32_C(1)

struct qbh_mlp_gate_up_handoff {
    uint8_t *middle_activation;
    const uint16_t *activation_lut;
    const uint8_t *output_multipliers;
    uint8_t *activation_gather_scratch;
    uint32_t activation_direct_arithmetic;
    uint32_t pair_slot_count;
    uint32_t *pair_publish_count;
    uint32_t *pair_consume_count;
    uint32_t *arithmetic_activation_task_count;
    uint64_t *activation_ticks;
};

struct qbh_w4_hmx_request {
    const uint8_t *activation_tiles;
    const int8_t *expanded_weight_tiles;
    const uint32_t *bias_words;
    uint8_t *output_tiles;
    uint32_t chunk_tiles;
    uint32_t begin_output;
    uint32_t store_output;
    uint32_t streaming;
    const volatile uint32_t *ready_generations;
    uint32_t expected_generation;
    uint32_t stream_count;
    volatile int32_t *abort_status;
    uint64_t timeout_ticks;
    uint64_t *ready_wait_ticks;
    volatile uint32_t *hmx_consumption_started;
    uint32_t *executed_stream_count;
    uint32_t batch_output_count;
    struct {
        const int8_t *expanded_weight_tiles;
        const uint32_t *bias_words;
        uint8_t *output_tiles;
        const volatile uint32_t *ready_generations;
        uint32_t expected_generation;
    } batch_outputs[QBH_W4_HMX_MAX_BATCH_OUTPUTS];
    uint32_t continuation_chunk_count;
    struct {
        const uint8_t *activation_tiles;
        const int8_t *expanded_weight_tiles;
        uint32_t chunk_tiles;
        void *ready_semaphore;
    } continuation_chunks[QBH_W4_HMX_MAX_CONTINUATION_CHUNKS];
};

struct qbh_w4_hmx_runner {
    void *context;
    uint32_t max_batch_outputs;
    uint32_t max_chunks_per_command;
    int (*submit)(void *context,
                  const struct qbh_w4_hmx_request *request);
};

struct qbh_w4_hvx_dispatch_runner {
    void *context;
    uint32_t max_workers;
    int (*start)(void *context, void *const *worker_contexts,
                 uint32_t worker_count);
    int (*wait)(void *context, uint32_t worker_count);
};

int qbh_run_chunked_w4_pipeline(
    struct qbh_probe_header *header,
    const struct qbh_projection_layout *layout,
    const uint8_t *stored_weights, const uint8_t *activation_tiles,
    uint8_t *vtcm, uint32_t hmx_context_id);

int qbh_run_chunked_w4_pipeline_mlp(
    struct qbh_probe_header *header,
    const struct qbh_projection_layout *layout,
    const uint8_t *stored_weights, const uint8_t *activation_tiles,
    uint8_t *vtcm, uint32_t hmx_context_id,
    const struct qbh_mlp_gate_up_handoff *handoff);

int qbh_run_chunked_w4_pipeline_external(
    struct qbh_probe_header *header,
    const struct qbh_projection_layout *layout,
    const uint8_t *stored_weights, const uint8_t *activation_tiles,
    uint8_t *vtcm,
    const struct qbh_mlp_gate_up_handoff *handoff,
    const struct qbh_w4_hmx_runner *runner);

int qbh_run_chunked_w4_pipeline_external_hvx(
    struct qbh_probe_header *header,
    const struct qbh_projection_layout *layout,
    const uint8_t *stored_weights, const uint8_t *activation_tiles,
    uint8_t *vtcm,
    const struct qbh_mlp_gate_up_handoff *handoff,
    const struct qbh_w4_hmx_runner *hmx_runner,
    const struct qbh_w4_hvx_dispatch_runner *hvx_runner);

int qbh_run_chunked_w4_external_hvx_worker(void *worker_context);
int qbh_run_chunked_w4_managed_hvx_worker(void *worker_context);

#endif
