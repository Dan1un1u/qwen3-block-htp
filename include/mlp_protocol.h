#ifndef QWEN3_BLOCK_HTP_MLP_PROTOCOL_H
#define QWEN3_BLOCK_HTP_MLP_PROTOCOL_H

#include <stdint.h>

#include "probe_protocol.h"

#define QBH_MLP_MAGIC UINT32_C(0x51424d4c)
#define QBH_MLP_ABI_VERSION UINT32_C(5)
#define QBH_MLP_EXPERIMENT UINT32_C(40)
#define QBH_MLP_INTERMEDIATE_BYTES \
    (QBH_PROJ_M * QBH_GATE_UP_N)
#define QBH_MLP_GATE_UP_MULTIPLIER_BYTES QBH_GATE_UP_PAIR_N
#define QBH_MLP_DOWN_MULTIPLIER_BYTES QBH_DOWN_N
#define QBH_MLP_HMX_INTERMEDIATE_ZERO_POINT INT32_C(128)
#define QBH_MLP_GATE_ZERO_POINT INT32_C(125)
#define QBH_MLP_UP_ZERO_POINT INT32_C(110)
#define QBH_MLP_DOWN_ZERO_POINT INT32_C(103)

enum qbh_mlp_status {
    QBH_MLP_STATUS_HOST_READY = 1,
    QBH_MLP_STATUS_DSP_RUNNING = 2,
    QBH_MLP_STATUS_OK = 3,
    QBH_MLP_STATUS_BAD_HEADER = -1,
    QBH_MLP_STATUS_CACHE_FAILED = -2,
    QBH_MLP_STATUS_SELF_TEST_FAILED = -3,
    QBH_MLP_STATUS_INPUT_DMA_FAILED = -4,
    QBH_MLP_STATUS_GATE_UP_FAILED = -5,
    QBH_MLP_STATUS_ACTIVATION_FAILED = -6,
    QBH_MLP_STATUS_DOWN_FAILED = -7,
    QBH_MLP_STATUS_OUTPUT_DMA_FAILED = -8,
};

struct qbh_mlp_header {
    uint32_t magic;
    uint32_t abi_version;
    uint32_t experiment;
    uint32_t header_bytes;
    uint32_t shared_bytes;
    uint32_t repeat_count;
    uint32_t run_activation_self_test;
    uint32_t pattern;
    uint32_t weight_storage_variant;
    uint32_t gate_up_worker_count;

    uint32_t input_offset;
    uint32_t gate_up_weight_offset;
    uint32_t down_weight_offset;
    uint32_t activation_lut_offset;
    uint32_t gate_up_multiplier_offset;
    uint32_t down_multiplier_offset;
    uint32_t output_offset;
    uint32_t input_bytes;
    uint32_t gate_up_weight_bytes;
    uint32_t down_weight_bytes;
    uint32_t activation_lut_bytes;
    uint32_t gate_up_multiplier_bytes;
    uint32_t down_multiplier_bytes;
    uint32_t output_bytes;

    int32_t dsp_status;
    int32_t cache_status;
    uint32_t prepared_session_run_index;
    uint32_t resource_vtcm_address;
    uint32_t resource_hmx_context_id;
    uint32_t vtcm_requested_bytes;
    uint32_t vtcm_acquired_bytes;
    uint32_t vtcm_peak_plan_bytes;
    uint32_t gate_up_output_vtcm_bytes;
    uint32_t middle_vtcm_bytes;
    uint32_t activation_lut_vtcm_bytes;
    uint32_t requant_metadata_vtcm_bytes;
    uint32_t activation_gather_scratch_vtcm_bytes;
    uint32_t final_output_vtcm_bytes;
    uint32_t gate_up_pair_slot_count;
    uint32_t gate_up_pair_publish_count;
    uint32_t gate_up_pair_consume_count;
    uint32_t gate_up_full_tensor_materialized;

    uint32_t activation_self_test_cases;
    uint32_t activation_self_test_mismatches;
    uint32_t intermediate_ddr_read_bytes;
    uint32_t intermediate_ddr_write_bytes;
    uint32_t intermediate_dma_descriptor_count;
    uint32_t intermediate_spill_fill_count;
    uint32_t gate_up_output_dma_descriptor_count;
    uint32_t middle_dma_descriptor_count;
    uint32_t final_output_dma_descriptor_count;
    uint32_t final_output_dma_timeout_count;
    int32_t final_output_dma_status;

    uint64_t qtimer_start;
    uint64_t qtimer_end;
    uint64_t total_ticks;
    uint64_t input_stage_ticks;
    uint64_t activation_lut_stage_ticks;
    uint64_t requant_metadata_stage_ticks;
    uint64_t gate_up_ticks;
    uint64_t activation_ticks;
    uint64_t down_ticks;
    uint64_t down_requant_ticks;
    uint64_t final_output_ticks;

    struct qbh_probe_header gate_up_phase;
    struct qbh_probe_header down_phase;
};

#endif
