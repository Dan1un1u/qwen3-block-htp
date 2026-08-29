#ifndef QWEN3_BLOCK_HTP_ATTENTION_PROTOCOL_H
#define QWEN3_BLOCK_HTP_ATTENTION_PROTOCOL_H

#include <stdint.h>

#include "probe_protocol.h"

#define QBH_ATTENTION_MAGIC UINT32_C(0x51424154)
#define QBH_ATTENTION_ABI_VERSION UINT32_C(1)
#define QBH_ATTENTION_EXPERIMENT UINT32_C(42)
#define QBH_ATTENTION_ALIGNMENT UINT32_C(4096)

#define QBH_ATTENTION_M UINT32_C(64)
#define QBH_ATTENTION_HEAD_DIM UINT32_C(128)
#define QBH_ATTENTION_Q_HEADS_PER_GROUP UINT32_C(2)
#define QBH_ATTENTION_SCORE_TILES UINT32_C(2)
#define QBH_ATTENTION_HEAD_DIM_TILES UINT32_C(4)
#define QBH_ATTENTION_Q_BYTES \
    (QBH_ATTENTION_Q_HEADS_PER_GROUP * QBH_ATTENTION_M * \
     QBH_ATTENTION_HEAD_DIM)
#define QBH_ATTENTION_KV_BYTES \
    (QBH_ATTENTION_M * QBH_ATTENTION_HEAD_DIM)
#define QBH_ATTENTION_SCORE_BYTES \
    (QBH_ATTENTION_Q_HEADS_PER_GROUP * QBH_ATTENTION_M * \
     QBH_ATTENTION_M)
#define QBH_ATTENTION_OUTPUT_BYTES QBH_ATTENTION_Q_BYTES

#define QBH_ATTENTION_HMX_CENTER UINT32_C(128)
#define QBH_ATTENTION_MAX_SHIFT UINT32_C(15)
#define QBH_ATTENTION_MAX_MULTIPLIER UINT32_C(18)

enum qbh_attention_division_mode {
    QBH_ATTENTION_DIVISION_EXACT = 1,
    QBH_ATTENTION_DIVISION_SOLE = 2,
    QBH_ATTENTION_DIVISION_ENDPOINT = 3,
};

enum qbh_attention_status {
    QBH_ATTENTION_STATUS_HOST_READY = 1,
    QBH_ATTENTION_STATUS_DSP_RUNNING = 2,
    QBH_ATTENTION_STATUS_OK = 0,
    QBH_ATTENTION_STATUS_BAD_HEADER = -1,
    QBH_ATTENTION_STATUS_CACHE_FAILED = -2,
    QBH_ATTENTION_STATUS_HMX_LOCK_FAILED = -3,
    QBH_ATTENTION_STATUS_HMX_UNLOCK_FAILED = -4,
    QBH_ATTENTION_STATUS_HVX_LOCK_FAILED = -5,
    QBH_ATTENTION_STATUS_HVX_UNLOCK_FAILED = -6,
    QBH_ATTENTION_STATUS_NUMERICAL_FAILED = -7,
    QBH_ATTENTION_STATUS_VTCM_PLAN_FAILED = -8,
    QBH_ATTENTION_STATUS_HMX_WORKER_FAILED = -9,
};

/* Binary package record.  It deliberately contains only fixed-point control
 * values used by the DSP path; floating-point qparams remain in the manifest
 * and are not consulted by the core implementation. */
struct qbh_attention_config {
    uint32_t abi_version;
    uint32_t group_index;
    uint32_t fraction_bits;
    uint32_t division_mode;

    int32_t q_zero_point;
    int32_t k_zero_point;
    int32_t v_zero_point;
    int32_t probability_zero_point;
    int32_t output_zero_point;

    uint32_t v_recenter_numerator;
    uint32_t v_recenter_denominator;
    uint32_t score_shift;
    uint32_t score_multiplier;
    uint32_t av_shift;
    uint32_t av_multiplier;
};

struct qbh_attention_header {
    uint32_t magic;
    uint32_t abi_version;
    uint32_t experiment;
    uint32_t header_bytes;
    uint32_t shared_bytes;
    uint32_t repeat_count;
    uint32_t run_self_test;
    uint32_t reserved0;

    uint32_t q_offset;
    uint32_t k_offset;
    uint32_t v_offset;
    uint32_t reference_score_offset;
    uint32_t reference_probability_offset;
    uint32_t reference_output_offset;
    uint32_t output_offset;
    uint32_t q_bytes;
    uint32_t k_bytes;
    uint32_t v_bytes;
    uint32_t reference_score_bytes;
    uint32_t reference_probability_bytes;
    uint32_t reference_output_bytes;
    uint32_t output_bytes;

    struct qbh_attention_config config;

    int32_t dsp_status;
    int32_t cache_status;
    int32_t hmx_lock_status;
    int32_t hmx_unlock_status;
    int32_t hvx_lock_status;
    int32_t hvx_unlock_status;
    uint32_t prepared_session_run_index;
    uint32_t resource_vtcm_address;
    uint32_t resource_hmx_context_id;
    uint32_t vtcm_acquired_bytes;
    uint32_t vtcm_peak_plan_bytes;

    uint32_t hmx_qk_execution_count;
    uint32_t hmx_qk_stream_count;
    uint32_t hmx_av_execution_count;
    uint32_t hmx_av_stream_count;
    uint32_t score_saturation_count;
    uint32_t v_recenter_saturation_count;
    uint32_t probability_mask_violation_count;
    uint32_t probability_row_sum_min;
    uint32_t probability_row_sum_max;
    uint32_t score_mismatch_count;
    uint32_t probability_mismatch_count;
    uint32_t output_mismatch_count;
    uint32_t output_max_abs_lsb;
    uint32_t first_score_mismatch_index;
    uint32_t first_score_actual;
    uint32_t first_score_expected;
    uint32_t first_probability_mismatch_index;
    uint32_t first_probability_actual;
    uint32_t first_probability_expected;
    int32_t debug_qk_accumulator0;
    uint32_t debug_qk_intermediate0;
    uint32_t debug_qk_expected_intermediate0;
    uint32_t debug_qk_post0;
    int32_t debug_qk_accumulator_row1_col0;
    uint32_t debug_qk_intermediate_row1_col0;
    uint32_t debug_qk_expected_intermediate_row1_col0;
    uint32_t debug_qk_post_row1_col0;
    int32_t debug_qk_accumulator_col32;
    uint32_t debug_qk_intermediate_col32;
    uint32_t debug_qk_expected_intermediate_col32;
    uint32_t debug_qk_post_col32;
    int32_t debug_qk_bias_upper0;
    int32_t debug_qk_bias_upper32;
    int32_t debug_qk_packed_accumulator_col32;

    uint32_t intermediate_ddr_read_bytes;
    uint32_t intermediate_ddr_write_bytes;
    uint32_t intermediate_dma_descriptor_count;
    uint32_t intermediate_spill_fill_count;
    uint32_t graph_split_count;
    uint32_t cpu_fallback_count;

    uint64_t qtimer_start;
    uint64_t qtimer_end;
    uint64_t total_ticks;
    uint64_t input_stage_ticks;
    uint64_t pack_ticks;
    uint64_t qk_hmx_ticks;
    uint64_t qk_requant_ticks;
    uint64_t softmax_ticks;
    uint64_t av_hmx_ticks;
    uint64_t av_requant_ticks;
    uint64_t self_test_ticks;
    uint64_t output_stage_ticks;
};

#endif
