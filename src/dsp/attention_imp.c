#include <AEEStdErr.h>
#include <HAP_compute_res.h>
#include <HAP_mem.h>
#include <HAP_perf.h>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include <qurt.h>
#include <qurt_hvx.h>
#include <remote.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "attention_imp.h"
#include "attention_protocol.h"
#include "hmx_int8_tile.h"
#include "hmx_u8s8_projection.h"

#define QBH_ATTN_VTCM_ALIGN UINT32_C(128)
#define QBH_ATTN_WEIGHT_ALIGN UINT32_C(256)
#define QBH_ATTN_SCORE_ZP INT32_C(128)
#define QBH_ATTN_EXP_FRAC_BITS UINT32_C(15)
#define QBH_ATTN_HMX_STACK_BYTES UINT32_C(16384)
#define QBH_ATTN_HVX_BYTES UINT32_C(128)
#define QBH_ATTN_SOFTMAX_SCRATCH_BYTES UINT32_C(256)

static const uint8_t qbh_attn_lane_index[QBH_ATTN_HVX_BYTES]
    __attribute__((aligned(QBH_ATTN_HVX_BYTES))) = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
        48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
        64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
        80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95,
        96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
        110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122,
        123, 124, 125, 126, 127,
    };

struct qbh_attention_hmx_worker {
    uint32_t hmx_context_id;
    qurt_sem_t command_ready;
    qurt_sem_t command_done;
    qurt_sem_t worker_started;
    volatile uint32_t stop;
    const uint8_t *activation;
    const int8_t *weight;
    const uint32_t *bias;
    uint8_t *output;
    uint32_t k_tiles;
    int32_t lock_status;
    int32_t unlock_status;
    int32_t command_status;
    uint32_t stream_count;
};

static uint8_t qbh_attn_hmx_stack[QBH_ATTN_HMX_STACK_BYTES]
    __attribute__((aligned(128)));

struct qbh_attention_buffers {
    uint8_t *q;
    uint8_t *k;
    uint8_t *v;
    uint8_t *q_tiles;
    int8_t *k_weight_tiles;
    int8_t *v_weight_tiles;
    uint32_t *qk_bias;
    uint32_t *av_bias;
    uint8_t *score_tiles;
    uint8_t *probability_tiles;
    uint8_t *output_tiles;
    uint8_t *output;
    uint8_t *softmax_scratch;
    uint32_t plan_bytes;
};

static uint32_t qbh_attn_align(uint32_t value, uint32_t alignment) {
    return ((value + alignment - 1U) / alignment) * alignment;
}

static int qbh_attn_range_valid(uint32_t offset, uint32_t bytes,
                                uint32_t shared_bytes) {
    return offset >= sizeof(struct qbh_attention_header) &&
           offset <= shared_bytes && bytes <= shared_bytes - offset;
}

static uint16_t qbh_attn_float_to_half_bits(float value) {
    __fp16 converted = (__fp16)value;
    uint16_t bits;
    memcpy(&bits, &converted, sizeof(bits));
    return bits;
}

static int32_t qbh_attn_clip_s8(int32_t value, uint32_t *saturations) {
    if (value < INT8_MIN) {
        ++*saturations;
        return INT8_MIN;
    }
    if (value > INT8_MAX) {
        ++*saturations;
        return INT8_MAX;
    }
    return value;
}

static uint8_t qbh_attn_clip_u8(int32_t value, uint32_t *saturations) {
    if (value < 0) {
        if (saturations != NULL) {
            ++*saturations;
        }
        return 0U;
    }
    if (value > UINT8_MAX) {
        if (saturations != NULL) {
            ++*saturations;
        }
        return UINT8_MAX;
    }
    return (uint8_t)value;
}

static int32_t qbh_attn_round_div_signed(int32_t numerator,
                                         int32_t denominator) {
    if (numerator >= 0) {
        return (numerator + denominator / 2) / denominator;
    }
    return -((-numerator + denominator / 2) / denominator);
}

static int qbh_attn_header_valid(
    const struct qbh_attention_header *header, uint32_t shared_bytes) {
    const struct qbh_attention_config *config;
    if (header == NULL) {
        return 0;
    }
    config = &header->config;
    return header->magic == QBH_ATTENTION_MAGIC &&
           header->abi_version == QBH_ATTENTION_ABI_VERSION &&
           header->experiment == QBH_ATTENTION_EXPERIMENT &&
           header->header_bytes == sizeof(*header) &&
           header->shared_bytes == shared_bytes &&
           header->repeat_count > 0U &&
           header->repeat_count <= QBH_HMX_MAX_REPEATS &&
           header->q_bytes == QBH_ATTENTION_Q_BYTES &&
           header->k_bytes == QBH_ATTENTION_KV_BYTES &&
           header->v_bytes == QBH_ATTENTION_KV_BYTES &&
           header->reference_score_bytes == QBH_ATTENTION_SCORE_BYTES &&
           header->reference_probability_bytes ==
               QBH_ATTENTION_SCORE_BYTES &&
           header->reference_output_bytes ==
               QBH_ATTENTION_OUTPUT_BYTES &&
           header->output_bytes == QBH_ATTENTION_OUTPUT_BYTES &&
           qbh_attn_range_valid(header->q_offset, header->q_bytes,
                                shared_bytes) &&
           qbh_attn_range_valid(header->k_offset, header->k_bytes,
                                shared_bytes) &&
           qbh_attn_range_valid(header->v_offset, header->v_bytes,
                                shared_bytes) &&
           qbh_attn_range_valid(header->reference_score_offset,
                                header->reference_score_bytes,
                                shared_bytes) &&
           qbh_attn_range_valid(header->reference_probability_offset,
                                header->reference_probability_bytes,
                                shared_bytes) &&
           qbh_attn_range_valid(header->reference_output_offset,
                                header->reference_output_bytes,
                                shared_bytes) &&
           qbh_attn_range_valid(header->output_offset,
                                header->output_bytes, shared_bytes) &&
           config->abi_version == QBH_ATTENTION_ABI_VERSION &&
           (config->fraction_bits == 3U ||
            config->fraction_bits == 4U) &&
           config->division_mode >= QBH_ATTENTION_DIVISION_EXACT &&
           config->division_mode <= QBH_ATTENTION_DIVISION_ENDPOINT &&
           config->q_zero_point >= 0 && config->q_zero_point <= 255 &&
           config->k_zero_point >= 0 && config->k_zero_point <= 255 &&
           config->v_zero_point >= 0 && config->v_zero_point <= 255 &&
           config->probability_zero_point == 0 &&
           config->output_zero_point >= 0 &&
           config->output_zero_point <= 255 &&
           config->v_recenter_numerator > 0U &&
           config->v_recenter_denominator > 0U &&
           config->score_shift <= QBH_ATTENTION_MAX_SHIFT &&
           config->score_multiplier > 0U &&
           config->score_multiplier <= QBH_ATTENTION_MAX_MULTIPLIER &&
           config->av_shift <= QBH_ATTENTION_MAX_SHIFT &&
           config->av_multiplier > 0U &&
           config->av_multiplier <= QBH_ATTENTION_MAX_MULTIPLIER;
}

static int qbh_attn_plan_buffers(uint8_t *vtcm, uint32_t vtcm_bytes,
                                 struct qbh_attention_buffers *buffers) {
    uint32_t cursor = 0U;
    if (vtcm == NULL || buffers == NULL) {
        return -1;
    }
#define QBH_ATTN_ALLOC(field, bytes, alignment)                         \
    do {                                                               \
        cursor = qbh_attn_align(cursor, (alignment));                  \
        buffers->field = (void *)(vtcm + cursor);                      \
        cursor += (bytes);                                             \
    } while (0)
    QBH_ATTN_ALLOC(q, QBH_ATTENTION_Q_BYTES, QBH_ATTN_VTCM_ALIGN);
    QBH_ATTN_ALLOC(k, QBH_ATTENTION_KV_BYTES, QBH_ATTN_VTCM_ALIGN);
    QBH_ATTN_ALLOC(v, QBH_ATTENTION_KV_BYTES, QBH_ATTN_VTCM_ALIGN);
    QBH_ATTN_ALLOC(q_tiles,
                   QBH_ATTENTION_Q_HEADS_PER_GROUP *
                       QBH_ATTENTION_HEAD_DIM_TILES *
                       QBH_HMX_ACTIVATION_BYTES,
                   QBH_ATTN_VTCM_ALIGN);
    QBH_ATTN_ALLOC(k_weight_tiles,
                   QBH_ATTENTION_SCORE_TILES *
                       QBH_ATTENTION_HEAD_DIM_TILES *
                       QBH_HMX_WEIGHT_BYTES,
                   QBH_ATTN_WEIGHT_ALIGN);
    QBH_ATTN_ALLOC(v_weight_tiles,
                   QBH_ATTENTION_HEAD_DIM_TILES *
                       QBH_ATTENTION_SCORE_TILES *
                       QBH_HMX_WEIGHT_BYTES,
                   QBH_ATTN_WEIGHT_ALIGN);
    QBH_ATTN_ALLOC(qk_bias,
                   QBH_ATTENTION_SCORE_TILES * QBH_HMX_BIAS_BYTES,
                   QBH_ATTN_WEIGHT_ALIGN);
    QBH_ATTN_ALLOC(av_bias,
                   QBH_ATTENTION_HEAD_DIM_TILES * QBH_HMX_BIAS_BYTES,
                   QBH_ATTN_WEIGHT_ALIGN);
    QBH_ATTN_ALLOC(score_tiles,
                   QBH_ATTENTION_Q_HEADS_PER_GROUP *
                       QBH_ATTENTION_SCORE_TILES *
                       QBH_HMX_OUTPUT_BYTES,
                   QBH_HMX_OUTPUT_BYTES);
    QBH_ATTN_ALLOC(probability_tiles,
                   QBH_ATTENTION_Q_HEADS_PER_GROUP *
                       QBH_ATTENTION_SCORE_TILES *
                       QBH_HMX_ACTIVATION_BYTES,
                   QBH_HMX_ACTIVATION_BYTES);
    QBH_ATTN_ALLOC(output_tiles,
                   QBH_ATTENTION_Q_HEADS_PER_GROUP *
                       QBH_ATTENTION_HEAD_DIM_TILES *
                       QBH_HMX_OUTPUT_BYTES,
                   QBH_HMX_OUTPUT_BYTES);
    QBH_ATTN_ALLOC(output, QBH_ATTENTION_OUTPUT_BYTES,
                   QBH_ATTN_VTCM_ALIGN);
    QBH_ATTN_ALLOC(softmax_scratch,
                   QBH_ATTN_SOFTMAX_SCRATCH_BYTES,
                   QBH_ATTN_HVX_BYTES);
#undef QBH_ATTN_ALLOC
    buffers->plan_bytes = cursor;
    return cursor <= vtcm_bytes ? 0 : -1;
}

static void qbh_attn_pack_q(const uint8_t *q, uint8_t *tiles) {
    for (uint32_t head = 0U;
         head < QBH_ATTENTION_Q_HEADS_PER_GROUP; ++head) {
        for (uint32_t k_tile = 0U;
             k_tile < QBH_ATTENTION_HEAD_DIM_TILES; ++k_tile) {
            uint8_t *tile = tiles +
                ((size_t)head * QBH_ATTENTION_HEAD_DIM_TILES + k_tile) *
                    QBH_HMX_ACTIVATION_BYTES;
            for (uint32_t row = 0U; row < QBH_ATTENTION_M; ++row) {
                memcpy(tile + (size_t)row * QBH_HMX_INPUT_CHANNELS,
                       q + ((size_t)head * QBH_ATTENTION_M + row) *
                               QBH_ATTENTION_HEAD_DIM +
                           k_tile * QBH_HMX_INPUT_CHANNELS,
                       QBH_HMX_INPUT_CHANNELS);
            }
        }
    }
}

static void qbh_attn_pack_k(
    const uint8_t *k, const struct qbh_attention_config *config,
    int8_t *tiles, uint32_t *bias) {
    const uint32_t divisor = UINT32_C(1) << config->score_shift;
    const int32_t rounding = config->score_shift == 0U
                                 ? 0
                                 : (int32_t)(divisor / 2U);
    const uint16_t conversion = qbh_attn_float_to_half_bits(
        512.0f / (float)divisor);
    for (uint32_t n_tile = 0U;
         n_tile < QBH_ATTENTION_SCORE_TILES; ++n_tile) {
        int32_t sums[QBH_HMX_OUTPUT_CHANNELS] = {0};
        for (uint32_t k_tile = 0U;
             k_tile < QBH_ATTENTION_HEAD_DIM_TILES; ++k_tile) {
            int8_t *tile = tiles +
                ((size_t)n_tile * QBH_ATTENTION_HEAD_DIM_TILES + k_tile) *
                    QBH_HMX_WEIGHT_BYTES;
            for (uint32_t input = 0U;
                 input < QBH_HMX_INPUT_CHANNELS; ++input) {
                for (uint32_t output = 0U;
                     output < QBH_HMX_OUTPUT_CHANNELS; ++output) {
                    uint32_t token =
                        n_tile * QBH_HMX_OUTPUT_CHANNELS + output;
                    int32_t centered =
                        (int32_t)k[(size_t)token *
                                       QBH_ATTENTION_HEAD_DIM +
                                   k_tile * QBH_HMX_INPUT_CHANNELS + input] -
                        config->k_zero_point;
                    tile[qbh_packed_weight_offset(input, output)] =
                        (int8_t)centered;
                    sums[output] += centered;
                }
            }
        }
        for (uint32_t output = 0U;
             output < QBH_HMX_OUTPUT_CHANNELS; ++output) {
            uint32_t *block = bias +
                (size_t)n_tile *
                    (QBH_HMX_BIAS_BYTES / sizeof(uint32_t));
            block[output] = conversion;
            block[QBH_HMX_OUTPUT_CHANNELS + output] =
                (uint32_t)(-config->q_zero_point * sums[output] +
                           QBH_ATTN_SCORE_ZP * (int32_t)divisor +
                           rounding);
        }
    }
}

static void qbh_attn_pack_v(
    const uint8_t *v, const struct qbh_attention_config *config,
    int8_t *tiles, uint32_t *bias, uint32_t *saturations) {
    const uint32_t divisor = UINT32_C(1) << config->av_shift;
    const int32_t rounding = config->av_shift == 0U
                                 ? 0
                                 : (int32_t)(divisor / 2U);
    const uint16_t conversion = qbh_attn_float_to_half_bits(
        512.0f / (float)divisor);
    for (uint32_t n_tile = 0U;
         n_tile < QBH_ATTENTION_HEAD_DIM_TILES; ++n_tile) {
        for (uint32_t k_tile = 0U;
             k_tile < QBH_ATTENTION_SCORE_TILES; ++k_tile) {
            int8_t *tile = tiles +
                ((size_t)n_tile * QBH_ATTENTION_SCORE_TILES + k_tile) *
                    QBH_HMX_WEIGHT_BYTES;
            for (uint32_t input = 0U;
                 input < QBH_HMX_INPUT_CHANNELS; ++input) {
                uint32_t token =
                    k_tile * QBH_HMX_INPUT_CHANNELS + input;
                for (uint32_t output = 0U;
                     output < QBH_HMX_OUTPUT_CHANNELS; ++output) {
                    uint32_t channel =
                        n_tile * QBH_HMX_OUTPUT_CHANNELS + output;
                    int32_t centered =
                        (int32_t)v[(size_t)token *
                                       QBH_ATTENTION_HEAD_DIM + channel] -
                        config->v_zero_point;
                    int32_t requantized = qbh_attn_round_div_signed(
                        centered *
                            (int32_t)config->v_recenter_numerator,
                        (int32_t)config->v_recenter_denominator);
                    requantized = qbh_attn_clip_s8(
                        requantized, saturations);
                    tile[qbh_packed_weight_offset(input, output)] =
                        (int8_t)requantized;
                }
            }
        }
        {
            uint32_t *block = bias +
                (size_t)n_tile *
                    (QBH_HMX_BIAS_BYTES / sizeof(uint32_t));
            for (uint32_t output = 0U;
                 output < QBH_HMX_OUTPUT_CHANNELS; ++output) {
                block[output] = conversion;
                block[QBH_HMX_OUTPUT_CHANNELS + output] =
                    QBH_ATTENTION_HMX_CENTER * divisor + rounding;
            }
        }
    }
}

static void qbh_attn_hmx_worker_main(void *opaque) {
    struct qbh_attention_hmx_worker *worker =
        (struct qbh_attention_hmx_worker *)opaque;
    worker->lock_status = HAP_compute_res_hmx_lock2(
        worker->hmx_context_id, HAP_COMPUTE_RES_HMX_SHARED);
    (void)qurt_sem_up(&worker->worker_started);
    if (worker->lock_status != AEE_SUCCESS) {
        qurt_thread_exit(worker->lock_status);
    }
    for (;;) {
        qurt_sem_down(&worker->command_ready);
        if (worker->stop != 0U) {
            break;
        }
        worker->command_status = AEE_SUCCESS;
        qbh_hmx_begin_u8s8_output(worker->bias);
        worker->stream_count = qbh_hmx_accumulate_u8s8_projection(
            worker->activation, worker->weight, worker->k_tiles);
        qbh_hmx_store_u8_output(worker->output);
        (void)qurt_sem_up(&worker->command_done);
    }
    worker->unlock_status = HAP_compute_res_hmx_unlock2(
        worker->hmx_context_id, HAP_COMPUTE_RES_HMX_SHARED);
    qurt_thread_exit(worker->unlock_status);
}

static int qbh_attn_hmx_output(
    struct qbh_attention_hmx_worker *worker,
    const uint8_t *activation, const int8_t *weight,
    const uint32_t *bias, uint8_t *output, uint32_t k_tiles,
    uint32_t *stream_count) {
    worker->activation = activation;
    worker->weight = weight;
    worker->bias = bias;
    worker->output = output;
    worker->k_tiles = k_tiles;
    worker->stream_count = 0U;
    worker->command_status = AEE_EFAILED;
    asm volatile("barrier" ::: "memory");
    (void)qurt_sem_up(&worker->command_ready);
    qurt_sem_down(&worker->command_done);
    asm volatile("barrier" ::: "memory");
    *stream_count += worker->stream_count;
    return worker->command_status == AEE_SUCCESS ? 0 : -1;
}

static int qbh_attn_run_qk(
    struct qbh_attention_header *header,
    const struct qbh_attention_buffers *buffers,
    struct qbh_attention_hmx_worker *worker) {
    for (uint32_t head = 0U;
         head < QBH_ATTENTION_Q_HEADS_PER_GROUP; ++head) {
        const uint8_t *q_tiles = buffers->q_tiles +
            (size_t)head * QBH_ATTENTION_HEAD_DIM_TILES *
                QBH_HMX_ACTIVATION_BYTES;
        for (uint32_t n_tile = 0U;
             n_tile < QBH_ATTENTION_SCORE_TILES; ++n_tile) {
            uint8_t *score = buffers->score_tiles +
                ((size_t)head * QBH_ATTENTION_SCORE_TILES + n_tile) *
                    QBH_HMX_OUTPUT_BYTES;
            const int8_t *weight = buffers->k_weight_tiles +
                (size_t)n_tile * QBH_ATTENTION_HEAD_DIM_TILES *
                    QBH_HMX_WEIGHT_BYTES;
            const uint32_t *bias = buffers->qk_bias +
                (size_t)n_tile *
                    (QBH_HMX_BIAS_BYTES / sizeof(uint32_t));
            if (qbh_attn_hmx_output(
                    worker, q_tiles, weight, bias,
                    score, QBH_ATTENTION_HEAD_DIM_TILES,
                    &header->hmx_qk_stream_count) != 0) {
                return -1;
            }
            header->hmx_qk_execution_count +=
                QBH_ATTENTION_HEAD_DIM_TILES;
        }
    }
    return 0;
}

static HVX_Vector qbh_attn_hvx_requant_centered_u8(
    HVX_Vector input, uint32_t multiplier, int32_t output_zero_point) {
    const HVX_Vector centered = Q6_V_vxor_VV(
        input, Q6_Vb_vsplat_R(0x80));
    const HVX_VectorPair product = Q6_Wh_vmpy_VbVb(
        centered, Q6_Vb_vsplat_R(multiplier));
    const HVX_VectorPair interleaved = Q6_W_vshuff_VVR(
        Q6_V_hi_W(product), Q6_V_lo_W(product), -2);
    const HVX_Vector output_zp =
        Q6_Vh_vsplat_R(output_zero_point);
    const HVX_Vector low = Q6_Vh_vadd_VhVh(
        Q6_V_lo_W(interleaved), output_zp);
    const HVX_Vector high = Q6_Vh_vadd_VhVh(
        Q6_V_hi_W(interleaved), output_zp);
    return Q6_Vub_vpack_VhVh_sat(high, low);
}

static void qbh_attn_requant_qk(
    struct qbh_attention_header *header,
    const struct qbh_attention_buffers *buffers) {
    const struct qbh_attention_config *config = &header->config;
    const uint32_t divisor = UINT32_C(1) << config->score_shift;
    const uint32_t rounding = config->score_shift == 0U
                                  ? 0U
                                  : divisor / 2U;

    if (header->run_self_test != 0U) {
        header->debug_qk_intermediate0 = buffers->score_tiles[0];
        header->debug_qk_intermediate_row1_col0 =
            buffers->score_tiles[QBH_HMX_OUTPUT_CHANNELS];
        header->debug_qk_intermediate_col32 =
            buffers->score_tiles[QBH_HMX_OUTPUT_BYTES];
        for (uint32_t input = 0U;
             input < QBH_ATTENTION_HEAD_DIM; ++input) {
            const int32_t q0 =
                (int32_t)buffers->q[input] - config->q_zero_point;
            const int32_t q1 =
                (int32_t)buffers->q[QBH_ATTENTION_HEAD_DIM + input] -
                config->q_zero_point;
            const int32_t k0 =
                (int32_t)buffers->k[input] - config->k_zero_point;
            const int32_t k32 =
                (int32_t)buffers->k[
                    (size_t)QBH_HMX_OUTPUT_CHANNELS *
                        QBH_ATTENTION_HEAD_DIM + input] -
                config->k_zero_point;
            const int8_t *weight32 = buffers->k_weight_tiles +
                (size_t)QBH_ATTENTION_HEAD_DIM_TILES *
                    QBH_HMX_WEIGHT_BYTES;
            header->debug_qk_accumulator0 += q0 * k0;
            header->debug_qk_accumulator_row1_col0 += q1 * k0;
            header->debug_qk_accumulator_col32 += q0 * k32;
            header->debug_qk_packed_accumulator_col32 +=
                q0 * (int32_t)weight32[
                    (size_t)(input / QBH_HMX_INPUT_CHANNELS) *
                        QBH_HMX_WEIGHT_BYTES +
                    qbh_packed_weight_offset(
                        input % QBH_HMX_INPUT_CHANNELS, 0U)];
        }
        header->debug_qk_expected_intermediate0 =
            (uint32_t)((header->debug_qk_accumulator0 +
                        QBH_ATTN_SCORE_ZP * (int32_t)divisor +
                        (int32_t)rounding) /
                       (int32_t)divisor);
        header->debug_qk_expected_intermediate_row1_col0 =
            (uint32_t)((header->debug_qk_accumulator_row1_col0 +
                        QBH_ATTN_SCORE_ZP * (int32_t)divisor +
                        (int32_t)rounding) /
                       (int32_t)divisor);
        header->debug_qk_expected_intermediate_col32 =
            (uint32_t)((header->debug_qk_accumulator_col32 +
                        QBH_ATTN_SCORE_ZP * (int32_t)divisor +
                        (int32_t)rounding) /
                       (int32_t)divisor);
        for (uint32_t element = 0U;
             element < QBH_ATTENTION_SCORE_BYTES; ++element) {
            const int32_t centered =
                (int32_t)buffers->score_tiles[element] -
                (int32_t)QBH_ATTENTION_HMX_CENTER;
            const int32_t value =
                centered * (int32_t)config->score_multiplier +
                QBH_ATTN_SCORE_ZP;
            if (value < 0 || value > UINT8_MAX) {
                ++header->score_saturation_count;
            }
        }
    }
    for (uint32_t offset = 0U;
         offset < QBH_ATTENTION_SCORE_BYTES;
         offset += sizeof(HVX_Vector)) {
        HVX_Vector *score = (HVX_Vector *)(
            buffers->score_tiles + offset);
        *score = qbh_attn_hvx_requant_centered_u8(
            *score, config->score_multiplier, QBH_ATTN_SCORE_ZP);
    }
    asm volatile("barrier" ::: "memory");
    header->debug_qk_post0 = buffers->score_tiles[0];
    header->debug_qk_post_row1_col0 =
        buffers->score_tiles[QBH_HMX_OUTPUT_CHANNELS];
    header->debug_qk_post_col32 =
        buffers->score_tiles[QBH_HMX_OUTPUT_BYTES];
}

static uint32_t qbh_attn_floor_log2_u32(uint32_t value) {
    uint32_t result = 0U;
    while (value > 1U) {
        value >>= 1U;
        ++result;
    }
    return result;
}

static uint8_t qbh_attn_divide_probability(
    uint32_t exponent, uint32_t sum,
    uint32_t mode, uint32_t valid_count) {
    const uint32_t weight = UINT32_C(1) <<
        (QBH_ATTN_EXP_FRAC_BITS - exponent);
    if (valid_count == 1U) {
        return UINT8_MAX;
    }
    if (mode == QBH_ATTENTION_DIVISION_EXACT) {
        return (uint8_t)(((uint64_t)weight * UINT8_MAX + sum / 2U) /
                         sum);
    }
    {
        const uint32_t leading = qbh_attn_floor_log2_u32(sum);
        const uint32_t next = leading == 0U
                                  ? 0U
                                  : ((sum >> (leading - 1U)) & 1U);
        uint32_t coefficient;
        int32_t shift = (int32_t)exponent + (int32_t)leading -
                        (int32_t)QBH_ATTN_EXP_FRAC_BITS;
        uint64_t numerator;
        uint32_t total_shift;

        if (mode == QBH_ATTENTION_DIVISION_SOLE) {
            coefficient = next != 0U ? 145U : 209U;
        } else {
            coefficient = next != 0U ? 171U : 256U;
        }
        numerator = (uint64_t)UINT8_MAX * coefficient;
        /* A Softmax row contains at least one 2^15 term, so leading >= 15
         * and shift is non-negative for every exponent code.  The original
         * SOLE/endpoint denominator is therefore exactly 2^(8 + shift).
         * Expressing the rounded quotient as a shift removes the variable
         * 64-bit divide from the per-row 16-entry LUT construction. */
        total_shift = 8U + (uint32_t)shift;
        return qbh_attn_clip_u8(
            (int32_t)((numerator +
                       (UINT64_C(1) << (total_shift - 1U))) >>
                      total_shift),
            NULL);
    }
}

static void qbh_attn_build_probability_lut(
    uint8_t *lut_bytes, uint32_t sum, uint32_t mode,
    uint32_t valid_count) {
    memset(lut_bytes, 0, QBH_ATTN_HVX_BYTES);
    if (valid_count == 1U) {
        for (uint32_t code = 0U; code <= 15U; ++code) {
            lut_bytes[2U * code] = UINT8_MAX;
        }
        return;
    }
    if (mode == QBH_ATTENTION_DIVISION_EXACT) {
        for (uint32_t code = 0U; code <= 15U; ++code) {
            lut_bytes[2U * code] = qbh_attn_divide_probability(
                code, sum, mode, valid_count);
        }
        return;
    }
    {
        const uint32_t leading = qbh_attn_floor_log2_u32(sum);
        const uint32_t next =
            (sum >> (leading - 1U)) & 1U;
        const uint32_t coefficient =
            mode == QBH_ATTENTION_DIVISION_SOLE
                ? (next != 0U ? 145U : 209U)
                : (next != 0U ? 171U : 256U);
        const uint64_t numerator =
            (uint64_t)UINT8_MAX * coefficient;
        const uint32_t base_shift =
            8U + leading - QBH_ATTN_EXP_FRAC_BITS;
        for (uint32_t code = 0U; code <= 15U; ++code) {
            const uint32_t shift = base_shift + code;
            lut_bytes[2U * code] = qbh_attn_clip_u8(
                (int32_t)((numerator +
                           (UINT64_C(1) << (shift - 1U))) >> shift),
                NULL);
        }
    }
}

static uint8_t qbh_attn_hvx_reduce_max_u8(HVX_Vector value) {
    const HVX_Vector zero = Q6_V_vzero();
    for (uint32_t shift = 64U; shift != 0U; shift >>= 1U) {
        value = Q6_Vub_vmax_VubVub(
            value, Q6_V_vlalign_VVR(value, zero, shift));
    }
    return (uint8_t)(Q6_R_vextract_VR(value, 124) >> 24U);
}

static uint32_t qbh_attn_hvx_reduce_sum_w(HVX_Vector value) {
    const HVX_Vector zero = Q6_V_vzero();
    for (uint32_t shift = 64U; shift >= 4U; shift >>= 1U) {
        value = Q6_Vw_vadd_VwVw(
            value, Q6_V_vlalign_VVR(value, zero, shift));
    }
    return (uint32_t)Q6_R_vextract_VR(value, 124);
}

static uint32_t qbh_attn_hvx_sum_log2_weights(HVX_Vector codes) {
    const HVX_VectorPair code_h = Q6_Wuh_vunpack_Vub(codes);
    const HVX_Vector one_h = Q6_Vh_vsplat_R(1);
    const HVX_Vector exponent_h = Q6_Vh_vsub_VhVh(
        Q6_Vh_vsplat_R(15), Q6_V_lo_W(code_h));
    const HVX_Vector weights_h =
        Q6_Vh_vasl_VhVh(one_h, exponent_h);
    const HVX_VectorPair weights_w =
        Q6_Wuw_vunpack_Vuh(weights_h);
    return qbh_attn_hvx_reduce_sum_w(
        Q6_Vw_vadd_VwVw(
            Q6_V_lo_W(weights_w), Q6_V_hi_W(weights_w)));
}

static uint32_t qbh_attn_hvx_sum_probability(HVX_Vector probability) {
    const HVX_VectorPair probability_h =
        Q6_Wuh_vunpack_Vub(probability);
    const HVX_VectorPair probability_w =
        Q6_Wuw_vunpack_Vuh(Q6_V_lo_W(probability_h));
    return qbh_attn_hvx_reduce_sum_w(
        Q6_Vw_vadd_VwVw(
            Q6_V_lo_W(probability_w),
            Q6_V_hi_W(probability_w)));
}

static HVX_Vector qbh_attn_hvx_log2_codes(
    HVX_Vector score, uint32_t row, uint32_t fraction_bits) {
    const HVX_Vector lane =
        *(const HVX_Vector *)qbh_attn_lane_index;
    const HVX_Vector row_vector = Q6_Vb_vsplat_R(row);
    const HVX_VectorPred valid = Q6_Q_not_Q(
        Q6_Q_vcmp_gt_VubVub(lane, row_vector));
    const HVX_Vector zero = Q6_V_vzero();
    const HVX_Vector masked_score = Q6_V_vmux_QVV(valid, score, zero);
    const uint8_t maximum = qbh_attn_hvx_reduce_max_u8(masked_score);
    const HVX_VectorPair score_h = Q6_Wuh_vunpack_Vub(masked_score);
    const HVX_Vector maximum_h = Q6_Vh_vsplat_R(maximum);
    const HVX_Vector rounding_h =
        Q6_Vh_vsplat_R(UINT32_C(1) << (fraction_bits - 1U));
    const HVX_Vector limit_h = Q6_Vh_vsplat_R(15);
    HVX_Vector code_lo = Q6_Vuh_vsub_VuhVuh_sat(
        maximum_h, Q6_V_lo_W(score_h));
    HVX_Vector code_hi = Q6_Vuh_vsub_VuhVuh_sat(
        maximum_h, Q6_V_hi_W(score_h));
    code_lo = Q6_Vh_vadd_VhVh(code_lo, rounding_h);
    code_hi = Q6_Vh_vadd_VhVh(code_hi, rounding_h);
    code_lo = Q6_Vuh_vlsr_VuhR(code_lo, fraction_bits);
    code_hi = Q6_Vuh_vlsr_VuhR(code_hi, fraction_bits);
    code_lo = Q6_Vuh_vmin_VuhVuh(code_lo, limit_h);
    code_hi = Q6_Vuh_vmin_VuhVuh(code_hi, limit_h);
    return Q6_V_vmux_QVV(
        valid, Q6_Vub_vpack_VhVh_sat(code_hi, code_lo),
        Q6_Vb_vsplat_R(16));
}

static void qbh_attn_softmax(
    struct qbh_attention_header *header,
    const struct qbh_attention_buffers *buffers) {
    uint8_t *row_scratch = buffers->softmax_scratch;
    uint8_t *lut_bytes = row_scratch + QBH_ATTN_HVX_BYTES;
    uint32_t row_sum_min = UINT32_MAX;
    uint32_t row_sum_max = 0U;

    for (uint32_t head = 0U;
         head < QBH_ATTENTION_Q_HEADS_PER_GROUP; ++head) {
        const uint8_t *score_base = buffers->score_tiles +
            (size_t)head * QBH_ATTENTION_SCORE_TILES *
                QBH_HMX_OUTPUT_BYTES;
        uint8_t *probability_base = buffers->probability_tiles +
            (size_t)head * QBH_ATTENTION_SCORE_TILES *
                QBH_HMX_ACTIVATION_BYTES;
        for (uint32_t row = 0U; row < QBH_ATTENTION_M; ++row) {
            const uint32_t valid_count = row + 1U;
            uint32_t sum = 0U;
            uint32_t probability_sum = 0U;
            HVX_Vector codes;
            HVX_Vector probabilities;

            memset(row_scratch, 0, QBH_ATTN_HVX_BYTES);
            memcpy(row_scratch,
                   score_base +
                       (size_t)row * QBH_HMX_OUTPUT_CHANNELS,
                   QBH_HMX_OUTPUT_CHANNELS);
            memcpy(row_scratch + QBH_HMX_OUTPUT_CHANNELS,
                   score_base + QBH_HMX_OUTPUT_BYTES +
                       (size_t)row * QBH_HMX_OUTPUT_CHANNELS,
                   QBH_HMX_OUTPUT_CHANNELS);
            codes = qbh_attn_hvx_log2_codes(
                *(const HVX_Vector *)row_scratch, row,
                header->config.fraction_bits);
            sum = qbh_attn_hvx_sum_log2_weights(codes);

            qbh_attn_build_probability_lut(
                lut_bytes, sum, header->config.division_mode,
                valid_count);
            probabilities = Q6_Vb_vlut32_VbVbR_nomatch(
                codes, *(const HVX_Vector *)lut_bytes, 0);
            *(HVX_Vector *)row_scratch = probabilities;
            probability_sum =
                qbh_attn_hvx_sum_probability(probabilities);
            memcpy(probability_base +
                       (size_t)row * QBH_HMX_INPUT_CHANNELS,
                   row_scratch, QBH_HMX_INPUT_CHANNELS);
            memcpy(probability_base + QBH_HMX_ACTIVATION_BYTES +
                       (size_t)row * QBH_HMX_INPUT_CHANNELS,
                   row_scratch + QBH_HMX_INPUT_CHANNELS,
                   QBH_HMX_INPUT_CHANNELS);
            if (probability_sum < row_sum_min) {
                row_sum_min = probability_sum;
            }
            if (probability_sum > row_sum_max) {
                row_sum_max = probability_sum;
            }
        }
    }
    asm volatile("barrier" ::: "memory");
    header->probability_row_sum_min = row_sum_min;
    header->probability_row_sum_max = row_sum_max;
}

static int qbh_attn_run_av(
    struct qbh_attention_header *header,
    const struct qbh_attention_buffers *buffers,
    struct qbh_attention_hmx_worker *worker) {
    for (uint32_t head = 0U;
         head < QBH_ATTENTION_Q_HEADS_PER_GROUP; ++head) {
        const uint8_t *probability = buffers->probability_tiles +
            (size_t)head * QBH_ATTENTION_SCORE_TILES *
                QBH_HMX_ACTIVATION_BYTES;
        for (uint32_t n_tile = 0U;
             n_tile < QBH_ATTENTION_HEAD_DIM_TILES; ++n_tile) {
            uint8_t *output_tile = buffers->output_tiles +
                ((size_t)head * QBH_ATTENTION_HEAD_DIM_TILES + n_tile) *
                    QBH_HMX_OUTPUT_BYTES;
            const int8_t *weight = buffers->v_weight_tiles +
                (size_t)n_tile * QBH_ATTENTION_SCORE_TILES *
                    QBH_HMX_WEIGHT_BYTES;
            const uint32_t *bias = buffers->av_bias +
                (size_t)n_tile *
                    (QBH_HMX_BIAS_BYTES / sizeof(uint32_t));
            if (qbh_attn_hmx_output(
                    worker, probability, weight, bias,
                    output_tile, QBH_ATTENTION_SCORE_TILES,
                    &header->hmx_av_stream_count) != 0) {
                return -1;
            }
            header->hmx_av_execution_count +=
                QBH_ATTENTION_SCORE_TILES;
        }
    }
    return 0;
}

static void qbh_attn_requant_av(
    struct qbh_attention_header *header,
    const struct qbh_attention_buffers *buffers) {
    const struct qbh_attention_config *config = &header->config;
    const uint32_t tile_bytes =
        QBH_ATTENTION_Q_HEADS_PER_GROUP *
        QBH_ATTENTION_HEAD_DIM_TILES * QBH_HMX_OUTPUT_BYTES;
    for (uint32_t offset = 0U; offset < tile_bytes;
         offset += sizeof(HVX_Vector)) {
        HVX_Vector *output = (HVX_Vector *)(
            buffers->output_tiles + offset);
        *output = qbh_attn_hvx_requant_centered_u8(
            *output, config->av_multiplier,
            config->output_zero_point);
    }
    asm volatile("barrier" ::: "memory");
    for (uint32_t head = 0U;
         head < QBH_ATTENTION_Q_HEADS_PER_GROUP; ++head) {
        for (uint32_t row = 0U; row < QBH_ATTENTION_M; ++row) {
            uint8_t *destination = buffers->output +
                ((size_t)head * QBH_ATTENTION_M + row) *
                    QBH_ATTENTION_HEAD_DIM;
            for (uint32_t n_tile = 0U;
                 n_tile < QBH_ATTENTION_HEAD_DIM_TILES; ++n_tile) {
                const uint8_t *source = buffers->output_tiles +
                    ((size_t)head * QBH_ATTENTION_HEAD_DIM_TILES +
                     n_tile) * QBH_HMX_OUTPUT_BYTES +
                    (size_t)row * QBH_HMX_OUTPUT_CHANNELS;
                memcpy(destination +
                           n_tile * QBH_HMX_OUTPUT_CHANNELS,
                       source, QBH_HMX_OUTPUT_CHANNELS);
            }
        }
    }
}

static void qbh_attn_self_test(
    struct qbh_attention_header *header,
    const struct qbh_attention_buffers *buffers,
    const uint8_t *reference_score,
    const uint8_t *reference_probability,
    const uint8_t *reference_output) {
    for (uint32_t head = 0U;
         head < QBH_ATTENTION_Q_HEADS_PER_GROUP; ++head) {
        for (uint32_t row = 0U; row < QBH_ATTENTION_M; ++row) {
            for (uint32_t column = 0U;
                 column < QBH_ATTENTION_M; ++column) {
                const size_t logical =
                    ((size_t)head * QBH_ATTENTION_M + row) *
                        QBH_ATTENTION_M + column;
                const size_t physical =
                    (size_t)head * QBH_ATTENTION_SCORE_TILES *
                        QBH_HMX_OUTPUT_BYTES +
                    (size_t)(column / QBH_HMX_OUTPUT_CHANNELS) *
                        QBH_HMX_OUTPUT_BYTES +
                    (size_t)row * QBH_HMX_OUTPUT_CHANNELS +
                    column % QBH_HMX_OUTPUT_CHANNELS;
                if (buffers->score_tiles[physical] !=
                    reference_score[logical]) {
                    if (header->score_mismatch_count == 0U) {
                        header->first_score_mismatch_index =
                            (uint32_t)logical;
                        header->first_score_actual =
                            buffers->score_tiles[physical];
                        header->first_score_expected =
                            reference_score[logical];
                    }
                    ++header->score_mismatch_count;
                }
                if (buffers->probability_tiles[physical] !=
                    reference_probability[logical]) {
                    if (header->probability_mismatch_count == 0U) {
                        header->first_probability_mismatch_index =
                            (uint32_t)logical;
                        header->first_probability_actual =
                            buffers->probability_tiles[physical];
                        header->first_probability_expected =
                            reference_probability[logical];
                    }
                    ++header->probability_mismatch_count;
                }
                if (column > row &&
                    buffers->probability_tiles[physical] != 0U) {
                    ++header->probability_mask_violation_count;
                }
            }
        }
    }
    for (uint32_t element = 0U;
         element < QBH_ATTENTION_OUTPUT_BYTES; ++element) {
        uint32_t actual = buffers->output[element];
        uint32_t expected = reference_output[element];
        uint32_t difference = actual > expected
                                  ? actual - expected
                                  : expected - actual;
        if (difference != 0U) {
            ++header->output_mismatch_count;
        }
        if (difference > header->output_max_abs_lsb) {
            header->output_max_abs_lsb = difference;
        }
    }
}

AEEResult qbh_run_attention_rpc(int32_t shared_fd, uint32_t shared_bytes,
                                uint8_t *vtcm, uint32_t vtcm_bytes,
                                uint32_t hmx_context_id,
                                uint32_t prepared_session_run_index) {
    struct qbh_attention_header *header = NULL;
    struct qbh_attention_buffers buffers;
    struct qbh_attention_hmx_worker worker;
    qurt_thread_attr_t attributes;
    qurt_thread_t thread;
    uint8_t *shared = NULL;
    int cache_status;
    int core_status = 0;
    int hvx_locked = 0;
    int thread_exit_status = AEE_EFAILED;
    AEEResult result;
    uint64_t start;

    result = HAP_mmap_get(shared_fd, (void **)&shared, NULL);
    if (result != AEE_SUCCESS || shared == NULL) {
        return result != AEE_SUCCESS ? result : AEE_EFAILED;
    }
    if (shared_bytes < sizeof(*header)) {
        (void)HAP_mmap_put(shared_fd);
        return AEE_EBADSIZE;
    }
    header = (struct qbh_attention_header *)shared;
    cache_status = qurt_mem_cache_clean(
        (qurt_addr_t)shared, (qurt_size_t)shared_bytes,
        QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
    if (cache_status != 0) {
        header->dsp_status = QBH_ATTENTION_STATUS_CACHE_FAILED;
        header->cache_status = cache_status;
        result = AEE_EFAILED;
        goto publish;
    }
    if (!qbh_attn_header_valid(header, shared_bytes)) {
        header->dsp_status = QBH_ATTENTION_STATUS_BAD_HEADER;
        result = AEE_EBADPARM;
        goto publish;
    }
    memset(&buffers, 0, sizeof(buffers));
    if (qbh_attn_plan_buffers(vtcm, vtcm_bytes, &buffers) != 0) {
        header->dsp_status = QBH_ATTENTION_STATUS_VTCM_PLAN_FAILED;
        result = AEE_ENOMEMORY;
        goto publish;
    }

    header->dsp_status = QBH_ATTENTION_STATUS_DSP_RUNNING;
    header->prepared_session_run_index = prepared_session_run_index;
    header->resource_vtcm_address = (uint32_t)(uintptr_t)vtcm;
    header->resource_hmx_context_id = hmx_context_id;
    header->vtcm_acquired_bytes = vtcm_bytes;
    header->vtcm_peak_plan_bytes = buffers.plan_bytes;
    header->qtimer_start = HAP_perf_get_qtimer_count();

    start = HAP_perf_get_qtimer_count();
    memcpy(buffers.q, shared + header->q_offset, header->q_bytes);
    memcpy(buffers.k, shared + header->k_offset, header->k_bytes);
    memcpy(buffers.v, shared + header->v_offset, header->v_bytes);
    header->input_stage_ticks = HAP_perf_get_qtimer_count() - start;

    start = HAP_perf_get_qtimer_count();
    qbh_attn_pack_q(buffers.q, buffers.q_tiles);
    qbh_attn_pack_k(buffers.k, &header->config,
                    buffers.k_weight_tiles, buffers.qk_bias);
    qbh_attn_pack_v(buffers.v, &header->config,
                    buffers.v_weight_tiles, buffers.av_bias,
                    &header->v_recenter_saturation_count);
    header->debug_qk_bias_upper0 = (int32_t)buffers.qk_bias[
        QBH_HMX_OUTPUT_CHANNELS];
    header->debug_qk_bias_upper32 = (int32_t)buffers.qk_bias[
        QBH_HMX_BIAS_BYTES / sizeof(uint32_t) +
        QBH_HMX_OUTPUT_CHANNELS];
    header->pack_ticks = HAP_perf_get_qtimer_count() - start;

    memset(&worker, 0, sizeof(worker));
    worker.hmx_context_id = hmx_context_id;
    qurt_sem_init_val(&worker.command_ready, 0U);
    qurt_sem_init_val(&worker.command_done, 0U);
    qurt_sem_init_val(&worker.worker_started, 0U);
    qurt_thread_attr_init(&attributes);
    qurt_thread_attr_set_name(&attributes, "qbh-attn-hmx");
    qurt_thread_attr_set_stack_addr(&attributes, qbh_attn_hmx_stack);
    qurt_thread_attr_set_stack_size(&attributes,
                                    QBH_ATTN_HMX_STACK_BYTES);
    qurt_thread_attr_set_priority(
        &attributes, qurt_thread_get_priority(qurt_thread_get_id()));
    qurt_thread_attr_set_detachstate(&attributes,
                                     QURT_THREAD_ATTR_CREATE_JOINABLE);
    result = qurt_thread_create(&thread, &attributes,
                                qbh_attn_hmx_worker_main, &worker);
    if (result != QURT_EOK) {
        header->dsp_status = QBH_ATTENTION_STATUS_HMX_WORKER_FAILED;
        result = AEE_EFAILED;
        qurt_sem_destroy(&worker.worker_started);
        qurt_sem_destroy(&worker.command_done);
        qurt_sem_destroy(&worker.command_ready);
        goto publish;
    }
    qurt_sem_down(&worker.worker_started);
    header->hmx_lock_status = worker.lock_status;
    if (worker.lock_status != AEE_SUCCESS) {
        (void)qurt_thread_join(thread, &thread_exit_status);
        qurt_sem_destroy(&worker.worker_started);
        qurt_sem_destroy(&worker.command_done);
        qurt_sem_destroy(&worker.command_ready);
        header->dsp_status = QBH_ATTENTION_STATUS_HMX_LOCK_FAILED;
        result = AEE_EFAILED;
        goto publish;
    }
    header->hvx_lock_status = qurt_hvx_lock(QURT_HVX_MODE_128B);
    if (header->hvx_lock_status != AEE_SUCCESS) {
        worker.stop = 1U;
        asm volatile("barrier" ::: "memory");
        (void)qurt_sem_up(&worker.command_ready);
        (void)qurt_thread_join(thread, &thread_exit_status);
        header->hmx_unlock_status = worker.unlock_status;
        qurt_sem_destroy(&worker.worker_started);
        qurt_sem_destroy(&worker.command_done);
        qurt_sem_destroy(&worker.command_ready);
        header->dsp_status = QBH_ATTENTION_STATUS_HVX_LOCK_FAILED;
        result = AEE_EFAILED;
        goto publish;
    }
    hvx_locked = 1;

    for (uint32_t repeat = 0U; repeat < header->repeat_count; ++repeat) {
        start = HAP_perf_get_qtimer_count();
        if (qbh_attn_run_qk(
                header, &buffers, &worker) != 0) {
            core_status = -1;
            break;
        }
        header->qk_hmx_ticks += HAP_perf_get_qtimer_count() - start;

        start = HAP_perf_get_qtimer_count();
        qbh_attn_requant_qk(header, &buffers);
        header->qk_requant_ticks +=
            HAP_perf_get_qtimer_count() - start;

        start = HAP_perf_get_qtimer_count();
        qbh_attn_softmax(header, &buffers);
        header->softmax_ticks += HAP_perf_get_qtimer_count() - start;

        start = HAP_perf_get_qtimer_count();
        if (qbh_attn_run_av(
                header, &buffers, &worker) != 0) {
            core_status = -1;
            break;
        }
        header->av_hmx_ticks += HAP_perf_get_qtimer_count() - start;

        start = HAP_perf_get_qtimer_count();
        qbh_attn_requant_av(header, &buffers);
        header->av_requant_ticks +=
            HAP_perf_get_qtimer_count() - start;
    }

    if (hvx_locked != 0) {
        header->hvx_unlock_status = qurt_hvx_unlock();
        hvx_locked = 0;
        if (header->hvx_unlock_status != AEE_SUCCESS) {
            core_status = -2;
        }
    }

    worker.stop = 1U;
    asm volatile("barrier" ::: "memory");
    (void)qurt_sem_up(&worker.command_ready);
    (void)qurt_thread_join(thread, &thread_exit_status);
    header->hmx_unlock_status = worker.unlock_status;
    qurt_sem_destroy(&worker.worker_started);
    qurt_sem_destroy(&worker.command_done);
    qurt_sem_destroy(&worker.command_ready);
    if (core_status != 0 || thread_exit_status != AEE_SUCCESS) {
        header->dsp_status = core_status == -2
            ? QBH_ATTENTION_STATUS_HVX_UNLOCK_FAILED
            : (worker.unlock_status != AEE_SUCCESS
                   ? QBH_ATTENTION_STATUS_HMX_UNLOCK_FAILED
                   : QBH_ATTENTION_STATUS_HMX_WORKER_FAILED);
        result = AEE_EFAILED;
        goto publish;
    }

    if (header->run_self_test != 0U) {
        start = HAP_perf_get_qtimer_count();
        qbh_attn_self_test(
            header, &buffers,
            shared + header->reference_score_offset,
            shared + header->reference_probability_offset,
            shared + header->reference_output_offset);
        header->self_test_ticks = HAP_perf_get_qtimer_count() - start;
    }

    start = HAP_perf_get_qtimer_count();
    memcpy(shared + header->output_offset, buffers.output,
           header->output_bytes);
    header->output_stage_ticks = HAP_perf_get_qtimer_count() - start;
    header->qtimer_end = HAP_perf_get_qtimer_count();
    header->total_ticks = header->qtimer_end - header->qtimer_start;
    header->dsp_status =
        header->run_self_test != 0U &&
                (header->score_mismatch_count != 0U ||
                 header->probability_mismatch_count != 0U ||
                 header->output_mismatch_count != 0U ||
                 header->probability_mask_violation_count != 0U)
            ? QBH_ATTENTION_STATUS_NUMERICAL_FAILED
            : QBH_ATTENTION_STATUS_OK;
    result = header->dsp_status == QBH_ATTENTION_STATUS_OK
                 ? AEE_SUCCESS
                 : AEE_EFAILED;

publish:
    if (header != NULL) {
        cache_status = qurt_mem_cache_clean(
            (qurt_addr_t)shared, (qurt_size_t)shared_bytes,
            QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
        if (cache_status != 0 && result == AEE_SUCCESS) {
            header->dsp_status = QBH_ATTENTION_STATUS_CACHE_FAILED;
            header->cache_status = cache_status;
            result = AEE_EFAILED;
        }
    }
    (void)HAP_mmap_put(shared_fd);
    return result;
}
