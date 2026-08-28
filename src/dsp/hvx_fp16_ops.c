#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "hvx_fp16_ops.h"
#include "hmx_fp16.h"
#include "qhmath_hvx_vector.h"

#define QBH_HVX_BYTES UINT32_C(128)
#define QBH_HVX_F16_LANES UINT32_C(64)

static const uint16_t qbh_hvx_lane_index[QBH_HVX_F16_LANES]
    __attribute__((aligned(QBH_HVX_BYTES))) = {
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
        48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
    };

static const int32_t qbh_hvx_crouton_vscatter_offsets[32]
    __attribute__((aligned(QBH_HVX_BYTES))) = {
        0, 128, 256, 384, 512, 640, 768, 896,
        1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920,
        2048, 2176, 2304, 2432, 2560, 2688, 2816, 2944,
        3072, 3200, 3328, 3456, 3584, 3712, 3840, 3968,
    };

void qbh_hvx_check_reset(struct qbh_hvx_check_metrics *metrics) {
    if (metrics == NULL) {
        return;
    }
    metrics->max_abs = 0.0f;
    metrics->dot = 0.0;
    metrics->actual_norm = 0.0;
    metrics->reference_norm = 0.0;
    metrics->nonfinite_count = 0U;
    metrics->mask_violation_count = 0U;
}

void qbh_hvx_residual_add_f16(__fp16 *residual,
                               const __fp16 *addition,
                               uint32_t elements) {
    HVX_Vector *residual_vectors = (HVX_Vector *)residual;
    const HVX_Vector *addition_vectors =
        (const HVX_Vector *)addition;
    const uint32_t vector_count = elements / QBH_HVX_F16_LANES;
    uint32_t index = 0U;

    for (; index + 4U <= vector_count; index += 4U) {
        const HVX_Vector residual0 = residual_vectors[index + 0U];
        const HVX_Vector residual1 = residual_vectors[index + 1U];
        const HVX_Vector residual2 = residual_vectors[index + 2U];
        const HVX_Vector residual3 = residual_vectors[index + 3U];
        const HVX_Vector addition0 = addition_vectors[index + 0U];
        const HVX_Vector addition1 = addition_vectors[index + 1U];
        const HVX_Vector addition2 = addition_vectors[index + 2U];
        const HVX_Vector addition3 = addition_vectors[index + 3U];
        residual_vectors[index + 0U] =
            Q6_Vhf_vadd_VhfVhf(residual0, addition0);
        residual_vectors[index + 1U] =
            Q6_Vhf_vadd_VhfVhf(residual1, addition1);
        residual_vectors[index + 2U] =
            Q6_Vhf_vadd_VhfVhf(residual2, addition2);
        residual_vectors[index + 3U] =
            Q6_Vhf_vadd_VhfVhf(residual3, addition3);
    }
    for (; index < vector_count; ++index) {
        residual_vectors[index] = Q6_Vhf_vadd_VhfVhf(
            residual_vectors[index], addition_vectors[index]);
    }
    for (uint32_t element =
             vector_count * QBH_HVX_F16_LANES;
         element < elements; ++element) {
        residual[element] = (__fp16)((float)residual[element] +
                                     (float)addition[element]);
    }
}

float qbh_hvx_check_cosine(const struct qbh_hvx_check_metrics *metrics) {
    if (metrics == NULL || metrics->actual_norm <= 0.0 ||
        metrics->reference_norm <= 0.0) {
        return 0.0f;
    }
    return (float)(metrics->dot /
        sqrt(metrics->actual_norm * metrics->reference_norm));
}

static void qbh_hvx_check_add(struct qbh_hvx_check_metrics *metrics,
                              float actual, float reference) {
    if (metrics == NULL) {
        return;
    }
    if (!isfinite(actual) || !isfinite(reference)) {
        ++metrics->nonfinite_count;
        return;
    }
    float difference = fabsf(actual - reference);
    if (difference > metrics->max_abs) {
        metrics->max_abs = difference;
    }
    metrics->dot += (double)actual * (double)reference;
    metrics->actual_norm += (double)actual * (double)actual;
    metrics->reference_norm += (double)reference * (double)reference;
}

static float qbh_hvx_reduce_sum_qf32(HVX_Vector sum) {
    float lanes[32] __attribute__((aligned(QBH_HVX_BYTES)));
    *(HVX_Vector *)lanes = Q6_Vsf_equals_Vqf32(sum);
    float result = 0.0f;
    for (uint32_t lane = 0; lane < 32U; ++lane) {
        result += lanes[lane];
    }
    return result;
}

static float qbh_hvx_reduce_sum_sf32(HVX_Vector sum) {
    float lanes[32] __attribute__((aligned(QBH_HVX_BYTES)));
    *(HVX_Vector *)lanes = sum;
    float result = 0.0f;
    for (uint32_t lane = 0; lane < 32U; ++lane) {
        result += lanes[lane];
    }
    return result;
}

static float qbh_hvx_sum_squares_f16(const __fp16 *input,
                                      uint32_t width) {
    const HVX_Vector *vectors = (const HVX_Vector *)input;
    HVX_Vector sum_lo = Q6_V_vzero();
    HVX_Vector sum_hi = Q6_V_vzero();
    uint32_t vector_count = width / QBH_HVX_F16_LANES;
    for (uint32_t index = 0; index < vector_count; ++index) {
        HVX_Vector value = vectors[index];
        HVX_VectorPair square = Q6_Wqf32_vmpy_VhfVhf(value, value);
        sum_lo = Q6_Vsf_vadd_VsfVsf(
            sum_lo, Q6_Vsf_equals_Vqf32(Q6_V_lo_W(square)));
        sum_hi = Q6_Vsf_vadd_VsfVsf(
            sum_hi, Q6_Vsf_equals_Vqf32(Q6_V_hi_W(square)));
    }
    return qbh_hvx_reduce_sum_sf32(
        Q6_Vsf_vadd_VsfVsf(sum_lo, sum_hi));
}

static float qbh_hvx_reduce_max_f16(HVX_Vector value) {
    const HVX_Vector negative_max = Q6_Vh_vsplat_R(0xfbff);
    __fp16 lanes[QBH_HVX_F16_LANES]
        __attribute__((aligned(QBH_HVX_BYTES)));
    for (int shift = 64; shift >= 2; shift >>= 1) {
        value = Q6_Vhf_vmax_VhfVhf(
            value, Q6_V_vlalign_VVR(value, negative_max, shift));
    }
    *(HVX_Vector *)lanes = value;
    return (float)lanes[QBH_HVX_F16_LANES - 1U];
}

static float qbh_hvx_reduce_sum_f16(HVX_Vector value) {
    const HVX_Vector one = Q6_Vh_vsplat_R(0x3c00);
    HVX_VectorPair widened = Q6_Wqf32_vmpy_VhfVhf(value, one);
    return qbh_hvx_reduce_sum_qf32(
        Q6_Vqf32_vadd_Vqf32Vqf32(
            Q6_V_lo_W(widened), Q6_V_hi_W(widened)));
}

static HVX_Vector qbh_hvx_multiply_scale_f16_f32(
    HVX_Vector left, HVX_Vector right, float scale) {
    HVX_DV scaled;
    const HVX_VectorPair product =
        Q6_Wqf32_vmpy_VhfVhf(left, right);
    const HVX_Vector scale_vector = Q6_Vsf_vadd_VsfVsf(
        Q6_V_vsplat_R(*(const int32_t *)&scale), Q6_V_vzero());
    const HVX_Vector product_lo =
        Q6_Vsf_equals_Vqf32(Q6_V_lo_W(product));
    const HVX_Vector product_hi =
        Q6_Vsf_equals_Vqf32(Q6_V_hi_W(product));
    scaled.V.lo = Q6_Vqf32_vmpy_VsfVsf(
        product_lo, scale_vector);
    scaled.V.hi = Q6_Vqf32_vmpy_VsfVsf(
        product_hi, scale_vector);
    return Q6_Vhf_equals_Wqf32(scaled.VV);
}

static HVX_Vector qbh_hvx_scale_then_multiply_f16_f32(
    HVX_Vector value, float scale, HVX_Vector multiplier) {
    const HVX_Vector one_half = Q6_Vh_vsplat_R(0x3c00);
    const HVX_Vector scale_vector = Q6_Vsf_vadd_VsfVsf(
        Q6_V_vsplat_R(*(const int32_t *)&scale), Q6_V_vzero());
    const HVX_VectorPair value_qf32 =
        Q6_Wqf32_vmpy_VhfVhf(value, one_half);
    const HVX_VectorPair multiplier_qf32 =
        Q6_Wqf32_vmpy_VhfVhf(multiplier, one_half);
    HVX_DV result;
    HVX_Vector scaled_lo = Q6_Vqf32_vmpy_VsfVsf(
        Q6_Vsf_equals_Vqf32(Q6_V_lo_W(value_qf32)), scale_vector);
    HVX_Vector scaled_hi = Q6_Vqf32_vmpy_VsfVsf(
        Q6_Vsf_equals_Vqf32(Q6_V_hi_W(value_qf32)), scale_vector);
    result.V.lo = Q6_Vqf32_vmpy_VsfVsf(
        Q6_Vsf_equals_Vqf32(scaled_lo),
        Q6_Vsf_equals_Vqf32(Q6_V_lo_W(multiplier_qf32)));
    result.V.hi = Q6_Vqf32_vmpy_VsfVsf(
        Q6_Vsf_equals_Vqf32(scaled_hi),
        Q6_Vsf_equals_Vqf32(Q6_V_hi_W(multiplier_qf32)));
    return Q6_Vhf_equals_Wqf32(result.VV);
}

void qbh_hvx_rms_norm_f16(const __fp16 *input, const __fp16 *gamma,
                           __fp16 *output, uint32_t rows,
                           uint32_t width,
                           struct qbh_hvx_check_metrics *check) {
    uint32_t vector_count = width / QBH_HVX_F16_LANES;
    const HVX_Vector *gamma_vectors = (const HVX_Vector *)gamma;
    for (uint32_t row = 0; row < rows; ++row) {
        const __fp16 *input_row = input + (size_t)row * width;
        __fp16 *output_row = output + (size_t)row * width;
        const HVX_Vector *input_vectors =
            (const HVX_Vector *)input_row;
        HVX_Vector *output_vectors = (HVX_Vector *)output_row;
        float sum = qbh_hvx_sum_squares_f16(input_row, width);
        float inverse =
            1.0f / sqrtf(sum / (float)width + 1.0e-6f);
        for (uint32_t index = 0; index < vector_count; ++index) {
            output_vectors[index] =
                qbh_hvx_scale_then_multiply_f16_f32(
                    input_vectors[index], inverse, gamma_vectors[index]);
        }
        if (check != NULL) {
            float reference_sum = 0.0f;
            for (uint32_t channel = 0; channel < width; ++channel) {
                float value = (float)input_row[channel];
                reference_sum += value * value;
            }
            float reference_inverse = 1.0f / sqrtf(
                reference_sum / (float)width + 1.0e-6f);
            for (uint32_t channel = 0; channel < width; ++channel) {
                __fp16 reference = (__fp16)(
                    (float)input_row[channel] * reference_inverse *
                    (float)gamma[channel]);
                qbh_hvx_check_add(
                    check, (float)output_row[channel], (float)reference);
            }
        }
    }
}

static void qbh_hvx_store_f16_pair_crouton(
    __fp16 *output_tiles, uint32_t column_tiles,
    uint32_t row, uint32_t first_column,
    HVX_Vector row0, HVX_Vector row1) {
    const uint32_t row_tile = row / QBH_HMX_FP16_ROWS;
    const uint32_t row_pair =
        (row % QBH_HMX_FP16_ROWS) / 2U;
    const size_t tile = qbh_hmx_fp16_matrix_tile_offset(
        row_tile, first_column / QBH_HMX_FP16_COLS,
        column_tiles);
    const HVX_VectorPair packed = Q6_W_vshuff_VVR(
        row1, row0, -2);
    HVX_Vector *output0 =
        (HVX_Vector *)(output_tiles + tile) + row_pair;
    HVX_Vector *output1 = output0 +
        QBH_HMX_FP16_TILE_BYTES / sizeof(HVX_Vector);
    *output0 = Q6_V_lo_W(packed);
    *output1 = Q6_V_hi_W(packed);
}

void qbh_hvx_rms_norm_f16_crouton(
    const __fp16 *input, const __fp16 *gamma,
    __fp16 *output_tiles, uint32_t rows, uint32_t width) {
    const uint32_t vector_count = width / QBH_HVX_F16_LANES;
    const uint32_t column_tiles = width / QBH_HMX_FP16_COLS;
    const HVX_Vector *gamma_vectors = (const HVX_Vector *)gamma;

    for (uint32_t row = 0U; row < rows; row += 2U) {
        const __fp16 *input0 = input + (size_t)row * width;
        const __fp16 *input1 = input + (size_t)(row + 1U) * width;
        const HVX_Vector *vectors0 = (const HVX_Vector *)input0;
        const HVX_Vector *vectors1 = (const HVX_Vector *)input1;
        const float inverse0 = 1.0f / sqrtf(
            qbh_hvx_sum_squares_f16(input0, width) /
                (float)width + 1.0e-6f);
        const float inverse1 = 1.0f / sqrtf(
            qbh_hvx_sum_squares_f16(input1, width) /
                (float)width + 1.0e-6f);

        for (uint32_t index = 0U; index < vector_count; ++index) {
            qbh_hvx_store_f16_pair_crouton(
                output_tiles, column_tiles, row,
                index * QBH_HVX_F16_LANES,
                qbh_hvx_scale_then_multiply_f16_f32(
                    vectors0[index], inverse0, gamma_vectors[index]),
                qbh_hvx_scale_then_multiply_f16_f32(
                    vectors1[index], inverse1, gamma_vectors[index]));
        }
    }
    asm volatile("barrier" ::: "memory");
}

void qbh_hvx_residual_rms_norm_f16(
    __fp16 *residual, const __fp16 *addition,
    const __fp16 *gamma, __fp16 *output,
    uint32_t rows, uint32_t width) {
    const uint32_t vector_count = width / QBH_HVX_F16_LANES;
    const uint32_t vector_elements =
        vector_count * QBH_HVX_F16_LANES;
    const HVX_Vector *gamma_vectors = (const HVX_Vector *)gamma;

    for (uint32_t row = 0U; row < rows; ++row) {
        __fp16 *residual_row = residual + (size_t)row * width;
        const __fp16 *addition_row = addition + (size_t)row * width;
        __fp16 *output_row = output + (size_t)row * width;
        HVX_Vector *residual_vectors = (HVX_Vector *)residual_row;
        const HVX_Vector *addition_vectors =
            (const HVX_Vector *)addition_row;
        HVX_Vector *output_vectors = (HVX_Vector *)output_row;
        HVX_Vector sum_lo = Q6_V_vzero();
        HVX_Vector sum_hi = Q6_V_vzero();

        for (uint32_t index = 0U; index < vector_count; ++index) {
            const HVX_Vector value = Q6_Vhf_vadd_VhfVhf(
                residual_vectors[index], addition_vectors[index]);
            const HVX_VectorPair square =
                Q6_Wqf32_vmpy_VhfVhf(value, value);
            residual_vectors[index] = value;
            sum_lo = Q6_Vsf_vadd_VsfVsf(
                sum_lo, Q6_Vsf_equals_Vqf32(Q6_V_lo_W(square)));
            sum_hi = Q6_Vsf_vadd_VsfVsf(
                sum_hi, Q6_Vsf_equals_Vqf32(Q6_V_hi_W(square)));
        }
        float sum = qbh_hvx_reduce_sum_sf32(
            Q6_Vsf_vadd_VsfVsf(sum_lo, sum_hi));
        for (uint32_t channel = vector_elements;
             channel < width; ++channel) {
            residual_row[channel] = (__fp16)(
                (float)residual_row[channel] +
                (float)addition_row[channel]);
            const float value = (float)residual_row[channel];
            sum += value * value;
        }
        const float inverse =
            1.0f / sqrtf(sum / (float)width + 1.0e-6f);
        for (uint32_t index = 0U; index < vector_count; ++index) {
            output_vectors[index] =
                qbh_hvx_scale_then_multiply_f16_f32(
                    residual_vectors[index], inverse,
                    gamma_vectors[index]);
        }
        for (uint32_t channel = vector_elements;
             channel < width; ++channel) {
            output_row[channel] = (__fp16)(
                (float)residual_row[channel] * inverse *
                (float)gamma[channel]);
        }
    }
}

void qbh_hvx_residual_rms_norm_f16_crouton(
    __fp16 *residual, const __fp16 *addition,
    const __fp16 *gamma, __fp16 *output_tiles,
    uint32_t rows, uint32_t width) {
    const uint32_t vector_count = width / QBH_HVX_F16_LANES;
    const uint32_t column_tiles = width / QBH_HMX_FP16_COLS;
    const HVX_Vector *gamma_vectors = (const HVX_Vector *)gamma;

    for (uint32_t row = 0U; row < rows; row += 2U) {
        HVX_Vector *residual0 = (HVX_Vector *)(residual +
            (size_t)row * width);
        HVX_Vector *residual1 = (HVX_Vector *)(residual +
            (size_t)(row + 1U) * width);
        const HVX_Vector *addition0 = (const HVX_Vector *)(addition +
            (size_t)row * width);
        const HVX_Vector *addition1 = (const HVX_Vector *)(addition +
            (size_t)(row + 1U) * width);
        HVX_Vector sum0_lo = Q6_V_vzero();
        HVX_Vector sum0_hi = Q6_V_vzero();
        HVX_Vector sum1_lo = Q6_V_vzero();
        HVX_Vector sum1_hi = Q6_V_vzero();

        for (uint32_t index = 0U; index < vector_count; ++index) {
            const HVX_Vector value0 = Q6_Vhf_vadd_VhfVhf(
                residual0[index], addition0[index]);
            const HVX_Vector value1 = Q6_Vhf_vadd_VhfVhf(
                residual1[index], addition1[index]);
            const HVX_VectorPair square0 =
                Q6_Wqf32_vmpy_VhfVhf(value0, value0);
            const HVX_VectorPair square1 =
                Q6_Wqf32_vmpy_VhfVhf(value1, value1);
            residual0[index] = value0;
            residual1[index] = value1;
            sum0_lo = Q6_Vsf_vadd_VsfVsf(
                sum0_lo, Q6_Vsf_equals_Vqf32(Q6_V_lo_W(square0)));
            sum0_hi = Q6_Vsf_vadd_VsfVsf(
                sum0_hi, Q6_Vsf_equals_Vqf32(Q6_V_hi_W(square0)));
            sum1_lo = Q6_Vsf_vadd_VsfVsf(
                sum1_lo, Q6_Vsf_equals_Vqf32(Q6_V_lo_W(square1)));
            sum1_hi = Q6_Vsf_vadd_VsfVsf(
                sum1_hi, Q6_Vsf_equals_Vqf32(Q6_V_hi_W(square1)));
        }
        const float inverse0 = 1.0f / sqrtf(
            qbh_hvx_reduce_sum_sf32(
                Q6_Vsf_vadd_VsfVsf(sum0_lo, sum0_hi)) /
                (float)width + 1.0e-6f);
        const float inverse1 = 1.0f / sqrtf(
            qbh_hvx_reduce_sum_sf32(
                Q6_Vsf_vadd_VsfVsf(sum1_lo, sum1_hi)) /
                (float)width + 1.0e-6f);
        for (uint32_t index = 0U; index < vector_count; ++index) {
            qbh_hvx_store_f16_pair_crouton(
                output_tiles, column_tiles, row,
                index * QBH_HVX_F16_LANES,
                qbh_hvx_scale_then_multiply_f16_f32(
                    residual0[index], inverse0, gamma_vectors[index]),
                qbh_hvx_scale_then_multiply_f16_f32(
                    residual1[index], inverse1, gamma_vectors[index]));
        }
    }
    asm volatile("barrier" ::: "memory");
}

void qbh_hvx_qk_norm_rope_f16(__fp16 *tensor, uint32_t rows,
                               uint32_t heads, uint32_t row_stride,
                               uint32_t head_dim, const __fp16 *gamma,
                               const __fp16 *cosine,
                               const __fp16 *sine,
                               struct qbh_hvx_check_metrics *check) {
    const uint32_t half_dim = head_dim / 2U;
    for (uint32_t row = 0; row < rows; ++row) {
        const HVX_Vector cosine_first =
            *(const HVX_Vector *)(cosine + (size_t)row * head_dim);
        const HVX_Vector cosine_second = *(const HVX_Vector *)(
            cosine + (size_t)row * head_dim + half_dim);
        const HVX_Vector sine_first =
            *(const HVX_Vector *)(sine + (size_t)row * head_dim);
        const HVX_Vector sine_second = *(const HVX_Vector *)(
            sine + (size_t)row * head_dim + half_dim);
        const HVX_Vector gamma_first = *(const HVX_Vector *)gamma;
        const HVX_Vector gamma_second =
            *(const HVX_Vector *)(gamma + half_dim);
        for (uint32_t head = 0; head < heads; ++head) {
            __fp16 original[128] __attribute__((aligned(QBH_HVX_BYTES)));
            __fp16 *values = tensor + (size_t)row * row_stride +
                             (size_t)head * head_dim;
            if (check != NULL) {
                for (uint32_t channel = 0; channel < head_dim; ++channel) {
                    original[channel] = values[channel];
                }
            }
            float sum = qbh_hvx_sum_squares_f16(values, head_dim);
            __fp16 inverse = (__fp16)(
                1.0f / sqrtf(sum / (float)head_dim + 1.0e-6f));
            HVX_Vector inverse_vector =
                Q6_Vh_vsplat_R(*(const uint16_t *)&inverse);
            HVX_Vector first = *(const HVX_Vector *)values;
            HVX_Vector second =
                *(const HVX_Vector *)(values + half_dim);
            HVX_Vector first_norm = Q6_Vqf16_vmpy_VhfVhf(
                first, gamma_first);
            HVX_Vector second_norm = Q6_Vqf16_vmpy_VhfVhf(
                second, gamma_second);
            first = Q6_Vhf_equals_Vqf16(
                Q6_Vqf16_vmpy_Vqf16Vhf(
                    first_norm, inverse_vector));
            second = Q6_Vhf_equals_Vqf16(
                Q6_Vqf16_vmpy_Vqf16Vhf(
                    second_norm, inverse_vector));

            HVX_Vector first_rotated = Q6_Vqf16_vsub_Vqf16Vqf16(
                Q6_Vqf16_vmpy_VhfVhf(first, cosine_first),
                Q6_Vqf16_vmpy_VhfVhf(second, sine_first));
            HVX_Vector second_rotated = Q6_Vqf16_vadd_Vqf16Vqf16(
                Q6_Vqf16_vmpy_VhfVhf(second, cosine_second),
                Q6_Vqf16_vmpy_VhfVhf(first, sine_second));
            *(HVX_Vector *)values =
                Q6_Vhf_equals_Vqf16(first_rotated);
            *(HVX_Vector *)(values + half_dim) =
                Q6_Vhf_equals_Vqf16(second_rotated);
            if (check != NULL) {
                float reference_sum = 0.0f;
                for (uint32_t channel = 0; channel < head_dim; ++channel) {
                    float value = (float)original[channel];
                    reference_sum += value * value;
                }
                float reference_inverse = 1.0f / sqrtf(
                    reference_sum / (float)head_dim + 1.0e-6f);
                for (uint32_t channel = 0; channel < half_dim; ++channel) {
                    float first = (float)original[channel] *
                        reference_inverse * (float)gamma[channel];
                    float second = (float)original[channel + half_dim] *
                        reference_inverse * (float)gamma[channel + half_dim];
                    __fp16 first_reference = (__fp16)(
                        first * (float)cosine[(size_t)row * head_dim + channel] -
                        second * (float)sine[(size_t)row * head_dim + channel]);
                    __fp16 second_reference = (__fp16)(
                        second * (float)cosine[
                            (size_t)row * head_dim + channel + half_dim] +
                        first * (float)sine[
                            (size_t)row * head_dim + channel + half_dim]);
                    qbh_hvx_check_add(
                        check, (float)values[channel],
                        (float)first_reference);
                    qbh_hvx_check_add(
                        check, (float)values[channel + half_dim],
                        (float)second_reference);
                }
            }
        }
    }
}

void qbh_hvx_qk_norm_rope_f16_head(
    __fp16 *tensor, uint32_t rows, uint32_t row_stride,
    uint32_t head_dim, uint32_t head, const __fp16 *gamma,
    const __fp16 *cosine, const __fp16 *sine) {
    const uint32_t half_dim = head_dim / 2U;
    const HVX_Vector gamma_first = *(const HVX_Vector *)gamma;
    const HVX_Vector gamma_second =
        *(const HVX_Vector *)(gamma + half_dim);

    for (uint32_t row = 0U; row < rows; ++row) {
        const HVX_Vector cosine_first =
            *(const HVX_Vector *)(cosine + (size_t)row * head_dim);
        const HVX_Vector cosine_second = *(const HVX_Vector *)(
            cosine + (size_t)row * head_dim + half_dim);
        const HVX_Vector sine_first =
            *(const HVX_Vector *)(sine + (size_t)row * head_dim);
        const HVX_Vector sine_second = *(const HVX_Vector *)(
            sine + (size_t)row * head_dim + half_dim);
        __fp16 *values = tensor + (size_t)row * row_stride +
                         (size_t)head * head_dim;
        float sum = qbh_hvx_sum_squares_f16(values, head_dim);
        __fp16 inverse = (__fp16)(
            1.0f / sqrtf(sum / (float)head_dim + 1.0e-6f));
        HVX_Vector inverse_vector =
            Q6_Vh_vsplat_R(*(const uint16_t *)&inverse);
        HVX_Vector first = *(const HVX_Vector *)values;
        HVX_Vector second =
            *(const HVX_Vector *)(values + half_dim);
        HVX_Vector first_norm = Q6_Vqf16_vmpy_VhfVhf(
            first, gamma_first);
        HVX_Vector second_norm = Q6_Vqf16_vmpy_VhfVhf(
            second, gamma_second);
        first = Q6_Vhf_equals_Vqf16(
            Q6_Vqf16_vmpy_Vqf16Vhf(
                first_norm, inverse_vector));
        second = Q6_Vhf_equals_Vqf16(
            Q6_Vqf16_vmpy_Vqf16Vhf(
                second_norm, inverse_vector));

        HVX_Vector first_rotated = Q6_Vqf16_vsub_Vqf16Vqf16(
            Q6_Vqf16_vmpy_VhfVhf(first, cosine_first),
            Q6_Vqf16_vmpy_VhfVhf(second, sine_first));
        HVX_Vector second_rotated = Q6_Vqf16_vadd_Vqf16Vqf16(
            Q6_Vqf16_vmpy_VhfVhf(second, cosine_second),
            Q6_Vqf16_vmpy_VhfVhf(first, sine_second));
        *(HVX_Vector *)values =
            Q6_Vhf_equals_Vqf16(first_rotated);
        *(HVX_Vector *)(values + half_dim) =
            Q6_Vhf_equals_Vqf16(second_rotated);
    }
}

static void qbh_hvx_qk_norm_rope_vectors(
    HVX_Vector first, HVX_Vector second,
    HVX_Vector gamma_first, HVX_Vector gamma_second,
    HVX_Vector cosine_first, HVX_Vector cosine_second,
    HVX_Vector sine_first, HVX_Vector sine_second,
    HVX_Vector *first_output, HVX_Vector *second_output) {
    HVX_Vector sum_lo = Q6_V_vzero();
    HVX_Vector sum_hi = Q6_V_vzero();
    HVX_VectorPair square = Q6_Wqf32_vmpy_VhfVhf(first, first);
    __fp16 inverse;
    HVX_Vector inverse_vector;
    HVX_Vector first_norm;
    HVX_Vector second_norm;

    sum_lo = Q6_Vsf_vadd_VsfVsf(
        sum_lo, Q6_Vsf_equals_Vqf32(Q6_V_lo_W(square)));
    sum_hi = Q6_Vsf_vadd_VsfVsf(
        sum_hi, Q6_Vsf_equals_Vqf32(Q6_V_hi_W(square)));
    square = Q6_Wqf32_vmpy_VhfVhf(second, second);
    sum_lo = Q6_Vsf_vadd_VsfVsf(
        sum_lo, Q6_Vsf_equals_Vqf32(Q6_V_lo_W(square)));
    sum_hi = Q6_Vsf_vadd_VsfVsf(
        sum_hi, Q6_Vsf_equals_Vqf32(Q6_V_hi_W(square)));
    inverse = (__fp16)(1.0f / sqrtf(
        qbh_hvx_reduce_sum_sf32(
            Q6_Vsf_vadd_VsfVsf(sum_lo, sum_hi)) /
            128.0f + 1.0e-6f));
    inverse_vector = Q6_Vh_vsplat_R(*(const uint16_t *)&inverse);
    first_norm = Q6_Vqf16_vmpy_VhfVhf(first, gamma_first);
    second_norm = Q6_Vqf16_vmpy_VhfVhf(second, gamma_second);
    first = Q6_Vhf_equals_Vqf16(
        Q6_Vqf16_vmpy_Vqf16Vhf(first_norm, inverse_vector));
    second = Q6_Vhf_equals_Vqf16(
        Q6_Vqf16_vmpy_Vqf16Vhf(second_norm, inverse_vector));
    *first_output = Q6_Vhf_equals_Vqf16(
        Q6_Vqf16_vsub_Vqf16Vqf16(
            Q6_Vqf16_vmpy_VhfVhf(first, cosine_first),
            Q6_Vqf16_vmpy_VhfVhf(second, sine_first)));
    *second_output = Q6_Vhf_equals_Vqf16(
        Q6_Vqf16_vadd_Vqf16Vqf16(
            Q6_Vqf16_vmpy_VhfVhf(second, cosine_second),
            Q6_Vqf16_vmpy_VhfVhf(first, sine_second)));
}

void qbh_hvx_qk_norm_rope_f16_crouton_head(
    const __fp16 *source_group_tiles, __fp16 *destination_tiles,
    uint32_t head, uint32_t destination_is_weight,
    const __fp16 *gamma, const __fp16 *cosine,
    const __fp16 *sine) {
    const uint32_t head_tiles = 8U;
    const uint32_t group_tiles = 4U;
    const __fp16 *source = source_group_tiles +
        (size_t)head * head_tiles * QBH_HMX_FP16_TILE_ELEMENTS;
    __fp16 *destination = destination_tiles +
        (size_t)head * head_tiles * QBH_HMX_FP16_TILE_ELEMENTS;
    const HVX_Vector gamma_first = *(const HVX_Vector *)gamma;
    const HVX_Vector gamma_second =
        *(const HVX_Vector *)(gamma + QBH_HVX_F16_LANES);
    const HVX_Vector offset_base =
        *(const HVX_Vector *)qbh_hvx_crouton_vscatter_offsets;

    for (uint32_t row = 0U; row < 64U; row += 2U) {
        const uint32_t row_tile = row / QBH_HMX_FP16_ROWS;
        const uint32_t row_pair =
            (row % QBH_HMX_FP16_ROWS) / 2U;
        const HVX_Vector *source_first0 =
            (const HVX_Vector *)(source +
                (size_t)(row_tile * 2U) *
                    QBH_HMX_FP16_TILE_ELEMENTS) + row_pair;
        const HVX_Vector *source_first1 = source_first0 +
            QBH_HMX_FP16_TILE_BYTES / QBH_HVX_BYTES;
        const HVX_Vector *source_second0 =
            (const HVX_Vector *)(source +
                (size_t)(group_tiles + row_tile * 2U) *
                    QBH_HMX_FP16_TILE_ELEMENTS) + row_pair;
        const HVX_Vector *source_second1 = source_second0 +
            QBH_HMX_FP16_TILE_BYTES / QBH_HVX_BYTES;
        const HVX_VectorPair first_rows = Q6_W_vdeal_VVR(
            *source_first1, *source_first0, -2);
        const HVX_VectorPair second_rows = Q6_W_vdeal_VVR(
            *source_second1, *source_second0, -2);
        HVX_Vector row_first[2];
        HVX_Vector row_second[2];

        qbh_hvx_qk_norm_rope_vectors(
            Q6_V_lo_W(first_rows), Q6_V_lo_W(second_rows),
            gamma_first, gamma_second,
            *(const HVX_Vector *)(cosine + (size_t)row * 128U),
            *(const HVX_Vector *)(cosine +
                (size_t)row * 128U + QBH_HVX_F16_LANES),
            *(const HVX_Vector *)(sine + (size_t)row * 128U),
            *(const HVX_Vector *)(sine +
                (size_t)row * 128U + QBH_HVX_F16_LANES),
            &row_first[0], &row_second[0]);
        qbh_hvx_qk_norm_rope_vectors(
            Q6_V_hi_W(first_rows), Q6_V_hi_W(second_rows),
            gamma_first, gamma_second,
            *(const HVX_Vector *)(cosine + (size_t)(row + 1U) * 128U),
            *(const HVX_Vector *)(cosine +
                (size_t)(row + 1U) * 128U + QBH_HVX_F16_LANES),
            *(const HVX_Vector *)(sine + (size_t)(row + 1U) * 128U),
            *(const HVX_Vector *)(sine +
                (size_t)(row + 1U) * 128U + QBH_HVX_F16_LANES),
            &row_first[1], &row_second[1]);

        if (destination_is_weight == 0U) {
            const HVX_VectorPair first_packed = Q6_W_vshuff_VVR(
                row_first[1], row_first[0], -2);
            const HVX_VectorPair second_packed = Q6_W_vshuff_VVR(
                row_second[1], row_second[0], -2);
            HVX_Vector *destination_first =
                (HVX_Vector *)(destination +
                    (size_t)(row_tile * 4U) *
                        QBH_HMX_FP16_TILE_ELEMENTS) + row_pair;
            HVX_Vector *destination_second =
                (HVX_Vector *)(destination +
                    (size_t)(row_tile * 4U + 2U) *
                        QBH_HMX_FP16_TILE_ELEMENTS) + row_pair;
            destination_first[0] = Q6_V_lo_W(first_packed);
            destination_first[QBH_HMX_FP16_TILE_BYTES /
                              QBH_HVX_BYTES] =
                Q6_V_hi_W(first_packed);
            destination_second[0] = Q6_V_lo_W(second_packed);
            destination_second[QBH_HMX_FP16_TILE_BYTES /
                               QBH_HVX_BYTES] =
                Q6_V_hi_W(second_packed);
        } else {
            for (uint32_t local_row = 0U;
                 local_row < 2U; ++local_row) {
                const uint32_t output_row = row + local_row;
                const uint32_t n_tile =
                    output_row / QBH_HMX_FP16_COLS;
                const uint32_t local_output =
                    output_row % QBH_HMX_FP16_COLS;
                const HVX_Vector offsets = Q6_Vw_vadd_VwVw(
                    offset_base, Q6_V_vsplat_R(local_output * 4U));
                Q6_vscatter_RMVwV(
                    (uint32_t)(uintptr_t)(destination +
                        (size_t)(n_tile * 4U) *
                            QBH_HMX_FP16_TILE_ELEMENTS),
                    2U * QBH_HMX_FP16_TILE_BYTES - 1U,
                    offsets, row_first[local_row]);
                Q6_vscatter_RMVwV(
                    (uint32_t)(uintptr_t)(destination +
                        (size_t)(n_tile * 4U + 2U) *
                            QBH_HMX_FP16_TILE_ELEMENTS),
                    2U * QBH_HMX_FP16_TILE_BYTES - 1U,
                    offsets, row_second[local_row]);
            }
        }
    }
    asm volatile("barrier" ::: "memory");
}

static HVX_Vector qbh_hvx_silu_multiply_vector(
    HVX_Vector gate_value, HVX_Vector up_value) {
    const HVX_Vector sign_mask = Q6_Vh_vsplat_R(0x8000);
    const HVX_Vector magnitude_mask = Q6_Vh_vsplat_R(0x7fff);
    const HVX_Vector one = Q6_Vh_vsplat_R(0x3c00);
    const HVX_Vector zero = Q6_V_vzero();
    HVX_Vector negative_abs = Q6_V_vor_VV(
        Q6_V_vand_VV(gate_value, magnitude_mask), sign_mask);
    HVX_Vector exponential = qhmath_hvx_exp_vhf(negative_abs);
    HVX_Vector denominator = Q6_Vhf_equals_Vqf16(
        Q6_Vqf16_vadd_VhfVhf(one, exponential));
    HVX_Vector reciprocal = qhmath_hvx_inv_vhf(denominator);
    HVX_Vector negative_sigmoid = Q6_Vhf_equals_Vqf16(
        Q6_Vqf16_vmpy_VhfVhf(exponential, reciprocal));
    HVX_VectorPred negative =
        Q6_Q_vcmp_gt_VhfVhf(zero, gate_value);
    HVX_Vector sigmoid = Q6_V_vmux_QVV(
        negative, negative_sigmoid, reciprocal);
    HVX_Vector silu = Q6_Vqf16_vmpy_VhfVhf(
        gate_value, sigmoid);
    silu = Q6_Vqf16_vmpy_Vqf16Vhf(silu, up_value);
    return Q6_Vhf_equals_Vqf16(silu);
}

void qbh_hvx_silu_multiply_f16_vectors(
    const __fp16 *gate, const __fp16 *up, __fp16 *middle,
    uint32_t first_vector, uint32_t vector_count) {
    uint32_t last_vector = first_vector + vector_count;
    for (uint32_t index = first_vector; index < last_vector; ++index) {
        HVX_Vector gate_value = ((const HVX_Vector *)gate)[index];
        HVX_Vector up_value = ((const HVX_Vector *)up)[index];
        ((HVX_Vector *)middle)[index] =
            qbh_hvx_silu_multiply_vector(gate_value, up_value);
    }
}

void qbh_hvx_silu_multiply_f16_channel64(
    const __fp16 *gate, const __fp16 *up, __fp16 *middle,
    uint32_t rows, uint32_t row_stride, uint32_t first_channel) {
    for (uint32_t row = 0U; row < rows; ++row) {
        size_t offset = (size_t)row * row_stride + first_channel;
        const HVX_Vector gate_value =
            *(const HVX_Vector *)(gate + offset);
        const HVX_Vector up_value =
            *(const HVX_Vector *)(up + offset);
        *(HVX_Vector *)(middle + offset) =
            qbh_hvx_silu_multiply_vector(gate_value, up_value);
    }
}

void qbh_hvx_silu_multiply_f16_crouton_tiles(
    const __fp16 *gate_tiles, const __fp16 *up_tiles,
    __fp16 *down_tiles, uint32_t m_tiles, uint32_t n_tiles,
    uint32_t down_k_tiles, uint32_t first_k_tile) {
    const uint32_t vectors_per_tile =
        QBH_HMX_FP16_TILE_BYTES / sizeof(HVX_Vector);

    for (uint32_t row_tile = 0U; row_tile < m_tiles; ++row_tile) {
        for (uint32_t column_tile = 0U;
             column_tile < n_tiles; ++column_tile) {
            const size_t source_tile =
                qbh_hmx_fp16_matrix_tile_offset(
                    row_tile, column_tile, n_tiles);
            const size_t destination_tile =
                qbh_hmx_fp16_matrix_tile_offset(
                    row_tile, first_k_tile + column_tile,
                    down_k_tiles);
            const HVX_Vector *gate_vectors =
                (const HVX_Vector *)(gate_tiles + source_tile);
            const HVX_Vector *up_vectors =
                (const HVX_Vector *)(up_tiles + source_tile);
            HVX_Vector *down_vectors =
                (HVX_Vector *)(down_tiles + destination_tile);

            for (uint32_t vector = 0U;
                 vector < vectors_per_tile; ++vector) {
                down_vectors[vector] =
                    qbh_hvx_silu_multiply_vector(
                        gate_vectors[vector], up_vectors[vector]);
            }
        }
    }
    asm volatile("barrier" ::: "memory");
}

void qbh_hvx_silu_multiply_f16_audit(
    const __fp16 *gate, const __fp16 *up, const __fp16 *middle,
    uint32_t elements, struct qbh_hvx_check_metrics *check) {
    if (check == NULL) {
        return;
    }
    for (uint32_t index = 0; index < elements; ++index) {
        float gate_value = (float)gate[index];
        __fp16 reference = (__fp16)(
            gate_value / (1.0f + expf(-gate_value)) *
            (float)up[index]);
        qbh_hvx_check_add(
            check, (float)middle[index], (float)reference);
    }
}

void qbh_hvx_silu_multiply_f16(const __fp16 *gate, const __fp16 *up,
                                __fp16 *middle, uint32_t elements,
                                struct qbh_hvx_check_metrics *check) {
    qbh_hvx_silu_multiply_f16_vectors(
        gate, up, middle, 0U, elements / QBH_HVX_F16_LANES);
    qbh_hvx_silu_multiply_f16_audit(
        gate, up, middle, elements, check);
}

void qbh_hvx_stable_causal_softmax_f16(__fp16 *scores,
                                        __fp16 *probability,
                                        uint32_t groups, uint32_t rows,
                                        uint32_t width,
                                        float score_scale,
                                        struct qbh_hvx_check_metrics *check) {
    const HVX_Vector lane_index =
        *(const HVX_Vector *)qbh_hvx_lane_index;
    const HVX_Vector negative_max = Q6_Vh_vsplat_R(0xfbff);
    const HVX_Vector zero = Q6_V_vzero();
    const HVX_Vector one_half = Q6_Vh_vsplat_R(0x3c00);
    __fp16 scale_half = (__fp16)score_scale;
    const HVX_Vector scale =
        Q6_Vh_vsplat_R(*(const uint16_t *)&scale_half);
    if (width != QBH_HVX_F16_LANES) {
        return;
    }
    for (uint32_t group = 0; group < groups; ++group) {
        for (uint32_t row = 0; row < rows; ++row) {
            __fp16 original[QBH_HVX_F16_LANES]
                __attribute__((aligned(QBH_HVX_BYTES)));
            size_t offset = ((size_t)group * rows + row) * width;
            HVX_Vector score = *(const HVX_Vector *)(scores + offset);
            if (check != NULL) {
                *(HVX_Vector *)original = score;
            }
            HVX_Vector row_limit = Q6_Vh_vsplat_R((int)row);
            HVX_VectorPred masked =
                Q6_Q_vcmp_gt_VuhVuh(lane_index, row_limit);
            score = Q6_Vhf_equals_Vqf16(
                Q6_Vqf16_vmpy_VhfVhf(score, scale));
            score = Q6_V_vmux_QVV(masked, negative_max, score);
            *(HVX_Vector *)(scores + offset) = score;

            float maximum_value = qbh_hvx_reduce_max_f16(score);
            __fp16 maximum_half = (__fp16)maximum_value;
            HVX_Vector maximum =
                Q6_Vh_vsplat_R(*(const uint16_t *)&maximum_half);
            HVX_Vector shifted = Q6_Vhf_equals_Vqf16(
                Q6_Vqf16_vsub_VhfVhf(score, maximum));
            HVX_Vector exponential = qhmath_hvx_exp_vhf(shifted);
            exponential = Q6_V_vmux_QVV(masked, zero, exponential);
            float sum = qbh_hvx_reduce_sum_f16(exponential);
            *(HVX_Vector *)(probability + offset) =
                qbh_hvx_multiply_scale_f16_f32(
                    exponential, one_half, 1.0f / sum);
            if (check != NULL) {
                float reference_score[QBH_HVX_F16_LANES];
                float reference_maximum = -INFINITY;
                float reference_sum = 0.0f;
                for (uint32_t column = 0; column < width; ++column) {
                    float value = column <= row
                        ? (float)original[column] * score_scale
                        : -INFINITY;
                    reference_score[column] = value;
                    if (value > reference_maximum) {
                        reference_maximum = value;
                    }
                }
                for (uint32_t column = 0; column <= row; ++column) {
                    __fp16 stored_score = (__fp16)reference_score[column];
                    reference_score[column] = expf(
                        (float)stored_score - reference_maximum);
                    reference_sum += reference_score[column];
                }
                for (uint32_t column = 0; column < width; ++column) {
                    __fp16 reference = column <= row
                        ? (__fp16)(reference_score[column] / reference_sum)
                        : (__fp16)0.0f;
                    float actual = (float)probability[offset + column];
                    qbh_hvx_check_add(check, actual, (float)reference);
                    if (column > row && actual != 0.0f) {
                        ++check->mask_violation_count;
                    }
                }
            }
        }
    }
}
