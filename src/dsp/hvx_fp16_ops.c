#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "hvx_fp16_ops.h"
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

void qbh_hvx_rms_norm_f16(const __fp16 *input, const __fp16 *gamma,
                           __fp16 *output, uint32_t rows,
                           uint32_t width) {
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
            output_vectors[index] = qbh_hvx_multiply_scale_f16_f32(
                input_vectors[index], gamma_vectors[index], inverse);
        }
    }
}

void qbh_hvx_qk_norm_rope_f16(__fp16 *tensor, uint32_t rows,
                               uint32_t heads, uint32_t row_stride,
                               uint32_t head_dim, const __fp16 *gamma,
                               const __fp16 *cosine,
                               const __fp16 *sine) {
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
}

void qbh_hvx_silu_multiply_f16(const __fp16 *gate, const __fp16 *up,
                                __fp16 *middle, uint32_t elements) {
    const HVX_Vector sign_mask = Q6_Vh_vsplat_R(0x8000);
    const HVX_Vector magnitude_mask = Q6_Vh_vsplat_R(0x7fff);
    const HVX_Vector one = Q6_Vh_vsplat_R(0x3c00);
    const HVX_Vector zero = Q6_V_vzero();
    uint32_t vector_count = elements / QBH_HVX_F16_LANES;
    for (uint32_t index = 0; index < vector_count; ++index) {
        HVX_Vector gate_value = ((const HVX_Vector *)gate)[index];
        HVX_Vector up_value = ((const HVX_Vector *)up)[index];
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
        ((HVX_Vector *)middle)[index] = Q6_Vhf_equals_Vqf16(silu);
    }
}

void qbh_hvx_stable_causal_softmax_f16(__fp16 *scores,
                                        __fp16 *probability,
                                        uint32_t groups, uint32_t rows,
                                        uint32_t width,
                                        float score_scale) {
    const HVX_Vector lane_index =
        *(const HVX_Vector *)qbh_hvx_lane_index;
    const HVX_Vector negative_max = Q6_Vh_vsplat_R(0xfbff);
    const HVX_Vector zero = Q6_V_vzero();
    __fp16 scale_half = (__fp16)score_scale;
    const HVX_Vector scale =
        Q6_Vh_vsplat_R(*(const uint16_t *)&scale_half);
    if (width != QBH_HVX_F16_LANES) {
        return;
    }
    for (uint32_t group = 0; group < groups; ++group) {
        for (uint32_t row = 0; row < rows; ++row) {
            size_t offset = ((size_t)group * rows + row) * width;
            HVX_Vector score = *(const HVX_Vector *)(scores + offset);
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
            __fp16 inverse_sum = (__fp16)(1.0f / sum);
            HVX_Vector inverse =
                Q6_Vh_vsplat_R(*(const uint16_t *)&inverse_sum);
            *(HVX_Vector *)(probability + offset) =
                Q6_Vhf_equals_Vqf16(
                    Q6_Vqf16_vmpy_VhfVhf(exponential, inverse));
        }
    }
}
