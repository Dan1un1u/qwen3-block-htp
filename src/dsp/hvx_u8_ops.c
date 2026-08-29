#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "hvx_u8_ops.h"
#include "qhmath_hvx_vector.h"

#define QBH_HVX_BYTES UINT32_C(128)
#define QBH_HVX_HALF_LANES UINT32_C(64)
#define QBH_HVX_WORD_LANES UINT32_C(32)

struct qbh_qf32_quad {
    HVX_Vector lane[4];
};

static HVX_Vector qbh_splat_sf(float value) {
    return Q6_Vsf_vadd_VsfVsf(
        Q6_V_vsplat_R(*(const int32_t *)&value), Q6_V_vzero());
}

static HVX_VectorPair qbh_centered_half_pair(
    HVX_Vector input, int32_t zero_point) {
    const HVX_VectorPair unpacked = Q6_Wuh_vunpack_Vub(input);
    const __fp16 zero_point_half = (__fp16)zero_point;
    const HVX_Vector offset =
        Q6_Vh_vsplat_R(*(const uint16_t *)&zero_point_half);
    HVX_DV centered;

    centered.V.lo = Q6_Vhf_vsub_VhfVhf(
        Q6_Vhf_vcvt_Vuh(Q6_V_lo_W(unpacked)), offset);
    centered.V.hi = Q6_Vhf_vsub_VhfVhf(
        Q6_Vhf_vcvt_Vuh(Q6_V_hi_W(unpacked)), offset);
    return centered.VV;
}

static void qbh_unary_gamma_sf32(
    HVX_Vector centered, HVX_Vector gamma,
    float coefficient, float offset,
    HVX_Vector *result_lo, HVX_Vector *result_hi) {
    const HVX_VectorPair centered_sf = Q6_Wsf_vcvt_Vhf(centered);
    const HVX_VectorPair gamma_sf = Q6_Wsf_vcvt_Vhf(gamma);
    const HVX_Vector coefficient_sf = qbh_splat_sf(coefficient);
    const HVX_Vector offset_sf = qbh_splat_sf(offset);
    HVX_Vector product;

    product = Q6_Vqf32_vmpy_VsfVsf(
        Q6_V_lo_W(centered_sf), Q6_V_lo_W(gamma_sf));
    product = Q6_Vqf32_vmpy_VsfVsf(
        Q6_Vsf_equals_Vqf32(product), coefficient_sf);
    *result_lo = Q6_Vsf_vadd_VsfVsf(
        Q6_Vsf_equals_Vqf32(product), offset_sf);

    product = Q6_Vqf32_vmpy_VsfVsf(
        Q6_V_hi_W(centered_sf), Q6_V_hi_W(gamma_sf));
    product = Q6_Vqf32_vmpy_VsfVsf(
        Q6_Vsf_equals_Vqf32(product), coefficient_sf);
    *result_hi = Q6_Vsf_vadd_VsfVsf(
        Q6_Vsf_equals_Vqf32(product), offset_sf);
}

static void qbh_affine_half_sf32(
    HVX_Vector value, float coefficient, float offset,
    HVX_Vector *result_lo, HVX_Vector *result_hi) {
    const HVX_VectorPair value_sf = Q6_Wsf_vcvt_Vhf(value);
    const HVX_Vector coefficient_sf = qbh_splat_sf(coefficient);
    const HVX_Vector offset_sf = qbh_splat_sf(offset);
    HVX_Vector product = Q6_Vqf32_vmpy_VsfVsf(
        Q6_V_lo_W(value_sf), coefficient_sf);

    *result_lo = Q6_Vsf_vadd_VsfVsf(
        Q6_Vsf_equals_Vqf32(product), offset_sf);
    product = Q6_Vqf32_vmpy_VsfVsf(
        Q6_V_hi_W(value_sf), coefficient_sf);
    *result_hi = Q6_Vsf_vadd_VsfVsf(
        Q6_Vsf_equals_Vqf32(product), offset_sf);
}

static HVX_Vector qbh_pack_qf32_to_u8(
    const struct qbh_qf32_quad *values) {
    HVX_Vector words[4];
    HVX_Vector halves[2];
    const HVX_Vector rounding = qbh_splat_sf(0.5f);

    for (uint32_t index = 0U; index < 4U; ++index) {
        words[index] = Q6_Vw_equals_Vsf(
            Q6_Vsf_vadd_VsfVsf(values->lane[index], rounding));
    }
    halves[0] = Q6_Vh_vpack_VwVw_sat(words[1], words[0]);
    halves[1] = Q6_Vh_vpack_VwVw_sat(words[3], words[2]);
    return Q6_Vub_vpack_VhVh_sat(halves[1], halves[0]);
}

static uint64_t qbh_reduce_word_sum(HVX_Vector value) {
    int32_t lanes[QBH_HVX_WORD_LANES]
        __attribute__((aligned(QBH_HVX_BYTES)));
    uint64_t sum = 0U;

    *(HVX_Vector *)lanes = value;
    for (uint32_t lane = 0U; lane < QBH_HVX_WORD_LANES; ++lane) {
        sum += (uint32_t)lanes[lane];
    }
    return sum;
}

static uint64_t qbh_centered_square_sum(
    const uint8_t *input, uint32_t elements, int32_t zero_point) {
    const HVX_Vector offset = Q6_Vh_vsplat_R(zero_point);
    HVX_Vector sums[4] = {
        Q6_V_vzero(), Q6_V_vzero(),
        Q6_V_vzero(), Q6_V_vzero(),
    };

    for (uint32_t element = 0U; element < elements;
         element += QBH_HVX_BYTES) {
        const HVX_Vector input_vector =
            *(const HVX_Vector *)(input + element);
        const HVX_VectorPair unpacked =
            Q6_Wuh_vunpack_Vub(input_vector);
        const HVX_Vector centered0 = Q6_Vh_vsub_VhVh(
            Q6_V_lo_W(unpacked), offset);
        const HVX_Vector centered1 = Q6_Vh_vsub_VhVh(
            Q6_V_hi_W(unpacked), offset);
        const HVX_VectorPair square0 =
            Q6_Ww_vmpy_VhVh(centered0, centered0);
        const HVX_VectorPair square1 =
            Q6_Ww_vmpy_VhVh(centered1, centered1);

        sums[0] = Q6_Vw_vadd_VwVw(sums[0], Q6_V_lo_W(square0));
        sums[1] = Q6_Vw_vadd_VwVw(sums[1], Q6_V_hi_W(square0));
        sums[2] = Q6_Vw_vadd_VwVw(sums[2], Q6_V_lo_W(square1));
        sums[3] = Q6_Vw_vadd_VwVw(sums[3], Q6_V_hi_W(square1));
    }
    return qbh_reduce_word_sum(sums[0]) +
           qbh_reduce_word_sum(sums[1]) +
           qbh_reduce_word_sum(sums[2]) +
           qbh_reduce_word_sum(sums[3]);
}

void qbh_hvx_rms_norm_u8(
    const uint8_t *input,
    const struct qbh_block_qparam *input_qparam,
    const __fp16 *gamma, uint8_t *output,
    const struct qbh_block_qparam *output_qparam,
    uint32_t rows, uint32_t width) {
    for (uint32_t row = 0U; row < rows; ++row) {
        const uint8_t *input_row = input + (size_t)row * width;
        uint8_t *output_row = output + (size_t)row * width;
        const uint64_t centered_square_sum = qbh_centered_square_sum(
            input_row, width, input_qparam->zero_point);
        const float input_scale = input_qparam->scale;
        const float real_square_sum =
            (float)centered_square_sum * input_scale * input_scale;
        const float inverse = 1.0f / sqrtf(
            real_square_sum / (float)width + 1.0e-6f);
        const float coefficient =
            input_scale * inverse / output_qparam->scale;

        for (uint32_t channel = 0U; channel < width;
             channel += QBH_HVX_BYTES) {
            const HVX_Vector input_vector =
                *(const HVX_Vector *)(input_row + channel);
            const HVX_VectorPair centered = qbh_centered_half_pair(
                input_vector, input_qparam->zero_point);
            struct qbh_qf32_quad encoded;

            qbh_unary_gamma_sf32(
                Q6_V_lo_W(centered),
                *(const HVX_Vector *)(gamma + channel),
                coefficient, (float)output_qparam->zero_point,
                &encoded.lane[0], &encoded.lane[1]);
            qbh_unary_gamma_sf32(
                Q6_V_hi_W(centered),
                *(const HVX_Vector *)(gamma + channel +
                                      QBH_HVX_HALF_LANES),
                coefficient, (float)output_qparam->zero_point,
                &encoded.lane[2], &encoded.lane[3]);
            *(HVX_Vector *)(output_row + channel) =
                qbh_pack_qf32_to_u8(&encoded);
        }
    }
    asm volatile("barrier" ::: "memory");
}

#define QBH_U8_RESIDUAL_FRAC_BITS UINT32_C(14)

static int32_t qbh_repeat_signed_half(int32_t value) {
    const uint32_t half = (uint16_t)value;
    return (int32_t)(half | (half << 16U));
}

static HVX_Vector qbh_residual_half_q14(
    HVX_Vector left, HVX_Vector right,
    int32_t left_coefficient, int32_t right_coefficient,
    int32_t output_zero_point) {
    HVX_VectorPair accumulated = Q6_Ww_vmpy_VhRh(
        left, qbh_repeat_signed_half(left_coefficient));
    accumulated = Q6_Ww_vmpyacc_WwVhRh(
        accumulated, right,
        qbh_repeat_signed_half(right_coefficient));
    const HVX_Vector offset = Q6_V_vsplat_R(
        output_zero_point << QBH_U8_RESIDUAL_FRAC_BITS);
    const HVX_Vector lower = Q6_Vw_vadd_VwVw(
        Q6_V_lo_W(accumulated), offset);
    const HVX_Vector upper = Q6_Vw_vadd_VwVw(
        Q6_V_hi_W(accumulated), offset);
    return Q6_Vh_vasr_VwVwR_rnd_sat(
        upper, lower, QBH_U8_RESIDUAL_FRAC_BITS);
}

void qbh_hvx_residual_add_u8(
    const uint8_t *left,
    const struct qbh_block_qparam *left_qparam,
    const uint8_t *right,
    const struct qbh_block_qparam *right_qparam,
    uint8_t *output,
    const struct qbh_block_qparam *output_qparam,
    uint32_t elements) {
    const float fixed_scale =
        (float)(UINT32_C(1) << QBH_U8_RESIDUAL_FRAC_BITS);
    const int32_t left_coefficient = (int32_t)roundf(
        left_qparam->scale / output_qparam->scale * fixed_scale);
    const int32_t right_coefficient = (int32_t)roundf(
        right_qparam->scale / output_qparam->scale * fixed_scale);
    const HVX_Vector left_offset = Q6_Vh_vsplat_R(
        left_qparam->zero_point);
    const HVX_Vector right_offset = Q6_Vh_vsplat_R(
        right_qparam->zero_point);

    for (uint32_t element = 0U; element < elements;
         element += QBH_HVX_BYTES) {
        const HVX_VectorPair left_unpacked = Q6_Wuh_vunpack_Vub(
            *(const HVX_Vector *)(left + element));
        const HVX_VectorPair right_unpacked = Q6_Wuh_vunpack_Vub(
            *(const HVX_Vector *)(right + element));
        const HVX_Vector left_lower = Q6_Vh_vsub_VhVh(
            Q6_V_lo_W(left_unpacked), left_offset);
        const HVX_Vector left_upper = Q6_Vh_vsub_VhVh(
            Q6_V_hi_W(left_unpacked), left_offset);
        const HVX_Vector right_lower = Q6_Vh_vsub_VhVh(
            Q6_V_lo_W(right_unpacked), right_offset);
        const HVX_Vector right_upper = Q6_Vh_vsub_VhVh(
            Q6_V_hi_W(right_unpacked), right_offset);
        const HVX_Vector result_lower = qbh_residual_half_q14(
            left_lower, right_lower, left_coefficient,
            right_coefficient, output_qparam->zero_point);
        const HVX_Vector result_upper = qbh_residual_half_q14(
            left_upper, right_upper, left_coefficient,
            right_coefficient, output_qparam->zero_point);
        *(HVX_Vector *)(output + element) = Q6_Vub_vpack_VhVh_sat(
            result_upper, result_lower);
    }
    asm volatile("barrier" ::: "memory");
}

void qbh_hvx_residual_rms_norm_u8(
    uint8_t *residual,
    const struct qbh_block_qparam *residual_qparam,
    const uint8_t *addition,
    const struct qbh_block_qparam *addition_qparam,
    const struct qbh_block_qparam *sum_qparam,
    const __fp16 *gamma, uint8_t *normalized,
    const struct qbh_block_qparam *normalized_qparam,
    uint32_t rows, uint32_t width) {
    qbh_hvx_residual_add_u8(
        residual, residual_qparam, addition, addition_qparam,
        residual, sum_qparam, rows * width);
    qbh_hvx_rms_norm_u8(
        residual, sum_qparam, gamma, normalized,
        normalized_qparam, rows, width);
}

static HVX_Vector qbh_sf32_scale_and_offset(
    HVX_Vector value, float scale, float offset) {
    const HVX_Vector scaled = Q6_Vqf32_vmpy_VsfVsf(
        value, qbh_splat_sf(scale));
    return Q6_Vsf_vadd_VsfVsf(
        Q6_Vsf_equals_Vqf32(scaled), qbh_splat_sf(offset));
}

static void qbh_qk_norm_rope_one_head_u8(
    uint8_t *values, const struct qbh_block_qparam *input_qparam,
    const struct qbh_block_qparam *output_qparam,
    const __fp16 *gamma, const __fp16 *cosine,
    const __fp16 *sine) {
    const HVX_VectorPair centered = qbh_centered_half_pair(
        *(const HVX_Vector *)values, input_qparam->zero_point);
    const uint64_t centered_square_sum = qbh_centered_square_sum(
        values, QBH_HVX_BYTES, input_qparam->zero_point);
    const float input_scale = input_qparam->scale;
    const float inverse = 1.0f / sqrtf(
        (float)centered_square_sum * input_scale * input_scale /
            (float)QBH_HVX_BYTES + 1.0e-6f);
    const float norm_coefficient = input_scale * inverse;
    HVX_Vector first[2];
    HVX_Vector second[2];
    const HVX_VectorPair cosine_first = Q6_Wsf_vcvt_Vhf(
        *(const HVX_Vector *)cosine);
    const HVX_VectorPair cosine_second = Q6_Wsf_vcvt_Vhf(
        *(const HVX_Vector *)(cosine + QBH_HVX_HALF_LANES));
    const HVX_VectorPair sine_first = Q6_Wsf_vcvt_Vhf(
        *(const HVX_Vector *)sine);
    const HVX_VectorPair sine_second = Q6_Wsf_vcvt_Vhf(
        *(const HVX_Vector *)(sine + QBH_HVX_HALF_LANES));
    struct qbh_qf32_quad encoded;

    qbh_unary_gamma_sf32(
        Q6_V_lo_W(centered), *(const HVX_Vector *)gamma,
        norm_coefficient, 0.0f, &first[0], &first[1]);
    qbh_unary_gamma_sf32(
        Q6_V_hi_W(centered),
        *(const HVX_Vector *)(gamma + QBH_HVX_HALF_LANES),
        norm_coefficient, 0.0f, &second[0], &second[1]);

    for (uint32_t part = 0U; part < 2U; ++part) {
        const HVX_Vector first_sf = first[part];
        const HVX_Vector second_sf = second[part];
        const HVX_Vector first_rotated = Q6_Vsf_equals_Vqf32(
            Q6_Vqf32_vsub_Vqf32Vqf32(
            Q6_Vqf32_vmpy_VsfVsf(
                first_sf,
                part == 0U ? Q6_V_lo_W(cosine_first)
                           : Q6_V_hi_W(cosine_first)),
            Q6_Vqf32_vmpy_VsfVsf(
                second_sf,
                part == 0U ? Q6_V_lo_W(sine_first)
                           : Q6_V_hi_W(sine_first))));
        const HVX_Vector second_rotated = Q6_Vsf_equals_Vqf32(
            Q6_Vqf32_vadd_Vqf32Vqf32(
            Q6_Vqf32_vmpy_VsfVsf(
                second_sf,
                part == 0U ? Q6_V_lo_W(cosine_second)
                           : Q6_V_hi_W(cosine_second)),
            Q6_Vqf32_vmpy_VsfVsf(
                first_sf,
                part == 0U ? Q6_V_lo_W(sine_second)
                           : Q6_V_hi_W(sine_second))));

        encoded.lane[part] = qbh_sf32_scale_and_offset(
            first_rotated, 1.0f / output_qparam->scale,
            (float)output_qparam->zero_point);
        encoded.lane[part + 2U] = qbh_sf32_scale_and_offset(
            second_rotated, 1.0f / output_qparam->scale,
            (float)output_qparam->zero_point);
    }
    *(HVX_Vector *)values = qbh_pack_qf32_to_u8(&encoded);
}

void qbh_hvx_qk_norm_rope_u8(
    uint8_t *tensor, uint32_t rows, uint32_t heads,
    uint32_t row_stride, uint32_t head_dim,
    const struct qbh_block_qparam *input_qparam,
    const struct qbh_block_qparam *output_qparam,
    const __fp16 *gamma, const __fp16 *cosine,
    const __fp16 *sine) {
    if (head_dim != QBH_HVX_BYTES) {
        return;
    }
    for (uint32_t row = 0U; row < rows; ++row) {
        for (uint32_t head = 0U; head < heads; ++head) {
            qbh_qk_norm_rope_one_head_u8(
                tensor + (size_t)row * row_stride +
                    (size_t)head * head_dim,
                input_qparam, output_qparam, gamma,
                cosine + (size_t)row * head_dim,
                sine + (size_t)row * head_dim);
        }
    }
    asm volatile("barrier" ::: "memory");
}

void qbh_hvx_expand_u8_to_f16_in_place(
    uint8_t *buffer, uint32_t elements,
    const struct qbh_block_qparam *qparam) {
    for (uint32_t end = elements; end != 0U;
         end -= QBH_HVX_BYTES) {
        const uint32_t input_offset = end - QBH_HVX_BYTES;
        const HVX_VectorPair centered = qbh_centered_half_pair(
            *(const HVX_Vector *)(buffer + input_offset),
            qparam->zero_point);
        HVX_Vector output0_lo;
        HVX_Vector output0_hi;
        HVX_Vector output1_lo;
        HVX_Vector output1_hi;

        qbh_affine_half_sf32(
            Q6_V_lo_W(centered), qparam->scale, 0.0f,
            &output0_lo, &output0_hi);
        qbh_affine_half_sf32(
            Q6_V_hi_W(centered), qparam->scale, 0.0f,
            &output1_lo, &output1_hi);
        *(HVX_Vector *)((__fp16 *)buffer + input_offset) =
            Q6_Vhf_vcvt_VsfVsf(output0_lo, output0_hi);
        *(HVX_Vector *)((__fp16 *)buffer + input_offset +
                        QBH_HVX_HALF_LANES) =
            Q6_Vhf_vcvt_VsfVsf(output1_lo, output1_hi);
    }
    asm volatile("barrier" ::: "memory");
}

void qbh_hvx_quantize_f16_to_u8(
    const __fp16 *input, uint8_t *output, uint32_t elements,
    const struct qbh_block_qparam *qparam) {
    for (uint32_t element = 0U; element < elements;
         element += QBH_HVX_BYTES) {
        struct qbh_qf32_quad encoded;

        qbh_affine_half_sf32(
            *(const HVX_Vector *)(input + element),
            1.0f / qparam->scale, (float)qparam->zero_point,
            &encoded.lane[0], &encoded.lane[1]);
        qbh_affine_half_sf32(
            *(const HVX_Vector *)(input + element +
                                  QBH_HVX_HALF_LANES),
            1.0f / qparam->scale, (float)qparam->zero_point,
            &encoded.lane[2], &encoded.lane[3]);
        *(HVX_Vector *)(output + element) =
            qbh_pack_qf32_to_u8(&encoded);
    }
    asm volatile("barrier" ::: "memory");
}
