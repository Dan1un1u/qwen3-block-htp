#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "hvx_u8_ops.h"
#include "qhmath_hvx_vector.h"

#define QBH_HVX_BYTES UINT32_C(128)
#define QBH_HVX_HALF_LANES UINT32_C(64)
#define QBH_HVX_WORD_LANES UINT32_C(32)

/* One packed word contains four adjacent K channels.  The 32 output-token
 * words in an integer-HMX tile are spaced by 128 bytes. */
static const int32_t qbh_u8_k_vscatter_offsets[QBH_HVX_WORD_LANES]
    __attribute__((aligned(QBH_HVX_BYTES))) = {
        0, 128, 256, 384, 512, 640, 768, 896,
        1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920,
        2048, 2176, 2304, 2432, 2560, 2688, 2816, 2944,
        3072, 3200, 3328, 3456, 3584, 3712, 3840, 3968,
    };

/* Scatter one row's 128 adjacent channels into four consecutive integer-HMX
 * activation tiles.  Each tile stores 64 rows of 32 channels. */
static const int32_t qbh_u8_activation_vscatter_offsets[QBH_HVX_WORD_LANES]
    __attribute__((aligned(QBH_HVX_BYTES))) = {
        0, 4, 8, 12, 16, 20, 24, 28,
        2048, 2052, 2056, 2060, 2064, 2068, 2072, 2076,
        4096, 4100, 4104, 4108, 4112, 4116, 4120, 4124,
        6144, 6148, 6152, 6156, 6160, 6164, 6168, 6172,
    };

struct qbh_qf32_quad {
    HVX_Vector lane[4];
};

static uint32_t qbh_u8_norm_reduction_mode =
    QBH_BLOCK_U8_NORM_REDUCTION_SCALAR;
static uint32_t qbh_u8_qk_pair_kernel_mode =
    QBH_BLOCK_W4U8_QK_PAIR_SERIAL_INNER;

void qbh_hvx_u8_set_norm_reduction_mode(uint32_t mode) {
    qbh_u8_norm_reduction_mode = mode;
    asm volatile("barrier" ::: "memory");
}

void qbh_hvx_u8_set_qk_pair_kernel_mode(uint32_t mode) {
    qbh_u8_qk_pair_kernel_mode = mode;
    asm volatile("barrier" ::: "memory");
}

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
    HVX_VectorPair ordered_words;
    const HVX_Vector rounding = qbh_splat_sf(0.5f);

    for (uint32_t index = 0U; index < 4U; ++index) {
        words[index] = Q6_Vw_equals_Vsf(
            Q6_Vsf_vadd_VsfVsf(values->lane[index], rounding));
    }
    /* SF32 widening separates adjacent FP16 lanes into even/odd word
     * vectors.  Re-interleave at word granularity before narrowing;
     * otherwise each 64-channel group is emitted as all even channels
     * followed by all odd channels instead of the canonical HMX carrier. */
    ordered_words = Q6_W_vshuff_VVR(words[1], words[0], -4);
    halves[0] = Q6_Vh_vpack_VwVw_sat(
        Q6_V_hi_W(ordered_words), Q6_V_lo_W(ordered_words));
    ordered_words = Q6_W_vshuff_VVR(words[3], words[2], -4);
    halves[1] = Q6_Vh_vpack_VwVw_sat(
        Q6_V_hi_W(ordered_words), Q6_V_lo_W(ordered_words));
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

static uint32_t qbh_reduce_word_sum_hvx(HVX_Vector value) {
    const HVX_Vector zero = Q6_V_vzero();
    for (uint32_t shift = 64U; shift >= 4U; shift >>= 1U) {
        value = Q6_Vw_vadd_VwVw(
            value, Q6_V_vlalign_VVR(value, zero, shift));
    }
    return (uint32_t)Q6_R_vextract_VR(value, 124);
}

static HVX_Vector qbh_center_u8_to_s8(
    HVX_Vector value, int32_t zero_point) {
    const HVX_VectorPair unpacked = Q6_Wuh_vunpack_Vub(value);
    const HVX_Vector offset = Q6_Vh_vsplat_R(zero_point);
    const HVX_Vector low = Q6_Vh_vsub_VhVh(
        Q6_V_lo_W(unpacked), offset);
    const HVX_Vector high = Q6_Vh_vsub_VhVh(
        Q6_V_hi_W(unpacked), offset);
    return Q6_Vb_vpack_VhVh_sat(high, low);
}

static int32_t qbh_reduce_signed_byte_sum(HVX_Vector value) {
    const HVX_VectorPair halves = Q6_Wh_vunpack_Vb(value);
    const HVX_VectorPair words0 =
        Q6_Ww_vunpack_Vh(Q6_V_lo_W(halves));
    const HVX_VectorPair words1 =
        Q6_Ww_vunpack_Vh(Q6_V_hi_W(halves));
    HVX_Vector sum = Q6_Vw_vadd_VwVw(
        Q6_Vw_vadd_VwVw(
            Q6_V_lo_W(words0), Q6_V_hi_W(words0)),
        Q6_Vw_vadd_VwVw(
            Q6_V_lo_W(words1), Q6_V_hi_W(words1)));
    const HVX_Vector zero = Q6_V_vzero();

    for (uint32_t shift = 64U; shift >= 4U; shift >>= 1U) {
        sum = Q6_Vw_vadd_VwVw(
            sum, Q6_V_vlalign_VVR(sum, zero, shift));
    }
    return (int32_t)Q6_R_vextract_VR(sum, 124);
}

static uint16_t qbh_half_bits(float value) {
    const __fp16 converted = (__fp16)value;
    uint16_t bits;

    memcpy(&bits, &converted, sizeof(bits));
    return bits;
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
    if (qbh_u8_norm_reduction_mode !=
        QBH_BLOCK_U8_NORM_REDUCTION_SCALAR) {
        const HVX_Vector combined = Q6_Vw_vadd_VwVw(
            Q6_Vw_vadd_VwVw(sums[0], sums[1]),
            Q6_Vw_vadd_VwVw(sums[2], sums[3]));
        return qbh_reduce_word_sum_hvx(combined);
    }
    return qbh_reduce_word_sum(sums[0]) +
           qbh_reduce_word_sum(sums[1]) +
           qbh_reduce_word_sum(sums[2]) +
           qbh_reduce_word_sum(sums[3]);
}

static void qbh_store_u8_hmx_activation(
    uint8_t *output_tiles, uint32_t row,
    uint32_t first_channel, HVX_Vector value) {
    const HVX_Vector offsets = Q6_Vw_vadd_VwVw(
        *(const HVX_Vector *)qbh_u8_activation_vscatter_offsets,
        Q6_V_vsplat_R(row * QBH_HMX_INPUT_CHANNELS));
    uint8_t *destination = output_tiles +
        (size_t)(first_channel / QBH_HMX_INPUT_CHANNELS) *
            QBH_HMX_ACTIVATION_BYTES;

    Q6_vscatter_RMVwV(
        (uint32_t)(uintptr_t)destination,
        4U * QBH_HMX_ACTIVATION_BYTES - 1U,
        offsets, value);
}

static void qbh_hvx_rms_norm_u8_impl(
    const uint8_t *input,
    const struct qbh_block_qparam *input_qparam,
    const __fp16 *gamma, uint8_t *output,
    const struct qbh_block_qparam *output_qparam,
    uint32_t first_row, uint32_t row_count, uint32_t width,
    uint32_t native_hmx_activation) {
    const uint32_t last_row = first_row + row_count;

    for (uint32_t row = first_row; row < last_row; ++row) {
        const uint8_t *input_row = input + (size_t)row * width;
        uint8_t *output_row = native_hmx_activation == 0U
            ? output + (size_t)row * width : NULL;
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
            const HVX_Vector result = qbh_pack_qf32_to_u8(&encoded);
            if (native_hmx_activation != 0U) {
                qbh_store_u8_hmx_activation(
                    output, row, channel, result);
            } else {
                *(HVX_Vector *)(output_row + channel) = result;
            }
        }
    }
    asm volatile("barrier" ::: "memory");
}

void qbh_hvx_rms_norm_u8(
    const uint8_t *input,
    const struct qbh_block_qparam *input_qparam,
    const __fp16 *gamma, uint8_t *output,
    const struct qbh_block_qparam *output_qparam,
    uint32_t rows, uint32_t width) {
    qbh_hvx_rms_norm_u8_impl(
        input, input_qparam, gamma, output, output_qparam,
        0U, rows, width, 0U);
}

void qbh_hvx_rms_norm_u8_native_activation(
    const uint8_t *input,
    const struct qbh_block_qparam *input_qparam,
    const __fp16 *gamma, uint8_t *output_tiles,
    const struct qbh_block_qparam *output_qparam,
    uint32_t rows, uint32_t width) {
    qbh_hvx_rms_norm_u8_impl(
        input, input_qparam, gamma, output_tiles, output_qparam,
        0U, rows, width, 1U);
}

void qbh_hvx_rms_norm_u8_native_activation_rows(
    const uint8_t *input,
    const struct qbh_block_qparam *input_qparam,
    const __fp16 *gamma, uint8_t *output_tiles,
    const struct qbh_block_qparam *output_qparam,
    uint32_t first_row, uint32_t row_count, uint32_t width) {
    qbh_hvx_rms_norm_u8_impl(
        input, input_qparam, gamma, output_tiles, output_qparam,
        first_row, row_count, width, 1U);
}

#define QBH_U8_RESIDUAL_FRAC_BITS UINT32_C(14)

static int32_t qbh_repeat_signed_half(int32_t value) {
    const uint32_t half = (uint16_t)value;
    return (int32_t)(half | (half << 16U));
}

static HVX_Vector qbh_residual_half_q14(
    HVX_Vector left, HVX_Vector right,
    int32_t left_coefficient, int32_t right_coefficient,
    int32_t output_zero_point, uint32_t fraction_bits) {
    HVX_VectorPair accumulated = Q6_Ww_vmpy_VhRh(
        left, qbh_repeat_signed_half(left_coefficient));
    accumulated = Q6_Ww_vmpyacc_WwVhRh(
        accumulated, right,
        qbh_repeat_signed_half(right_coefficient));
    const HVX_Vector offset = Q6_V_vsplat_R(
        output_zero_point << fraction_bits);
    const HVX_Vector lower = Q6_Vw_vadd_VwVw(
        Q6_V_lo_W(accumulated), offset);
    const HVX_Vector upper = Q6_Vw_vadd_VwVw(
        Q6_V_hi_W(accumulated), offset);
    return Q6_Vh_vasr_VwVwR_rnd_sat(
        upper, lower, fraction_bits);
}

static uint32_t qbh_residual_fraction_bits(
    const struct qbh_block_qparam *left_qparam,
    const struct qbh_block_qparam *right_qparam,
    const struct qbh_block_qparam *output_qparam) {
    const float left_ratio = fabsf(
        left_qparam->scale / output_qparam->scale);
    const float right_ratio = fabsf(
        right_qparam->scale / output_qparam->scale);
    const float maximum_ratio = fmaxf(left_ratio, right_ratio);
    uint32_t fraction_bits = QBH_U8_RESIDUAL_FRAC_BITS;

    while (fraction_bits != 0U &&
           roundf(maximum_ratio *
                  (float)(UINT32_C(1) << fraction_bits)) > 32767.0f) {
        --fraction_bits;
    }
    return fraction_bits;
}

static HVX_Vector qbh_residual_add_u8_vector(
    HVX_Vector left, HVX_Vector right,
    HVX_Vector left_offset, HVX_Vector right_offset,
    int32_t left_coefficient, int32_t right_coefficient,
    int32_t output_zero_point, uint32_t fraction_bits) {
    const HVX_VectorPair left_unpacked = Q6_Wuh_vunpack_Vub(left);
    const HVX_VectorPair right_unpacked = Q6_Wuh_vunpack_Vub(right);
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
        right_coefficient, output_zero_point, fraction_bits);
    const HVX_Vector result_upper = qbh_residual_half_q14(
        left_upper, right_upper, left_coefficient,
        right_coefficient, output_zero_point, fraction_bits);

    return Q6_Vub_vpack_VhVh_sat(result_upper, result_lower);
}

void qbh_hvx_residual_add_u8(
    const uint8_t *left,
    const struct qbh_block_qparam *left_qparam,
    const uint8_t *right,
    const struct qbh_block_qparam *right_qparam,
    uint8_t *output,
    const struct qbh_block_qparam *output_qparam,
    uint32_t elements) {
    const uint32_t fraction_bits = qbh_residual_fraction_bits(
        left_qparam, right_qparam, output_qparam);
    const float fixed_scale =
        (float)(UINT32_C(1) << fraction_bits);
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
        *(HVX_Vector *)(output + element) = qbh_residual_add_u8_vector(
            *(const HVX_Vector *)(left + element),
            *(const HVX_Vector *)(right + element),
            left_offset, right_offset, left_coefficient,
            right_coefficient, output_qparam->zero_point,
            fraction_bits);
    }
    asm volatile("barrier" ::: "memory");
}

void qbh_hvx_residual_add_u8_native_output_rows(
    uint8_t *residual,
    const struct qbh_block_qparam *residual_qparam,
    const uint8_t *addition_tiles,
    const struct qbh_block_qparam *addition_qparam,
    const struct qbh_block_qparam *output_qparam,
    uint32_t first_row, uint32_t row_count, uint32_t width) {
    const uint32_t fraction_bits = qbh_residual_fraction_bits(
        residual_qparam, addition_qparam, output_qparam);
    const float fixed_scale =
        (float)(UINT32_C(1) << fraction_bits);
    const int32_t residual_coefficient = (int32_t)roundf(
        residual_qparam->scale / output_qparam->scale * fixed_scale);
    const int32_t addition_coefficient = (int32_t)roundf(
        addition_qparam->scale / output_qparam->scale * fixed_scale);
    const HVX_Vector residual_offset = Q6_Vh_vsplat_R(
        residual_qparam->zero_point);
    const HVX_Vector addition_offset = Q6_Vh_vsplat_R(
        addition_qparam->zero_point);
    const HVX_Vector tile_offsets =
        *(const HVX_Vector *)qbh_u8_activation_vscatter_offsets;
    const HVX_VectorPred all_lanes = Q6_Q_vcmp_eq_VwVw(
        tile_offsets, tile_offsets);

    const uint32_t last_row = first_row + row_count;

    for (uint32_t row = first_row; row < last_row; ++row) {
        uint8_t *residual_row = residual + (size_t)row * width;
        const HVX_Vector offsets = Q6_Vw_vadd_VwVw(
            tile_offsets,
            Q6_V_vsplat_R(row * QBH_HMX_OUTPUT_CHANNELS));

        for (uint32_t channel = 0U; channel < width;
             channel += QBH_HVX_BYTES) {
            HVX_Vector *destination =
                (HVX_Vector *)(residual_row + channel);
            const HVX_Vector residual_value = *destination;
            const uint8_t *tile_group = addition_tiles +
                (size_t)(channel / QBH_HMX_OUTPUT_CHANNELS) *
                    QBH_HMX_OUTPUT_BYTES;

            Q6_vgather_AQRMVw(
                destination, all_lanes,
                (int32_t)(uintptr_t)tile_group,
                4U * QBH_HMX_OUTPUT_BYTES - 1U,
                offsets);
            *destination = qbh_residual_add_u8_vector(
                residual_value, *(volatile HVX_Vector *)destination,
                residual_offset, addition_offset,
                residual_coefficient, addition_coefficient,
                output_qparam->zero_point, fraction_bits);
        }
    }
    asm volatile("barrier" ::: "memory");
}

void qbh_hvx_residual_add_u8_native_output_rows_shuffle4(
    uint8_t *residual,
    const struct qbh_block_qparam *residual_qparam,
    const uint8_t *addition_tiles,
    const struct qbh_block_qparam *addition_qparam,
    const struct qbh_block_qparam *output_qparam,
    uint32_t first_row, uint32_t row_count, uint32_t width) {
    const uint32_t fraction_bits = qbh_residual_fraction_bits(
        residual_qparam, addition_qparam, output_qparam);
    const float fixed_scale =
        (float)(UINT32_C(1) << fraction_bits);
    const int32_t residual_coefficient = (int32_t)roundf(
        residual_qparam->scale / output_qparam->scale * fixed_scale);
    const int32_t addition_coefficient = (int32_t)roundf(
        addition_qparam->scale / output_qparam->scale * fixed_scale);
    const HVX_Vector residual_offset = Q6_Vh_vsplat_R(
        residual_qparam->zero_point);
    const HVX_Vector addition_offset = Q6_Vh_vsplat_R(
        addition_qparam->zero_point);

    if ((first_row & 3U) != 0U || (row_count & 3U) != 0U ||
        (width & (QBH_HVX_BYTES - 1U)) != 0U) {
        qbh_hvx_residual_add_u8_native_output_rows(
            residual, residual_qparam, addition_tiles,
            addition_qparam, output_qparam,
            first_row, row_count, width);
        return;
    }

    const uint32_t last_row = first_row + row_count;
    for (uint32_t row = first_row; row < last_row; row += 4U) {
        for (uint32_t channel = 0U; channel < width;
             channel += QBH_HVX_BYTES) {
            const uint8_t *tile_group = addition_tiles +
                (size_t)(channel / QBH_HMX_OUTPUT_CHANNELS) *
                    QBH_HMX_OUTPUT_BYTES +
                (size_t)row * QBH_HMX_OUTPUT_CHANNELS;
            const HVX_Vector tile0 =
                *(const HVX_Vector *)(tile_group +
                    0U * QBH_HMX_OUTPUT_BYTES);
            const HVX_Vector tile1 =
                *(const HVX_Vector *)(tile_group +
                    1U * QBH_HMX_OUTPUT_BYTES);
            const HVX_Vector tile2 =
                *(const HVX_Vector *)(tile_group +
                    2U * QBH_HMX_OUTPUT_BYTES);
            const HVX_Vector tile3 =
                *(const HVX_Vector *)(tile_group +
                    3U * QBH_HMX_OUTPUT_BYTES);
            const HVX_VectorPair tile01 = Q6_W_vshuff_VVR(
                tile1, tile0, -32);
            const HVX_VectorPair tile23 = Q6_W_vshuff_VVR(
                tile3, tile2, -32);
            const HVX_VectorPair rows01 = Q6_W_vshuff_VVR(
                Q6_V_lo_W(tile23), Q6_V_lo_W(tile01), -64);
            const HVX_VectorPair rows23 = Q6_W_vshuff_VVR(
                Q6_V_hi_W(tile23), Q6_V_hi_W(tile01), -64);
            HVX_Vector *residual0 = (HVX_Vector *)(residual +
                (size_t)(row + 0U) * width + channel);
            HVX_Vector *residual1 = (HVX_Vector *)(residual +
                (size_t)(row + 1U) * width + channel);
            HVX_Vector *residual2 = (HVX_Vector *)(residual +
                (size_t)(row + 2U) * width + channel);
            HVX_Vector *residual3 = (HVX_Vector *)(residual +
                (size_t)(row + 3U) * width + channel);

            *residual0 = qbh_residual_add_u8_vector(
                *residual0, Q6_V_lo_W(rows01),
                residual_offset, addition_offset,
                residual_coefficient, addition_coefficient,
                output_qparam->zero_point, fraction_bits);
            *residual1 = qbh_residual_add_u8_vector(
                *residual1, Q6_V_hi_W(rows01),
                residual_offset, addition_offset,
                residual_coefficient, addition_coefficient,
                output_qparam->zero_point, fraction_bits);
            *residual2 = qbh_residual_add_u8_vector(
                *residual2, Q6_V_lo_W(rows23),
                residual_offset, addition_offset,
                residual_coefficient, addition_coefficient,
                output_qparam->zero_point, fraction_bits);
            *residual3 = qbh_residual_add_u8_vector(
                *residual3, Q6_V_hi_W(rows23),
                residual_offset, addition_offset,
                residual_coefficient, addition_coefficient,
                output_qparam->zero_point, fraction_bits);
        }
    }
    asm volatile("barrier" ::: "memory");
}

void qbh_hvx_residual_add_u8_native_output(
    uint8_t *residual,
    const struct qbh_block_qparam *residual_qparam,
    const uint8_t *addition_tiles,
    const struct qbh_block_qparam *addition_qparam,
    const struct qbh_block_qparam *output_qparam,
    uint32_t rows, uint32_t width) {
    qbh_hvx_residual_add_u8_native_output_rows(
        residual, residual_qparam, addition_tiles, addition_qparam,
        output_qparam, 0U, rows, width);
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

void qbh_hvx_residual_rms_norm_u8_native_activation(
    uint8_t *residual,
    const struct qbh_block_qparam *residual_qparam,
    const uint8_t *addition,
    const struct qbh_block_qparam *addition_qparam,
    const struct qbh_block_qparam *sum_qparam,
    const __fp16 *gamma, uint8_t *normalized_tiles,
    const struct qbh_block_qparam *normalized_qparam,
    uint32_t rows, uint32_t width) {
    qbh_hvx_residual_add_u8(
        residual, residual_qparam, addition, addition_qparam,
        residual, sum_qparam, rows * width);
    qbh_hvx_rms_norm_u8_native_activation(
        residual, sum_qparam, gamma, normalized_tiles,
        normalized_qparam, rows, width);
}

void qbh_hvx_residual_rms_norm_u8_native_io(
    uint8_t *residual,
    const struct qbh_block_qparam *residual_qparam,
    const uint8_t *addition_tiles,
    const struct qbh_block_qparam *addition_qparam,
    const struct qbh_block_qparam *sum_qparam,
    const __fp16 *gamma, uint8_t *normalized_tiles,
    const struct qbh_block_qparam *normalized_qparam,
    uint32_t rows, uint32_t width) {
    qbh_hvx_residual_rms_norm_u8_native_io_rows(
        residual, residual_qparam, addition_tiles, addition_qparam,
        sum_qparam, gamma, normalized_tiles, normalized_qparam,
        0U, rows, width);
}

void qbh_hvx_residual_rms_norm_u8_native_io_rows(
    uint8_t *residual,
    const struct qbh_block_qparam *residual_qparam,
    const uint8_t *addition_tiles,
    const struct qbh_block_qparam *addition_qparam,
    const struct qbh_block_qparam *sum_qparam,
    const __fp16 *gamma, uint8_t *normalized_tiles,
    const struct qbh_block_qparam *normalized_qparam,
    uint32_t first_row, uint32_t row_count, uint32_t width) {
    qbh_hvx_residual_add_u8_native_output_rows(
        residual, residual_qparam, addition_tiles, addition_qparam,
        sum_qparam, first_row, row_count, width);
    qbh_hvx_rms_norm_u8_native_activation_rows(
        residual, sum_qparam, gamma, normalized_tiles,
        normalized_qparam, first_row, row_count, width);
}

void qbh_hvx_residual_rms_norm_u8_native_io_rows_shuffle4(
    uint8_t *residual,
    const struct qbh_block_qparam *residual_qparam,
    const uint8_t *addition_tiles,
    const struct qbh_block_qparam *addition_qparam,
    const struct qbh_block_qparam *sum_qparam,
    const __fp16 *gamma, uint8_t *normalized_tiles,
    const struct qbh_block_qparam *normalized_qparam,
    uint32_t first_row, uint32_t row_count, uint32_t width) {
    qbh_hvx_residual_add_u8_native_output_rows_shuffle4(
        residual, residual_qparam, addition_tiles, addition_qparam,
        sum_qparam, first_row, row_count, width);
    qbh_hvx_rms_norm_u8_native_activation_rows(
        residual, sum_qparam, gamma, normalized_tiles,
        normalized_qparam, first_row, row_count, width);
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

struct qbh_qk_norm_rope_gamma_sf32 {
    HVX_Vector lane[4];
};

struct qbh_qk_norm_rope_row_sf32 {
    HVX_Vector cosine[4];
    HVX_Vector sine[4];
};

static void qbh_qk_norm_rope_load_gamma_sf32(
    const __fp16 *gamma,
    struct qbh_qk_norm_rope_gamma_sf32 *converted) {
    const HVX_VectorPair first = Q6_Wsf_vcvt_Vhf(
        *(const HVX_Vector *)gamma);
    const HVX_VectorPair second = Q6_Wsf_vcvt_Vhf(
        *(const HVX_Vector *)(gamma + QBH_HVX_HALF_LANES));

    converted->lane[0] = Q6_V_lo_W(first);
    converted->lane[1] = Q6_V_hi_W(first);
    converted->lane[2] = Q6_V_lo_W(second);
    converted->lane[3] = Q6_V_hi_W(second);
}

static void qbh_qk_norm_rope_load_row_sf32(
    const __fp16 *cosine, const __fp16 *sine,
    struct qbh_qk_norm_rope_row_sf32 *converted) {
    const HVX_VectorPair cosine_first = Q6_Wsf_vcvt_Vhf(
        *(const HVX_Vector *)cosine);
    const HVX_VectorPair cosine_second = Q6_Wsf_vcvt_Vhf(
        *(const HVX_Vector *)(cosine + QBH_HVX_HALF_LANES));
    const HVX_VectorPair sine_first = Q6_Wsf_vcvt_Vhf(
        *(const HVX_Vector *)sine);
    const HVX_VectorPair sine_second = Q6_Wsf_vcvt_Vhf(
        *(const HVX_Vector *)(sine + QBH_HVX_HALF_LANES));

    converted->cosine[0] = Q6_V_lo_W(cosine_first);
    converted->cosine[1] = Q6_V_hi_W(cosine_first);
    converted->cosine[2] = Q6_V_lo_W(cosine_second);
    converted->cosine[3] = Q6_V_hi_W(cosine_second);
    converted->sine[0] = Q6_V_lo_W(sine_first);
    converted->sine[1] = Q6_V_hi_W(sine_first);
    converted->sine[2] = Q6_V_lo_W(sine_second);
    converted->sine[3] = Q6_V_hi_W(sine_second);
}

void qbh_hvx_qk_rope_preconvert_sf32(
    const __fp16 *cosine, const __fp16 *sine,
    uint8_t *rope_sf32_cache) {
    struct qbh_qk_norm_rope_row_sf32 *cache =
        (struct qbh_qk_norm_rope_row_sf32 *)rope_sf32_cache;

    _Static_assert(
        sizeof(struct qbh_qk_norm_rope_row_sf32) ==
            8U * QBH_HVX_BYTES,
        "RoPE SF32 cache row must contain eight HVX vectors");
    for (uint32_t row = 0U; row < QBH_BLOCK_M; ++row) {
        qbh_qk_norm_rope_load_row_sf32(
            cosine + (size_t)row * QBH_BLOCK_HEAD_DIM,
            sine + (size_t)row * QBH_BLOCK_HEAD_DIM,
            cache + row);
    }
    asm volatile("barrier" ::: "memory");
}

static void qbh_unary_gamma_preconverted_sf32(
    HVX_Vector centered, HVX_Vector gamma_lo,
    HVX_Vector gamma_hi, float coefficient,
    HVX_Vector *result_lo, HVX_Vector *result_hi) {
    const HVX_VectorPair centered_sf = Q6_Wsf_vcvt_Vhf(centered);
    const HVX_Vector coefficient_sf = qbh_splat_sf(coefficient);
    const HVX_Vector offset_sf = qbh_splat_sf(0.0f);
    HVX_Vector product;

    product = Q6_Vqf32_vmpy_VsfVsf(
        Q6_V_lo_W(centered_sf), gamma_lo);
    product = Q6_Vqf32_vmpy_VsfVsf(
        Q6_Vsf_equals_Vqf32(product), coefficient_sf);
    *result_lo = Q6_Vsf_vadd_VsfVsf(
        Q6_Vsf_equals_Vqf32(product), offset_sf);

    product = Q6_Vqf32_vmpy_VsfVsf(
        Q6_V_hi_W(centered_sf), gamma_hi);
    product = Q6_Vqf32_vmpy_VsfVsf(
        Q6_Vsf_equals_Vqf32(product), coefficient_sf);
    *result_hi = Q6_Vsf_vadd_VsfVsf(
        Q6_Vsf_equals_Vqf32(product), offset_sf);
}

static void qbh_qk_norm_rope_one_head_u8_preconverted_coefficient(
    uint8_t *values,
    int32_t input_zero_point,
    const struct qbh_block_qparam *output_qparam,
    const struct qbh_qk_norm_rope_gamma_sf32 *gamma,
    const struct qbh_qk_norm_rope_row_sf32 *rope,
    float norm_coefficient) {
    const HVX_VectorPair centered = qbh_centered_half_pair(
        *(const HVX_Vector *)values, input_zero_point);
    HVX_Vector first[2];
    HVX_Vector second[2];
    struct qbh_qf32_quad encoded;

    qbh_unary_gamma_preconverted_sf32(
        Q6_V_lo_W(centered), gamma->lane[0], gamma->lane[1],
        norm_coefficient, &first[0], &first[1]);
    qbh_unary_gamma_preconverted_sf32(
        Q6_V_hi_W(centered), gamma->lane[2], gamma->lane[3],
        norm_coefficient, &second[0], &second[1]);

    for (uint32_t part = 0U; part < 2U; ++part) {
        const HVX_Vector first_sf = first[part];
        const HVX_Vector second_sf = second[part];
        const HVX_Vector first_rotated = Q6_Vsf_equals_Vqf32(
            Q6_Vqf32_vsub_Vqf32Vqf32(
            Q6_Vqf32_vmpy_VsfVsf(
                first_sf, rope->cosine[part]),
            Q6_Vqf32_vmpy_VsfVsf(
                second_sf, rope->sine[part])));
        const HVX_Vector second_rotated = Q6_Vsf_equals_Vqf32(
            Q6_Vqf32_vadd_Vqf32Vqf32(
            Q6_Vqf32_vmpy_VsfVsf(
                second_sf, rope->cosine[part + 2U]),
            Q6_Vqf32_vmpy_VsfVsf(
                first_sf, rope->sine[part + 2U])));

        encoded.lane[part] = qbh_sf32_scale_and_offset(
            first_rotated, 1.0f / output_qparam->scale,
            (float)output_qparam->zero_point);
        encoded.lane[part + 2U] = qbh_sf32_scale_and_offset(
            second_rotated, 1.0f / output_qparam->scale,
            (float)output_qparam->zero_point);
    }
    *(HVX_Vector *)values = qbh_pack_qf32_to_u8(&encoded);
}

static inline HVX_Vector qbh_qk_norm_quarter_sf32(
    HVX_Vector centered_half, HVX_Vector gamma,
    HVX_Vector coefficient, HVX_Vector zero,
    uint32_t upper) {
    const HVX_VectorPair centered_sf = Q6_Wsf_vcvt_Vhf(centered_half);
    HVX_Vector product = Q6_Vqf32_vmpy_VsfVsf(
        upper == 0U ? Q6_V_lo_W(centered_sf) : Q6_V_hi_W(centered_sf),
        gamma);

    product = Q6_Vqf32_vmpy_VsfVsf(
        Q6_Vsf_equals_Vqf32(product), coefficient);
    return Q6_Vsf_vadd_VsfVsf(
        Q6_Vsf_equals_Vqf32(product), zero);
}

static inline HVX_Vector qbh_qk_requant_quarter_sf32(
    HVX_Vector value, HVX_Vector inverse_scale,
    HVX_Vector output_zero_point) {
    const HVX_Vector scaled = Q6_Vqf32_vmpy_VsfVsf(
        value, inverse_scale);
    return Q6_Vsf_vadd_VsfVsf(
        Q6_Vsf_equals_Vqf32(scaled), output_zero_point);
}

static void qbh_qk_norm_rope_two_heads_u8_quarter_tiled(
    uint8_t *first_values, uint8_t *second_values,
    int32_t input_zero_point,
    const struct qbh_block_qparam *output_qparam,
    const struct qbh_qk_norm_rope_gamma_sf32 *gamma,
    const struct qbh_qk_norm_rope_row_sf32 *rope,
    float first_norm_coefficient,
    float second_norm_coefficient) {
    const __fp16 zero_point_half = (__fp16)input_zero_point;
    const HVX_Vector input_zero_point_half = Q6_Vh_vsplat_R(
        *(const uint16_t *)&zero_point_half);
    const HVX_Vector first_coefficient =
        qbh_splat_sf(first_norm_coefficient);
    const HVX_Vector second_coefficient =
        qbh_splat_sf(second_norm_coefficient);
    const HVX_Vector inverse_scale =
        qbh_splat_sf(1.0f / output_qparam->scale);
    const HVX_Vector output_zero_point =
        qbh_splat_sf((float)output_qparam->zero_point);
    const HVX_Vector zero = qbh_splat_sf(0.0f);
    const HVX_Vector rounding = qbh_splat_sf(0.5f);
    HVX_Vector first_part_0;
    HVX_Vector first_part_1;
    HVX_Vector second_part_0;
    HVX_Vector second_part_1;

    {
        HVX_Vector first_a;
        HVX_Vector first_b;
        HVX_Vector second_a;
        HVX_Vector second_b;
        {
            const HVX_VectorPair unpacked = Q6_Wuh_vunpack_Vub(
                *(const HVX_Vector *)first_values);
            first_a = qbh_qk_norm_quarter_sf32(
                Q6_Vhf_vsub_VhfVhf(
                    Q6_Vhf_vcvt_Vuh(Q6_V_lo_W(unpacked)),
                    input_zero_point_half),
                gamma->lane[0], first_coefficient, zero, 0U);
            first_b = qbh_qk_norm_quarter_sf32(
                Q6_Vhf_vsub_VhfVhf(
                    Q6_Vhf_vcvt_Vuh(Q6_V_hi_W(unpacked)),
                    input_zero_point_half),
                gamma->lane[2], first_coefficient, zero, 0U);
        }
        {
            const HVX_VectorPair unpacked = Q6_Wuh_vunpack_Vub(
                *(const HVX_Vector *)second_values);
            second_a = qbh_qk_norm_quarter_sf32(
                Q6_Vhf_vsub_VhfVhf(
                    Q6_Vhf_vcvt_Vuh(Q6_V_lo_W(unpacked)),
                    input_zero_point_half),
                gamma->lane[0], second_coefficient, zero, 0U);
            second_b = qbh_qk_norm_quarter_sf32(
                Q6_Vhf_vsub_VhfVhf(
                    Q6_Vhf_vcvt_Vuh(Q6_V_hi_W(unpacked)),
                    input_zero_point_half),
                gamma->lane[2], second_coefficient, zero, 0U);
        }
        const HVX_Vector first_rotated_a = Q6_Vsf_equals_Vqf32(
            Q6_Vqf32_vsub_Vqf32Vqf32(
                Q6_Vqf32_vmpy_VsfVsf(first_a, rope->cosine[0]),
                Q6_Vqf32_vmpy_VsfVsf(first_b, rope->sine[0])));
        const HVX_Vector second_rotated_a = Q6_Vsf_equals_Vqf32(
            Q6_Vqf32_vsub_Vqf32Vqf32(
                Q6_Vqf32_vmpy_VsfVsf(second_a, rope->cosine[0]),
                Q6_Vqf32_vmpy_VsfVsf(second_b, rope->sine[0])));
        const HVX_Vector first_rotated_b = Q6_Vsf_equals_Vqf32(
            Q6_Vqf32_vadd_Vqf32Vqf32(
                Q6_Vqf32_vmpy_VsfVsf(first_b, rope->cosine[2]),
                Q6_Vqf32_vmpy_VsfVsf(first_a, rope->sine[2])));
        const HVX_Vector second_rotated_b = Q6_Vsf_equals_Vqf32(
            Q6_Vqf32_vadd_Vqf32Vqf32(
                Q6_Vqf32_vmpy_VsfVsf(second_b, rope->cosine[2]),
                Q6_Vqf32_vmpy_VsfVsf(second_a, rope->sine[2])));

        const HVX_Vector first_words_a = Q6_Vw_equals_Vsf(
            Q6_Vsf_vadd_VsfVsf(
                qbh_qk_requant_quarter_sf32(
                    first_rotated_a, inverse_scale, output_zero_point),
                rounding));
        const HVX_Vector second_words_a = Q6_Vw_equals_Vsf(
            Q6_Vsf_vadd_VsfVsf(
                qbh_qk_requant_quarter_sf32(
                    second_rotated_a, inverse_scale, output_zero_point),
                rounding));
        const HVX_Vector first_words_b = Q6_Vw_equals_Vsf(
            Q6_Vsf_vadd_VsfVsf(
                qbh_qk_requant_quarter_sf32(
                    first_rotated_b, inverse_scale, output_zero_point),
                rounding));
        const HVX_Vector second_words_b = Q6_Vw_equals_Vsf(
            Q6_Vsf_vadd_VsfVsf(
                qbh_qk_requant_quarter_sf32(
                    second_rotated_b, inverse_scale, output_zero_point),
                rounding));

        first_part_0 = Q6_Vh_vpack_VwVw_sat(first_words_b, first_words_a);
        second_part_0 = Q6_Vh_vpack_VwVw_sat(
            second_words_b, second_words_a);
    }
    {
        HVX_Vector first_a;
        HVX_Vector first_b;
        HVX_Vector second_a;
        HVX_Vector second_b;
        {
            const HVX_VectorPair unpacked = Q6_Wuh_vunpack_Vub(
                *(const HVX_Vector *)first_values);
            first_a = qbh_qk_norm_quarter_sf32(
                Q6_Vhf_vsub_VhfVhf(
                    Q6_Vhf_vcvt_Vuh(Q6_V_lo_W(unpacked)),
                    input_zero_point_half),
                gamma->lane[1], first_coefficient, zero, 1U);
            first_b = qbh_qk_norm_quarter_sf32(
                Q6_Vhf_vsub_VhfVhf(
                    Q6_Vhf_vcvt_Vuh(Q6_V_hi_W(unpacked)),
                    input_zero_point_half),
                gamma->lane[3], first_coefficient, zero, 1U);
        }
        {
            const HVX_VectorPair unpacked = Q6_Wuh_vunpack_Vub(
                *(const HVX_Vector *)second_values);
            second_a = qbh_qk_norm_quarter_sf32(
                Q6_Vhf_vsub_VhfVhf(
                    Q6_Vhf_vcvt_Vuh(Q6_V_lo_W(unpacked)),
                    input_zero_point_half),
                gamma->lane[1], second_coefficient, zero, 1U);
            second_b = qbh_qk_norm_quarter_sf32(
                Q6_Vhf_vsub_VhfVhf(
                    Q6_Vhf_vcvt_Vuh(Q6_V_hi_W(unpacked)),
                    input_zero_point_half),
                gamma->lane[3], second_coefficient, zero, 1U);
        }
        const HVX_Vector first_rotated_a = Q6_Vsf_equals_Vqf32(
            Q6_Vqf32_vsub_Vqf32Vqf32(
                Q6_Vqf32_vmpy_VsfVsf(first_a, rope->cosine[1]),
                Q6_Vqf32_vmpy_VsfVsf(first_b, rope->sine[1])));
        const HVX_Vector second_rotated_a = Q6_Vsf_equals_Vqf32(
            Q6_Vqf32_vsub_Vqf32Vqf32(
                Q6_Vqf32_vmpy_VsfVsf(second_a, rope->cosine[1]),
                Q6_Vqf32_vmpy_VsfVsf(second_b, rope->sine[1])));
        const HVX_Vector first_rotated_b = Q6_Vsf_equals_Vqf32(
            Q6_Vqf32_vadd_Vqf32Vqf32(
                Q6_Vqf32_vmpy_VsfVsf(first_b, rope->cosine[3]),
                Q6_Vqf32_vmpy_VsfVsf(first_a, rope->sine[3])));
        const HVX_Vector second_rotated_b = Q6_Vsf_equals_Vqf32(
            Q6_Vqf32_vadd_Vqf32Vqf32(
                Q6_Vqf32_vmpy_VsfVsf(second_b, rope->cosine[3]),
                Q6_Vqf32_vmpy_VsfVsf(second_a, rope->sine[3])));

        const HVX_Vector first_words_a = Q6_Vw_equals_Vsf(
            Q6_Vsf_vadd_VsfVsf(
                qbh_qk_requant_quarter_sf32(
                    first_rotated_a, inverse_scale, output_zero_point),
                rounding));
        const HVX_Vector second_words_a = Q6_Vw_equals_Vsf(
            Q6_Vsf_vadd_VsfVsf(
                qbh_qk_requant_quarter_sf32(
                    second_rotated_a, inverse_scale, output_zero_point),
                rounding));
        const HVX_Vector first_words_b = Q6_Vw_equals_Vsf(
            Q6_Vsf_vadd_VsfVsf(
                qbh_qk_requant_quarter_sf32(
                    first_rotated_b, inverse_scale, output_zero_point),
                rounding));
        const HVX_Vector second_words_b = Q6_Vw_equals_Vsf(
            Q6_Vsf_vadd_VsfVsf(
                qbh_qk_requant_quarter_sf32(
                    second_rotated_b, inverse_scale, output_zero_point),
                rounding));

        first_part_1 = Q6_Vh_vpack_VwVw_sat(first_words_b, first_words_a);
        second_part_1 = Q6_Vh_vpack_VwVw_sat(
            second_words_b, second_words_a);
    }
    {
        /* The two passes carry alternating 16-bit lanes.  Shuffle them at
         * halfword granularity before the final U8 narrowing so Q/K retain
         * canonical channel order for the following integer-HMX QK. */
        const HVX_VectorPair first_halves = Q6_W_vshuff_VVR(
            first_part_1, first_part_0, -2);
        const HVX_VectorPair second_halves = Q6_W_vshuff_VVR(
            second_part_1, second_part_0, -2);

        *(HVX_Vector *)first_values = Q6_Vub_vpack_VhVh_sat(
            Q6_V_hi_W(first_halves), Q6_V_lo_W(first_halves));
        *(HVX_Vector *)second_values = Q6_Vub_vpack_VhVh_sat(
            Q6_V_hi_W(second_halves), Q6_V_lo_W(second_halves));
    }
}

static void qbh_qk_norm_rope_one_head_u8_preconverted(
    uint8_t *values,
    const struct qbh_block_qparam *input_qparam,
    const struct qbh_block_qparam *output_qparam,
    const struct qbh_qk_norm_rope_gamma_sf32 *gamma,
    const struct qbh_qk_norm_rope_row_sf32 *rope) {
    const uint64_t centered_square_sum = qbh_centered_square_sum(
        values, QBH_HVX_BYTES, input_qparam->zero_point);
    const float input_scale = input_qparam->scale;
    const float inverse = 1.0f / sqrtf(
        (float)centered_square_sum * input_scale * input_scale /
            (float)QBH_HVX_BYTES + 1.0e-6f);
    const float norm_coefficient = input_scale * inverse;
    qbh_qk_norm_rope_one_head_u8_preconverted_coefficient(
        values, input_qparam->zero_point, output_qparam,
        gamma, rope, norm_coefficient);
}

/* Convert four adjacent 32-byte rows from each of four native HMX tiles into
 * four contiguous 128-byte logical rows.  This is the same byte transpose as
 * the proven residual shuffle4 path, and replaces 16 small memcpy calls per
 * four-row pair with aligned HVX loads, vshuff, and stores. */
static __attribute__((noinline)) void
qbh_qk_norm_rope_pair_pack_rows_shuffle4(
    const uint8_t *first_head_tiles,
    const uint8_t *second_head_tiles,
    uint32_t first_row, uint32_t row_count,
    uint8_t *row_cache) {
    const uint8_t *head_tiles[2] = {
        first_head_tiles, second_head_tiles,
    };

    for (uint32_t pair = 0U; pair < 2U; ++pair) {
        uint8_t *pair_cache = row_cache +
            (size_t)pair * QBH_QK_PAIR_RSQRT_ROWS * QBH_HVX_BYTES;
        for (uint32_t local_row = 0U;
             local_row < row_count; local_row += 4U) {
            const size_t row_offset =
                (size_t)(first_row + local_row) *
                    QBH_HMX_INPUT_CHANNELS;
            const uint8_t *tile_group = head_tiles[pair] + row_offset;
            const HVX_Vector tile0 = *(const HVX_Vector *)(
                tile_group + 0U * QBH_HMX_ACTIVATION_BYTES);
            const HVX_Vector tile1 = *(const HVX_Vector *)(
                tile_group + 1U * QBH_HMX_ACTIVATION_BYTES);
            const HVX_Vector tile2 = *(const HVX_Vector *)(
                tile_group + 2U * QBH_HMX_ACTIVATION_BYTES);
            const HVX_Vector tile3 = *(const HVX_Vector *)(
                tile_group + 3U * QBH_HMX_ACTIVATION_BYTES);
            const HVX_VectorPair tile01 = Q6_W_vshuff_VVR(
                tile1, tile0, -32);
            const HVX_VectorPair tile23 = Q6_W_vshuff_VVR(
                tile3, tile2, -32);
            const HVX_VectorPair rows01 = Q6_W_vshuff_VVR(
                Q6_V_lo_W(tile23), Q6_V_lo_W(tile01), -64);
            const HVX_VectorPair rows23 = Q6_W_vshuff_VVR(
                Q6_V_hi_W(tile23), Q6_V_hi_W(tile01), -64);
            HVX_Vector *rows = (HVX_Vector *)(pair_cache +
                (size_t)local_row * QBH_HVX_BYTES);

            rows[0] = Q6_V_lo_W(rows01);
            rows[1] = Q6_V_hi_W(rows01);
            rows[2] = Q6_V_lo_W(rows23);
            rows[3] = Q6_V_hi_W(rows23);
        }
    }
}

/* The four-by-four quarter transpose is symmetric.  Apply it to four
 * completed logical rows to recreate four native HMX activation tiles with
 * one aligned vector store per tile. */
static __attribute__((noinline)) void
qbh_qk_norm_rope_pair_store_rows_shuffle4(
    uint8_t *first_head_tiles, uint8_t *second_head_tiles,
    uint32_t first_row, uint32_t row_count,
    const uint8_t *row_cache) {
    uint8_t *head_tiles[2] = {
        first_head_tiles, second_head_tiles,
    };

    for (uint32_t pair = 0U; pair < 2U; ++pair) {
        const uint8_t *pair_cache = row_cache +
            (size_t)pair * QBH_QK_PAIR_RSQRT_ROWS * QBH_HVX_BYTES;
        for (uint32_t local_row = 0U;
             local_row < row_count; local_row += 4U) {
            const HVX_Vector *rows = (const HVX_Vector *)(pair_cache +
                (size_t)local_row * QBH_HVX_BYTES);
            const HVX_VectorPair row01 = Q6_W_vshuff_VVR(
                rows[1], rows[0], -32);
            const HVX_VectorPair row23 = Q6_W_vshuff_VVR(
                rows[3], rows[2], -32);
            const HVX_VectorPair tiles01 = Q6_W_vshuff_VVR(
                Q6_V_lo_W(row23), Q6_V_lo_W(row01), -64);
            const HVX_VectorPair tiles23 = Q6_W_vshuff_VVR(
                Q6_V_hi_W(row23), Q6_V_hi_W(row01), -64);
            const size_t row_offset =
                (size_t)(first_row + local_row) *
                    QBH_HMX_INPUT_CHANNELS;
            uint8_t *tile_group = head_tiles[pair] + row_offset;

            *(HVX_Vector *)(tile_group +
                0U * QBH_HMX_ACTIVATION_BYTES) = Q6_V_lo_W(tiles01);
            *(HVX_Vector *)(tile_group +
                1U * QBH_HMX_ACTIVATION_BYTES) = Q6_V_hi_W(tiles01);
            *(HVX_Vector *)(tile_group +
                2U * QBH_HMX_ACTIVATION_BYTES) = Q6_V_lo_W(tiles23);
            *(HVX_Vector *)(tile_group +
                3U * QBH_HMX_ACTIVATION_BYTES) = Q6_V_hi_W(tiles23);
        }
    }
}

static void qbh_qk_norm_rope_pair_batched_rsqrt(
    const uint8_t *first_head_tiles,
    const uint8_t *second_head_tiles,
    const struct qbh_block_qparam *input_qparam,
    uint32_t first_row, uint32_t row_count,
    uint8_t *row_cache,
    float inverse_sqrt[2][QBH_QK_PAIR_RSQRT_ROWS]) {
    const float input_scale = input_qparam->scale;
    const uint32_t simd_row_pack =
        qbh_u8_qk_pair_kernel_mode >=
        QBH_BLOCK_W4U8_QK_PAIR_QUARTER_TILED_SIMD_ROW_PACK;

    if (simd_row_pack != 0U) {
        qbh_qk_norm_rope_pair_pack_rows_shuffle4(
            first_head_tiles, second_head_tiles, first_row,
            row_count, row_cache);
    }

    for (uint32_t local_row = 0U;
         local_row < row_count; ++local_row) {
        const uint32_t row = first_row + local_row;
        uint8_t *row_values[2] = {
            row_cache +
                (size_t)local_row * QBH_HVX_BYTES,
            row_cache +
                (size_t)(QBH_QK_PAIR_RSQRT_ROWS + local_row) *
                    QBH_HVX_BYTES,
        };
        if (simd_row_pack == 0U) {
            for (uint32_t tile = 0U;
                 tile < QBH_BLOCK_HEAD_DIM / QBH_HMX_INPUT_CHANNELS;
                 ++tile) {
                const size_t tile_offset =
                    (size_t)tile * QBH_HMX_ACTIVATION_BYTES +
                    (size_t)row * QBH_HMX_INPUT_CHANNELS;
                memcpy(row_values[0] + tile * QBH_HMX_INPUT_CHANNELS,
                       first_head_tiles + tile_offset,
                       QBH_HMX_INPUT_CHANNELS);
                memcpy(row_values[1] + tile * QBH_HMX_INPUT_CHANNELS,
                       second_head_tiles + tile_offset,
                       QBH_HMX_INPUT_CHANNELS);
            }
        }
        for (uint32_t pair = 0U; pair < 2U; ++pair) {
            const uint64_t sum = qbh_centered_square_sum(
                row_values[pair], QBH_HVX_BYTES,
                input_qparam->zero_point);
            inverse_sqrt[pair][local_row] =
                (float)sum * input_scale * input_scale /
                    (float)QBH_HVX_BYTES +
                1.0e-6f;
        }
    }
    *(HVX_Vector *)inverse_sqrt =
        qhmath_hvx_rsqrt_vf(*(const HVX_Vector *)inverse_sqrt);
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

void qbh_hvx_qk_norm_rope_u8_native_head(
    uint8_t *head_tiles,
    const struct qbh_block_qparam *input_qparam,
    const struct qbh_block_qparam *output_qparam,
    const __fp16 *gamma, const __fp16 *cosine,
    const __fp16 *sine) {
    uint8_t row_values[QBH_HVX_BYTES]
        __attribute__((aligned(QBH_HVX_BYTES)));

    for (uint32_t row = 0U; row < QBH_BLOCK_M; ++row) {
        for (uint32_t tile = 0U;
             tile < QBH_BLOCK_HEAD_DIM / 32U; ++tile) {
            memcpy(row_values + tile * 32U,
                   head_tiles + (size_t)tile * 2048U +
                       (size_t)row * 32U,
                   32U);
        }
        qbh_qk_norm_rope_one_head_u8(
            row_values, input_qparam, output_qparam, gamma,
            cosine + (size_t)row * QBH_BLOCK_HEAD_DIM,
            sine + (size_t)row * QBH_BLOCK_HEAD_DIM);
        for (uint32_t tile = 0U;
             tile < QBH_BLOCK_HEAD_DIM / 32U; ++tile) {
            memcpy(head_tiles + (size_t)tile * 2048U +
                       (size_t)row * 32U,
                   row_values + tile * 32U, 32U);
        }
    }
    asm volatile("barrier" ::: "memory");
}

static void qbh_hvx_qk_norm_rope_u8_native_head_pair_impl(
    uint8_t *first_head_tiles, uint8_t *second_head_tiles,
    const struct qbh_block_qparam *input_qparam,
    const struct qbh_block_qparam *output_qparam,
    const __fp16 *gamma, const __fp16 *cosine,
    const __fp16 *sine, uint8_t *rsqrt_scratch,
    const uint8_t *rope_sf32_cache, uint32_t rows) {
    uint8_t row_values[2][QBH_HVX_BYTES]
        __attribute__((aligned(QBH_HVX_BYTES)));
    float inverse_sqrt[2][QBH_QK_PAIR_RSQRT_ROWS]
        __attribute__((aligned(QBH_HVX_BYTES)));
    struct qbh_qk_norm_rope_gamma_sf32 gamma_sf32
        __attribute__((aligned(QBH_HVX_BYTES)));
    const uint32_t batched_rsqrt =
        qbh_u8_norm_reduction_mode >=
        QBH_BLOCK_U8_NORM_REDUCTION_HVX_TREE_QK_BATCHED_RSQRT;
    const uint32_t shared_rope =
        qbh_u8_norm_reduction_mode >=
        QBH_BLOCK_U8_NORM_REDUCTION_HVX_TREE_QK_BATCHED_RSQRT_SHARED_ROPE;

    qbh_qk_norm_rope_load_gamma_sf32(gamma, &gamma_sf32);
    for (uint32_t first_row = 0U; first_row < rows;
         first_row += QBH_QK_PAIR_RSQRT_ROWS) {
        const uint32_t row_count =
            rows - first_row < QBH_QK_PAIR_RSQRT_ROWS
                ? rows - first_row
                : QBH_QK_PAIR_RSQRT_ROWS;
        if (batched_rsqrt != 0U) {
            if (row_count != QBH_QK_PAIR_RSQRT_ROWS) {
                *(HVX_Vector *)inverse_sqrt =
                    Q6_V_vsplat_R(0x3f800000);
            }
            qbh_qk_norm_rope_pair_batched_rsqrt(
                first_head_tiles, second_head_tiles, input_qparam,
                first_row, row_count, rsqrt_scratch, inverse_sqrt);
        }
        for (uint32_t local_row = 0U;
             local_row < row_count; ++local_row) {
            const uint32_t row = first_row + local_row;
            uint8_t *active_values[2] = {
                batched_rsqrt != 0U
                    ? rsqrt_scratch +
                          (size_t)local_row * QBH_HVX_BYTES
                    : row_values[0],
                batched_rsqrt != 0U
                    ? rsqrt_scratch +
                          (size_t)(QBH_QK_PAIR_RSQRT_ROWS + local_row) *
                              QBH_HVX_BYTES
                    : row_values[1],
            };
            struct qbh_qk_norm_rope_row_sf32 rope_sf32
                __attribute__((aligned(QBH_HVX_BYTES)));
            const struct qbh_qk_norm_rope_row_sf32 *rope = &rope_sf32;

            if (batched_rsqrt == 0U) {
                for (uint32_t tile = 0U;
                     tile < QBH_BLOCK_HEAD_DIM / 32U; ++tile) {
                    const size_t tile_offset = (size_t)tile * 2048U +
                        (size_t)row * 32U;
                    memcpy(active_values[0] + tile * 32U,
                           first_head_tiles + tile_offset, 32U);
                    memcpy(active_values[1] + tile * 32U,
                           second_head_tiles + tile_offset, 32U);
                }
            }
            if (shared_rope != 0U) {
                rope =
                    (const struct qbh_qk_norm_rope_row_sf32 *)
                        rope_sf32_cache + row;
            } else {
                qbh_qk_norm_rope_load_row_sf32(
                    cosine + (size_t)row * QBH_BLOCK_HEAD_DIM,
                    sine + (size_t)row * QBH_BLOCK_HEAD_DIM,
                    &rope_sf32);
            }
            if (batched_rsqrt != 0U &&
                qbh_u8_qk_pair_kernel_mode >=
                    QBH_BLOCK_W4U8_QK_PAIR_QUARTER_TILED) {
                qbh_qk_norm_rope_two_heads_u8_quarter_tiled(
                    active_values[0], active_values[1],
                    input_qparam->zero_point, output_qparam,
                    &gamma_sf32, rope,
                    input_qparam->scale * inverse_sqrt[0][local_row],
                    input_qparam->scale * inverse_sqrt[1][local_row]);
            } else for (uint32_t pair = 0U; pair < 2U; ++pair) {
                if (batched_rsqrt != 0U) {
                    qbh_qk_norm_rope_one_head_u8_preconverted_coefficient(
                        active_values[pair], input_qparam->zero_point,
                        output_qparam, &gamma_sf32, rope,
                        input_qparam->scale *
                            inverse_sqrt[pair][local_row]);
                } else {
                    qbh_qk_norm_rope_one_head_u8_preconverted(
                        active_values[pair], input_qparam, output_qparam,
                        &gamma_sf32, rope);
                }
            }
            if (qbh_u8_qk_pair_kernel_mode <
                QBH_BLOCK_W4U8_QK_PAIR_QUARTER_TILED_SIMD_IO) {
                for (uint32_t tile = 0U;
                     tile < QBH_BLOCK_HEAD_DIM / 32U; ++tile) {
                    const size_t tile_offset = (size_t)tile * 2048U +
                        (size_t)row * 32U;
                    memcpy(first_head_tiles + tile_offset,
                           active_values[0] + tile * 32U, 32U);
                    memcpy(second_head_tiles + tile_offset,
                           active_values[1] + tile * 32U, 32U);
                }
            }
        }
        if (qbh_u8_qk_pair_kernel_mode >=
            QBH_BLOCK_W4U8_QK_PAIR_QUARTER_TILED_SIMD_IO) {
            qbh_qk_norm_rope_pair_store_rows_shuffle4(
                first_head_tiles, second_head_tiles,
                first_row, row_count, rsqrt_scratch);
        }
    }
    asm volatile("barrier" ::: "memory");
}

void qbh_hvx_qk_norm_rope_u8_native_head_pair(
    uint8_t *first_head_tiles, uint8_t *second_head_tiles,
    const struct qbh_block_qparam *input_qparam,
    const struct qbh_block_qparam *output_qparam,
    const __fp16 *gamma, const __fp16 *cosine,
    const __fp16 *sine, uint8_t *rsqrt_scratch,
    const uint8_t *rope_sf32_cache) {
    qbh_hvx_qk_norm_rope_u8_native_head_pair_impl(
        first_head_tiles, second_head_tiles,
        input_qparam, output_qparam, gamma, cosine, sine,
        rsqrt_scratch, rope_sf32_cache, QBH_BLOCK_M);
}

void qbh_hvx_qk_norm_rope_u8_native_head_pair_rows(
    uint8_t *first_head_tiles, uint8_t *second_head_tiles,
    const struct qbh_block_qparam *input_qparam,
    const struct qbh_block_qparam *output_qparam,
    const __fp16 *gamma, const __fp16 *cosine,
    const __fp16 *sine, uint8_t *rsqrt_scratch,
    const uint8_t *rope_sf32_cache, uint32_t rows) {
    if (rows == 0U || rows > QBH_BLOCK_M || rows % 4U != 0U) {
        return;
    }
    qbh_hvx_qk_norm_rope_u8_native_head_pair_impl(
        first_head_tiles, second_head_tiles,
        input_qparam, output_qparam, gamma, cosine, sine,
        rsqrt_scratch, rope_sf32_cache, rows);
}

void qbh_hvx_poison_u8_native_head_pair_padding(
    uint8_t *first_head_tiles, uint8_t *second_head_tiles,
    uint32_t first_padding_row) {
    const HVX_Vector poison = Q6_V_vsplat_R(0x5ac39e71);
    uint8_t *heads[2] = {first_head_tiles, second_head_tiles};
    const uint32_t first_byte =
        first_padding_row * QBH_HMX_INPUT_CHANNELS;

    if (first_padding_row >= QBH_BLOCK_M ||
        first_byte % sizeof(HVX_Vector) != 0U) {
        return;
    }
    for (uint32_t head = 0U; head < 2U; ++head) {
        for (uint32_t tile = 0U;
             tile < QBH_BLOCK_HEAD_DIM / QBH_HMX_INPUT_CHANNELS;
             ++tile) {
            uint8_t *tile_base = heads[head] +
                (size_t)tile * QBH_HMX_ACTIVATION_BYTES;
            for (uint32_t offset = first_byte;
                 offset < QBH_HMX_ACTIVATION_BYTES;
                 offset += sizeof(HVX_Vector)) {
                *(HVX_Vector *)(tile_base + offset) = poison;
            }
        }
    }
    asm volatile("barrier" ::: "memory");
}

void qbh_hvx_qk_norm_rope_u8_native_k_head(
    uint8_t *head_tiles,
    const struct qbh_block_qparam *input_qparam,
    const struct qbh_block_qparam *output_qparam,
    const __fp16 *gamma, const __fp16 *cosine,
    const __fp16 *sine,
    const struct qbh_attention_config *config,
    int8_t *weight_tiles, uint32_t *bias_words) {
    uint8_t row_values[QBH_HVX_BYTES]
        __attribute__((aligned(QBH_HVX_BYTES)));
    int32_t signed_sums[QBH_ATTENTION_M];
    const HVX_Vector offsets_base =
        *(const HVX_Vector *)qbh_u8_k_vscatter_offsets;
    const uint32_t divisor = UINT32_C(1) << config->score_shift;
    const int32_t rounding = config->score_shift == 0U
                                 ? 0
                                 : (int32_t)(divisor / 2U);
    const uint16_t conversion = qbh_half_bits(
        512.0f / (float)divisor);

    for (uint32_t row = 0U; row < QBH_BLOCK_M; ++row) {
        const uint32_t n_tile = row / QBH_HMX_OUTPUT_CHANNELS;
        const uint32_t output = row % QBH_HMX_OUTPUT_CHANNELS;
        int8_t *destination = weight_tiles +
            (size_t)n_tile * QBH_ATTENTION_HEAD_DIM_TILES *
                QBH_HMX_WEIGHT_BYTES;
        const HVX_Vector offsets = Q6_Vw_vadd_VwVw(
            offsets_base, Q6_V_vsplat_R(output * 4U));

        for (uint32_t tile = 0U;
             tile < QBH_BLOCK_HEAD_DIM / QBH_HMX_INPUT_CHANNELS;
             ++tile) {
            memcpy(row_values + tile * QBH_HMX_INPUT_CHANNELS,
                   head_tiles +
                       (size_t)tile * QBH_HMX_ACTIVATION_BYTES +
                       (size_t)row * QBH_HMX_INPUT_CHANNELS,
                   QBH_HMX_INPUT_CHANNELS);
        }
        qbh_qk_norm_rope_one_head_u8(
            row_values, input_qparam, output_qparam, gamma,
            cosine + (size_t)row * QBH_BLOCK_HEAD_DIM,
            sine + (size_t)row * QBH_BLOCK_HEAD_DIM);
        for (uint32_t tile = 0U;
             tile < QBH_BLOCK_HEAD_DIM / QBH_HMX_INPUT_CHANNELS;
             ++tile) {
            memcpy(head_tiles +
                       (size_t)tile * QBH_HMX_ACTIVATION_BYTES +
                       (size_t)row * QBH_HMX_INPUT_CHANNELS,
                   row_values + tile * QBH_HMX_INPUT_CHANNELS,
                   QBH_HMX_INPUT_CHANNELS);
        }
        {
            const HVX_Vector centered = qbh_center_u8_to_s8(
                *(const HVX_Vector *)row_values,
                output_qparam->zero_point);
            signed_sums[row] = qbh_reduce_signed_byte_sum(centered);
            Q6_vscatter_RMVwV(
                (uint32_t)(uintptr_t)destination,
                QBH_ATTENTION_HEAD_DIM_TILES *
                        QBH_HMX_WEIGHT_BYTES -
                    1U,
                offsets, centered);
        }
    }

    for (uint32_t n_tile = 0U;
         n_tile < QBH_ATTENTION_SCORE_TILES; ++n_tile) {
        uint32_t *bias = bias_words +
            (size_t)n_tile *
                (QBH_HMX_BIAS_BYTES / sizeof(uint32_t));
        for (uint32_t output = 0U;
             output < QBH_HMX_OUTPUT_CHANNELS; ++output) {
            const uint32_t token =
                n_tile * QBH_HMX_OUTPUT_CHANNELS + output;
            bias[output] = conversion;
            bias[QBH_HMX_OUTPUT_CHANNELS + output] = (uint32_t)(
                -config->q_zero_point * signed_sums[token] +
                (int32_t)QBH_ATTENTION_HMX_CENTER *
                    (int32_t)divisor +
                rounding);
        }
    }
    asm volatile("barrier" ::: "memory");
}

void qbh_hvx_qk_norm_rope_u8_native_k_head_pair(
    uint8_t *first_head_tiles, uint8_t *second_head_tiles,
    const struct qbh_block_qparam *input_qparam,
    const struct qbh_block_qparam *output_qparam,
    const __fp16 *gamma, const __fp16 *cosine,
    const __fp16 *sine,
    const struct qbh_attention_config *first_config,
    const struct qbh_attention_config *second_config,
    int8_t *first_weight_tiles, int8_t *second_weight_tiles,
    uint32_t *first_bias_words, uint32_t *second_bias_words,
    uint8_t *rsqrt_scratch, const uint8_t *rope_sf32_cache) {
    uint8_t row_values[2][QBH_HVX_BYTES]
        __attribute__((aligned(QBH_HVX_BYTES)));
    float inverse_sqrt[2][QBH_QK_PAIR_RSQRT_ROWS]
        __attribute__((aligned(QBH_HVX_BYTES)));
    int32_t signed_sums[2][QBH_ATTENTION_M];
    struct qbh_qk_norm_rope_gamma_sf32 gamma_sf32
        __attribute__((aligned(QBH_HVX_BYTES)));
    const HVX_Vector offsets_base =
        *(const HVX_Vector *)qbh_u8_k_vscatter_offsets;
    const uint32_t batched_rsqrt =
        qbh_u8_norm_reduction_mode >=
        QBH_BLOCK_U8_NORM_REDUCTION_HVX_TREE_QK_BATCHED_RSQRT;
    const uint32_t shared_rope =
        qbh_u8_norm_reduction_mode >=
        QBH_BLOCK_U8_NORM_REDUCTION_HVX_TREE_QK_BATCHED_RSQRT_SHARED_ROPE;

    qbh_qk_norm_rope_load_gamma_sf32(gamma, &gamma_sf32);
    for (uint32_t first_row = 0U; first_row < QBH_BLOCK_M;
         first_row += QBH_QK_PAIR_RSQRT_ROWS) {
        if (batched_rsqrt != 0U) {
            qbh_qk_norm_rope_pair_batched_rsqrt(
                first_head_tiles, second_head_tiles, input_qparam,
                first_row, QBH_QK_PAIR_RSQRT_ROWS,
                rsqrt_scratch, inverse_sqrt);
        }
        for (uint32_t local_row = 0U;
             local_row < QBH_QK_PAIR_RSQRT_ROWS; ++local_row) {
            const uint32_t row = first_row + local_row;
            const uint32_t n_tile = row / QBH_HMX_OUTPUT_CHANNELS;
            const uint32_t output = row % QBH_HMX_OUTPUT_CHANNELS;
            const HVX_Vector offsets = Q6_Vw_vadd_VwVw(
                offsets_base, Q6_V_vsplat_R(output * 4U));
            uint8_t *active_values[2] = {
                batched_rsqrt != 0U
                    ? rsqrt_scratch +
                          (size_t)local_row * QBH_HVX_BYTES
                    : row_values[0],
                batched_rsqrt != 0U
                    ? rsqrt_scratch +
                          (size_t)(QBH_QK_PAIR_RSQRT_ROWS + local_row) *
                              QBH_HVX_BYTES
                    : row_values[1],
            };
            struct qbh_qk_norm_rope_row_sf32 rope_sf32
                __attribute__((aligned(QBH_HVX_BYTES)));
            const struct qbh_qk_norm_rope_row_sf32 *rope = &rope_sf32;

            if (batched_rsqrt == 0U) {
                for (uint32_t tile = 0U;
                     tile < QBH_BLOCK_HEAD_DIM /
                                QBH_HMX_INPUT_CHANNELS;
                     ++tile) {
                    const size_t tile_offset =
                        (size_t)tile * QBH_HMX_ACTIVATION_BYTES +
                        (size_t)row * QBH_HMX_INPUT_CHANNELS;
                    memcpy(active_values[0] +
                               tile * QBH_HMX_INPUT_CHANNELS,
                           first_head_tiles + tile_offset,
                           QBH_HMX_INPUT_CHANNELS);
                    memcpy(active_values[1] +
                               tile * QBH_HMX_INPUT_CHANNELS,
                           second_head_tiles + tile_offset,
                           QBH_HMX_INPUT_CHANNELS);
                }
            }
            if (shared_rope != 0U) {
                rope =
                    (const struct qbh_qk_norm_rope_row_sf32 *)
                        rope_sf32_cache + row;
            } else {
                qbh_qk_norm_rope_load_row_sf32(
                    cosine + (size_t)row * QBH_BLOCK_HEAD_DIM,
                    sine + (size_t)row * QBH_BLOCK_HEAD_DIM,
                    &rope_sf32);
            }
            if (batched_rsqrt != 0U &&
                qbh_u8_qk_pair_kernel_mode >=
                    QBH_BLOCK_W4U8_QK_PAIR_QUARTER_TILED) {
                qbh_qk_norm_rope_two_heads_u8_quarter_tiled(
                    active_values[0], active_values[1],
                    input_qparam->zero_point, output_qparam,
                    &gamma_sf32, rope,
                    input_qparam->scale * inverse_sqrt[0][local_row],
                    input_qparam->scale * inverse_sqrt[1][local_row]);
            } else for (uint32_t pair = 0U; pair < 2U; ++pair) {
                if (batched_rsqrt != 0U) {
                    qbh_qk_norm_rope_one_head_u8_preconverted_coefficient(
                        active_values[pair], input_qparam->zero_point,
                        output_qparam, &gamma_sf32, rope,
                        input_qparam->scale *
                            inverse_sqrt[pair][local_row]);
                } else {
                    qbh_qk_norm_rope_one_head_u8_preconverted(
                        active_values[pair], input_qparam, output_qparam,
                        &gamma_sf32, rope);
                }
            }

            for (uint32_t pair = 0U; pair < 2U; ++pair) {
                uint8_t *head_tiles = pair == 0U
                    ? first_head_tiles : second_head_tiles;
                int8_t *weight_tiles = pair == 0U
                    ? first_weight_tiles : second_weight_tiles;
                int8_t *destination = weight_tiles +
                    (size_t)n_tile * QBH_ATTENTION_HEAD_DIM_TILES *
                        QBH_HMX_WEIGHT_BYTES;

                if (qbh_u8_qk_pair_kernel_mode <
                    QBH_BLOCK_W4U8_QK_PAIR_QUARTER_TILED_SIMD_IO) {
                    for (uint32_t tile = 0U;
                         tile < QBH_BLOCK_HEAD_DIM /
                                    QBH_HMX_INPUT_CHANNELS;
                         ++tile) {
                        const size_t tile_offset =
                            (size_t)tile * QBH_HMX_ACTIVATION_BYTES +
                            (size_t)row * QBH_HMX_INPUT_CHANNELS;
                        memcpy(head_tiles + tile_offset,
                               active_values[pair] +
                                   tile * QBH_HMX_INPUT_CHANNELS,
                               QBH_HMX_INPUT_CHANNELS);
                    }
                }
                {
                    const HVX_Vector centered = qbh_center_u8_to_s8(
                        *(const HVX_Vector *)active_values[pair],
                        output_qparam->zero_point);
                    signed_sums[pair][row] =
                        qbh_reduce_signed_byte_sum(centered);
                    Q6_vscatter_RMVwV(
                        (uint32_t)(uintptr_t)destination,
                        QBH_ATTENTION_HEAD_DIM_TILES *
                                QBH_HMX_WEIGHT_BYTES -
                            1U,
                        offsets, centered);
                }
            }
        }
        if (qbh_u8_qk_pair_kernel_mode >=
            QBH_BLOCK_W4U8_QK_PAIR_QUARTER_TILED_SIMD_IO) {
            qbh_qk_norm_rope_pair_store_rows_shuffle4(
                first_head_tiles, second_head_tiles,
                first_row, QBH_QK_PAIR_RSQRT_ROWS,
                rsqrt_scratch);
        }
    }

    for (uint32_t pair = 0U; pair < 2U; ++pair) {
        const struct qbh_attention_config *config = pair == 0U
            ? first_config : second_config;
        uint32_t *bias_words = pair == 0U
            ? first_bias_words : second_bias_words;
        const uint32_t divisor = UINT32_C(1) << config->score_shift;
        const int32_t rounding = config->score_shift == 0U
                                     ? 0
                                     : (int32_t)(divisor / 2U);
        const uint16_t conversion = qbh_half_bits(
            512.0f / (float)divisor);

        for (uint32_t n_tile = 0U;
             n_tile < QBH_ATTENTION_SCORE_TILES; ++n_tile) {
            uint32_t *bias = bias_words +
                (size_t)n_tile *
                    (QBH_HMX_BIAS_BYTES / sizeof(uint32_t));
            for (uint32_t output = 0U;
                 output < QBH_HMX_OUTPUT_CHANNELS; ++output) {
                const uint32_t token =
                    n_tile * QBH_HMX_OUTPUT_CHANNELS + output;
                bias[output] = conversion;
                bias[QBH_HMX_OUTPUT_CHANNELS + output] =
                    (uint32_t)(
                        -config->q_zero_point *
                            signed_sums[pair][token] +
                        (int32_t)QBH_ATTENTION_HMX_CENTER *
                            (int32_t)divisor +
                        rounding);
            }
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
