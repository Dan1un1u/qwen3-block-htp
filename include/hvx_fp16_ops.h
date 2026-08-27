#ifndef QWEN3_BLOCK_HTP_HVX_FP16_OPS_H
#define QWEN3_BLOCK_HTP_HVX_FP16_OPS_H

#include <stdint.h>

void qbh_hvx_rms_norm_f16(const __fp16 *input, const __fp16 *gamma,
                           __fp16 *output, uint32_t rows,
                           uint32_t width);

void qbh_hvx_qk_norm_rope_f16(__fp16 *tensor, uint32_t rows,
                               uint32_t heads, uint32_t row_stride,
                               uint32_t head_dim, const __fp16 *gamma,
                               const __fp16 *cosine,
                               const __fp16 *sine);

void qbh_hvx_silu_multiply_f16(const __fp16 *gate, const __fp16 *up,
                                __fp16 *middle, uint32_t elements);

void qbh_hvx_stable_causal_softmax_f16(__fp16 *scores,
                                        __fp16 *probability,
                                        uint32_t groups, uint32_t rows,
                                        uint32_t width, float score_scale);

#endif
