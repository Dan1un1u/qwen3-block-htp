#!/usr/bin/env python3
"""Exact CPU reference helpers for the project's integer-HMX W4U8 path."""

from __future__ import annotations

import ctypes
import functools
import math
import struct
from pathlib import Path

import numpy as np
import torch


HMX_CHANNELS = 32
U8_RESIDUAL_FRACTION_BITS = 14
U8_ATTENTION_EXP_FRACTION_BITS = 15
QPARAM_RECORD = struct.Struct("<32sfi2f")


def load_qparams_bin(path: Path) -> dict[str, dict[str, object]]:
    payload = path.read_bytes()
    if len(payload) % QPARAM_RECORD.size != 0:
        raise ValueError(f"invalid qparam bytes: {path} ({len(payload)})")
    result: dict[str, dict[str, object]] = {}
    for offset in range(0, len(payload), QPARAM_RECORD.size):
        raw_name, scale, zero_point, minimum, maximum = (
            QPARAM_RECORD.unpack_from(payload, offset)
        )
        name = raw_name.split(b"\0", 1)[0].decode("ascii")
        result[name] = {
            "scale": scale,
            "zero_point": zero_point,
            "minimum": minimum,
            "maximum": maximum,
        }
    return result


class HmxU8Converter:
    """Apply V79 U8 HMX conversion through the SDK CPU emulator."""

    def __init__(self, library: Path) -> None:
        self.library_path = library.resolve()
        if not self.library_path.is_file():
            raise FileNotFoundError(self.library_path)
        self.library = ctypes.CDLL(str(self.library_path))
        self.convert_function = self.library.qbh_hmx_u8_reference_convert
        self.convert_function.argtypes = [
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_uint8),
        ]
        self.convert_function.restype = ctypes.c_int

    def convert(
        self,
        accumulator: np.ndarray,
        lower_words: np.ndarray,
        upper_biases: np.ndarray,
    ) -> np.ndarray:
        accumulator_i64 = np.ascontiguousarray(accumulator, dtype="<i8")
        if accumulator_i64.ndim != 2:
            raise ValueError(
                f"accumulator must be rank 2, got {accumulator_i64.shape}"
            )
        rows, channels = accumulator_i64.shape
        lower_u32 = np.ascontiguousarray(lower_words, dtype="<u4")
        upper_i32 = np.ascontiguousarray(upper_biases, dtype="<i4")
        if lower_u32.shape != (channels,) or upper_i32.shape != (channels,):
            raise ValueError(
                "bias shape mismatch: "
                f"lower={lower_u32.shape} upper={upper_i32.shape} "
                f"channels={channels}"
            )
        output = np.empty((rows, channels), dtype=np.uint8)
        status = self.convert_function(
            accumulator_i64.ctypes.data_as(
                ctypes.POINTER(ctypes.c_int64)
            ),
            rows,
            channels,
            lower_u32.ctypes.data_as(
                ctypes.POINTER(ctypes.c_uint32)
            ),
            upper_i32.ctypes.data_as(
                ctypes.POINTER(ctypes.c_int32)
            ),
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        )
        if status != 0:
            raise RuntimeError(f"libnative HMX conversion failed: {status}")
        return output


def unpack_w4_codes(root: Path, name: str, n: int, k: int) -> np.ndarray:
    packed = np.fromfile(
        root / f"{name}_weight_w4_hmx.bin", dtype=np.uint8
    ).reshape(n // HMX_CHANNELS, k // HMX_CHANNELS, 512)
    flat = np.empty(
        (n // HMX_CHANNELS, k // HMX_CHANNELS, 1024), dtype=np.uint8
    )
    flat[..., 0::2] = packed & 0x0f
    flat[..., 1::2] = packed >> 4
    signed = flat.astype(np.int8)
    signed[signed >= 8] -= 16
    physical = signed.reshape(
        n // HMX_CHANNELS, k // HMX_CHANNELS, 8, HMX_CHANNELS, 4
    )
    return np.ascontiguousarray(
        physical.transpose(0, 1, 2, 4, 3)
        .reshape(n // HMX_CHANNELS, k // HMX_CHANNELS,
                 HMX_CHANNELS, HMX_CHANNELS)
        .transpose(0, 3, 1, 2)
        .reshape(n, k)
    )


@functools.lru_cache(maxsize=None)
def _cached_w4_projection(
    root_name: str, name: str, n: int, k: int,
) -> tuple[np.ndarray, np.ndarray]:
    root = Path(root_name)
    weights = unpack_w4_codes(root, name, n, k)
    scales = np.fromfile(
        root / f"{name}_weight_w4_scale_f32.bin", dtype="<f4"
    )
    if scales.shape != (n,):
        raise ValueError(f"{name} scale shape {scales.shape}, expected {(n,)}")
    return weights, scales


def round_half_away_from_zero(value: np.ndarray) -> np.ndarray:
    value64 = np.asarray(value, dtype=np.float64)
    return np.where(
        value64 >= 0.0,
        np.floor(value64 + 0.5),
        np.ceil(value64 - 0.5),
    ).astype(np.int64)


def pack_u8_hmx_activation(values: np.ndarray) -> np.ndarray:
    """Pack logical [M,N] U8 values into consecutive 64x32 HMX tiles."""
    logical = np.ascontiguousarray(values, dtype=np.uint8)
    if logical.ndim != 2 or logical.shape[0] != 64 or (
        logical.shape[1] % HMX_CHANNELS != 0
    ):
        raise ValueError(f"invalid U8 HMX activation shape {logical.shape}")
    return np.ascontiguousarray(
        logical.reshape(64, -1, HMX_CHANNELS).transpose(1, 0, 2)
    ).reshape(-1)


def unpack_u8_hmx_activation(
    carrier: np.ndarray, channels: int,
) -> np.ndarray:
    """Unpack consecutive 64x32 HMX tiles into logical [64,N] values."""
    physical = np.ascontiguousarray(carrier, dtype=np.uint8)
    expected = 64 * channels
    if channels % HMX_CHANNELS != 0 or physical.size != expected:
        raise ValueError(
            f"invalid U8 HMX carrier bytes {physical.size}, expected {expected}"
        )
    return np.ascontiguousarray(
        physical.reshape(channels // HMX_CHANNELS, 64, HMX_CHANNELS)
        .transpose(1, 0, 2)
        .reshape(64, channels)
    )


def exact_rms_norm_u8(
    input_u8: np.ndarray,
    input_qparam: dict[str, object],
    gamma_f16: np.ndarray,
    output_qparam: dict[str, object],
) -> np.ndarray:
    """Reproduce the DSP U8 RMSNorm SF32 operation and rounding order."""
    values = np.ascontiguousarray(input_u8, dtype=np.uint8)
    gamma = np.ascontiguousarray(gamma_f16, dtype=np.float16)
    if values.ndim != 2 or gamma.shape != (values.shape[1],):
        raise ValueError(
            f"RMSNorm shape mismatch values={values.shape} gamma={gamma.shape}"
        )
    centered = values.astype(np.int32) - int(input_qparam["zero_point"])
    square_sum = np.sum(
        centered.astype(np.int64) * centered.astype(np.int64),
        axis=1,
        dtype=np.int64,
    )
    input_scale = np.float32(input_qparam["scale"])
    real_square_sum = np.float32(square_sum.astype(np.float32) * input_scale)
    real_square_sum = np.float32(real_square_sum * input_scale)
    mean_square = np.float32(
        real_square_sum / np.float32(values.shape[1])
    )
    denominator = np.float32(mean_square + np.float32(1.0e-6))
    inverse = np.float32(
        np.float32(1.0) / np.sqrt(denominator, dtype=np.float32)
    )
    coefficient = np.float32(input_scale * inverse)
    coefficient = np.float32(
        coefficient / np.float32(output_qparam["scale"])
    )
    encoded = np.float32(
        centered.astype(np.float32) * gamma.astype(np.float32)[None, :]
    )
    encoded = np.float32(encoded * coefficient[:, None])
    encoded = np.float32(
        encoded + np.float32(output_qparam["zero_point"])
    )
    encoded = np.float32(encoded + np.float32(0.5))
    return np.clip(np.trunc(encoded), 0, 255).astype(np.uint8)


def exact_qk_norm_rope_u8(
    projected_u8: np.ndarray,
    heads: int,
    input_qparam: dict[str, object],
    output_qparam: dict[str, object],
    gamma_f16: np.ndarray,
    cosine_f16: np.ndarray,
    sine_f16: np.ndarray,
) -> np.ndarray:
    """Reproduce the DSP 128-channel U8 Q/K RMSNorm and RoPE kernel."""
    projected = np.ascontiguousarray(projected_u8, dtype=np.uint8)
    if projected.ndim != 2 or projected.shape[1] != heads * 128:
        raise ValueError(
            "Q/K projection shape "
            f"{projected.shape}, expected [rows,{heads * 128}]"
        )
    rows = projected.shape[0]
    gamma = np.ascontiguousarray(gamma_f16, dtype=np.float16)
    cosine_all = np.ascontiguousarray(
        cosine_f16, dtype=np.float16
    ).reshape(-1, 128)
    sine_all = np.ascontiguousarray(
        sine_f16, dtype=np.float16
    ).reshape(-1, 128)
    if cosine_all.shape[0] < rows or sine_all.shape[0] < rows:
        raise ValueError(
            "RoPE row count is smaller than projection row count: "
            f"cos={cosine_all.shape} sin={sine_all.shape} rows={rows}"
        )
    cosine = cosine_all[:rows]
    sine = sine_all[:rows]
    if gamma.shape != (128,):
        raise ValueError(f"Q/K gamma shape {gamma.shape}, expected {(128,)}")

    source = projected.reshape(rows, heads, 128)
    output = np.empty_like(source)
    input_scale = np.float32(input_qparam["scale"])
    inverse_output_scale = np.float32(
        np.float32(1.0) / np.float32(output_qparam["scale"])
    )
    gamma_f32 = gamma.astype(np.float32)
    cosine_f32 = cosine.astype(np.float32)
    sine_f32 = sine.astype(np.float32)
    for row in range(rows):
        for head in range(heads):
            centered = (
                source[row, head].astype(np.int32) -
                int(input_qparam["zero_point"])
            )
            square_sum = np.sum(
                centered.astype(np.int64) * centered.astype(np.int64),
                dtype=np.int64,
            )
            real_square_sum = np.float32(np.float32(square_sum) * input_scale)
            real_square_sum = np.float32(real_square_sum * input_scale)
            mean_square = np.float32(real_square_sum / np.float32(128.0))
            denominator = np.float32(mean_square + np.float32(1.0e-6))
            inverse = np.float32(
                np.float32(1.0) / np.sqrt(denominator, dtype=np.float32)
            )
            coefficient = np.float32(input_scale * inverse)
            normalized = np.float32(
                centered.astype(np.float32) * gamma_f32
            )
            normalized = np.float32(normalized * coefficient)
            first = normalized[:64]
            second = normalized[64:]
            first_rotated = np.float32(
                np.float32(first * cosine_f32[row, :64]) -
                np.float32(second * sine_f32[row, :64])
            )
            second_rotated = np.float32(
                np.float32(second * cosine_f32[row, 64:]) +
                np.float32(first * sine_f32[row, 64:])
            )
            rotated = np.concatenate((first_rotated, second_rotated))
            encoded = np.float32(rotated * inverse_output_scale)
            encoded = np.float32(
                encoded + np.float32(output_qparam["zero_point"])
            )
            encoded = np.float32(encoded + np.float32(0.5))
            output[row, head] = np.clip(
                np.trunc(encoded), 0, 255
            ).astype(np.uint8)
    return output.reshape(rows, heads * 128)


def _residual_coefficient(
    input_scale: object, output_scale: object, fraction_bits: int,
) -> int:
    ratio = np.float32(
        np.float32(input_scale) / np.float32(output_scale)
    )
    fixed = np.float32(
        ratio * np.float32(1 << fraction_bits)
    )
    if fixed >= 0:
        return int(np.floor(np.float32(fixed + np.float32(0.5))))
    return int(np.ceil(np.float32(fixed - np.float32(0.5))))


def _residual_q14_coefficient(input_scale: object, output_scale: object) -> int:
    return _residual_coefficient(
        input_scale, output_scale, U8_RESIDUAL_FRACTION_BITS
    )


def _residual_fixed_parameters(
    left_scale: object, right_scale: object, output_scale: object,
) -> tuple[int, int, int]:
    left_ratio = np.abs(np.float32(
        np.float32(left_scale) / np.float32(output_scale)
    ))
    right_ratio = np.abs(np.float32(
        np.float32(right_scale) / np.float32(output_scale)
    ))
    maximum_ratio = np.maximum(left_ratio, right_ratio)
    fraction_bits = U8_RESIDUAL_FRACTION_BITS
    while fraction_bits:
        scaled = np.float32(
            maximum_ratio * np.float32(1 << fraction_bits)
        )
        rounded = np.floor(np.float32(scaled + np.float32(0.5)))
        if rounded <= 32767.0:
            break
        fraction_bits -= 1
    return (
        fraction_bits,
        _residual_coefficient(left_scale, output_scale, fraction_bits),
        _residual_coefficient(right_scale, output_scale, fraction_bits),
    )


def exact_residual_add_u8(
    left_u8: np.ndarray,
    left_qparam: dict[str, object],
    right_u8: np.ndarray,
    right_qparam: dict[str, object],
    output_qparam: dict[str, object],
) -> np.ndarray:
    """Reproduce the DSP Q14 HVX residual add including signed rounding."""
    left = np.ascontiguousarray(left_u8, dtype=np.uint8)
    right = np.ascontiguousarray(right_u8, dtype=np.uint8)
    if left.shape != right.shape:
        raise ValueError(
            f"residual shape mismatch left={left.shape} right={right.shape}"
        )
    fraction_bits, left_coefficient, right_coefficient = (
        _residual_fixed_parameters(
            left_qparam["scale"], right_qparam["scale"],
            output_qparam["scale"],
        )
    )
    accumulator = (
        (left.astype(np.int64) - int(left_qparam["zero_point"])) *
        left_coefficient +
        (right.astype(np.int64) - int(right_qparam["zero_point"])) *
        right_coefficient +
        (int(output_qparam["zero_point"]) << fraction_bits)
    )
    rounded = accumulator if fraction_bits == 0 else (
        accumulator + (1 << (fraction_bits - 1))
    ) >> fraction_bits
    return np.clip(rounded, 0, 255).astype(np.uint8)


def projection_bias_words(
    weights: np.ndarray,
    weight_scales: np.ndarray,
    input_qparam: dict[str, object],
    output_qparam: dict[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    scales_f32 = np.asarray(weight_scales, dtype=np.float32)
    ratio = np.asarray(
        np.float32(input_qparam["scale"]) * scales_f32 /
        np.float32(output_qparam["scale"]),
        dtype=np.float32,
    )
    if np.any(~np.isfinite(ratio)) or np.any(ratio <= 0.0):
        raise ValueError("invalid projection ratio")
    conversion_f32 = np.asarray(np.float32(512.0) * ratio, dtype=np.float32)
    lower = conversion_f32.astype("<f2").view("<u2").astype("<u4")
    sums = weights.astype(np.int32).sum(axis=1, dtype=np.int32)
    offset = (
        -float(int(input_qparam["zero_point"])) * sums.astype(np.float64) +
        float(int(output_qparam["zero_point"])) / ratio.astype(np.float64)
    )
    upper64 = round_half_away_from_zero(offset)
    if np.any(upper64 < np.iinfo(np.int32).min) or np.any(
        upper64 > np.iinfo(np.int32).max
    ):
        raise OverflowError("projection HMX bias exceeds int32")
    return lower, upper64.astype("<i4")


def raw_u8s8_accumulator(
    activation_u8: np.ndarray, weights_s8: np.ndarray
) -> np.ndarray:
    activation = np.ascontiguousarray(activation_u8, dtype=np.uint8)
    weights = np.ascontiguousarray(weights_s8, dtype=np.int8)
    if activation.ndim != 2 or weights.ndim != 2 or (
        activation.shape[1] != weights.shape[1]
    ):
        raise ValueError(
            f"invalid matmul shapes {activation.shape} and {weights.shape}"
        )
    signed_activation = np.ascontiguousarray(
        (activation.astype(np.int16) - 128).astype(np.int8)
    )
    left = torch.from_numpy(signed_activation)
    right = torch.from_numpy(np.ascontiguousarray(weights.T))
    signed_product = torch._int_mm(left, right).numpy().astype(np.int64)
    weight_sums = weights.astype(np.int32).sum(axis=1, dtype=np.int32)
    return signed_product + 128 * weight_sums.astype(np.int64)[None, :]


def project_w4u8(
    activation_u8: np.ndarray,
    package: Path,
    name: str,
    n: int,
    k: int,
    input_qparam: dict[str, object],
    output_qparam: dict[str, object],
    converter: HmxU8Converter,
) -> np.ndarray:
    activation = np.ascontiguousarray(activation_u8, dtype=np.uint8)
    if activation.ndim != 2 or activation.shape[1] != k:
        raise ValueError(
            f"{name} activation shape {activation.shape}, expected [M,{k}]"
        )
    weights, scales = _cached_w4_projection(
        str(package.resolve()), name, n, k
    )
    lower, upper = projection_bias_words(
        weights, scales, input_qparam, output_qparam
    )
    accumulator = raw_u8s8_accumulator(activation, weights)
    return converter.convert(accumulator, lower, upper)


def power_of_two_bias_words(
    channels: int, shift: int, output_zero_point: int,
) -> tuple[np.ndarray, np.ndarray]:
    divisor = 1 << shift
    rounding = divisor // 2 if shift else 0
    conversion = np.asarray(
        np.float32(512.0) / np.float32(divisor), dtype=np.float32
    ).astype("<f2").view("<u2").item()
    return (
        np.full(channels, conversion, dtype="<u4"),
        np.full(
            channels,
            output_zero_point * divisor + rounding,
            dtype="<i4",
        ),
    )


def post_centered_requant(
    intermediate: np.ndarray, multiplier: int, output_zero_point: int,
) -> np.ndarray:
    value = (
        (intermediate.astype(np.int32) - 128) * int(multiplier) +
        int(output_zero_point)
    )
    return np.clip(value, 0, 255).astype(np.uint8)


def signed_round_divide(
    value: np.ndarray, numerator: int, denominator: int,
) -> np.ndarray:
    product = value.astype(np.int64) * int(numerator)
    magnitude = np.abs(product)
    rounded = (magnitude + denominator // 2) // denominator
    return np.where(product >= 0, rounded, -rounded)


def exact_attention_prefill(
    q_u8: np.ndarray,
    k_u8: np.ndarray,
    v_u8: np.ndarray,
    configs: list[tuple[int, ...]],
    converter: HmxU8Converter,
    log2_softmax,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the M=64 integer Attention path with exact HMX conversion."""
    rows, heads, head_dim = q_u8.shape
    kv_heads = k_u8.shape[1]
    if rows != 64 or head_dim != 128 or heads != 2 * kv_heads:
        raise ValueError(
            f"unexpected Attention shapes q={q_u8.shape} k={k_u8.shape}"
        )
    score_all = np.empty((heads, rows, rows), dtype=np.uint8)
    probability_all = np.empty_like(score_all)
    output = np.empty((rows, heads, head_dim), dtype=np.uint8)
    for group, fields in enumerate(configs):
        (
            _abi, config_group, fraction_bits, division_mode,
            q_zero_point, k_zero_point, v_zero_point,
            _probability_zero_point, output_zero_point,
            v_numerator, v_denominator, score_shift, score_multiplier,
            av_shift, av_multiplier,
        ) = fields
        if config_group != group:
            raise ValueError(f"Attention config group {config_group} != {group}")
        first_head = 2 * group
        k_signed = np.clip(
            k_u8[:, group, :].astype(np.int16) - int(k_zero_point),
            -128, 127,
        ).astype(np.int8)
        k_sums = k_signed.astype(np.int32).sum(axis=1, dtype=np.int32)
        score_divisor = 1 << int(score_shift)
        score_rounding = score_divisor // 2 if score_shift else 0
        score_lower, _ = power_of_two_bias_words(
            rows, int(score_shift), 128
        )
        score_upper = (
            -int(q_zero_point) * k_sums + 128 * score_divisor +
            score_rounding
        ).astype("<i4")
        for local_head in range(2):
            head = first_head + local_head
            accumulator = raw_u8s8_accumulator(
                q_u8[:, head, :], k_signed
            )
            intermediate = converter.convert(
                accumulator, score_lower, score_upper
            )
            score_all[head] = post_centered_requant(
                intermediate, int(score_multiplier), 128
            )
        division_name = {1: "exact", 2: "sole", 3: "endpoint"}.get(
            int(division_mode)
        )
        if division_name is None:
            raise ValueError(f"invalid Softmax division mode {division_mode}")
        probability, _ = log2_softmax(
            score_all[first_head:first_head + 2],
            int(fraction_bits),
            division_name,
        )
        probability_all[first_head:first_head + 2] = probability
        centered_v = (
            v_u8[:, group, :].astype(np.int32) - int(v_zero_point)
        )
        signed_v = np.clip(
            signed_round_divide(
                centered_v, int(v_numerator), int(v_denominator)
            ),
            -128, 127,
        ).astype(np.int8)
        av_hmx_zero_point = (
            int(output_zero_point) if int(av_multiplier) == 1 else 128
        )
        av_lower, av_upper = power_of_two_bias_words(
            head_dim, int(av_shift), av_hmx_zero_point
        )
        for local_head in range(2):
            head = first_head + local_head
            accumulator = raw_u8s8_accumulator(
                probability_all[head], signed_v.T
            )
            intermediate = converter.convert(
                accumulator, av_lower, av_upper
            )
            output[:, head, :] = (
                intermediate if int(av_multiplier) == 1 else
                post_centered_requant(
                    intermediate, int(av_multiplier), int(output_zero_point)
                )
            )
    return output, score_all, probability_all


def exact_attention_dynamic(
    q_u8: np.ndarray,
    k_cache_u8: np.ndarray,
    v_cache_u8: np.ndarray,
    past_tokens: int,
    configs: list[tuple[int, ...]],
    converter: HmxU8Converter,
    divide_probability,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the cache-native integer Attention path for logical decode rows.

    K/V use the runtime cache ABI: head-major logical rows
    ``[kv_heads, valid_tokens, 128]``.  Only valid logical columns are
    modeled; the device's HMX padding is zero and therefore cannot change
    these outputs.
    """
    q = np.ascontiguousarray(q_u8, dtype=np.uint8)
    k_cache = np.ascontiguousarray(k_cache_u8, dtype=np.uint8)
    v_cache = np.ascontiguousarray(v_cache_u8, dtype=np.uint8)
    if q.ndim != 3 or q.shape[2] != 128:
        raise ValueError(f"invalid dynamic Q shape {q.shape}")
    query_rows, heads, head_dim = q.shape
    if k_cache.ndim != 3 or v_cache.shape != k_cache.shape or (
        k_cache.shape[2] != head_dim
    ):
        raise ValueError(
            f"invalid dynamic cache shapes k={k_cache.shape} v={v_cache.shape}"
        )
    kv_heads, valid_tokens, _ = k_cache.shape
    if heads != 2 * kv_heads or valid_tokens != past_tokens + query_rows:
        raise ValueError(
            "dynamic Attention contract mismatch: "
            f"q={q.shape} cache={k_cache.shape} past={past_tokens}"
        )

    score_all = np.zeros(
        (heads, query_rows, valid_tokens), dtype=np.uint8
    )
    probability_all = np.zeros_like(score_all)
    output = np.empty((query_rows, heads, head_dim), dtype=np.uint8)
    for group, fields in enumerate(configs):
        (
            _abi, config_group, fraction_bits, division_mode,
            q_zero_point, k_zero_point, v_zero_point,
            _probability_zero_point, output_zero_point,
            v_numerator, v_denominator, score_shift, score_multiplier,
            av_shift, av_multiplier,
        ) = fields
        if config_group != group:
            raise ValueError(f"Attention config group {config_group} != {group}")
        first_head = 2 * group
        k_signed = np.clip(
            k_cache[group].astype(np.int16) - int(k_zero_point),
            -128, 127,
        ).astype(np.int8)
        k_sums = k_signed.astype(np.int32).sum(axis=1, dtype=np.int32)
        score_divisor = 1 << int(score_shift)
        score_rounding = score_divisor // 2 if score_shift else 0
        score_lower, _ = power_of_two_bias_words(
            valid_tokens, int(score_shift), 128
        )
        score_upper = (
            -int(q_zero_point) * k_sums + 128 * score_divisor +
            score_rounding
        ).astype("<i4")
        for local_head in range(2):
            head = first_head + local_head
            accumulator = raw_u8s8_accumulator(q[:, head, :], k_signed)
            intermediate = converter.convert(
                accumulator, score_lower, score_upper
            )
            score_all[head] = post_centered_requant(
                intermediate, int(score_multiplier), 128
            )

        division_name = {1: "exact", 2: "sole", 3: "endpoint"}.get(
            int(division_mode)
        )
        if division_name is None:
            raise ValueError(f"invalid Softmax division mode {division_mode}")
        rounding = 1 << (int(fraction_bits) - 1)
        for local_head in range(2):
            head = first_head + local_head
            for row in range(query_rows):
                valid_count = past_tokens + row + 1
                scores = score_all[head, row, :valid_count]
                maximum = int(scores.max())
                exponents = np.minimum(
                    15,
                    (maximum - scores.astype(np.int32) + rounding) >>
                    int(fraction_bits),
                ).astype(np.uint8)
                total = sum(
                    1 << (
                        U8_ATTENTION_EXP_FRACTION_BITS - int(exponent)
                    ) for exponent in exponents
                )
                for column, exponent in enumerate(exponents):
                    probability_all[head, row, column] = divide_probability(
                        int(exponent), total, division_name, valid_count
                    )

        centered_v = (
            v_cache[group].astype(np.int32) - int(v_zero_point)
        )
        signed_v = np.clip(
            signed_round_divide(
                centered_v, int(v_numerator), int(v_denominator)
            ),
            -128, 127,
        ).astype(np.int8)
        av_hmx_zero_point = (
            int(output_zero_point) if int(av_multiplier) == 1 else 128
        )
        av_lower, av_upper = power_of_two_bias_words(
            head_dim, int(av_shift), av_hmx_zero_point
        )
        for local_head in range(2):
            head = first_head + local_head
            accumulator = raw_u8s8_accumulator(
                probability_all[head], signed_v.T
            )
            intermediate = converter.convert(
                accumulator, av_lower, av_upper
            )
            output[:, head, :] = (
                intermediate if int(av_multiplier) == 1 else
                post_centered_requant(
                    intermediate, int(av_multiplier), int(output_zero_point)
                )
            )
    return output, score_all, probability_all
