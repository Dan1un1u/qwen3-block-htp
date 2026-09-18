"""SDK-v79 reciprocal sqrt via independent ISA simulator; never device-output injection."""
from pathlib import Path
import os,subprocess,json
import numpy as np
SDK=Path('/home/daniuniu/toolchains/hexagon-sdk-6.6.0.0-exp0001/Hexagon_SDK/6.6.0.0')
SIM=SDK/'tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-sim'
EXE=Path('/home/daniuniu/work/qwen3-block-htp/build/reference/qhl_rsqrt_sim')
_cache={};_calls=0
def rsqrt(values):
 global _calls
 a=np.asarray(values,dtype="<f4");assert np.isfinite(a).all() and (a>0).all()
 words=a.reshape(-1).view("<u4");missing=sorted(set(words.tolist())-_cache.keys())
 if missing:
  root=Path(os.environ['QBH_SDK_REFERENCE_DIR']);root.mkdir(parents=True,exist_ok=True)
  stem=root/f"call{_calls:05d}";_calls+=1
  ip=stem.with_suffix(".input.bin");op=stem.with_suffix(".output.bin")
  np.asarray(missing,dtype="<u4").tofile(ip)
  z=subprocess.run([str(SIM),'-mv79',str(EXE),'--',str(ip),str(op)],check=True,capture_output=True,text=True)
  stem.with_suffix(".log").write_text(z.stdout+z.stderr)
  out=np.fromfile(op,dtype="<f4");assert len(out)==len(missing) and np.isfinite(out).all()
  _cache.update(zip(missing,out))
 return np.asarray([_cache[int(v)] for v in words],dtype="f4").reshape(a.shape)

def exact_qk_norm_rope_u8(
    projected_u8: np.ndarray,
    heads: int,
    input_qparam: dict[str, object],
    output_qparam: dict[str, object],
    gamma_f16: np.ndarray,
    cosine_f16: np.ndarray,
    sine_f16: np.ndarray,
) -> np.ndarray:
    """Preserve Q/K arithmetic and obtain SDK rsqrt from an independent ISA simulator.

    Mathematical reciprocal sqrt is not bit-identical to the SDK's three
    Newton steps. The simulator sees only reference-computed denominators.
    """
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
    centered_all=source.astype("i8")-int(input_qparam["zero_point"])
    square_all=np.sum(centered_all*centered_all,axis=-1,dtype="i8")
    scale=np.float32(input_qparam["scale"])
    denominator_all=np.float32(np.float32(np.float32(square_all)*scale)*scale)/np.float32(128)+np.float32(1e-6)
    inverse_all=rsqrt(denominator_all)
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
            inverse = inverse_all[row,head]
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
