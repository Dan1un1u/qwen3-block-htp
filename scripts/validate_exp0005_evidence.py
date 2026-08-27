#!/usr/bin/env python3
import json
import pathlib
import sys


def load_jsonl(path: pathlib.Path):
    records = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise SystemExit(f"{path}:{line_number}: invalid JSON: {error}")
    return records


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} RESULT_DIR")
    result_dir = pathlib.Path(sys.argv[1])
    expected_keys = {
        (storage, projection, pattern)
        for storage in ("expanded_s8_control", "packed_w4u8")
        for projection in ("gate_up", "down")
        for pattern in ("identity", "signed", "structured", "boundary")
    }
    by_repeat = {}
    for repeat, filename in ((1, "correctness_repeat1.jsonl"),
                             (10, "timing_repeat10.jsonl")):
        records = load_jsonl(result_dir / filename)
        if len(records) != len(expected_keys):
            raise SystemExit(
                f"{filename}: expected {len(expected_keys)} records, "
                f"got {len(records)}")
        keyed = {}
        for record in records:
            key = (record["weight_storage"], record["projection"],
                   record["pattern"])
            if key in keyed:
                raise SystemExit(f"{filename}: duplicate record {key}")
            if record["repeat_count"] != repeat:
                raise SystemExit(f"{filename}: wrong repeat for {key}")
            if record["rpc_result"] != 0 or record["dsp_status"] != 0:
                raise SystemExit(f"{filename}: runtime failure for {key}")
            if record["mismatches"] != 0:
                raise SystemExit(f"{filename}: mismatch for {key}")
            keyed[key] = record
        if set(keyed) != expected_keys:
            raise SystemExit(f"{filename}: missing or unexpected matrix keys")
        by_repeat[repeat] = keyed

    for repeat, keyed in by_repeat.items():
        for projection in ("gate_up", "down"):
            for pattern in ("identity", "signed", "structured", "boundary"):
                control = keyed[("expanded_s8_control", projection, pattern)]
                packed = keyed[("packed_w4u8", projection, pattern)]
                comparable = (
                    "reference_checksum", "reference_min", "reference_max",
                    "expanded_carrier_checksum", "projection_m",
                    "projection_k", "projection_n", "hmx_execution_count",
                    "hmx_stream_count", "output_tile_count")
                for field in comparable:
                    if control[field] != packed[field]:
                        raise SystemExit(
                            f"repeat {repeat} {projection}/{pattern}: "
                            f"A/B differs at {field}")
                if not (packed["stored_weight_bytes_per_repeat"] <
                        control["stored_weight_bytes_per_repeat"]):
                    raise SystemExit(
                        f"repeat {repeat} {projection}/{pattern}: "
                        "packed W4 did not reduce stored weight bytes")
                if packed["expanded_weight_bytes_per_repeat"] != \
                        control["expanded_weight_bytes_per_repeat"]:
                    raise SystemExit(
                        f"repeat {repeat} {projection}/{pattern}: "
                        "expanded carrier size differs")

    print(json.dumps({
        "experiment": "EXP-0005",
        "records": 32,
        "exact_reference": True,
        "ab_equivalent": True,
        "packed_w4_reduces_native_weight_bytes": True,
    }, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
