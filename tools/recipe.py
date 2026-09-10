#!/usr/bin/env python3
"""Inspect migration recipes and verify frozen provenance. Never launches a model."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
RECIPES = ("w16a16", "w4a16", "w4a8")
ROTATIONS = ("none", "r3", "r3-r4")


def read(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def baseline_id(recipe, rotation):
    require(recipe in RECIPES, "Unknown recipe")
    require(rotation in ROTATIONS, "Unknown rotation")
    if recipe != "w4a8":
        require(rotation == "none", "Rotation is only configured for W4A8")
        return recipe
    return {"none": "w4a8-off", "r3": "w4a8-r3", "r3-r4": "w4a8-r3-r4"}[rotation]


def resolve(recipe=None, rotation=None, model=None):
    branch = read(ROOT / "config/branch.json")
    recipe = recipe or branch["default_recipe"]
    model = model or branch["model"]
    require(model in ("llama32", "qwen3-frozen"), "Unknown model")
    rotation = rotation if rotation is not None else (branch["w4a8_rotation"] if recipe == "w4a8" else "none")
    name = baseline_id(recipe, rotation)
    require(name in branch["allowed_baselines"], "Baseline is not enabled on this branch")
    spec = read(ROOT / "recipes" / (name + ".json"))
    manifest = read(ROOT / "baselines/qwen3-frozen/manifest.json")
    baseline = manifest["baselines"][name]
    require(spec["baseline_id"] == name and spec["rotation"] == rotation, "Recipe identity mismatch")
    require(baseline["rotation"] == rotation, "Package/rotation identity mismatch")
    env = spec["schedule"]["env"]
    require(env["QBH_DENSE_R3"] == ("0" if rotation == "none" else "1"), "R3 mode mismatch")
    require(env["QBH_R3_OPT"] == ("0" if rotation == "none" else "2"), "R3 optimization mismatch")
    require(env["QBH_DENSE_R4"] == ("1" if rotation == "r3-r4" else "0"), "R4 mode mismatch")
    require(env["QBH_R4_OPT"] == ("6" if rotation == "r3-r4" else "0"), "R4 optimization mismatch")
    require(env["QBH_W4F16_DECODE_OPT"] == ("2" if recipe == "w4a16" else "0"), "W4A16 decode optimization mismatch")
    require(env["QBH_LPBQ32"] == "0", "Grouped weights are not a migration baseline")
    model_spec = read(ROOT / "models" / model / "model.json")
    return {
        "kind": "configuration_plan_only",
        "branch": branch["branch"], "model": model_spec,
        "recipe": recipe, "rotation": rotation, "baseline_id": name,
        "execution_enabled": False,
        "reason": ("Use the registered Llama experiment entrypoints; this command only inspects configuration. "
                   "W4A16/A8 port validation is pending") if model == "llama32" else "Qwen3 research is frozen; retained recipes are provenance references",
        "intended_weight_format": spec["weight_format"],
        "intended_activation_format": spec["activation_format"],
        "qwen3_reference_schedule": spec["schedule"],
        "qwen3_reference_package": baseline["package"],
        "qwen3_historical_measurement": baseline["measurement"],
        "quality_status": baseline["quality_status"],
    }


def verify(artifacts=False):
    manifest = read(ROOT / "baselines/qwen3-frozen/manifest.json")
    expected = manifest["native_build_file_sha256"]
    paths = ["src", "include", "CMakeLists.txt", "CMakePresets.json"]
    # Verify the sealed historical commit, not the actively adapted Llama files.
    # The manifest hashes are immutable; no replacement hash is accepted here.
    frozen = manifest["frozen_source_commit"]
    tracked = subprocess.check_output(["git", "-C", str(ROOT), "ls-tree", "-r", "--name-only", frozen, "--", *paths], text=True).splitlines()
    require(set(tracked) == set(expected), "Frozen native/build inventory mismatch")
    for name, digest in expected.items():
        payload = subprocess.check_output(["git", "-C", str(ROOT), "show", frozen + ":" + name])
        require(hashlib.sha256(payload).hexdigest() == digest, "Frozen native file hash mismatch: " + name)
    current_differences = [name for name, digest in expected.items() if not (ROOT/name).is_file() or sha(ROOT/name) != digest]
    branch = read(ROOT / "config/branch.json")
    require(branch["qwen3_frozen_commit"] == manifest["frozen_source_commit"], "Frozen parent mismatch")
    count = 0
    for name in branch["allowed_baselines"]:
        spec = read(ROOT / "recipes" / (name + ".json"))
        for model in ("llama32", "qwen3-frozen"):
            resolve(spec["recipe"], spec["rotation"], model)
            count += 1
    verified = []
    if artifacts:
        for name, baseline in manifest["baselines"].items():
            for key in ("profile_report", "artifact_provenance"):
                ref = baseline[key]
                require(sha(Path(ref["path"])) == ref["sha256"], "Evidence mismatch: " + name + "/" + key)
            package = baseline["package"]
            require(sha(Path(package["path"]) / "manifest.json") == package["manifest_sha256"], "Package manifest mismatch: " + name)
            ref = baseline["launch_evidence"]
            path = ROOT / ref["path"] if ref["kind"] == "frozen_runner_f16f16_speed_case" else Path(ref["path"])
            require(sha(path) == ref["sha256"], "Launch evidence mismatch: " + name)
            verified.append(name)
    return {"pass": True, "frozen_native_build_files_verified": len(expected), "current_model_adapter_differences": current_differences, "model_recipe_plans_checked": count, "compact_evidence_and_package_manifests_verified": verified, "weight_payload_rehash": False, "new_device_or_quality_measurement": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("list", "show", "verify"))
    parser.add_argument("--recipe", choices=RECIPES)
    parser.add_argument("--rotation", choices=ROTATIONS)
    parser.add_argument("--model", choices=("llama32", "qwen3-frozen"))
    parser.add_argument("--artifacts", action="store_true", help="Verify compact retained evidence and package manifests; does not hash large weights")
    args = parser.parse_args()
    try:
        if args.action == "list":
            result = read(ROOT / "config/branch.json")
        elif args.action == "verify":
            result = verify(args.artifacts)
        else:
            result = resolve(args.recipe, args.rotation, args.model)
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(2, "ERROR: " + str(error) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
