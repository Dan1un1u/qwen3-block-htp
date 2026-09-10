# Standalone HTP runtime — Llama 3.2 development

Qwen3 research is frozen at `48eb1ea7f9db0eb197a7c7908ab954d5a6635fc5`
(EXP-0265). This branch prepares Llama 3.2 support using the existing FastRPC,
HVX, HMX and VTCM runtime. **Llama inference is not implemented yet**; the exact
checkpoint, dimensions, tokenizer and fresh quantization artifacts remain to
be selected and validated.

## Development branches

| Branch | Shared recipes | W4A8 default | Optional |
|---|---|---|---|
| `codex/llama32-no-rotation` | W16A16, C64 W4A16 OPT2 | No rotation | — |
| `codex/llama32-rotation` | Same | Dense R3 OPT2 | Dense R3 + R4 OPT6 |

Both descend from frozen EXP-0265 and share the runtime. Their only intended
file difference at setup is `config/branch.json`. R3/R4 remain research
implementations with explicitly retained quality/numerical limitations.

## Start here

Run the Llama authority bootstrap before project work:

```sh
/home/daniuniu/work/llama32-htp-project-memory/scripts/bootstrap.sh "$PWD"
```

Read the four authority files in its printed order. The old Qwen3 authority
remains frozen historical evidence and must not authorize new Llama work.

Inspect configurations and source/evidence identity without executing hardware:

```sh
python3 tools/recipe.py list
python3 tools/recipe.py show --recipe w4a16
python3 tools/recipe.py show --recipe w4a8
python3 tools/recipe.py verify --artifacts
```

`show` emits a **plan**, including the frozen Qwen3 schedule provenance. It does
not turn Qwen3 weights, prefix, calibration or hardcoded dimensions into Llama
inputs. On the rotation branch, `--rotation r3-r4` selects the optional reference.
No new launcher is enabled before a model-specific port is validated.

## Layout

| Path | Responsibility |
|---|---|
| `src/host`, `src/dsp`, `include` | Existing runtime and ABI; paths and native code preserved during organization |
| `models/qwen3-frozen` | Frozen model dimensions and artifact ownership |
| `models/llama32` | New model/checkpoint contract and port prerequisites |
| `recipes` | Complete frozen schedule references, numerical format and explicit optimization flags |
| `config/branch.json` | Branch default and allowed W4A8 variants |
| `baselines/qwen3-frozen` | Pinned commits, builds, model manifests, report hashes and historical speed |
| `tools/recipe.py` | Read-only configuration resolver and identity verification |
| `experiments/qwen3-frozen` | Inventory of historical experiment entrypoints |
| `docs/LLAMA32_DEVELOPMENT.md` | Worktrees, storage, architecture boundary and next port steps |

The existing CMake entrypoints, `scripts/` and old tools remain at their original
paths so historical imports and evidence references continue to resolve. They
are cataloged as Qwen3 research, not Llama launchers. Historical tutorial/build
notes are retained verbatim in [the archived README](docs/archive/QWEN3_RESEARCH_README.md).

See [frozen baselines](baselines/qwen3-frozen/README.md) for measured speed and
scope. No QNN execution path is introduced.
