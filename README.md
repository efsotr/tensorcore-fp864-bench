# tensorcore-fp864-bench

Microbenchmark for **single-PTX-instruction Tensor Core peak FLOPS** on NVIDIA **GeForce RTX 5090** and **RTX PRO 6000 Blackwell** (Compute Capability **12.0 / SM120**).

The hot loop repeatedly issues one PTX MMA spelling with register-resident operands, multiple semantically independent accumulator chains, and four-way inner unrolling. There is no global/shared-memory traffic inside the repeated MMA body; the target is PTX-level Tensor Core instruction throughput rather than GEMM/library/end-to-end performance.

## SM120 instruction path

For RTX 5090 / RTX PRO 6000, the relevant public low-precision PTX interface is warp-level `mma.sync` and sparse `mma.sp{::ordered_metadata}.sync`. NVIDIA's family-specific PTX feature table exposes `tcgen05.mma` to the SM100/SM110 family, while the FP6/FP4 `.e3m2`, `.e2m3`, `.e2m1`, `.kind`, `.block_scale`, and `.scale_vec` extensions of warp-level `mma{.sp}` are exposed to SM120.

Generated PTX uses `.version 9.1` + `.target sm_120a`, covering the documented `mxf4nvf4 + ue8m0 + scale_vec::4X` form introduced in PTX ISA 9.1.

Primary references:

- https://docs.nvidia.com/cuda/parallel-thread-execution/
- https://docs.nvidia.com/cuda/pdf/ptx_isa_9.1.pdf
- https://developer.nvidia.com/cuda/gpus
- https://github.com/NVIDIA/cutlass/blob/main/python/CuTeDSL/cutlass/cute/nvgpu/warp/mma.py

## Coverage classes

The PTX manifest separates:

1. **DOCUMENTED** — directly defined by normative PTX syntax / valid-combination tables and applicable SM120 target notes. These are benchmarked.
2. **DOC_AMBIGUOUS** — NVIDIA ISA notes/prose claim support, but current normative syntax does not define the spelling. These are JIT-probed only.
3. **UNDOCUMENTED** — plausible-looking combinations not defined by NVIDIA's normative grammar/table. These are JIT-probed only.

The canonical base manifest contains **234 documented cases, 8 documentation-ambiguous probes, and 17 deliberately undocumented probes**. `--all-operands` additionally expands all documented sparse and block-scale selector values.

## Documented FP8 / FP6 / FP4 space

Dense unscaled coverage includes legacy FP8 (`e4m3` / `e5m2`, `m16n8k16` and `m16n8k32`) and SM120 `kind::f8f6f4` (`m16n8k32`) with A/B independently selected from `e4m3`, `e5m2`, `e3m2`, `e2m3`, `e2m1`, with F16/F32 C/D where defined.

Block-scaled coverage follows NVIDIA's valid-combination table:

| kind | A/B element types | scale type | scale vector |
|---|---|---|---|
| `mxf8f6f4` | `e4m3`, `e5m2`, `e3m2`, `e2m3`, `e2m1` | `ue8m0` | `1X` |
| `mxf4` | `e2m1` | `ue8m0` | `2X` |
| `mxf4nvf4` | `e2m1` | `ue8m0` | `2X`, `4X` |
| `mxf4nvf4` | `e2m1` | `ue4m3` | `4X` |

Sparse coverage follows the current normative grammar: legacy FP8 at `m16n8k64/f32`; `kind::f8f6f4` at `m16n8k64`; block-scaled `mxf8f6f4` at `m16n8k64`; and block-scaled `mxf4` / `mxf4nvf4` at `m16n8k128`. Kind-qualified and block-scaled sparse forms are ordered-metadata-only where the grammar requires it.

The sparse ISA notes separately claim SM120 support for FP8 `m16n8k32 + f16`, while the current normative sparse syntax contains no such production. Those 8 spellings remain `DOC_AMBIGUOUS` probes.

## Human-readable and machine-readable output

Each normal run writes:

```text
results/<UTC timestamp>_<GPU>/
  summary.md
  cases.log
  run.json
  results.csv
  peak_summary.json
  ptx/*.ptx
  cubin/*.cubin
  sass/*.sass.txt
  sass/*.sass.json
  sass/*.nvdisasm.log
  sass/nvdisasm-version.txt
```

`summary.md` and the terminal headline table are intended for humans. `run.json`, `results.csv`, and `peak_summary.json` are intended for automation; JSON schemas are under `schemas/`.

## Headline Peak extraction

The benchmark extracts dense Peak TFLOP/s for:

```text
FP8
FP6
FP4
FP8 x FP6
FP8 x FP4
FP6 x FP4
```

Each category is reported as:

- `without_mx`: documented dense unscaled PTX;
- `with_mx`: documented dense MX PTX (`mxf8f6f4` / `mxf4`).

Mixed-width categories consider both A×B orientations and all documented subformats. The winning case preserves exact A/B formats, orientation, accumulator, PTX spelling, measured TFLOP/s, and SASS audit result.

`mxf4nvf4` is intentionally not folded into MX; the best NVFP4-family FP4 result is reported separately as `nvfp4_reference`.

## SASS audit: recorded, never gating

SASS auditing is enabled by default. For every documented PTX case, the runner:

1. links PTX with the CUDA Driver API and obtains the exact CUBIN;
2. saves that CUBIN;
3. disassembles it with NVIDIA `nvdisasm` in text and JSON forms;
4. compares the observed Tensor Core SASS family, shape, accumulator, A/B formats, and static MMA count against the expected mapping;
5. **records the result, then runs the benchmark regardless of whether the SASS matches the expectation**.

Therefore `SASS_OK`, `SASS_MISMATCH`, `SASS_TOOL_ERROR`, and `SASS_DISASM_ERROR` are audit metadata, not execution status. A documented case that JIT-compiles successfully can still be timed and can still win a headline Peak category even if its SASS audit does not match the current expectation. `peak_summary.json` records the winning case's `sass_status` and `sass_matches_expected` alongside the measured Peak.

This distinction is deliberate because NVIDIA does not publish a normative SM120 SASS ISA equivalent to PTX. Current expected families (`QMMA`, `QMMA.SF`, `OMMA.SF`, and sparse `.SP` forms) are an audit hypothesis based on current `nvdisasm` observations, not the definition of PTX legality.

`--no-sass-check` disables the audit entirely; it does not change which documented cases are benchmarked.

## Peak-FLOPS methodology

A dense `m16n8kK` warp MMA is counted as `2 * 16 * 8 * K` FLOPs. Sparse results report both logical dense-equivalent TFLOP/s and nonzero-work TFLOP/s (`logical / 2`). Sparse values are not allowed to win the dense headline categories.

Each accumulator chain begins from a different C value so equivalent chains cannot legally collapse into one common subexpression. The hot loop issues `chains × 4` MMAs per iteration; with the default 8 chains this is 32 MMA instructions per warp per loop iteration.

## Build / run

Requires a CUDA toolkit/driver capable of JIT-compiling PTX 9.1 for SM120.

```bash
cmake -S . -B build
cmake --build build -j
./build/tensorcore-fp864-bench
```

Important options:

```text
--device N
--iters N
--blocks-per-sm N
--chains {1,2,4,8}
--repeats N
--filter TEXT
--all-operands
--include-probes
--probes-only
--output-dir DIR
--nvdisasm PATH
--no-sass-check
--quiet-cases
--list
--verbose-jit
```

See `docs/ptx-coverage.md` for PTX classification and `docs/output-and-sass.md` for output/SASS semantics.
