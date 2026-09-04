# tensorcore-fp864-bench

Microbenchmark for **single-PTX-instruction Tensor Core peak FLOPS** on NVIDIA **GeForce RTX 5090** and **RTX PRO 6000 Blackwell** (Compute Capability **12.0 / SM120**).

The hot loop repeats exactly one PTX MMA spelling with register-resident operands and multiple independent accumulator chains. No global/shared-memory access occurs inside the repeated MMA body; the benchmark therefore targets PTX-level Tensor Core instruction throughput rather than GEMM/library/end-to-end performance.

## Architecture rule: SM120 uses warp-level `mma`, not `tcgen05`

For RTX 5090 / RTX PRO 6000, the relevant low-precision PTX interface is `mma.sync` / `mma.sp::ordered_metadata.sync`. NVIDIA's family-specific PTX feature table exposes `tcgen05.mma` to the SM100/SM110 family, while the FP6/FP4 `.e3m2`, `.e2m3`, `.e2m1`, `.kind`, `.block_scale`, and `.scale_vec` extensions of warp-level `mma{.sp}` are exposed to SM120.

Primary references:

- Current PTX ISA: https://docs.nvidia.com/cuda/parallel-thread-execution/
- PTX ISA 9.1 PDF: https://docs.nvidia.com/cuda/pdf/ptx_isa_9.1.pdf
- NVIDIA CUDA GPU Compute Capability table: https://developer.nvidia.com/cuda/gpus
- CUTLASS SM120 warp MMA implementation, secondary cross-check only: https://github.com/NVIDIA/cutlass/blob/main/python/CuTeDSL/cutlass/cute/nvgpu/warp/mma.py

The generated PTX uses **`.version 9.1` + `.target sm_120a`**. PTX 9.1 is required to include the documented `mxf4nvf4 + ue8m0 + scale_vec::4X` form introduced in PTX ISA 9.1.

## Coverage classes

`src/bench.cpp` separates three classes:

1. **DOCUMENTED** — directly present in the normative PTX syntax/valid-combination tables and applicable SM120 target notes. These are benchmarked.
2. **DOC_AMBIGUOUS** — NVIDIA prose/ISA notes claim support, but the normative syntax table does not define the spelling. These are compile-probed only.
3. **UNDOCUMENTED** — plausible-looking combinations not defined by NVIDIA's normative grammar/table. These are compile-probed only.

A JIT accepting a non-documented spelling is reported as an observation; it never upgrades that spelling to documented ISA support.

## Documented FP8 / FP6 / FP4 space

### Dense unscaled

- Legacy FP8: `e4m3` / `e5m2`, `m16n8k16` and `m16n8k32`, independent A/B FP8 format selection, F16/F32 C/D where defined.
- SM120 `kind::f8f6f4`: fixed shape `m16n8k32`, A/B independently selected from `e4m3`, `e5m2`, `e3m2`, `e2m3`, `e2m1`, with F16/F32 C/D.

### Dense block-scaled

| kind | A/B element types | scale type | scale vector |
|---|---|---|---|
| `mxf8f6f4` | `e4m3`, `e5m2`, `e3m2`, `e2m3`, `e2m1` | `ue8m0` | `1X` |
| `mxf4` | `e2m1` | `ue8m0` | `2X` |
| `mxf4nvf4` | `e2m1` | `ue8m0` | `2X`, `4X` |
| `mxf4nvf4` | `e2m1` | `ue4m3` | `4X` |

The manifest includes both the explicit and documented default spellings where applicable: omitting `scale_vec` means `1X` for `mxf8f6f4` and `2X` for `mxf4`; `mxf4nvf4` requires an explicit `scale_vec`. Block-scaled C/D are F32 in the documented grammar.

### Sparse

The normative PTX 9.3 grammar currently defines:

- legacy FP8 sparse: `m16n8k64`, F32 C/D, both `.sp` and `.sp::ordered_metadata`;
- `kind::f8f6f4`: `m16n8k64`, **`sp::ordered_metadata` only**, F16/F32 C/D;
- block-scaled `mxf8f6f4`: `m16n8k64`, **`sp::ordered_metadata` only**;
- block-scaled `mxf4` / `mxf4nvf4`: `m16n8k128`, **`sp::ordered_metadata` only**.

There is one NVIDIA-documentation inconsistency worth preserving as a probe: the sparse PTX ISA Notes say SM120 adds FP8 `m16n8k32 + f16` support, but the normative sparse syntax table still contains no FP8 `m16n8k32/f16` production. The repository classifies those spellings as `DOC_AMBIGUOUS`, not `DOCUMENTED`.

For ordered metadata, the source uses `0x44444444`; each 4-bit nibble encodes increasing indices `(0,1)`. Block-scale selectors use the valid canonical `{0,0}` values, and scale bytes use `0x38` so neither UE8M0 (`0xff`) nor UE4M3 (`0x7f`) NaN encodings are introduced.

## Plausible but undefined probes

The repository explicitly probes important boundaries, including:

- `tcgen05.mma` on `sm_120a`;
- dense `kind::f8f6f4` with `k16` / `k64` instead of the defined `k32`;
- FP6/FP4 (`e3m2`, `e2m3`, `e2m1`) used without `kind::f8f6f4`;
- invalid block-scale table pairings such as `mxf8f6f4 + 2X`, `mxf8f6f4 + ue4m3`, `mxf4 + 1X`, `mxf4 + ue4m3`, `mxf4nvf4 + ue4m3 + 2X`;
- `mxf4nvf4` with omitted `scale_vec`;
- block-scaled MMA with F16 C/D;
- plain `.sp` versions of sparse `f8f6f4`, `mxf8f6f4`, and `mxf4` where the normative grammar is ordered-metadata-only;
- `row.row` low-precision layout where the grammar is fixed to `row.col`.

The invalid-string space is unbounded, so probes target meaningful grammar/table boundaries rather than arbitrary malformed PTX.

## FLOP accounting

A dense `m16n8kK` warp MMA is counted as `2 * 16 * 8 * K` FLOPs. Sparse results report both `peak_logical_tflops = 2*M*N*K` and `peak_nonzero_tflops = logical/2`, avoiding ambiguity between dense-equivalent sparse throughput and actual nonzero multiply-add work.

Each documented case is warmed up, timed repeatedly (default 5 repetitions), and the best device-event time is used for `peak_logical_tflops`; the mean is also reported. RTX 5090 and RTX PRO 6000 share the same SM120 PTX capability class but can have different device-level peaks because SM count, clocks, power, and thermal limits differ.

## Build / run

Requires a CUDA toolkit/driver capable of JIT-compiling PTX 9.1 for SM120.

```bash
cmake -S . -B build
cmake --build build -j
./build/tensorcore-fp864-bench
```

Options:

```text
--device N                CUDA device index (default: 0)
--iters N                 loop iterations (default: 4000)
--blocks-per-sm N         work multiplier per SM (default: 4)
--chains N                independent accumulator chains: 1,2,4,8 (default: 8)
--repeats N               timed repetitions; best is peak (default: 5)
--filter TEXT             select case names containing TEXT
--include-probes          also compile DOC_AMBIGUOUS / UNDOCUMENTED cases
--probes-only             compile only non-documented probes
--list                    list manifest without CUDA initialization
--verbose-jit             print CUDA JIT logs
```

Result tags include `PASS`, `FAIL_DOCUMENTED`, `ACCEPTED_DOC_AMBIGUOUS`, `REJECTED_DOC_AMBIGUOUS`, `ACCEPTED_UNDOCUMENTED`, and `REJECTED_UNDOCUMENTED`.

See `docs/ptx-coverage.md` for the audit rationale. The source was statically reviewed against NVIDIA PTX documentation; no GPU benchmark or local command was run while creating the repository.