# tensorcore-fp864-bench

Microbenchmark for **single-PTX-instruction Tensor Core peak FLOPS** on NVIDIA **GeForce RTX 5090** and **RTX PRO 6000 Blackwell** (Compute Capability **12.0 / SM120**).

The hot loop repeatedly issues one PTX MMA spelling with register-resident operands, multiple semantically independent accumulator chains, and four-way inner unrolling. There is no global/shared-memory traffic inside the repeated MMA body; this measures PTX-level Tensor Core instruction throughput rather than GEMM/library/end-to-end performance.

## Architecture rule: SM120 uses warp-level `mma`, not `tcgen05`

For RTX 5090 / RTX PRO 6000, the relevant public low-precision PTX interface is warp-level `mma.sync` and sparse `mma.sp{::ordered_metadata}.sync`. NVIDIA's family-specific PTX feature table exposes `tcgen05.mma` to the SM100/SM110 family, while the FP6/FP4 `.e3m2`, `.e2m3`, `.e2m1`, `.kind`, `.block_scale`, and `.scale_vec` extensions of warp-level `mma{.sp}` are exposed to SM120.

Primary references:

- Current PTX ISA: https://docs.nvidia.com/cuda/parallel-thread-execution/
- PTX ISA 9.1 PDF: https://docs.nvidia.com/cuda/pdf/ptx_isa_9.1.pdf
- NVIDIA CUDA GPU Compute Capability table: https://developer.nvidia.com/cuda/gpus
- CUTLASS SM120 warp MMA implementation, secondary cross-check only: https://github.com/NVIDIA/cutlass/blob/main/python/CuTeDSL/cutlass/cute/nvgpu/warp/mma.py

Generated PTX uses **`.version 9.1` + `.target sm_120a`**. PTX 9.1 is required to cover the documented `mxf4nvf4 + ue8m0 + scale_vec::4X` form introduced in PTX ISA 9.1.

## Coverage classes

`src/bench.cpp` uses three classes:

1. **DOCUMENTED** — directly defined by normative PTX syntax / valid-combination tables and applicable SM120 target notes. These are benchmarked.
2. **DOC_AMBIGUOUS** — NVIDIA ISA notes/prose claim support, but the current normative syntax does not define the spelling. These are JIT-probed only.
3. **UNDOCUMENTED** — plausible-looking combinations not defined by NVIDIA's normative grammar/table. These are JIT-probed only.

A JIT accepting a non-documented spelling is reported as an observation and is never treated as an ISA guarantee or executed by the benchmark.

The canonical base manifest contains **234 documented cases, 8 documentation-ambiguous probes, and 17 deliberately undocumented probes**.

## Documented FP8 / FP6 / FP4 instruction space

### Dense unscaled

- Legacy FP8: `e4m3` / `e5m2`, `m16n8k16` and `m16n8k32`, independent A/B format selection, F16/F32 C/D where defined.
- SM120 `kind::f8f6f4`: `m16n8k32`, A/B independently selected from `e4m3`, `e5m2`, `e3m2`, `e2m3`, `e2m1`, with F16/F32 C/D.

### Block-scaled

| kind | A/B element types | scale type | scale vector |
|---|---|---|---|
| `mxf8f6f4` | `e4m3`, `e5m2`, `e3m2`, `e2m3`, `e2m1` | `ue8m0` | `1X` |
| `mxf4` | `e2m1` | `ue8m0` | `2X` |
| `mxf4nvf4` | `e2m1` | `ue8m0` | `2X`, `4X` |
| `mxf4nvf4` | `e2m1` | `ue4m3` | `4X` |

Both documented explicit/default spellings are included where applicable: omitted `scale_vec` means `1X` for `mxf8f6f4` and `2X` for `mxf4`; it is mandatory for `mxf4nvf4`. Block-scaled C/D are F32.

### Sparse

Current normative PTX syntax defines:

- legacy FP8: `m16n8k64`, F32 C/D, both `.sp` and `.sp::ordered_metadata`;
- `kind::f8f6f4`: `m16n8k64`, **ordered metadata only**, F16/F32 C/D;
- `mxf8f6f4`: `m16n8k64`, **ordered metadata only**;
- `mxf4` / `mxf4nvf4`: `m16n8k128`, **ordered metadata only**.

The sparse ISA notes separately claim SM120 support for FP8 `m16n8k32 + f16`, while the current normative sparse syntax contains no such FP8 production. Those 8 variants are intentionally classified `DOC_AMBIGUOUS` rather than silently promoted to supported syntax.

## Exhaustive operand selectors

By default, each documented opcode uses canonical legal operand selectors (`sparse f=0`, block-scale selectors `{0,0}`). With `--all-operands`, the manifest additionally expands **every selector value explicitly defined by PTX**:

- sparse selector `f = 0,1,2,3`;
- `scale_vec::1X`: `byte-id-a/b = 0..3`, `thread-id-a = 0..1`, `thread-id-b = 0..3`;
- `scale_vec::2X`: `byte-id-a/b = {0,2}`, `thread-id-a = 0..1`, `thread-id-b = 0..3`;
- `scale_vec::4X`: `byte-id-a/b = 0`, `thread-id-a = 0..1`, `thread-id-b = 0..3`.

This keeps the normal benchmark practical while retaining an exhaustive mode for the documented PTX operand space. Values outside NVIDIA's selector table are not executed because PTX defines their behavior as undefined.

## Plausible but undefined probes

Representative negative probes include:

- `tcgen05.mma` targeting `sm_120a`;
- dense `kind::f8f6f4` with `k16` / `k64` instead of `k32`;
- FP6/FP4 types without `kind::f8f6f4`;
- invalid block-scale pairings (`mxf8f6f4 + 2X`, `mxf8f6f4 + ue4m3`, `mxf4 + 1X`, `mxf4 + ue4m3`, `mxf4nvf4 + ue4m3 + 2X`);
- `mxf4nvf4` without mandatory `scale_vec`;
- block-scaled F16 C/D;
- plain `.sp` versions of kind-qualified / block-scaled sparse forms whose normative grammar is ordered-metadata-only;
- `row.row` for low-precision forms whose grammar is fixed to `row.col`.

The syntactically imaginable invalid space is unbounded, so probes target meaningful ISA/table boundaries rather than arbitrary malformed strings.

## Peak-FLOPS methodology

A dense `m16n8kK` warp MMA is counted as `2 * 16 * 8 * K` FLOPs. Sparse results report both `peak_logical_tflops = 2*M*N*K` and `peak_nonzero_tflops = logical/2`, separating dense-equivalent sparse throughput from nonzero multiply-add work.

Each chain starts from a different accumulator seed so equivalent MMA chains cannot legally be collapsed into one common subexpression. The loop issues `chains × 4` MMAs per iteration before branching; by default that is 32 MMA instructions per warp per loop iteration. The kernel is warmed up, timed 5 times with CUDA device events, and reports both the best (peak) and mean throughput. A/B, accumulator, sparse metadata, and scale metadata remain register-resident throughout the repeated body.

The input byte patterns obey the required FP6/FP4 padding rules. Ordered sparse metadata uses `0x44444444`; scale bytes use `0x38`, avoiding the reserved UE8M0 (`0xff`) and UE4M3 (`0x7f`) NaN encodings.

RTX 5090 and RTX PRO 6000 share the SM120 PTX capability class but can produce different device-level peaks due to SM count, clocks, power limits, and thermal behavior.

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
--all-operands            expand all documented sparse/block-scale selector values
--include-probes          also JIT-probe DOC_AMBIGUOUS / UNDOCUMENTED cases
--probes-only             JIT-probe only non-documented cases
--list                    list the selected manifest without CUDA initialization
--verbose-jit             print CUDA JIT logs
```

Result tags include `PASS`, `FAIL_DOCUMENTED`, `ACCEPTED_DOC_AMBIGUOUS`, `REJECTED_DOC_AMBIGUOUS`, `ACCEPTED_UNDOCUMENTED`, and `REJECTED_UNDOCUMENTED`.

See `docs/ptx-coverage.md` for the classification audit. The repository was created by static inspection only: **no local command, compiler invocation, CUDA JIT, RTX 5090 benchmark, or RTX PRO 6000 benchmark was run while creating it**.