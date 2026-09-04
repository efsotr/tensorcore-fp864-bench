# tensorcore-fp864-bench

Microbenchmark for **single-PTX-instruction Tensor Core peak FLOPS** on NVIDIA **GeForce RTX 5090** and **RTX PRO 6000 Blackwell**, both Compute Capability **12.0 / SM120**.

The timed inner loop repeatedly issues one Tensor Core PTX instruction family, keeps operands in registers, uses multiple independent accumulator chains to reduce dependency-latency effects, and performs no global/shared-memory traffic inside the repeated MMA body. The target metric is Tensor Core instruction throughput, not GEMM/library/end-to-end performance.

## SM120: use `mma`, not `tcgen05`

RTX 5090 and RTX PRO 6000 Blackwell are SM120 devices. For FP8/FP6/FP4 Tensor Core operations, the relevant public PTX path is warp-level `mma.sync` / `mma.sp...sync` with the SM120 low-precision extensions. NVIDIA's PTX family-specific feature table exposes `tcgen05` MMA to the SM100/SM110-family targets, while `.e3m2`, `.e2m3`, `.e2m1`, `.kind`, `.block_scale`, and `.scale_vec` extensions for `mma{.sp}` are exposed to the SM120 family.

Primary references:

- Current PTX ISA: https://docs.nvidia.com/cuda/parallel-thread-execution/
- PTX ISA 9.0 PDF: https://docs.nvidia.com/cuda/pdf/ptx_isa_9.0.pdf
- PTX ISA 9.1 PDF: https://docs.nvidia.com/cuda/pdf/ptx_isa_9.1.pdf
- NVIDIA CUDA GPU Compute Capability table: https://developer.nvidia.com/cuda/gpus
- CUTLASS SM120 warp MMA implementation, used only as a secondary cross-check: https://github.com/NVIDIA/cutlass/blob/main/python/CuTeDSL/cutlass/cute/nvgpu/warp/mma.py

## Coverage model

`src/main.cpp` generates two classes of cases:

1. **Documented** — combinations directly covered by NVIDIA PTX grammar, valid-combination tables, and SM120 target notes. These are JIT-compiled and benchmarked.
2. **Undocumented probes** — combinations that look plausible but are not defined as valid by the NVIDIA PTX documentation. These are compile-probed only. If a current driver happens to accept one, it is reported as `ACCEPTED_UNDOCUMENTED`; that is implementation behavior, not an ISA guarantee.

The documented manifest covers:

- legacy FP8 `mma.sync` using `e4m3` / `e5m2`, including mixed E4M3/E5M2 forms;
- SM120 `.kind::f8f6f4` dense MMA using A/B types from `e4m3`, `e5m2`, `e3m2`, `e2m3`, `e2m1`;
- block-scaled `.kind::mxf8f6f4`, `.kind::mxf4`, and `.kind::mxf4nvf4` MMA;
- sparse `mma.sp.sync` and `mma.sp::ordered_metadata.sync` variants where the PTX ISA defines them;
- F16/F32 accumulator variants where the corresponding PTX grammar permits them.

For block scaling, NVIDIA PTX Table 37 defines the following combinations:

| kind | A/B element types | scale type | scale vector |
|---|---|---|---|
| `mxf8f6f4` | `e4m3`, `e5m2`, `e3m2`, `e2m3`, `e2m1` | `ue8m0` | `1X` |
| `mxf4` | `e2m1` | `ue8m0` | `2X` |
| `mxf4nvf4` | `e2m1` | `ue8m0` | `2X` or `4X` |
| `mxf4nvf4` | `e2m1` | `ue4m3` | `4X` |

A and B type qualifiers are enumerated independently where the PTX grammar permits independent `atype` / `btype` selection.

## Deliberately included undocumented / negative probes

Representative probes include:

- `tcgen05.mma` targeting `sm_120a`;
- dense `.kind::f8f6f4` with `m16n8k64`;
- FP6/FP4 types used in the older FP8 spelling without `.kind::f8f6f4`;
- `.kind::mxf8f6f4` with `scale_vec::2X` / `4X` or UE4M3 scales;
- `.kind::mxf4` with `scale_vec::1X` / `4X` or UE4M3 scales;
- `.kind::mxf4nvf4` with UE4M3 + `scale_vec::2X`, or with the mandatory scale-vector qualifier omitted;
- block-scaled MMA with F16 accumulator/result types where the PTX grammar fixes them to F32;
- plain sparse `.sp` with `mxf4` / `mxf4nvf4`, for which the documented SM120 path is the ordered-metadata form;
- alternative layouts such as `row.row` where the documented low-precision grammar is fixed to `row.col`.

These cases intentionally separate three questions: **is the spelling plausible, does the current CUDA JIT accept it, and does NVIDIA document it as supported?** Only the last one determines the benchmark's documented-support classification.

## FLOP accounting

For dense `m16n8kK`, one warp-level MMA instruction is counted as:

```text
2 * 16 * 8 * K FLOPs
```

For sparse MMA, the program reports both:

- `logical_tflops`: `2*M*N*K`, corresponding to the logical dense operation represented by the sparse instruction;
- `nonzero_tflops`: half of the logical value, corresponding to the 50%-sparse A operand's nonzero multiply-add work.

This avoids silently mixing NVIDIA-style sparse logical throughput with physical nonzero work.

## Measurement method

Each benchmark kernel is generated as PTX and JIT-loaded through the CUDA Driver API. The hot loop contains multiple independent accumulator chains of the **same PTX MMA spelling**, with fixed register-resident A/B fragments. The loop performs no memory accesses; only a final liveness store is emitted after the timed repeated MMA body.

The default launch creates enough warps to occupy all SMs and uses a long loop so launch/event overhead is negligible relative to Tensor Core work. Device-level peak results can differ between RTX 5090 and RTX PRO 6000 despite identical SM120 instruction support because SM count, clocks, power limits, and thermal behavior differ.

## Build

Requires a CUDA 13.x-era toolkit/driver with SM120 support.

```bash
cmake -S . -B build
cmake --build build -j
```

## Run

```bash
./build/tensorcore-fp864-bench
```

Options:

```text
--iters N                 loop iterations per kernel (default: 2000)
--blocks-per-sm N         work multiplier per SM (default: 4)
--chains N                independent accumulator chains: 1,2,4,8 (default: 8)
--filter TEXT             select case names containing TEXT
--documented-only         skip undocumented compile probes
--probes-only             compile/probe without timing documented cases
--verbose-jit             print CUDA JIT logs
```

## Result interpretation

- `PASS`: documented combination JIT-compiled and ran.
- `FAIL_DOCUMENTED`: documented combination was rejected or failed to launch; record CUDA/PTX/driver versions before drawing an ISA conclusion.
- `REJECTED_UNDOCUMENTED`: expected result for a non-documented combination.
- `ACCEPTED_UNDOCUMENTED`: current toolchain accepted a non-documented spelling; do **not** treat this as portable or specified support.

The documentation-status classification is intentionally independent from the observed compiler/JIT result.