# SM120 low-precision PTX coverage audit

This note records the documentation basis used by the benchmark. It deliberately distinguishes **NVIDIA PTX ISA support** from higher-level library exposure.

## 1. Target architecture

NVIDIA's CUDA GPU Compute Capability table places both GeForce RTX 5090 and RTX PRO 6000 Blackwell in Compute Capability 12.0. The corresponding PTX/CUDA architecture target is SM120.

The PTX 8.8 family-specific feature table is the key architectural discriminator:

- `tcgen05` MMA instructions and Tensor Memory operations are listed for the SM100/SM110-family targets, not the SM120 family.
- warp-level `mma{.sp}` gains `.e3m2`, `.e2m3`, `.e2m1`, `.kind`, `.block_scale`, and `.scale_vec_size` support on the SM120 family.

Therefore the RTX 5090 / RTX PRO 6000 benchmark uses `mma.sync` / `mma.sp...sync`, not `tcgen05.mma`.

References:

- https://developer.nvidia.com/cuda/gpus
- https://docs.nvidia.com/cuda/archive/12.9.2/parallel-thread-execution/
- https://docs.nvidia.com/cuda/parallel-thread-execution/

## 2. Dense unscaled low-precision MMA

### Legacy FP8 path

Documented warp-level FP8 forms use `e4m3` and `e5m2` with `m16n8k16` and `m16n8k32`. F16 and F32 accumulator/result forms are included where the PTX grammar permits them. A/B format qualifiers are enumerated independently, including E4M3 x E5M2 and E5M2 x E4M3.

Representative spelling:

```text
mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e4m3.f32
```

### SM120 `kind::f8f6f4`

PTX extends warp MMA for SM120 with:

```text
mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.<dtype>.<atype>.<btype>.<ctype>
```

with A/B type choices from:

```text
e4m3, e5m2, e3m2, e2m3, e2m1
```

and F16/F32 C/D types as permitted by the grammar. For these FP6/FP4 entries, PTX specifies an 8-bit container representation for individual 6-bit/4-bit elements in `kind::f8f6f4`.

A dense `kind::f8f6f4` `m16n8k64` form is not defined; it is included only as an undocumented compile probe.

## 3. Dense block-scaled MMA

The PTX block-scaling valid-combination table defines:

| `.kind` | A/B types | `.stype` | `.scale_vec_size` | dense shape used here |
|---|---|---|---|---|
| `mxf8f6f4` | `e4m3/e5m2/e3m2/e2m3/e2m1` | `ue8m0` | `1X` | `m16n8k32` |
| `mxf4` | `e2m1` | `ue8m0` | `2X` | `m16n8k64` |
| `mxf4nvf4` | `e2m1` | `ue8m0` | `2X` | `m16n8k64` |
| `mxf4nvf4` | `e2m1` | `ue8m0` | `4X` | `m16n8k64` |
| `mxf4nvf4` | `e2m1` | `ue4m3` | `4X` | `m16n8k64` |

For `mxf4`, omission of the scale-vector qualifier defaults to 2X; for `mxf8f6f4`, omission defaults to 1X. The benchmark uses explicit qualifiers so output names remain unambiguous. For `mxf4nvf4`, the scale-vector qualifier is mandatory.

Block-scaled warp MMA uses F32 accumulator/result in the documented grammar. An F16 block-scaled spelling is therefore an undocumented negative probe.

Reference: PTX ISA 9.1, block scaling Table 37:
https://docs.nvidia.com/cuda/pdf/ptx_isa_9.1.pdf

## 4. Sparse MMA

Sparse warp-level MMA represents a logical MxNxK multiplication with a 50%-sparse A matrix. The benchmark reports both the logical dense-equivalent FLOP rate and half-rate nonzero work.

The manifest covers:

- legacy FP8 `mma.sp.sync` and `mma.sp::ordered_metadata.sync`;
- SM120 `kind::f8f6f4` sparse forms, with logical K doubled relative to the dense `m16n8k32` form (`m16n8k64`);
- block-scaled `mxf8f6f4` sparse forms at `m16n8k64`;
- block-scaled `mxf4` / `mxf4nvf4` ordered-metadata forms at `m16n8k128`.

NVIDIA's target notes explicitly carve out plain `.sp` for `mxf4` / `mxf4nvf4`; the ordered-metadata examples are documented. Consequently plain `mma.sp...kind::mxf4` is treated as an undocumented probe, not a benchmark case.

For `mma.sp::ordered_metadata`, the metadata indices must be sorted in increasing order. The benchmark uses a repeated `0b0100`-style metadata nibble (`0x44444444`) to stay on a documented ordered pattern rather than feeding arbitrary metadata.

## 5. Why raw PTX, not CUTLASS coverage, defines the matrix

CUTLASS/CuTe is useful as a second implementation cross-check but is not the ISA specification. Current CUTLASS SM120 warp-MMA wrappers expose a narrower set of block-scaled mixed-type combinations than the PTX grammar; for example, some wrapper code uses explicit allow-lists and does not expose the full FP6 space even though the PTX ISA defines FP6 alternate formats and `mxf8f6f4` grammar.

The benchmark therefore applies this priority:

1. NVIDIA PTX ISA grammar + valid-combination tables + Target ISA Notes.
2. NVIDIA CUTLASS/PTX examples as a cross-check for fragment/register shapes and spelling.
3. Current JIT acceptance only as an observation.

A current driver accepting an undocumented spelling never upgrades that spelling to `documented`.

CUTLASS cross-check:
https://github.com/NVIDIA/cutlass/blob/main/python/CuTeDSL/cutlass/cute/nvgpu/warp/mma.py

## 6. Undocumented probes intentionally kept in the repository

The probe set is finite and representative; the space of syntactically imaginable invalid strings is unbounded. Included probes exercise the important boundaries:

| Probe | Why it looks plausible | Why it is not classified as documented |
|---|---|---|
| `tcgen05.mma` on `sm_120a` | Blackwell has fifth-generation Tensor Cores | PTX target-feature table does not expose tcgen05 MMA to SM120 |
| dense `f8f6f4` `m16n8k64` | lower bit-width often increases K | dense SM120 grammar defines `m16n8k32` |
| FP6 without `kind::f8f6f4` | legacy FP8 spelling looks structurally identical | FP6/FP4 extension is documented through the kind-qualified grammar |
| `mxf8f6f4` + `2X` | larger scale vector is a known qualifier | valid-combination table fixes it to `1X` |
| `mxf8f6f4` + UE4M3 | UE4M3 is a valid scale format elsewhere | table fixes this kind to UE8M0 |
| `mxf4` + `1X` | same scale qualifier family exists | table fixes `mxf4` to `2X` |
| `mxf4` + UE4M3 | UE4M3 is valid for NVF4 | not defined for `mxf4` |
| `mxf4nvf4` + UE4M3 + `2X` | both UE4M3 and `2X` individually appear in the table | UE4M3 is paired only with `4X` |
| `mxf4nvf4` with omitted scale vector | other block kinds have defaults | PTX states it is mandatory for this kind |
| block-scaled F16 accumulator | unscaled low-precision MMA permits F16 | block-scaled grammar fixes F32 |
| plain sparse `mxf4` | ordinary sparse MMA has `.sp` | target notes exclude this combination; ordered metadata is documented |
| low-precision `row.row` | matrix layout modifiers exist broadly in PTX | documented low-precision warp form is `row.col` |

## 7. Remaining empirical validation

The repository source has been checked against the NVIDIA documentation and CUTLASS examples, but no command or GPU execution was performed while creating it. Final empirical validation on both target boards should record at minimum:

- GPU exact product name;
- driver version;
- CUDA toolkit/PTX version;
- observed GPU clock/power state during the run;
- JIT acceptance/rejection table;
- measured TFLOPS for each documented case.

This separation is intentional: documentation correctness can be reviewed statically, while throughput and toolchain acceptance are hardware/toolchain observations.