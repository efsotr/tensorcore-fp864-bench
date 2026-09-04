# SM120 FP8 / FP6 / FP4 PTX coverage audit

This file records the rules used to classify benchmark cases. The source of truth is NVIDIA's **normative PTX syntax + valid-combination tables + Target ISA Notes**; CUTLASS is only a secondary implementation cross-check, and JIT acceptance is only an empirical observation.

## 1. Target: RTX 5090 / RTX PRO 6000 = SM120

NVIDIA lists GeForce RTX 5090 and RTX PRO 6000 Blackwell as Compute Capability 12.0. PTX's family-specific feature table makes the key distinction:

- `tcgen05.mma` belongs to the SM100/SM110 family, not SM120;
- SM120 receives warp-level `mma{.sp}` extensions for `.e3m2`, `.e2m3`, `.e2m1`, `.kind`, `.block_scale`, and `.scale_vec`.

Therefore this repository targets `.target sm_120a` and benchmarks warp-level `mma.sync` / sparse ordered-metadata MMA rather than `tcgen05.mma`.

References:

- https://developer.nvidia.com/cuda/gpus
- https://docs.nvidia.com/cuda/parallel-thread-execution/
- https://docs.nvidia.com/cuda/pdf/ptx_isa_9.1.pdf

## 2. Dense unscaled grammar

### Legacy FP8

Normative form:

```text
mma.sync.aligned.shape.row.col.dtype.f8type.f8type.ctype
```

with:

```text
shape  = m16n8k16 | m16n8k32
f8type = e4m3 | e5m2
dtype/ctype = f16 | f32   (D and C use the same type for these shapes)
```

A and B `f8type` positions are independent, so same-format and mixed E4M3/E5M2 cases are included.

### `kind::f8f6f4`

Normative SM120 form:

```text
mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.dtype.atype.btype.ctype
```

with A/B independently selected from:

```text
e4m3, e5m2, e3m2, e2m3, e2m1
```

and matching F16 or F32 C/D. The shape is fixed to `m16n8k32`; `k16` and `k64` variants are useful negative probes but are not defined by the grammar.

For `kind::f8f6f4`, FP6 values occupy the lower six bits of an 8-bit container with two upper padding bits; FP4 E2M1 occupies the central four bits with two padding bits on each side. The benchmark's repeated A/B byte patterns satisfy those padding requirements.

## 3. Block scaling

Current PTX valid-combination table:

| `.kind` | A/B type | `.stype` | `.scale_vec_size` |
|---|---|---|---|
| `mxf8f6f4` | `e4m3/e5m2/e3m2/e2m3/e2m1` | `ue8m0` | `1X` |
| `mxf4` | `e2m1` | `ue8m0` | `2X` |
| `mxf4nvf4` | `e2m1` | `ue8m0` | `2X`, `4X` |
| `mxf4nvf4` | `e2m1` | `ue4m3` | `4X` |

Dense shapes are `m16n8k32` for `mxf8f6f4` and `m16n8k64` for `mxf4` / `mxf4nvf4`. C/D are F32.

`scale_vec` omission is itself a documented spelling for two kinds: `mxf8f6f4` defaults to `1X`, and `mxf4` defaults to `2X`. It is mandatory for `mxf4nvf4`. The benchmark therefore includes both explicit and default spellings for the first two kinds.

PTX ISA 9.1 added `scale_vec::4X` with UE8M0 for `mxf4nvf4`; this is why generated PTX declares `.version 9.1` rather than 9.0.

Selector operands are not swept as separate throughput cases because they do not change the MMA opcode/shape. The benchmark uses canonical valid selectors `{0,0}`. PTX's selector table confirms zero is valid for 1X, 2X and 4X. Scale bytes use `0x38`: UE8M0 reserves `0xff` as NaN and UE4M3 reserves `0x7f`, so `0x38` avoids injecting either NaN encoding.

## 4. Sparse grammar — important restrictions

### Legacy FP8

The **normative PTX 9.3 syntax** explicitly defines:

```text
mma.spvariant.sync.aligned.m16n8k64.row.col.f32.f8type.f8type.f32
spvariant = sp | sp::ordered_metadata
f8type    = e4m3 | e5m2
```

Thus the documented benchmark matrix uses `m16n8k64`, F32, both sparse variants, and all four independent FP8 A/B pairs.

There is a documentation inconsistency: the same sparse section's ISA Notes state that PTX 8.7 / SM120 added **`m16n8k32` + F16 C/D with E4M3/E5M2**, but the normative sparse syntax table in current PTX 9.3 still has no corresponding FP8 `m16n8k32/f16` production. The repository therefore labels these cases `DOC_AMBIGUOUS` and compile-probes them instead of claiming them as documented benchmark cases.

### `kind::f8f6f4`

Normative syntax is explicitly:

```text
mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.kind::f8f6f4.dtype.atype.btype.ctype
```

There is **no `spvariant` placeholder** here: plain `.sp` is not a documented form. A/B are independently selected from all five F8/F6/F4 types; matching F16/F32 C/D are defined.

### Block-scaled sparse

All three block-scaled sparse grammars set:

```text
spvariant = sp::ordered_metadata
```

only. Shapes are:

- `mxf8f6f4`: `m16n8k64`;
- `mxf4`: `m16n8k128`;
- `mxf4nvf4`: `m16n8k128`.

Plain `.sp` spellings are therefore intentionally classified as undocumented probes.

For ordered metadata, PTX requires indices to be sorted increasingly from the LSB. `0x44444444` repeats nibble `0100b`, i.e. two 2-bit indices `0,1`, satisfying that ordering requirement.

## 5. Documentation classes

The executable uses three classes:

- `DOCUMENTED`: normative grammar/table defines the case and SM120 target notes permit it;
- `DOC_AMBIGUOUS`: NVIDIA prose/ISA notes suggest support but the normative production is absent or inconsistent;
- `UNDOCUMENTED`: plausible but the normative grammar/table does not define it.

Only `DOCUMENTED` cases are timed by default. `--include-probes` JIT-compiles the other two classes; `--probes-only` restricts execution to those compile probes. Undocumented/ambiguous cases are never executed after JIT acceptance because the goal is to observe toolchain acceptance without relying on unspecified execution semantics.

## 6. Boundary probes included

The finite probe set targets meaningful ISA boundaries rather than arbitrary malformed strings:

| Probe | Reason it looks plausible | Why not documented |
|---|---|---|
| `tcgen05.mma` on `sm_120a` | both are Blackwell-era concepts | target-feature table exposes tcgen05 MMA to SM100/SM110 family, not SM120 |
| dense `f8f6f4` k16/k64 | legacy FP8 has multiple K shapes | kind-qualified grammar is fixed to k32 |
| FP6/FP4 without `kind::f8f6f4` | same token positions exist for FP8 | normative FP6/FP4 warp-MMA syntax is kind-qualified |
| `mxf8f6f4 + 2X` | 2X is a valid scale-vector token | table fixes this kind to 1X |
| `mxf8f6f4 + ue4m3` | UE4M3 is a valid scale type elsewhere | table fixes this kind to UE8M0 |
| `mxf4 + 1X` | 1X exists for another kind | table fixes mxf4 to 2X |
| `mxf4 + ue4m3` | UE4M3 is valid for mxf4nvf4 | not defined for mxf4 |
| `mxf4nvf4 + ue4m3 + 2X` | both tokens exist individually | UE4M3 is paired only with 4X |
| `mxf4nvf4` without scale vector | omission is valid for two other kinds | mandatory for mxf4nvf4 |
| block-scaled F16 C/D | F16 is valid in unscaled low precision | block-scaled grammar fixes C/D to F32 |
| plain `.sp` `f8f6f4` | legacy sparse FP8 allows `.sp` | kind-qualified sparse grammar is ordered-metadata-only |
| plain `.sp` block scaling | generic sparse family has `.sp` | block-scaled sparse productions are ordered-metadata-only |
| `row.row` low precision | layout tokens exist elsewhere | these low-precision productions are fixed to `row.col` |

A JIT acceptance result is reported as `ACCEPTED_UNDOCUMENTED` or `ACCEPTED_DOC_AMBIGUOUS`; it is not promoted to ISA support.

## 7. CUTLASS cross-check policy

CUTLASS/CuTe SM120 wrappers are useful for checking fragment register counts, shapes and known spellings, but their exposed type allow-lists can be narrower than the PTX grammar. Consequently this repository does **not** infer unsupported hardware from a missing CUTLASS wrapper. Priority is:

1. NVIDIA PTX normative grammar + tables + target notes;
2. NVIDIA examples/CUTLASS for implementation cross-checks;
3. JIT behavior as empirical evidence only.

CUTLASS reference:
https://github.com/NVIDIA/cutlass/blob/main/python/CuTeDSL/cutlass/cute/nvgpu/warp/mma.py

## 8. What has and has not been validated

The repository was created by static inspection of NVIDIA PTX documentation and NVIDIA implementation references. **No local command, compiler, CUDA JIT, RTX 5090 run, or RTX PRO 6000 run was executed during creation.**

Actual hardware validation should record the exact GPU, driver/toolkit version, clock/power state, JIT acceptance table, register count, best/mean kernel time, and resulting logical/nonzero TFLOPS. A documented case rejected by a specific JIT is reported as `FAIL_DOCUMENTED` and should be treated as a toolchain/version observation until diagnosed, not as an automatic correction to the ISA classification.