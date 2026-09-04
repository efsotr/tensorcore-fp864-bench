# Output, headline peaks, and SASS verification

The benchmark treats reproducibility as part of the measurement. A normal run writes both human-readable and machine-readable results and preserves the PTX -> CUBIN -> SASS trail for every selected case.

## Output layout

Unless `--output-dir DIR` is supplied, a run is written under:

```text
results/<UTC timestamp>_<GPU name>/
```

The directory contains:

```text
summary.md             human-readable headline report
cases.log              human-readable per-case log
run.json               complete machine-readable run record
results.csv            flat machine-readable per-case table
peak_summary.json      machine-readable headline Peak FP8/FP6/FP4 table
ptx/*.ptx              exact generated PTX for each selected case
cubin/*.cubin          exact Driver-JIT-linked cubin used for SASS audit/timing
sass/*.sass.txt        nvdisasm human-readable SASS
sass/*.sass.json       nvdisasm native JSON SASS output
sass/*.nvdisasm.log    disassembler diagnostics
sass/nvdisasm-version.txt
```

`run.json` is the source of truth for automation. `results.csv` is intended for pandas/R/spreadsheet workflows. `summary.md` and the final terminal table are intended for quick human inspection.

## Headline Peak extraction

The headline table is deliberately **dense only**. Sparse throughput remains in `run.json` / `results.csv` and reports both logical dense-equivalent TFLOP/s and nonzero-work TFLOP/s, but sparse values are not allowed to win a dense Peak category.

The six headline precision categories are:

```text
FP8
FP6
FP4
FP8 x FP6
FP8 x FP4
FP6 x FP4
```

Each is reported in two modes:

- `without_mx`: documented dense unscaled PTX (`legacy FP8` and/or `kind::f8f6f4` as applicable);
- `with_mx`: documented dense MX PTX (`kind::mxf8f6f4` and `kind::mxf4`).

For mixed-width categories, both operand orientations are eligible. For example, `FP8 x FP4` searches both FP8(A) x FP4(B) and FP4(A) x FP8(B), all documented format subtypes, and all documented accumulator choices. The highest measured result wins, while `peak_summary.json` preserves the exact A/B formats, orientation, accumulator type, PTX case, and SASS family of the winner.

`mxf4nvf4` is intentionally **not** folded into `with_mx`: NVFP4 and MXFP4 are different scaling formats. The best documented `mxf4nvf4` FP4 result is emitted separately as `nvfp4_reference`.

## SASS is a default execution gate

By default, a documented PTX case is **not timed immediately after JIT compilation**. The runner first:

1. compiles/links the generated PTX with the CUDA Driver API;
2. obtains the resulting CUBIN from `cuLinkComplete`;
3. writes that exact CUBIN to disk;
4. disassembles it using NVIDIA `nvdisasm` in both text and JSON forms;
5. checks that the resulting SASS has the expected Tensor Core family, MMA shape, accumulator type, A/B data formats, and static Tensor Core MMA instruction count;
6. only then launches and times the kernel.

Therefore, with the default policy, a `PASS` result has already passed its SASS structural audit. If the SASS does not match, the case is recorded as `SASS_MISMATCH` and is **not executed**. If `nvdisasm` is unavailable, cases are recorded as `SASS_TOOL_ERROR` and are not executed; the run still writes partial reports and exits non-zero. `--no-sass-check` is an explicit opt-out and is recorded in `run.json` / `peak_summary.json`.

NVIDIA documents that `cuLinkComplete` returns the cubin image produced by a Driver API link and that the image can be loaded directly with `cuModuleLoadData`. NVIDIA also documents `nvdisasm` support for standalone cubins, `-c` code-only output, and native `-json` disassembly. These are the mechanisms used here:

- https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html
- https://docs.nvidia.com/cuda/cuda-binary-utilities/

## Expected SM120 SASS families

NVIDIA does not publish a normative SM120 SASS ISA specification equivalent to PTX, so the SASS names below are treated as an **audit expectation**, not as the source of truth for PTX legality. They are cross-checked against `nvdisasm` observations from current SM120 work and should be revisited if NVIDIA changes disassembler spelling.

| PTX path | Expected SASS structural family |
|---|---|
| FP8 / `kind::f8f6f4`, dense | `QMMA.168K.<acc>.<A>.<B>` |
| `kind::mxf8f6f4.block_scale`, dense | `QMMA.SF.16832.F32.<A>.<B>...` |
| `kind::mxf4` / `kind::mxf4nvf4`, dense | `OMMA.SF.16864.F32.E2M1.E2M1...` |
| sparse FP8 / `kind::f8f6f4` | `QMMA.SP.168K...` |
| sparse `mxf8f6f4` | `QMMA.SF.SP.16864...` |
| sparse `mxf4` / `mxf4nvf4` | `OMMA.SF.SP.168128...` |

Empirical SM120 references that expose the same `nvdisasm` family names include:

- https://github.com/florianmattana/sass-king/blob/main/corpus/tensor_cores/README.md
- https://zartbot.github.io/micro_arch/nvidia/sm_120/04_tensorcore_architecture.html

The current automated gate checks the SASS family (`QMMA` vs `OMMA`, plus `SF` / `SP` modifiers), shape, accumulator type, A/B element format, and exact static Tensor Core MMA count. The complete SASS text/JSON is always preserved so scale-mode/control-code details can be audited independently; this avoids pretending that reverse-engineered SASS modifier semantics are part of NVIDIA's public PTX contract.

## Why the static MMA count is checked

The hot PTX body contains `chains x inner_unroll` MMA instructions. Each accumulator chain begins from a different C value and remains observable after the loop, preventing legal common-subexpression elimination from collapsing identical chains. The SASS gate then verifies that exactly the expected number of Tensor Core MMA instructions remains in the compiled kernel and that none has been replaced by another Tensor Core family.

This is important for a peak-throughput microbenchmark: measuring elapsed time is insufficient if the compiler silently changes the number or class of operations being timed.
