# Output, headline peaks, and SASS audit

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
cubin/*.cubin          exact Driver-JIT-linked cubin used for audit/timing
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

For mixed-width categories, both operand orientations are eligible. `FP8 x FP4`, for example, searches both FP8(A) x FP4(B) and FP4(A) x FP8(B), all documented format subtypes, and all documented accumulator choices. The highest measured result wins.

`mxf4nvf4` is intentionally **not** folded into `with_mx`: NVFP4 and MXFP4 are different scaling formats. The best documented `mxf4nvf4` FP4 result is emitted separately as `nvfp4_reference`.

## SASS audit is observational

SASS auditing is enabled by default, but it is **not an execution gate** and **not a Peak-selection gate**.

For every documented PTX case, the runner:

1. compiles/links the generated PTX with the CUDA Driver API;
2. obtains the resulting CUBIN from `cuLinkComplete`;
3. writes that exact CUBIN to disk;
4. attempts to disassemble it using NVIDIA `nvdisasm` in both text and JSON forms;
5. compares the resulting SASS with the expected Tensor Core structural mapping;
6. records the SASS result;
7. executes and times the documented case regardless of whether the SASS matched the expectation.

The important statuses are:

- `SASS_OK`: observed SASS matches the current structural expectation;
- `SASS_MISMATCH`: disassembly succeeded but one or more expected structural properties did not match;
- `SASS_TOOL_ERROR`: `nvdisasm` was unavailable;
- `SASS_DISASM_ERROR`: `nvdisasm` failed on that CUBIN;
- `SASS_DISABLED`: audit explicitly disabled with `--no-sass-check`.

All of these are **audit metadata**. They do not replace the benchmark status. A documented case that JIT-compiles and executes successfully is reported as `PASS` even when its SASS status is `SASS_MISMATCH`, `SASS_TOOL_ERROR`, or `SASS_DISASM_ERROR`.

The machine-readable output makes this explicit:

```text
run.json:
  sass.audit_enabled
  sass.gates_execution = false
  cases[i].sass.status
  cases[i].sass.matches_expected = true | false | null

peak_summary.json:
  sass_audit_enabled
  sass_gates_peak_selection = false
  headline[i].sass_status
  headline[i].sass_matches_expected
```

Thus a headline Peak always represents the fastest successful documented benchmark case in that category, independent of SASS expectation matching; the SASS audit result is attached to the winning measurement so a reviewer can immediately see whether the PTX -> SASS lowering matched the current expectation.

## Expected SM120 SASS families

NVIDIA does not publish a normative SM120 SASS ISA specification equivalent to PTX, so these names are treated as **audit expectations**, not as the source of truth for PTX legality.

| PTX path | Expected SASS structural family |
|---|---|
| FP8 / `kind::f8f6f4`, dense | `QMMA.168K.<acc>.<A>.<B>` |
| `kind::mxf8f6f4.block_scale`, dense | `QMMA.SF.16832.F32.<A>.<B>...` |
| `kind::mxf4` / `kind::mxf4nvf4`, dense | `OMMA.SF.16864.F32.E2M1.E2M1...` |
| sparse FP8 / `kind::f8f6f4` | `QMMA.SP.168K...` |
| sparse `mxf8f6f4` | `QMMA.SF.SP.16864...` |
| sparse `mxf4` / `mxf4nvf4` | `OMMA.SF.SP.168128...` |

The automated audit currently checks family (`QMMA` vs `OMMA`, plus `SF` / `SP` modifiers), shape, accumulator type, A/B element format, and static Tensor Core MMA count. The complete SASS text and NVIDIA JSON disassembly are always preserved when available so lower-level scale/control details can be audited separately.

## Why record the static MMA count

The hot PTX body contains `chains x inner_unroll` MMA instructions. Each accumulator chain begins from a different C value and remains observable after the loop, reducing the opportunity for legal common-subexpression elimination.

The SASS audit records whether the expected number and class of Tensor Core MMA instructions survived lowering. A mismatch is useful diagnostic evidence, but it does not suppress the actual measured throughput; this is important because the SASS expectation itself is not a normative NVIDIA contract.

NVIDIA references for the CUBIN/disassembly mechanisms:

- https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html
- https://docs.nvidia.com/cuda/cuda-binary-utilities/
