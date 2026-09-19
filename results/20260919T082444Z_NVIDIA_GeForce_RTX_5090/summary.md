# Tensor Core FP8/FP6/FP4 peak summary

Device: **NVIDIA GeForce RTX 5090** · CC 12.0 · 170 SMs · PTX 9.1 / sm_120a

SASS audit: **enabled**. SASS is observational and never gates execution or headline Peak selection.

## Headline dense peaks (TFLOP/s)

| Precision pair | Without MX | With MX |
|---|---:|---:|
| fp8 | 1010.44 | 1006.07 |
| fp6 | 510.67 | 507.61 |
| fp4 | 510.48 | 2003.39 |
| fp8xfp6 | 1010.80 | 1002.07 |
| fp8xfp4 | 1010.61 | 1001.92 |
| fp6xfp4 | 510.75 | 507.61 |

`with_mx` means the documented MX paths (`mxf8f6f4` / `mxf4`). `mxf4nvf4` is reported separately. Mixed categories take the maximum over both A×B orientations and all documented subformats.

## Winning cases

| Mode | Pair | Peak TFLOP/s | A×B | Acc | SASS audit | PTX case |
|---|---|---:|---|---|---|---|
| without_mx | fp8 | 1010.44 | e5m2 × e4m3 | f16 | `QMMA` / SASS_OK | `dense/f8f6f4/e5m2xe4m3/acc-f16` |
| without_mx | fp6 | 510.67 | e3m2 × e2m3 | f16 | `QMMA` / SASS_OK | `dense/f8f6f4/e3m2xe2m3/acc-f16` |
| without_mx | fp4 | 510.48 | e2m1 × e2m1 | f16 | `QMMA` / SASS_OK | `dense/f8f6f4/e2m1xe2m1/acc-f16` |
| without_mx | fp8xfp6 | 1010.80 | e5m2 × e2m3 | f16 | `QMMA` / SASS_OK | `dense/f8f6f4/e5m2xe2m3/acc-f16` |
| without_mx | fp8xfp4 | 1010.61 | e5m2 × e2m1 | f16 | `QMMA` / SASS_OK | `dense/f8f6f4/e5m2xe2m1/acc-f16` |
| without_mx | fp6xfp4 | 510.75 | e2m3 × e2m1 | f16 | `QMMA` / SASS_OK | `dense/f8f6f4/e2m3xe2m1/acc-f16` |
| with_mx | fp8 | 1006.07 | e4m3 × e4m3 | f32 | `QMMA.SF` / SASS_OK | `dense/mxf8f6f4/e4m3xe4m3/ue8m0-1X` |
| with_mx | fp6 | 507.61 | e2m3 × e2m3 | f32 | `QMMA.SF` / SASS_OK | `dense/mxf8f6f4/e2m3xe2m3/ue8m0-default-1X` |
| with_mx | fp4 | 2003.39 | e2m1 × e2m1 | f32 | `OMMA.SF` / SASS_OK | `dense/mxf4/e2m1xe2m1/ue8m0-default-2X` |
| with_mx | fp8xfp6 | 1002.07 | e5m2 × e3m2 | f32 | `QMMA.SF` / SASS_OK | `dense/mxf8f6f4/e5m2xe3m2/ue8m0-1X` |
| with_mx | fp8xfp4 | 1001.92 | e5m2 × e2m1 | f32 | `QMMA.SF` / SASS_OK | `dense/mxf8f6f4/e5m2xe2m1/ue8m0-1X` |
| with_mx | fp6xfp4 | 507.61 | e2m3 × e2m1 | f32 | `QMMA.SF` / SASS_OK | `dense/mxf8f6f4/e2m3xe2m1/ue8m0-1X` |

## NVFP4 reference

Best NVFP4-family FP4 result: **2004.11 TFLOP/s**, case `dense/mxf4nvf4/e2m1xe2m1/ue4m3-4X`, SASS `OMMA.SF` / SASS_OK.

## Run health

- Timed PASS cases: 232
- SASS_OK: 232
- SASS_MISMATCH (still timed): 0
- Timed cases without usable SASS audit: 0
- Documented cases not reaching timing for non-SASS reasons: 2
- Full machine-readable records: `run.json` and `results.csv`
- Headline machine-readable summary: `peak_summary.json`
- Per-case PTX/CUBIN/SASS: `ptx/`, `cubin/`, `sass/`
