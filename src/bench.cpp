#include <cuda.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr unsigned kInnerUnroll = 4;

enum class DocClass { Documented, DocAmbiguous, Undocumented };

struct Options {
  int device = 0;
  unsigned iters = 4000;
  unsigned blocks_per_sm = 4;
  unsigned chains = 8;
  unsigned repeats = 5;
  std::string filter;
  bool include_probes = false;
  bool probes_only = false;
  bool all_operands = false;
  bool list_only = false;
  bool verbose_jit = false;
};

struct Case {
  std::string name;
  std::string opcode;
  std::string note;
  DocClass doc = DocClass::Documented;
  bool sparse = false;
  bool block_scale = false;
  bool acc_f16 = false;
  bool custom_ptx = false;
  std::string custom_source;
  int m = 16;
  int n = 8;
  int k = 32;
  int a_regs = 4;
  int b_regs = 2;
  int sparse_selector = 0;
  int scale_vec = 0;  // 0, 1, 2, 4; 0 means no block scaling.
  int byte_a = 0;
  int thread_a = 0;
  int byte_b = 0;
  int thread_b = 0;
};

const char* doc_name(DocClass d) {
  switch (d) {
    case DocClass::Documented: return "DOCUMENTED";
    case DocClass::DocAmbiguous: return "DOC_AMBIGUOUS";
    default: return "UNDOCUMENTED";
  }
}

std::string cuda_error(CUresult r) {
  const char* name = nullptr;
  const char* text = nullptr;
  cuGetErrorName(r, &name);
  cuGetErrorString(r, &text);
  std::ostringstream os;
  os << (name ? name : "CUDA_ERROR") << ": " << (text ? text : "unknown");
  return os.str();
}

void check(CUresult r, const char* what) {
  if (r != CUDA_SUCCESS) {
    throw std::runtime_error(std::string(what) + ": " + cuda_error(r));
  }
}

std::string reg_tuple(const char* prefix, int first, int count) {
  std::ostringstream os;
  os << "{";
  for (int i = 0; i < count; ++i) {
    if (i) os << ", ";
    os << "%" << prefix << (first + i);
  }
  os << "}";
  return os.str();
}

Case make_case(std::string name, std::string opcode, int k, int a_regs, int b_regs,
               bool f16 = false, bool sparse = false, int scale_vec = 0,
               DocClass doc = DocClass::Documented, std::string note = {}) {
  Case c;
  c.name = std::move(name);
  c.opcode = std::move(opcode);
  c.k = k;
  c.a_regs = a_regs;
  c.b_regs = b_regs;
  c.acc_f16 = f16;
  c.sparse = sparse;
  c.block_scale = scale_vec != 0;
  c.scale_vec = scale_vec;
  c.doc = doc;
  c.note = std::move(note);
  return c;
}

std::string tcgen05_sm120_probe() {
  // Deliberately target SM120 with a tcgen05 MMA spelling. The purpose is to
  // isolate the target-family boundary: tcgen05 MMA is documented for the
  // SM100/SM110 family, not SM120.
  return R"PTX(.version 9.1
.target sm_120a
.address_size 64

.visible .entry bench(.param .u64 out_ptr, .param .u32 iters) {
  .reg .u32 %taddr;
  .reg .b32 %idesc, %m0, %m1, %m2, %m3;
  .reg .b64 %adesc, %bdesc;
  .reg .pred %p;
  mov.u32 %taddr, 0;
  mov.b64 %adesc, 0;
  mov.b64 %bdesc, 0;
  mov.b32 %idesc, 0;
  mov.b32 %m0, 0;
  mov.b32 %m1, 0;
  mov.b32 %m2, 0;
  mov.b32 %m3, 0;
  setp.eq.u32 %p, %taddr, %taddr;
  tcgen05.mma.cta_group::1.kind::f8f6f4
      [%taddr], %adesc, %bdesc, %idesc, {%m0, %m1, %m2, %m3}, %p;
  ret;
})PTX";
}

std::string build_ptx(const Case& c, unsigned chains) {
  if (c.custom_ptx) return c.custom_source;

  const int acc_regs = c.acc_f16 ? 2 : 4;
  static const char* kF32Seeds[8] = {
      "0f3f800000", "0f40000000", "0f40400000", "0f40800000",
      "0f40a00000", "0f40c00000", "0f40e00000", "0f41000000"};
  static const char* kF16x2Seeds[8] = {
      "0x3c003c00", "0x40004000", "0x42004200", "0x44004400",
      "0x45004500", "0x46004600", "0x47004700", "0x48004800"};

  std::ostringstream p;
  p << ".version 9.1\n"
    << ".target sm_120a\n"
    << ".address_size 64\n\n"
    << ".visible .entry bench(.param .u64 out_ptr, .param .u32 iters) {\n"
    << "  .reg .pred %done, %skip;\n"
    << "  .reg .u32 %i, %lim, %lane, %tid, %cta, %ntid, %warp, %wpb, %wg;\n"
    << "  .reg .u64 %out, %off, %addr;\n"
    << "  .reg .b32 %a<" << c.a_regs << ">, %b<" << c.b_regs << ">;\n";

  if (c.acc_f16) {
    p << "  .reg .b32 %d<" << (chains * acc_regs) << ">, %live;\n";
  } else {
    p << "  .reg .f32 %d<" << (chains * acc_regs) << ">, %live;\n";
  }
  if (c.sparse) p << "  .reg .b32 %meta;\n";
  if (c.block_scale) p << "  .reg .b32 %sa, %sb;\n";

  p << "  ld.param.u64 %out, [out_ptr];\n"
    << "  ld.param.u32 %lim, [iters];\n";

  // These byte patterns are finite low-magnitude values for the FP8 formats
  // and obey the documented padding positions of kind::f8f6f4 FP6/FP4.
  for (int i = 0; i < c.a_regs; ++i) p << "  mov.b32 %a" << i << ", 0x14141414;\n";
  for (int i = 0; i < c.b_regs; ++i) p << "  mov.b32 %b" << i << ", 0x0c0c0c0c;\n";

  // Initialize every accumulator register, then seed the first register of
  // each chain differently. This makes the chains semantically non-equivalent
  // and prevents a legal common-subexpression optimization from collapsing
  // multiple identical MMA chains into one.
  for (unsigned r = 0; r < chains * static_cast<unsigned>(acc_regs); ++r) {
    if (c.acc_f16) p << "  mov.b32 %d" << r << ", 0;\n";
    else p << "  mov.f32 %d" << r << ", 0f00000000;\n";
  }
  for (unsigned chain = 0; chain < chains; ++chain) {
    const unsigned r = chain * static_cast<unsigned>(acc_regs);
    if (c.acc_f16) p << "  mov.b32 %d" << r << ", " << kF16x2Seeds[chain] << ";\n";
    else p << "  mov.f32 %d" << r << ", " << kF32Seeds[chain] << ";\n";
  }

  // Repeating nibble 0b0100 encodes increasing ordered-metadata indices.
  if (c.sparse) p << "  mov.b32 %meta, 0x44444444;\n";

  // 0x38 avoids the reserved NaN encodings of both UE8M0 (0xff) and UE4M3
  // (0x7f), so one source value works for every documented scale format.
  if (c.block_scale) {
    p << "  mov.b32 %sa, 0x38383838;\n"
      << "  mov.b32 %sb, 0x38383838;\n";
  }

  const std::string a = reg_tuple("a", 0, c.a_regs);
  const std::string b = reg_tuple("b", 0, c.b_regs);

  p << "  mov.u32 %i, 0;\n"
    << "LOOP:\n"
    << "  setp.ge.u32 %done, %i, %lim;\n"
    << "  @%done bra DONE;\n";

  for (unsigned u = 0; u < kInnerUnroll; ++u) {
    for (unsigned chain = 0; chain < chains; ++chain) {
      const int base = static_cast<int>(chain) * acc_regs;
      const std::string d = reg_tuple("d", base, acc_regs);
      p << "  " << c.opcode << "\n"
        << "    " << d << ", " << a << ", " << b << ", " << d;
      if (c.sparse) p << ", %meta, " << c.sparse_selector;
      if (c.block_scale) {
        p << ", %sa, {" << c.byte_a << ", " << c.thread_a << "}"
          << ", %sb, {" << c.byte_b << ", " << c.thread_b << "}";
      }
      p << ";\n";
    }
  }

  p << "  add.u32 %i, %i, 1;\n"
    << "  bra LOOP;\n"
    << "DONE:\n";

  // Keep one accumulator register from every chain observable. Since each MMA
  // is a multi-result warp instruction and the chains have distinct C inputs,
  // all repeated MMA chains remain semantically required.
  if (c.acc_f16) {
    p << "  mov.b32 %live, %d0;\n";
    for (unsigned chain = 1; chain < chains; ++chain) {
      p << "  xor.b32 %live, %live, %d" << (chain * acc_regs) << ";\n";
    }
  } else {
    p << "  mov.f32 %live, %d0;\n";
    for (unsigned chain = 1; chain < chains; ++chain) {
      p << "  add.f32 %live, %live, %d" << (chain * acc_regs) << ";\n";
    }
  }

  p << "  mov.u32 %lane, %laneid;\n"
    << "  setp.ne.u32 %skip, %lane, 0;\n"
    << "  @%skip bra EXIT;\n"
    << "  mov.u32 %tid, %tid.x;\n"
    << "  mov.u32 %cta, %ctaid.x;\n"
    << "  mov.u32 %ntid, %ntid.x;\n"
    << "  shr.u32 %warp, %tid, 5;\n"
    << "  shr.u32 %wpb, %ntid, 5;\n"
    << "  mad.lo.u32 %wg, %cta, %wpb, %warp;\n"
    << "  mul.wide.u32 %off, %wg, 4;\n"
    << "  add.u64 %addr, %out, %off;\n";
  if (c.acc_f16) p << "  st.global.b32 [%addr], %live;\n";
  else p << "  st.global.f32 [%addr], %live;\n";
  p << "EXIT:\n"
    << "  ret;\n"
    << "}\n";
  return p.str();
}

std::vector<Case> base_manifest() {
  std::vector<Case> v;
  const std::vector<std::string> f8 = {"e4m3", "e5m2"};
  const std::vector<std::string> f864 = {"e4m3", "e5m2", "e3m2", "e2m3", "e2m1"};
  const std::vector<std::string> acc = {"f16", "f32"};

  // Legacy dense FP8: both K shapes, both accumulator types, independent A/B formats.
  for (int k : {16, 32}) {
    for (const auto& a : f8) for (const auto& b : f8) for (const auto& d : acc) {
      const bool h = d == "f16";
      v.push_back(make_case(
          "dense/fp8/k" + std::to_string(k) + "/" + a + "x" + b + "/acc-" + d,
          "mma.sync.aligned.m16n8k" + std::to_string(k) + ".row.col." + d + "." + a + "." + b + "." + d,
          k, k == 16 ? 2 : 4, k == 16 ? 1 : 2, h));
    }
  }

  // SM120 unscaled F8/F6/F4: normative shape is m16n8k32.
  for (const auto& a : f864) for (const auto& b : f864) for (const auto& d : acc) {
    const bool h = d == "f16";
    v.push_back(make_case(
        "dense/f8f6f4/" + a + "x" + b + "/acc-" + d,
        "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4." + d + "." + a + "." + b + "." + d,
        32, 4, 2, h));
  }

  // Dense MXF8/F6/F4: explicit 1X and documented omitted-default spelling.
  for (const auto& a : f864) for (const auto& b : f864) {
    const std::string prefix = "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale";
    v.push_back(make_case(
        "dense/mxf8f6f4/" + a + "x" + b + "/ue8m0-1X",
        prefix + ".scale_vec::1X.f32." + a + "." + b + ".f32.ue8m0",
        32, 4, 2, false, false, 1));
    v.push_back(make_case(
        "dense/mxf8f6f4/" + a + "x" + b + "/ue8m0-default-1X",
        prefix + ".f32." + a + "." + b + ".f32.ue8m0",
        32, 4, 2, false, false, 1));
  }

  // Dense MXF4: explicit 2X and documented omitted-default spelling.
  v.push_back(make_case(
      "dense/mxf4/e2m1xe2m1/ue8m0-2X",
      "mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0",
      64, 4, 2, false, false, 2));
  v.push_back(make_case(
      "dense/mxf4/e2m1xe2m1/ue8m0-default-2X",
      "mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.f32.e2m1.e2m1.f32.ue8m0",
      64, 4, 2, false, false, 2));

  // Dense MXF4NVF4 valid rows of the PTX block-scaling table.
  v.push_back(make_case(
      "dense/mxf4nvf4/e2m1xe2m1/ue8m0-2X",
      "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0",
      64, 4, 2, false, false, 2));
  v.push_back(make_case(
      "dense/mxf4nvf4/e2m1xe2m1/ue8m0-4X",
      "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue8m0",
      64, 4, 2, false, false, 4));
  v.push_back(make_case(
      "dense/mxf4nvf4/e2m1xe2m1/ue4m3-4X",
      "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3",
      64, 4, 2, false, false, 4));

  // Normative legacy sparse FP8 syntax: m16n8k64, F32, both metadata variants.
  for (const char* sp_c : {"sp", "sp::ordered_metadata"}) {
    const std::string sp = sp_c;
    for (const auto& a : f8) for (const auto& b : f8) {
      v.push_back(make_case(
          "sparse/fp8/" + sp + "/k64/" + a + "x" + b + "/acc-f32",
          "mma." + sp + ".sync.aligned.m16n8k64.row.col.f32." + a + "." + b + ".f32",
          64, 4, 4, false, true));
    }
  }

  // NVIDIA ISA Notes state an SM120 FP8 sparse k32/F16 feature, but the current
  // normative sparse syntax does not contain that production. Keep it visible
  // as DOC_AMBIGUOUS rather than silently treating prose as grammar.
  for (const char* sp_c : {"sp", "sp::ordered_metadata"}) {
    const std::string sp = sp_c;
    for (const auto& a : f8) for (const auto& b : f8) {
      v.push_back(make_case(
          "probe/doc-ambiguous/sparse-fp8/" + sp + "/k32/" + a + "x" + b + "/acc-f16",
          "mma." + sp + ".sync.aligned.m16n8k32.row.col.f16." + a + "." + b + ".f16",
          32, 2, 2, true, true, 0, DocClass::DocAmbiguous,
          "PTX ISA Notes claim SM120 support, but current normative sparse FP8 syntax omits k32/f16"));
    }
  }

  // Sparse kind::f8f6f4 is normatively ordered-metadata-only.
  for (const auto& a : f864) for (const auto& b : f864) for (const auto& d : acc) {
    const bool h = d == "f16";
    v.push_back(make_case(
        "sparse/f8f6f4/sp::ordered_metadata/" + a + "x" + b + "/acc-" + d,
        "mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.kind::f8f6f4." + d + "." + a + "." + b + "." + d,
        64, 4, 4, h, true));
  }

  // Sparse block scaling is normatively ordered-metadata-only.
  for (const auto& a : f864) for (const auto& b : f864) {
    const std::string prefix = "mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.kind::mxf8f6f4.block_scale";
    v.push_back(make_case(
        "sparse/mxf8f6f4/" + a + "x" + b + "/ue8m0-1X",
        prefix + ".scale_vec::1X.f32." + a + "." + b + ".f32.ue8m0",
        64, 4, 4, false, true, 1));
    v.push_back(make_case(
        "sparse/mxf8f6f4/" + a + "x" + b + "/ue8m0-default-1X",
        prefix + ".f32." + a + "." + b + ".f32.ue8m0",
        64, 4, 4, false, true, 1));
  }

  v.push_back(make_case(
      "sparse/mxf4/e2m1xe2m1/ue8m0-2X",
      "mma.sp::ordered_metadata.sync.aligned.m16n8k128.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0",
      128, 4, 4, false, true, 2));
  v.push_back(make_case(
      "sparse/mxf4/e2m1xe2m1/ue8m0-default-2X",
      "mma.sp::ordered_metadata.sync.aligned.m16n8k128.row.col.kind::mxf4.block_scale.f32.e2m1.e2m1.f32.ue8m0",
      128, 4, 4, false, true, 2));
  v.push_back(make_case(
      "sparse/mxf4nvf4/e2m1xe2m1/ue8m0-2X",
      "mma.sp::ordered_metadata.sync.aligned.m16n8k128.row.col.kind::mxf4nvf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0",
      128, 4, 4, false, true, 2));
  v.push_back(make_case(
      "sparse/mxf4nvf4/e2m1xe2m1/ue8m0-4X",
      "mma.sp::ordered_metadata.sync.aligned.m16n8k128.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue8m0",
      128, 4, 4, false, true, 4));
  v.push_back(make_case(
      "sparse/mxf4nvf4/e2m1xe2m1/ue4m3-4X",
      "mma.sp::ordered_metadata.sync.aligned.m16n8k128.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3",
      128, 4, 4, false, true, 4));

  // ---- Plausible but not normatively defined ----
  Case tcgen;
  tcgen.name = "probe/undocumented/tcgen05-on-sm120";
  tcgen.doc = DocClass::Undocumented;
  tcgen.custom_ptx = true;
  tcgen.custom_source = tcgen05_sm120_probe();
  tcgen.note = "tcgen05 MMA is documented for the SM100/SM110 family, not SM120";
  v.push_back(std::move(tcgen));

  for (int k : {16, 64}) {
    v.push_back(make_case(
        "probe/undocumented/dense-f8f6f4-k" + std::to_string(k),
        "mma.sync.aligned.m16n8k" + std::to_string(k) + ".row.col.kind::f8f6f4.f32.e3m2.e2m3.f32",
        k, k == 16 ? 2 : 8, k == 16 ? 1 : 4, false, false, 0, DocClass::Undocumented,
        "dense kind::f8f6f4 grammar is fixed to m16n8k32"));
  }

  for (const char* t_c : {"e3m2", "e2m3", "e2m1"}) {
    const std::string t = t_c;
    v.push_back(make_case(
        "probe/undocumented/no-kind-" + t,
        "mma.sync.aligned.m16n8k32.row.col.f32." + t + "." + t + ".f32",
        32, 4, 2, false, false, 0, DocClass::Undocumented,
        "SM120 FP6/FP4 unscaled warp-MMA syntax is kind::f8f6f4-qualified"));
  }

  v.push_back(make_case(
      "probe/undocumented/mxf8f6f4-ue8m0-2X",
      "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::2X.f32.e4m3.e4m3.f32.ue8m0",
      32, 4, 2, false, false, 2, DocClass::Undocumented, "valid-combination table fixes mxf8f6f4 to 1X"));
  v.push_back(make_case(
      "probe/undocumented/mxf8f6f4-ue4m3-1X",
      "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue4m3",
      32, 4, 2, false, false, 1, DocClass::Undocumented, "valid-combination table fixes mxf8f6f4 scale type to ue8m0"));
  v.push_back(make_case(
      "probe/undocumented/mxf4-ue8m0-1X",
      "mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.scale_vec::1X.f32.e2m1.e2m1.f32.ue8m0",
      64, 4, 2, false, false, 1, DocClass::Undocumented, "valid-combination table fixes mxf4 to 2X"));
  v.push_back(make_case(
      "probe/undocumented/mxf4-ue4m3-2X",
      "mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue4m3",
      64, 4, 2, false, false, 2, DocClass::Undocumented, "ue4m3 scale is not defined for mxf4"));
  v.push_back(make_case(
      "probe/undocumented/mxf4nvf4-ue4m3-2X",
      "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue4m3",
      64, 4, 2, false, false, 2, DocClass::Undocumented, "ue4m3 is paired only with 4X for mxf4nvf4"));
  v.push_back(make_case(
      "probe/undocumented/mxf4nvf4-missing-scale-vec",
      "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.f32.e2m1.e2m1.f32.ue8m0",
      64, 4, 2, false, false, 2, DocClass::Undocumented, "scale_vec is mandatory for mxf4nvf4"));
  v.push_back(make_case(
      "probe/undocumented/block-scale-f16-acc",
      "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f16.e4m3.e4m3.f16.ue8m0",
      32, 4, 2, true, false, 1, DocClass::Undocumented, "block-scaled grammar fixes C/D to f32"));

  v.push_back(make_case(
      "probe/undocumented/sparse-f8f6f4-plain-sp",
      "mma.sp.sync.aligned.m16n8k64.row.col.kind::f8f6f4.f32.e3m2.e2m3.f32",
      64, 4, 4, false, true, 0, DocClass::Undocumented, "kind::f8f6f4 sparse grammar is ordered-metadata-only"));
  v.push_back(make_case(
      "probe/undocumented/sparse-mxf8f6f4-plain-sp",
      "mma.sp.sync.aligned.m16n8k64.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0",
      64, 4, 4, false, true, 1, DocClass::Undocumented, "block-scaled sparse grammar is ordered-metadata-only"));
  v.push_back(make_case(
      "probe/undocumented/sparse-mxf4-plain-sp",
      "mma.sp.sync.aligned.m16n8k128.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0",
      128, 4, 4, false, true, 2, DocClass::Undocumented, "block-scaled sparse grammar is ordered-metadata-only"));
  v.push_back(make_case(
      "probe/undocumented/f8f6f4-row-row",
      "mma.sync.aligned.m16n8k32.row.row.kind::f8f6f4.f32.e4m3.e4m3.f32",
      32, 4, 2, false, false, 0, DocClass::Undocumented, "low-precision warp-MMA grammar is fixed to row.col"));

  return v;
}

std::vector<int> byte_values(int scale_vec) {
  if (scale_vec == 1) return {0, 1, 2, 3};
  if (scale_vec == 2) return {0, 2};
  if (scale_vec == 4) return {0};
  return {0};
}

std::vector<Case> expand_documented_operands(const std::vector<Case>& base, bool all_operands) {
  if (!all_operands) return base;

  std::vector<Case> out;
  for (const Case& c : base) {
    // Keep ambiguous/undocumented probes canonical: their operand space is not
    // itself documented and must not be extrapolated.
    if (c.doc != DocClass::Documented) {
      out.push_back(c);
      continue;
    }

    const std::vector<int> sparse_f = c.sparse ? std::vector<int>{0, 1, 2, 3} : std::vector<int>{0};
    if (!c.block_scale) {
      for (int f : sparse_f) {
        Case x = c;
        x.sparse_selector = f;
        if (c.sparse) x.name += "/sp-f" + std::to_string(f);
        out.push_back(std::move(x));
      }
      continue;
    }

    const std::vector<int> ba = byte_values(c.scale_vec);
    const std::vector<int> bb = byte_values(c.scale_vec);
    const std::vector<int> ta = {0, 1};
    const std::vector<int> tb = {0, 1, 2, 3};
    for (int f : sparse_f) for (int a_byte : ba) for (int a_thread : ta)
      for (int b_byte : bb) for (int b_thread : tb) {
        Case x = c;
        x.sparse_selector = f;
        x.byte_a = a_byte;
        x.thread_a = a_thread;
        x.byte_b = b_byte;
        x.thread_b = b_thread;
        x.name += "/selA-" + std::to_string(a_byte) + "-" + std::to_string(a_thread)
                + "/selB-" + std::to_string(b_byte) + "-" + std::to_string(b_thread);
        if (c.sparse) x.name += "/sp-f" + std::to_string(f);
        out.push_back(std::move(x));
      }
  }
  return out;
}

Options parse_args(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto value = [&](const char* opt) -> std::string {
      if (++i >= argc) throw std::runtime_error(std::string("missing value for ") + opt);
      return argv[i];
    };
    if (a == "--device") o.device = std::stoi(value("--device"));
    else if (a == "--iters") o.iters = static_cast<unsigned>(std::stoul(value("--iters")));
    else if (a == "--blocks-per-sm") o.blocks_per_sm = static_cast<unsigned>(std::stoul(value("--blocks-per-sm")));
    else if (a == "--chains") o.chains = static_cast<unsigned>(std::stoul(value("--chains")));
    else if (a == "--repeats") o.repeats = static_cast<unsigned>(std::stoul(value("--repeats")));
    else if (a == "--filter") o.filter = value("--filter");
    else if (a == "--include-probes") o.include_probes = true;
    else if (a == "--probes-only") { o.include_probes = true; o.probes_only = true; }
    else if (a == "--all-operands") o.all_operands = true;
    else if (a == "--list") o.list_only = true;
    else if (a == "--verbose-jit") o.verbose_jit = true;
    else if (a == "-h" || a == "--help") {
      std::cout
          << "--device N --iters N --blocks-per-sm N --chains {1,2,4,8} --repeats N\n"
          << "--filter TEXT --include-probes --probes-only --all-operands --list --verbose-jit\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown option: " + a);
    }
  }
  if (!(o.chains == 1 || o.chains == 2 || o.chains == 4 || o.chains == 8)) {
    throw std::runtime_error("--chains must be one of 1,2,4,8");
  }
  if (!o.iters || !o.blocks_per_sm || !o.repeats) {
    throw std::runtime_error("iteration/work counts must be non-zero");
  }
  return o;
}

struct LoadedModule {
  CUmodule module = nullptr;
  CUfunction function = nullptr;
  std::string log;
};

CUresult load_ptx(const std::string& ptx, LoadedModule& out) {
  char error_log[16384] = {};
  char info_log[16384] = {};
  CUjit_option options[] = {
      CU_JIT_ERROR_LOG_BUFFER,
      CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
      CU_JIT_INFO_LOG_BUFFER,
      CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
      CU_JIT_LOG_VERBOSE,
      CU_JIT_OPTIMIZATION_LEVEL};
  void* values[] = {
      error_log,
      reinterpret_cast<void*>(static_cast<uintptr_t>(sizeof(error_log))),
      info_log,
      reinterpret_cast<void*>(static_cast<uintptr_t>(sizeof(info_log))),
      reinterpret_cast<void*>(static_cast<uintptr_t>(1)),
      reinterpret_cast<void*>(static_cast<uintptr_t>(4))};

  CUresult r = cuModuleLoadDataEx(&out.module, ptx.c_str(), 6, options, values);
  out.log = std::string(error_log) + std::string(info_log);
  if (r != CUDA_SUCCESS) return r;

  r = cuModuleGetFunction(&out.function, out.module, "bench");
  if (r != CUDA_SUCCESS) {
    cuModuleUnload(out.module);
    out.module = nullptr;
  }
  return r;
}

void unload(LoadedModule& m) {
  if (m.module) cuModuleUnload(m.module);
  m = {};
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options opt = parse_args(argc, argv);
    const std::vector<Case> cases = expand_documented_operands(base_manifest(), opt.all_operands);

    if (opt.list_only) {
      size_t shown = 0;
      for (const Case& c : cases) {
        if (!opt.filter.empty() && c.name.find(opt.filter) == std::string::npos) continue;
        if (c.doc != DocClass::Documented && !opt.include_probes) continue;
        if (opt.probes_only && c.doc == DocClass::Documented) continue;
        std::cout << doc_name(c.doc) << " " << c.name << " :: "
                  << (c.custom_ptx ? "<custom PTX>" : c.opcode) << "\n";
        ++shown;
      }
      std::cout << "listed=" << shown << " expanded_manifest=" << cases.size() << "\n";
      return 0;
    }

    check(cuInit(0), "cuInit");
    CUdevice dev;
    check(cuDeviceGet(&dev, opt.device), "cuDeviceGet");

    char device_name[256] = {};
    int major = 0, minor = 0, sms = 0, clock_khz = 0, driver_version = 0;
    check(cuDeviceGetName(device_name, sizeof(device_name), dev), "cuDeviceGetName");
    check(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev), "CC major");
    check(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev), "CC minor");
    check(cuDeviceGetAttribute(&sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev), "SM count");
    check(cuDeviceGetAttribute(&clock_khz, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, dev), "clock rate");
    check(cuDriverGetVersion(&driver_version), "driver version");

    std::cout << "device=\"" << device_name << "\" cc=" << major << "." << minor
              << " sm_count=" << sms
              << " reported_clock_mhz=" << (clock_khz / 1000.0)
              << " driver_api=" << driver_version
              << " ptx=9.1 target=sm_120a"
              << " inner_unroll=" << kInnerUnroll << "\n";

    if (major != 12 || minor != 0) {
      std::cerr << "out-of-scope device: this benchmark is for SM120\n";
      return 2;
    }

    CUcontext ctx;
    check(cuDevicePrimaryCtxRetain(&ctx, dev), "cuDevicePrimaryCtxRetain");
    check(cuCtxSetCurrent(ctx), "cuCtxSetCurrent");

    constexpr unsigned threads = 256;
    constexpr unsigned warps_per_block = threads / 32;
    const unsigned blocks = static_cast<unsigned>(sms) * opt.blocks_per_sm;
    CUdeviceptr output = 0;
    check(cuMemAlloc(&output, static_cast<size_t>(blocks) * warps_per_block * sizeof(uint32_t)), "cuMemAlloc");

    CUevent start, stop;
    check(cuEventCreate(&start, CU_EVENT_DEFAULT), "cuEventCreate(start)");
    check(cuEventCreate(&stop, CU_EVENT_DEFAULT), "cuEventCreate(stop)");

    size_t selected = 0, documented = 0, probes = 0;
    for (const Case& c : cases) {
      if (!opt.filter.empty() && c.name.find(opt.filter) == std::string::npos) continue;
      if (c.doc != DocClass::Documented && !opt.include_probes) continue;
      if (opt.probes_only && c.doc == DocClass::Documented) continue;

      ++selected;
      if (c.doc == DocClass::Documented) ++documented;
      else ++probes;

      LoadedModule mod;
      const CUresult jit_result = load_ptx(build_ptx(c, opt.chains), mod);
      if (jit_result != CUDA_SUCCESS) {
        const char* tag = c.doc == DocClass::Documented ? "FAIL_DOCUMENTED" :
                          c.doc == DocClass::DocAmbiguous ? "REJECTED_DOC_AMBIGUOUS" :
                                                           "REJECTED_UNDOCUMENTED";
        std::cout << tag << " name=" << c.name << " error=\"" << cuda_error(jit_result) << "\"";
        if (!c.note.empty()) std::cout << " note=\"" << c.note << "\"";
        std::cout << "\n";
        if (opt.verbose_jit && !mod.log.empty()) std::cout << mod.log << "\n";
        continue;
      }

      // Never execute unspecified/ambiguous instructions merely because one
      // toolchain accepted them. Acceptance is recorded as an observation only.
      if (c.doc != DocClass::Documented) {
        std::cout << (c.doc == DocClass::DocAmbiguous ? "ACCEPTED_DOC_AMBIGUOUS" : "ACCEPTED_UNDOCUMENTED")
                  << " name=" << c.name;
        if (!c.note.empty()) std::cout << " note=\"" << c.note << "\"";
        std::cout << "\n";
        if (opt.verbose_jit && !mod.log.empty()) std::cout << mod.log << "\n";
        unload(mod);
        continue;
      }

      unsigned warm_iters = std::min(opt.iters, 256u);
      void* warm_args[] = {&output, &warm_iters};
      check(cuLaunchKernel(mod.function, blocks, 1, 1, threads, 1, 1, 0, nullptr, warm_args, nullptr), "warmup launch");
      check(cuCtxSynchronize(), "warmup synchronize");

      float best_ms = std::numeric_limits<float>::infinity();
      float sum_ms = 0.0f;
      for (unsigned rep = 0; rep < opt.repeats; ++rep) {
        unsigned timed_iters = opt.iters;
        void* args[] = {&output, &timed_iters};
        check(cuEventRecord(start, nullptr), "event start");
        check(cuLaunchKernel(mod.function, blocks, 1, 1, threads, 1, 1, 0, nullptr, args, nullptr), "timed launch");
        check(cuEventRecord(stop, nullptr), "event stop");
        check(cuEventSynchronize(stop), "event synchronize");
        float ms = 0.0f;
        check(cuEventElapsedTime(&ms, start, stop), "event elapsed");
        best_ms = std::min(best_ms, ms);
        sum_ms += ms;
      }

      int regs_per_thread = 0;
      check(cuFuncGetAttribute(&regs_per_thread, CU_FUNC_ATTRIBUTE_NUM_REGS, mod.function), "register count");

      const long double warp_count = static_cast<long double>(blocks) * warps_per_block;
      const long double instruction_count = warp_count * opt.iters * opt.chains * kInnerUnroll;
      const long double flops_per_instruction = 2.0L * c.m * c.n * c.k;
      const long double best_seconds = static_cast<long double>(best_ms) / 1000.0L;
      const long double mean_seconds = (static_cast<long double>(sum_ms) / opt.repeats) / 1000.0L;
      const long double peak_logical = instruction_count * flops_per_instruction / best_seconds / 1.0e12L;
      const long double mean_logical = instruction_count * flops_per_instruction / mean_seconds / 1.0e12L;

      std::cout << std::fixed << std::setprecision(3)
                << "PASS name=" << c.name
                << " best_ms=" << best_ms
                << " peak_logical_tflops=" << static_cast<double>(peak_logical)
                << " mean_logical_tflops=" << static_cast<double>(mean_logical);
      if (c.sparse) {
        std::cout << " peak_nonzero_tflops=" << static_cast<double>(peak_logical * 0.5L);
      }
      std::cout << " regs_per_thread=" << regs_per_thread
                << " blocks=" << blocks
                << " threads=" << threads
                << " chains=" << opt.chains
                << " inner_unroll=" << kInnerUnroll
                << " iters=" << opt.iters
                << " repeats=" << opt.repeats
                << "\n";

      if (opt.verbose_jit && !mod.log.empty()) std::cout << mod.log << "\n";
      unload(mod);
    }

    std::cout << "selected=" << selected
              << " documented=" << documented
              << " probes=" << probes
              << " expanded_manifest=" << cases.size()
              << "\n";

    cuEventDestroy(start);
    cuEventDestroy(stop);
    cuMemFree(output);
    cuDevicePrimaryCtxRelease(dev);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "fatal: " << e.what() << "\n";
    return 1;
  }
}
