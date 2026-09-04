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
#include <vector>

namespace {

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
  int m = 16, n = 8, k = 32;
  int a_regs = 4, b_regs = 2;
  bool custom_ptx = false;
  std::string ptx;
};

std::string errstr(CUresult r) {
  const char *n = nullptr, *s = nullptr;
  cuGetErrorName(r, &n);
  cuGetErrorString(r, &s);
  std::ostringstream os;
  os << (n ? n : "CUDA_ERROR") << ": " << (s ? s : "unknown");
  return os.str();
}

void ck(CUresult r, const char* what) {
  if (r != CUDA_SUCCESS) throw std::runtime_error(std::string(what) + ": " + errstr(r));
}

const char* doc_name(DocClass d) {
  switch (d) {
    case DocClass::Documented: return "DOCUMENTED";
    case DocClass::DocAmbiguous: return "DOC_AMBIGUOUS";
    default: return "UNDOCUMENTED";
  }
}

std::string regs(const char* p, int first, int count) {
  std::ostringstream os;
  os << "{";
  for (int i = 0; i < count; ++i) {
    if (i) os << ", ";
    os << "%" << p << first + i;
  }
  os << "}";
  return os.str();
}

Case make_case(std::string name, std::string opcode, int k, int ar, int br,
               bool f16 = false, bool sparse = false, bool block = false,
               DocClass doc = DocClass::Documented, std::string note = {}) {
  Case c;
  c.name = std::move(name); c.opcode = std::move(opcode); c.k = k;
  c.a_regs = ar; c.b_regs = br; c.acc_f16 = f16; c.sparse = sparse;
  c.block_scale = block; c.doc = doc; c.note = std::move(note);
  return c;
}

std::string tcgen05_sm120_probe() {
  return R"PTX(.version 9.1
.target sm_120a
.address_size 64
.visible .entry bench(.param .u64 out_ptr, .param .u32 iters) {
  .reg .b32 %taddr, %idesc, %m0, %m1, %m2, %m3;
  .reg .b64 %adesc, %bdesc;
  .reg .pred %p;
  mov.b32 %taddr, 0;
  mov.b64 %adesc, 0;
  mov.b64 %bdesc, 0;
  mov.b32 %idesc, 0;
  mov.b32 %m0, 0; mov.b32 %m1, 0; mov.b32 %m2, 0; mov.b32 %m3, 0;
  setp.eq.u32 %p, %taddr, %taddr;
  tcgen05.mma.cta_group::1.kind::f8f6f4
      [%taddr], %adesc, %bdesc, %idesc, {%m0, %m1, %m2, %m3}, %p;
  ret;
})PTX";
}

std::string build_ptx(const Case& c, unsigned chains) {
  if (c.custom_ptx) return c.ptx;
  const int cr = c.acc_f16 ? 2 : 4;
  std::ostringstream p;
  p << ".version 9.1\n.target sm_120a\n.address_size 64\n\n"
    << ".visible .entry bench(.param .u64 out_ptr, .param .u32 iters) {\n"
    << "  .reg .pred %done, %skip;\n"
    << "  .reg .u32 %i, %lim, %lane, %tid, %cta, %ntid, %warp, %wpb, %wg;\n"
    << "  .reg .u64 %out, %off, %addr;\n"
    << "  .reg .b32 %a<" << c.a_regs << ">, %b<" << c.b_regs << ">;\n";
  if (c.acc_f16) p << "  .reg .b32 %d<" << chains * cr << ">, %live;\n";
  else p << "  .reg .f32 %d<" << chains * cr << ">, %live;\n";
  if (c.sparse) p << "  .reg .b32 %meta;\n";
  if (c.block_scale) p << "  .reg .b32 %sa, %sb;\n";

  p << "  ld.param.u64 %out, [out_ptr];\n  ld.param.u32 %lim, [iters];\n";
  for (int i = 0; i < c.a_regs; ++i) p << "  mov.b32 %a" << i << ", 0x3c3c3c3c;\n";
  for (int i = 0; i < c.b_regs; ++i) p << "  mov.b32 %b" << i << ", 0x34343434;\n";
  for (unsigned q = 0; q < chains * cr; ++q) {
    if (c.acc_f16) p << "  mov.b32 %d" << q << ", 0;\n";
    else p << "  mov.f32 %d" << q << ", 0f00000000;\n";
  }
  if (c.sparse) p << "  mov.b32 %meta, 0x44444444;\n"; // ordered nibble: indices 0,1
  if (c.block_scale) {
    // 0x38 avoids both UE8M0 NaN (0xff) and UE4M3 NaN (0x7f).
    p << "  mov.b32 %sa, 0x38383838;\n  mov.b32 %sb, 0x38383838;\n";
  }

  const std::string a = regs("a", 0, c.a_regs);
  const std::string b = regs("b", 0, c.b_regs);
  p << "  mov.u32 %i, 0;\nL0:\n  setp.ge.u32 %done, %i, %lim;\n  @%done bra L1;\n";
  for (unsigned chain = 0; chain < chains; ++chain) {
    const std::string d = regs("d", static_cast<int>(chain) * cr, cr);
    p << "  " << c.opcode << "\n    " << d << ", " << a << ", " << b << ", " << d;
    if (c.sparse) p << ", %meta, 0";
    if (c.block_scale) p << ", %sa, {0, 0}, %sb, {0, 0}";
    p << ";\n";
  }
  p << "  add.u32 %i, %i, 1;\n  bra L0;\nL1:\n";

  if (c.acc_f16) {
    p << "  mov.b32 %live, %d0;\n";
    for (unsigned chain = 1; chain < chains; ++chain)
      p << "  xor.b32 %live, %live, %d" << chain * cr << ";\n";
  } else {
    p << "  mov.f32 %live, %d0;\n";
    for (unsigned chain = 1; chain < chains; ++chain)
      p << "  add.f32 %live, %live, %d" << chain * cr << ";\n";
  }

  p << "  mov.u32 %lane, %laneid;\n  setp.ne.u32 %skip, %lane, 0;\n  @%skip bra L2;\n"
    << "  mov.u32 %tid, %tid.x; mov.u32 %cta, %ctaid.x; mov.u32 %ntid, %ntid.x;\n"
    << "  shr.u32 %warp, %tid, 5; shr.u32 %wpb, %ntid, 5;\n"
    << "  mad.lo.u32 %wg, %cta, %wpb, %warp; mul.wide.u32 %off, %wg, 4; add.u64 %addr, %out, %off;\n";
  if (c.acc_f16) p << "  st.global.b32 [%addr], %live;\n";
  else p << "  st.global.f32 [%addr], %live;\n";
  p << "L2:\n  ret;\n}\n";
  return p.str();
}

void add_block_dense(std::vector<Case>& v, const std::vector<std::string>& t) {
  for (const auto& a : t) for (const auto& b : t) {
    const std::string base = "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale";
    v.push_back(make_case("dense/mxf8f6f4/"+a+"x"+b+"/ue8m0-1X", base+".scale_vec::1X.f32."+a+"."+b+".f32.ue8m0", 32, 4, 2, false, false, true));
    v.push_back(make_case("dense/mxf8f6f4/"+a+"x"+b+"/ue8m0-default-1X", base+".f32."+a+"."+b+".f32.ue8m0", 32, 4, 2, false, false, true));
  }
  v.push_back(make_case("dense/mxf4/e2m1xe2m1/ue8m0-2X", "mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0", 64, 4, 2, false, false, true));
  v.push_back(make_case("dense/mxf4/e2m1xe2m1/ue8m0-default-2X", "mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.f32.e2m1.e2m1.f32.ue8m0", 64, 4, 2, false, false, true));
  for (const auto& x : std::vector<std::pair<std::string,std::string>>{{"ue8m0-2X","scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0"},{"ue8m0-4X","scale_vec::4X.f32.e2m1.e2m1.f32.ue8m0"},{"ue4m3-4X","scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3"}})
    v.push_back(make_case("dense/mxf4nvf4/e2m1xe2m1/"+x.first, "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale."+x.second, 64, 4, 2, false, false, true));
}

std::vector<Case> manifest() {
  std::vector<Case> v;
  const std::vector<std::string> f8 = {"e4m3", "e5m2"};
  const std::vector<std::string> f864 = {"e4m3", "e5m2", "e3m2", "e2m3", "e2m1"};
  const std::vector<std::string> acc = {"f16", "f32"};

  // Dense legacy FP8: grammar allows k16/k32, independent A/B formats, f16/f32 C/D.
  for (int k : {16, 32}) for (const auto& a : f8) for (const auto& b : f8) for (const auto& d : acc) {
    const bool h = d == "f16";
    v.push_back(make_case("dense/fp8/k"+std::to_string(k)+"/"+a+"x"+b+"/acc-"+d,
      "mma.sync.aligned.m16n8k"+std::to_string(k)+".row.col."+d+"."+a+"."+b+"."+d,
      k, k == 16 ? 2 : 4, k == 16 ? 1 : 2, h));
  }

  // Dense SM120 F8/F6/F4: kind-qualified grammar is fixed to m16n8k32.
  for (const auto& a : f864) for (const auto& b : f864) for (const auto& d : acc) {
    const bool h = d == "f16";
    v.push_back(make_case("dense/f8f6f4/"+a+"x"+b+"/acc-"+d,
      "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4."+d+"."+a+"."+b+"."+d,
      32, 4, 2, h));
  }
  add_block_dense(v, f864);

  // Sparse legacy FP8 grammar explicitly defines m16n8k64 + f32 only.
  for (const std::string sp : {"sp", "sp::ordered_metadata"}) for (const auto& a : f8) for (const auto& b : f8)
    v.push_back(make_case("sparse/fp8/"+sp+"/k64/"+a+"x"+b+"/acc-f32",
      "mma."+sp+".sync.aligned.m16n8k64.row.col.f32."+a+"."+b+".f32", 64, 4, 4, false, true));

  // PTX ISA Notes mention FP8 m16n8k32 + f16 on SM120, while the normative syntax line is absent.
  // Keep the inconsistency visible and probe it, but do not label it documented benchmark coverage.
  for (const std::string sp : {"sp", "sp::ordered_metadata"}) for (const auto& a : f8) for (const auto& b : f8)
    v.push_back(make_case("probe/doc-ambiguous/sparse-fp8/"+sp+"/k32/"+a+"x"+b+"/acc-f16",
      "mma."+sp+".sync.aligned.m16n8k32.row.col.f16."+a+"."+b+".f16", 32, 2, 2, true, true, false,
      DocClass::DocAmbiguous, "ISA Notes claim this feature, but PTX 9.3 sparse syntax does not define the FP8 k32/f16 form"));

  // Sparse kind::f8f6f4 is explicitly ordered-metadata-only.
  for (const auto& a : f864) for (const auto& b : f864) for (const auto& d : acc) {
    const bool h = d == "f16";
    v.push_back(make_case("sparse/f8f6f4/sp::ordered_metadata/"+a+"x"+b+"/acc-"+d,
      "mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.kind::f8f6f4."+d+"."+a+"."+b+"."+d,
      64, 4, 4, h, true));
  }

  // Sparse block-scaled forms are all ordered-metadata-only.
  for (const auto& a : f864) for (const auto& b : f864) {
    const std::string base = "mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.kind::mxf8f6f4.block_scale";
    v.push_back(make_case("sparse/mxf8f6f4/"+a+"x"+b+"/ue8m0-1X", base+".scale_vec::1X.f32."+a+"."+b+".f32.ue8m0", 64, 4, 4, false, true, true));
    v.push_back(make_case("sparse/mxf8f6f4/"+a+"x"+b+"/ue8m0-default-1X", base+".f32."+a+"."+b+".f32.ue8m0", 64, 4, 4, false, true, true));
  }
  v.push_back(make_case("sparse/mxf4/e2m1xe2m1/ue8m0-2X", "mma.sp::ordered_metadata.sync.aligned.m16n8k128.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0", 128, 4, 4, false, true, true));
  v.push_back(make_case("sparse/mxf4/e2m1xe2m1/ue8m0-default-2X", "mma.sp::ordered_metadata.sync.aligned.m16n8k128.row.col.kind::mxf4.block_scale.f32.e2m1.e2m1.f32.ue8m0", 128, 4, 4, false, true, true));
  for (const auto& x : std::vector<std::pair<std::string,std::string>>{{"ue8m0-2X","scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0"},{"ue8m0-4X","scale_vec::4X.f32.e2m1.e2m1.f32.ue8m0"},{"ue4m3-4X","scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3"}})
    v.push_back(make_case("sparse/mxf4nvf4/e2m1xe2m1/"+x.first, "mma.sp::ordered_metadata.sync.aligned.m16n8k128.row.col.kind::mxf4nvf4.block_scale."+x.second, 128, 4, 4, false, true, true));

  // ---- Plausible but not defined by NVIDIA's normative grammar/table ----
  Case tg; tg.name = "probe/undocumented/tcgen05-on-sm120"; tg.doc = DocClass::Undocumented;
  tg.custom_ptx = true; tg.ptx = tcgen05_sm120_probe();
  tg.note = "tcgen05 MMA is exposed to the SM100/SM110 family, not SM120"; v.push_back(std::move(tg));

  // kind::f8f6f4 has only dense k32; k16/k64 look natural but are not defined.
  for (int k : {16, 64})
    v.push_back(make_case("probe/undocumented/dense-f8f6f4-k"+std::to_string(k),
      "mma.sync.aligned.m16n8k"+std::to_string(k)+".row.col.kind::f8f6f4.f32.e3m2.e2m3.f32",
      k, k == 16 ? 2 : 8, k == 16 ? 1 : 4, false, false, false, DocClass::Undocumented,
      "kind::f8f6f4 dense grammar is fixed to m16n8k32"));

  // Sub-FP8 types without kind::f8f6f4.
  for (const std::string t : {"e3m2", "e2m3", "e2m1"})
    v.push_back(make_case("probe/undocumented/no-kind-"+t,
      "mma.sync.aligned.m16n8k32.row.col.f32."+t+"."+t+".f32", 32, 4, 2, false, false, false,
      DocClass::Undocumented, "FP6/FP4 unscaled SM120 syntax is kind::f8f6f4-qualified"));

  // Block scaling table-boundary probes.
  v.push_back(make_case("probe/undocumented/mxf8f6f4-ue8m0-2X", "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::2X.f32.e4m3.e4m3.f32.ue8m0", 32, 4, 2, false, false, true, DocClass::Undocumented, "Table 39 fixes mxf8f6f4 to 1X"));
  v.push_back(make_case("probe/undocumented/mxf8f6f4-ue4m3-1X", "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue4m3", 32, 4, 2, false, false, true, DocClass::Undocumented, "Table 39 fixes mxf8f6f4 scale type to ue8m0"));
  v.push_back(make_case("probe/undocumented/mxf4-ue8m0-1X", "mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.scale_vec::1X.f32.e2m1.e2m1.f32.ue8m0", 64, 4, 2, false, false, true, DocClass::Undocumented, "Table 39 fixes mxf4 to 2X"));
  v.push_back(make_case("probe/undocumented/mxf4-ue4m3-2X", "mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue4m3", 64, 4, 2, false, false, true, DocClass::Undocumented, "ue4m3 scale is not defined for mxf4"));
  v.push_back(make_case("probe/undocumented/mxf4nvf4-ue4m3-2X", "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue4m3", 64, 4, 2, false, false, true, DocClass::Undocumented, "ue4m3 pairs only with 4X for mxf4nvf4"));
  v.push_back(make_case("probe/undocumented/mxf4nvf4-missing-scale-vec", "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.f32.e2m1.e2m1.f32.ue8m0", 64, 4, 2, false, false, true, DocClass::Undocumented, "scale_vec is mandatory for mxf4nvf4"));
  v.push_back(make_case("probe/undocumented/block-scale-f16-acc", "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f16.e4m3.e4m3.f16.ue8m0", 32, 4, 2, true, false, true, DocClass::Undocumented, "block-scaled grammar fixes C/D to f32"));

  // The normative sparse kind-qualified grammars are ordered-metadata-only.
  v.push_back(make_case("probe/undocumented/sparse-f8f6f4-plain-sp", "mma.sp.sync.aligned.m16n8k64.row.col.kind::f8f6f4.f32.e3m2.e2m3.f32", 64, 4, 4, false, true, false, DocClass::Undocumented, "kind::f8f6f4 sparse syntax is ordered-metadata-only"));
  v.push_back(make_case("probe/undocumented/sparse-mxf8f6f4-plain-sp", "mma.sp.sync.aligned.m16n8k64.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0", 64, 4, 4, false, true, true, DocClass::Undocumented, "block-scaled sparse syntax is ordered-metadata-only"));
  v.push_back(make_case("probe/undocumented/sparse-mxf4-plain-sp", "mma.sp.sync.aligned.m16n8k128.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0", 128, 4, 4, false, true, true, DocClass::Undocumented, "block-scaled sparse syntax is ordered-metadata-only"));

  // Other layout spellings exist for other MMA classes, but this low-precision grammar is row.col.
  v.push_back(make_case("probe/undocumented/f8f6f4-row-row", "mma.sync.aligned.m16n8k32.row.row.kind::f8f6f4.f32.e4m3.e4m3.f32", 32, 4, 2, false, false, false, DocClass::Undocumented, "low-precision grammar is fixed to row.col"));

  return v;
}

Options parse(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto val = [&](const char* k) { if (++i >= argc) throw std::runtime_error(std::string("missing value for ")+k); return std::string(argv[i]); };
    if (a == "--device") o.device = std::stoi(val("--device"));
    else if (a == "--iters") o.iters = std::stoul(val("--iters"));
    else if (a == "--blocks-per-sm") o.blocks_per_sm = std::stoul(val("--blocks-per-sm"));
    else if (a == "--chains") o.chains = std::stoul(val("--chains"));
    else if (a == "--repeats") o.repeats = std::stoul(val("--repeats"));
    else if (a == "--filter") o.filter = val("--filter");
    else if (a == "--include-probes") o.include_probes = true;
    else if (a == "--probes-only") { o.include_probes = true; o.probes_only = true; }
    else if (a == "--list") o.list_only = true;
    else if (a == "--verbose-jit") o.verbose_jit = true;
    else if (a == "-h" || a == "--help") {
      std::cout << "--device N --iters N --blocks-per-sm N --chains {1,2,4,8} --repeats N --filter TEXT\n"
                   "--include-probes --probes-only --list --verbose-jit\n";
      std::exit(0);
    } else throw std::runtime_error("unknown option: "+a);
  }
  if (!(o.chains == 1 || o.chains == 2 || o.chains == 4 || o.chains == 8)) throw std::runtime_error("chains must be 1,2,4,8");
  if (!o.iters || !o.blocks_per_sm || !o.repeats) throw std::runtime_error("counts must be non-zero");
  return o;
}

struct Module { CUmodule m = nullptr; CUfunction f = nullptr; std::string log; };

CUresult load_ptx(const std::string& ptx, Module& m) {
  char elog[16384] = {}, ilog[16384] = {};
  CUjit_option op[] = {CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES, CU_JIT_INFO_LOG_BUFFER,
                       CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES, CU_JIT_LOG_VERBOSE, CU_JIT_OPTIMIZATION_LEVEL};
  void* va[] = {elog, reinterpret_cast<void*>(uintptr_t(sizeof(elog))), ilog, reinterpret_cast<void*>(uintptr_t(sizeof(ilog))),
                reinterpret_cast<void*>(uintptr_t(1)), reinterpret_cast<void*>(uintptr_t(4))};
  CUresult r = cuModuleLoadDataEx(&m.m, ptx.c_str(), 6, op, va);
  m.log = std::string(elog) + std::string(ilog);
  if (r == CUDA_SUCCESS) r = cuModuleGetFunction(&m.f, m.m, "bench");
  if (r != CUDA_SUCCESS && m.m) { cuModuleUnload(m.m); m.m = nullptr; }
  return r;
}

void unload(Module& m) { if (m.m) cuModuleUnload(m.m); m = {}; }

} // namespace

int main(int argc, char** argv) {
  try {
    const Options o = parse(argc, argv);
    auto cases = manifest();

    if (o.list_only) {
      for (const auto& c : cases) {
        if (!o.filter.empty() && c.name.find(o.filter) == std::string::npos) continue;
        std::cout << doc_name(c.doc) << " " << c.name << " :: " << (c.custom_ptx ? "<custom PTX>" : c.opcode) << "\n";
      }
      return 0;
    }

    ck(cuInit(0), "cuInit");
    CUdevice dev; ck(cuDeviceGet(&dev, o.device), "cuDeviceGet");
    char dn[256] = {}; int maj=0,min=0,sms=0,clk=0,drv=0;
    ck(cuDeviceGetName(dn, sizeof(dn), dev), "device name");
    ck(cuDeviceGetAttribute(&maj, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev), "cc major");
    ck(cuDeviceGetAttribute(&min, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev), "cc minor");
    ck(cuDeviceGetAttribute(&sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev), "sm count");
    ck(cuDeviceGetAttribute(&clk, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, dev), "clock");
    ck(cuDriverGetVersion(&drv), "driver version");
    std::cout << "device=\"" << dn << "\" cc=" << maj << "." << min << " sm_count=" << sms
              << " reported_clock_mhz=" << clk/1000.0 << " driver_api=" << drv << " ptx=9.1 target=sm_120a\n";
    if (maj != 12 || min != 0) { std::cerr << "out-of-scope device: benchmark is for SM120\n"; return 2; }

    CUcontext ctx; ck(cuDevicePrimaryCtxRetain(&ctx, dev), "retain context"); ck(cuCtxSetCurrent(ctx), "set context");
    const unsigned threads = 256, wpb = threads/32, blocks = unsigned(sms) * o.blocks_per_sm;
    CUdeviceptr out = 0; ck(cuMemAlloc(&out, size_t(blocks)*wpb*sizeof(uint32_t)), "cuMemAlloc");
    CUevent e0,e1; ck(cuEventCreate(&e0, CU_EVENT_DEFAULT), "event0"); ck(cuEventCreate(&e1, CU_EVENT_DEFAULT), "event1");

    size_t selected=0, documented=0, probes=0;
    for (const auto& c : cases) {
      if (!o.filter.empty() && c.name.find(o.filter) == std::string::npos) continue;
      if (c.doc != DocClass::Documented && !o.include_probes) continue;
      if (o.probes_only && c.doc == DocClass::Documented) continue;
      ++selected; if (c.doc == DocClass::Documented) ++documented; else ++probes;

      Module mod; CUresult jr = load_ptx(build_ptx(c, o.chains), mod);
      if (jr != CUDA_SUCCESS) {
        const char* tag = c.doc == DocClass::Documented ? "FAIL_DOCUMENTED" :
                          c.doc == DocClass::DocAmbiguous ? "REJECTED_DOC_AMBIGUOUS" : "REJECTED_UNDOCUMENTED";
        std::cout << tag << " name=" << c.name << " error=\"" << errstr(jr) << "\"";
        if (!c.note.empty()) std::cout << " note=\"" << c.note << "\"";
        std::cout << "\n"; if (o.verbose_jit && !mod.log.empty()) std::cout << mod.log << "\n"; continue;
      }
      if (c.doc != DocClass::Documented) {
        std::cout << (c.doc == DocClass::DocAmbiguous ? "ACCEPTED_DOC_AMBIGUOUS" : "ACCEPTED_UNDOCUMENTED")
                  << " name=" << c.name;
        if (!c.note.empty()) std::cout << " note=\"" << c.note << "\"";
        std::cout << "\n"; if (o.verbose_jit && !mod.log.empty()) std::cout << mod.log << "\n"; unload(mod); continue;
      }

      unsigned warm = std::min(o.iters, 32u); void* wa[] = {&out, &warm};
      ck(cuLaunchKernel(mod.f, blocks,1,1, threads,1,1, 0,nullptr,wa,nullptr), "warmup"); ck(cuCtxSynchronize(), "warmup sync");
      float best = std::numeric_limits<float>::infinity(), sum=0.0f;
      for (unsigned r=0; r<o.repeats; ++r) {
        unsigned it = o.iters; void* ka[] = {&out, &it};
        ck(cuEventRecord(e0, nullptr), "record start");
        ck(cuLaunchKernel(mod.f, blocks,1,1, threads,1,1, 0,nullptr,ka,nullptr), "launch");
        ck(cuEventRecord(e1, nullptr), "record stop"); ck(cuEventSynchronize(e1), "sync stop");
        float ms=0; ck(cuEventElapsedTime(&ms,e0,e1), "elapsed"); best=std::min(best,ms); sum+=ms;
      }
      int nr=0; cuFuncGetAttribute(&nr, CU_FUNC_ATTRIBUTE_NUM_REGS, mod.f);
      const long double inst = (long double)blocks*wpb*o.iters*o.chains;
      const long double flopi = 2.0L*c.m*c.n*c.k;
      const long double peak = inst*flopi/(best/1000.0L)/1.0e12L;
      const long double avg = inst*flopi/((sum/o.repeats)/1000.0L)/1.0e12L;
      std::cout << std::fixed << std::setprecision(3) << "PASS name=" << c.name
                << " best_ms=" << best << " peak_logical_tflops=" << double(peak)
                << " avg_logical_tflops=" << double(avg);
      if (c.sparse) std::cout << " peak_nonzero_tflops=" << double(peak*0.5L);
      std::cout << " regs_per_thread=" << nr << " blocks=" << blocks << " chains=" << o.chains
                << " iters=" << o.iters << " repeats=" << o.repeats << "\n";
      if (o.verbose_jit && !mod.log.empty()) std::cout << mod.log << "\n";
      unload(mod);
    }

    std::cout << "selected=" << selected << " documented=" << documented << " probes=" << probes
              << " manifest_total=" << cases.size() << "\n";
    cuEventDestroy(e0); cuEventDestroy(e1); cuMemFree(out); cuDevicePrimaryCtxRelease(dev);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "fatal: " << e.what() << "\n";
    return 1;
  }
}
