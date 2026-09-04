#include <cuda.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
  unsigned iters = 2000;
  unsigned blocks_per_sm = 4;
  unsigned chains = 8;
  std::string filter;
  bool documented_only = false;
  bool probes_only = false;
  bool verbose_jit = false;
};

struct TestCase {
  std::string name;
  std::string mnemonic;
  std::string note;
  bool documented = true;
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
};

std::string cu_error(CUresult r) {
  const char* name = nullptr;
  const char* msg = nullptr;
  cuGetErrorName(r, &name);
  cuGetErrorString(r, &msg);
  std::ostringstream os;
  os << (name ? name : "CUDA_ERROR") << ": " << (msg ? msg : "unknown");
  return os.str();
}

void check(CUresult r, const char* what) {
  if (r != CUDA_SUCCESS) {
    throw std::runtime_error(std::string(what) + ": " + cu_error(r));
  }
}

std::string join_regs(const std::string& prefix, int begin, int count) {
  std::ostringstream os;
  os << "{";
  for (int i = 0; i < count; ++i) {
    if (i) os << ", ";
    os << "%" << prefix << (begin + i);
  }
  os << "}";
  return os.str();
}

std::string make_tcgen05_negative_probe() {
  return R"PTX(.version 9.0
.target sm_120a
.address_size 64

.visible .entry bench(
    .param .u64 out_ptr,
    .param .u32 iters)
{
    .reg .b32 %taddr;
    .reg .b64 %adesc, %bdesc;
    .reg .b32 %idesc;
    .reg .b32 %m0, %m1, %m2, %m3;
    .reg .pred %p;
    mov.b32 %taddr, 0;
    mov.b64 %adesc, 0;
    mov.b64 %bdesc, 0;
    mov.b32 %idesc, 0;
    mov.b32 %m0, 0;
    mov.b32 %m1, 0;
    mov.b32 %m2, 0;
    mov.b32 %m3, 0;
    mov.pred %p, 0;
    tcgen05.mma.cta_group::1.kind::f8f6f4
        [%taddr], %adesc, %bdesc, %idesc, {%m0, %m1, %m2, %m3}, %p;
    ret;
}
)PTX";
}

std::string make_ptx(const TestCase& tc, unsigned chains) {
  if (tc.custom_ptx) return tc.custom_source;

  const int acc_regs = tc.acc_f16 ? 2 : 4;
  std::ostringstream p;
  p << ".version 9.0\n"
    << ".target sm_120a\n"
    << ".address_size 64\n\n"
    << ".visible .entry bench(\n"
    << "    .param .u64 out_ptr,\n"
    << "    .param .u32 iters)\n"
    << "{\n"
    << "    .reg .pred %done, %not_lane0;\n"
    << "    .reg .u32 %i, %limit, %lane, %tid, %cta, %ntid, %warp, %wpb, %wg;\n"
    << "    .reg .u64 %out, %off, %addr;\n"
    << "    .reg .b32 %a<" << tc.a_regs << ">;\n"
    << "    .reg .b32 %b<" << tc.b_regs << ">;\n";

  if (tc.acc_f16) {
    p << "    .reg .b32 %d<" << (chains * acc_regs) << ">;\n"
      << "    .reg .b32 %live;\n";
  } else {
    p << "    .reg .f32 %d<" << (chains * acc_regs) << ">;\n"
      << "    .reg .f32 %live;\n";
  }
  if (tc.sparse) p << "    .reg .b32 %meta;\n";
  if (tc.block_scale) p << "    .reg .b32 %sa, %sb;\n";

  p << "\n"
    << "    ld.param.u64 %out, [out_ptr];\n"
    << "    ld.param.u32 %limit, [iters];\n";

  for (int i = 0; i < tc.a_regs; ++i)
    p << "    mov.b32 %a" << i << ", 0x3c3c3c3c;\n";
  for (int i = 0; i < tc.b_regs; ++i)
    p << "    mov.b32 %b" << i << ", 0x34343434;\n";

  for (unsigned c = 0; c < chains; ++c) {
    for (int j = 0; j < acc_regs; ++j) {
      const int idx = static_cast<int>(c) * acc_regs + j;
      if (tc.acc_f16)
        p << "    mov.b32 %d" << idx << ", 0;\n";
      else
        p << "    mov.f32 %d" << idx << ", 0f00000000;\n";
    }
  }

  if (tc.sparse) p << "    mov.b32 %meta, 0x44444444;\n";
  if (tc.block_scale) {
    p << "    mov.b32 %sa, 0x7f7f7f7f;\n"
      << "    mov.b32 %sb, 0x7f7f7f7f;\n";
  }

  p << "    mov.u32 %i, 0;\n"
    << "LOOP:\n"
    << "    setp.ge.u32 %done, %i, %limit;\n"
    << "    @%done bra DONE;\n";

  const std::string a = join_regs("a", 0, tc.a_regs);
  const std::string b = join_regs("b", 0, tc.b_regs);
  for (unsigned c = 0; c < chains; ++c) {
    const int base = static_cast<int>(c) * acc_regs;
    const std::string d = join_regs("d", base, acc_regs);
    p << "    " << tc.mnemonic << "\n"
      << "        " << d << ", " << a << ", " << b << ", " << d;
    if (tc.sparse) p << ", %meta, 0";
    if (tc.block_scale) p << ", %sa, {0, 0}, %sb, {0, 0}";
    p << ";\n";
  }

  p << "    add.u32 %i, %i, 1;\n"
    << "    bra LOOP;\n"
    << "DONE:\n";

  if (tc.acc_f16) {
    p << "    mov.b32 %live, %d0;\n";
    for (unsigned c = 1; c < chains; ++c)
      p << "    xor.b32 %live, %live, %d" << (c * acc_regs) << ";\n";
  } else {
    p << "    mov.f32 %live, %d0;\n";
    for (unsigned c = 1; c < chains; ++c)
      p << "    add.f32 %live, %live, %d" << (c * acc_regs) << ";\n";
  }

  p << "    mov.u32 %lane, %laneid;\n"
    << "    setp.ne.u32 %not_lane0, %lane, 0;\n"
    << "    @%not_lane0 bra EXIT;\n"
    << "    mov.u32 %tid, %tid.x;\n"
    << "    mov.u32 %cta, %ctaid.x;\n"
    << "    mov.u32 %ntid, %ntid.x;\n"
    << "    shr.u32 %warp, %tid, 5;\n"
    << "    shr.u32 %wpb, %ntid, 5;\n"
    << "    mad.lo.u32 %wg, %cta, %wpb, %warp;\n"
    << "    mul.wide.u32 %off, %wg, 4;\n"
    << "    add.u64 %addr, %out, %off;\n";
  if (tc.acc_f16)
    p << "    st.global.b32 [%addr], %live;\n";
  else
    p << "    st.global.f32 [%addr], %live;\n";
  p << "EXIT:\n"
    << "    ret;\n"
    << "}\n";
  return p.str();
}

void add_case(std::vector<TestCase>& v, const std::string& name,
              const std::string& mnemonic, int k, int a_regs, int b_regs,
              bool f16, bool sparse, bool block, bool documented,
              const std::string& note = {}) {
  TestCase tc;
  tc.name = name;
  tc.mnemonic = mnemonic;
  tc.k = k;
  tc.a_regs = a_regs;
  tc.b_regs = b_regs;
  tc.acc_f16 = f16;
  tc.sparse = sparse;
  tc.block_scale = block;
  tc.documented = documented;
  tc.note = note;
  v.push_back(std::move(tc));
}

std::vector<TestCase> build_manifest() {
  std::vector<TestCase> v;
  const std::vector<std::string> fp8 = {"e4m3", "e5m2"};
  const std::vector<std::string> fp864 = {"e4m3", "e5m2", "e3m2", "e2m3", "e2m1"};
  const std::vector<std::string> accs = {"f16", "f32"};

  // Legacy FP8 dense forms already present before SM120, but still supported on SM120.
  for (int k : {16, 32}) {
    const int ar = (k == 16) ? 2 : 4;
    const int br = (k == 16) ? 1 : 2;
    for (const auto& a : fp8) for (const auto& b : fp8) for (const auto& acc : accs) {
      const bool f16 = acc == "f16";
      std::ostringstream mn, nm;
      mn << "mma.sync.aligned.m16n8k" << k << ".row.col." << acc << "." << a << "." << b << "." << acc;
      nm << "dense/legacy-fp8/k" << k << "/" << a << "x" << b << "/acc-" << acc;
      add_case(v, nm.str(), mn.str(), k, ar, br, f16, false, false, true);
    }
  }

  // SM120 dense F8/F6/F4 kind. A and B type qualifiers are independent.
  for (const auto& a : fp864) for (const auto& b : fp864) for (const auto& acc : accs) {
    const bool f16 = acc == "f16";
    const std::string mn = "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4." + acc + "." + a + "." + b + "." + acc;
    const std::string nm = "dense/f8f6f4/" + a + "x" + b + "/acc-" + acc;
    add_case(v, nm, mn, 32, 4, 2, f16, false, false, true);
  }

  // Block-scaled dense: MXF8/F6/F4, MXF4, NVFP4-style MXF4NVF4.
  for (const auto& a : fp864) for (const auto& b : fp864) {
    const std::string mn = "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32." + a + "." + b + ".f32.ue8m0";
    const std::string nm = "dense/mxf8f6f4/" + a + "x" + b + "/ue8m0-1X";
    add_case(v, nm, mn, 32, 4, 2, false, false, true, true);
  }
  add_case(v, "dense/mxf4/e2m1xe2m1/ue8m0-2X",
           "mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0",
           64, 4, 2, false, false, true, true);
  add_case(v, "dense/mxf4nvf4/e2m1xe2m1/ue8m0-2X",
           "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0",
           64, 4, 2, false, false, true, true);
  add_case(v, "dense/mxf4nvf4/e2m1xe2m1/ue8m0-4X",
           "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue8m0",
           64, 4, 2, false, false, true, true);
  add_case(v, "dense/mxf4nvf4/e2m1xe2m1/ue4m3-4X",
           "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3",
           64, 4, 2, false, false, true, true);

  // Legacy sparse FP8: sparse K is doubled relative to the corresponding dense form.
  for (int k : {32, 64}) {
    const int ar = (k == 32) ? 2 : 4;
    const int br = (k == 32) ? 2 : 4;
    for (const std::string sp : {"sp", "sp::ordered_metadata"}) {
      for (const auto& a : fp8) for (const auto& b : fp8) for (const auto& acc : accs) {
        const bool f16 = acc == "f16";
        std::ostringstream mn, nm;
        mn << "mma." << sp << ".sync.aligned.m16n8k" << k << ".row.col." << acc << "." << a << "." << b << "." << acc;
        nm << "sparse/legacy-fp8/" << sp << "/k" << k << "/" << a << "x" << b << "/acc-" << acc;
        add_case(v, nm.str(), mn.str(), k, ar, br, f16, true, false, true);
      }
    }
  }

  // SM120 sparse F8/F6/F4, both metadata forms.
  for (const std::string sp : {"sp", "sp::ordered_metadata"}) {
    for (const auto& a : fp864) for (const auto& b : fp864) for (const auto& acc : accs) {
      const bool f16 = acc == "f16";
      const std::string mn = "mma." + sp + ".sync.aligned.m16n8k64.row.col.kind::f8f6f4." + acc + "." + a + "." + b + "." + acc;
      const std::string nm = "sparse/f8f6f4/" + sp + "/" + a + "x" + b + "/acc-" + acc;
      add_case(v, nm, mn, 64, 4, 4, f16, true, false, true);
    }
  }

  // Block-scaled sparse MXF8/F6/F4. PTX exposes both sparse metadata spellings here.
  for (const std::string sp : {"sp", "sp::ordered_metadata"}) {
    for (const auto& a : fp864) for (const auto& b : fp864) {
      const std::string mn = "mma." + sp + ".sync.aligned.m16n8k64.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32." + a + "." + b + ".f32.ue8m0";
      const std::string nm = "sparse/mxf8f6f4/" + sp + "/" + a + "x" + b + "/ue8m0-1X";
      add_case(v, nm, mn, 64, 4, 4, false, true, true, true);
    }
  }

  // For MXF4/MXF4NVF4, the documented SM120 sparse form is ordered metadata.
  add_case(v, "sparse/mxf4/sp::ordered_metadata/e2m1xe2m1/ue8m0-2X",
           "mma.sp::ordered_metadata.sync.aligned.m16n8k128.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0",
           128, 4, 4, false, true, true, true);
  add_case(v, "sparse/mxf4nvf4/sp::ordered_metadata/e2m1xe2m1/ue8m0-2X",
           "mma.sp::ordered_metadata.sync.aligned.m16n8k128.row.col.kind::mxf4nvf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0",
           128, 4, 4, false, true, true, true);
  add_case(v, "sparse/mxf4nvf4/sp::ordered_metadata/e2m1xe2m1/ue8m0-4X",
           "mma.sp::ordered_metadata.sync.aligned.m16n8k128.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue8m0",
           128, 4, 4, false, true, true, true);
  add_case(v, "sparse/mxf4nvf4/sp::ordered_metadata/e2m1xe2m1/ue4m3-4X",
           "mma.sp::ordered_metadata.sync.aligned.m16n8k128.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3",
           128, 4, 4, false, true, true, true);

  // Plausible but undocumented / explicitly outside the documented combination tables.
  {
    TestCase tc;
    tc.name = "probe/undocumented/tcgen05-on-sm120";
    tc.documented = false;
    tc.custom_ptx = true;
    tc.custom_source = make_tcgen05_negative_probe();
    tc.note = "tcgen05 MMA is not exposed for the SM120 family in the PTX target-feature table";
    v.push_back(std::move(tc));
  }
  add_case(v, "probe/undocumented/f8f6f4-dense-k64",
           "mma.sync.aligned.m16n8k64.row.col.kind::f8f6f4.f32.e3m2.e2m3.f32",
           64, 8, 4, false, false, false, false,
           "documented dense kind::f8f6f4 shape is m16n8k32");
  add_case(v, "probe/undocumented/fp6-without-kind",
           "mma.sync.aligned.m16n8k32.row.col.f32.e3m2.e3m2.f32",
           32, 4, 2, false, false, false, false,
           "FP6/FP4 SM120 forms are documented with kind::f8f6f4");
  add_case(v, "probe/undocumented/mxf8f6f4-ue8m0-2X",
           "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::2X.f32.e4m3.e4m3.f32.ue8m0",
           32, 4, 2, false, false, true, false,
           "Table 37 defines mxf8f6f4 with scale_vec::1X");
  add_case(v, "probe/undocumented/mxf8f6f4-ue4m3-1X",
           "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue4m3",
           32, 4, 2, false, false, true, false,
           "Table 37 defines mxf8f6f4 scale type as ue8m0");
  add_case(v, "probe/undocumented/mxf4-ue8m0-1X",
           "mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.scale_vec::1X.f32.e2m1.e2m1.f32.ue8m0",
           64, 4, 2, false, false, true, false,
           "Table 37 defines mxf4 with scale_vec::2X");
  add_case(v, "probe/undocumented/mxf4-ue4m3-2X",
           "mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue4m3",
           64, 4, 2, false, false, true, false,
           "Table 37 defines mxf4 scale type as ue8m0");
  add_case(v, "probe/undocumented/mxf4nvf4-ue4m3-2X",
           "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue4m3",
           64, 4, 2, false, false, true, false,
           "ue4m3 is documented only with scale_vec::4X for mxf4nvf4");
  add_case(v, "probe/undocumented/mxf4nvf4-missing-scale-vec",
           "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.f32.e2m1.e2m1.f32.ue8m0",
           64, 4, 2, false, false, true, false,
           "PTX requires an explicit scale_vec qualifier for mxf4nvf4");
  add_case(v, "probe/undocumented/mxf8f6f4-f16-acc",
           "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f16.e4m3.e4m3.f16.ue8m0",
           32, 4, 2, true, false, true, false,
           "documented block-scaled grammar fixes accumulator/result to f32");
  add_case(v, "probe/undocumented/mxf4-plain-sp",
           "mma.sp.sync.aligned.m16n8k128.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0",
           128, 4, 4, false, true, true, false,
           "SM120 target notes exclude plain .sp for mxf4/mxf4nvf4; ordered metadata is documented");
  add_case(v, "probe/undocumented/f8f6f4-row-row",
           "mma.sync.aligned.m16n8k32.row.row.kind::f8f6f4.f32.e4m3.e4m3.f32",
           32, 4, 2, false, false, false, false,
           "documented low-precision warp MMA spelling is row.col");

  return v;
}

Options parse_args(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto need = [&](const char* opt) -> std::string {
      if (++i >= argc) throw std::runtime_error(std::string("missing value for ") + opt);
      return argv[i];
    };
    if (a == "--iters") o.iters = static_cast<unsigned>(std::stoul(need("--iters")));
    else if (a == "--blocks-per-sm") o.blocks_per_sm = static_cast<unsigned>(std::stoul(need("--blocks-per-sm")));
    else if (a == "--chains") o.chains = static_cast<unsigned>(std::stoul(need("--chains")));
    else if (a == "--filter") o.filter = need("--filter");
    else if (a == "--documented-only") o.documented_only = true;
    else if (a == "--probes-only") o.probes_only = true;
    else if (a == "--verbose-jit") o.verbose_jit = true;
    else if (a == "--help" || a == "-h") {
      std::cout << "tensorcore-fp864-bench options:\n"
                << "  --iters N\n"
                << "  --blocks-per-sm N\n"
                << "  --chains {1,2,4,8}\n"
                << "  --filter TEXT\n"
                << "  --documented-only\n"
                << "  --probes-only\n"
                << "  --verbose-jit\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown option: " + a);
    }
  }
  if (!(o.chains == 1 || o.chains == 2 || o.chains == 4 || o.chains == 8))
    throw std::runtime_error("--chains must be one of 1,2,4,8");
  if (o.iters == 0 || o.blocks_per_sm == 0) throw std::runtime_error("iteration/work counts must be nonzero");
  return o;
}

struct Loaded {
  CUmodule module = nullptr;
  CUfunction function = nullptr;
  std::string log;
};

CUresult jit_load(const std::string& ptx, Loaded& out) {
  char errlog[16384] = {};
  char infolog[16384] = {};
  CUjit_option opts[] = {
      CU_JIT_ERROR_LOG_BUFFER,
      CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
      CU_JIT_INFO_LOG_BUFFER,
      CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
      CU_JIT_LOG_VERBOSE,
      CU_JIT_OPTIMIZATION_LEVEL};
  void* vals[] = {
      errlog,
      reinterpret_cast<void*>(static_cast<uintptr_t>(sizeof(errlog))),
      infolog,
      reinterpret_cast<void*>(static_cast<uintptr_t>(sizeof(infolog))),
      reinterpret_cast<void*>(static_cast<uintptr_t>(1)),
      reinterpret_cast<void*>(static_cast<uintptr_t>(4))};

  CUresult r = cuModuleLoadDataEx(&out.module, ptx.c_str(), 6, opts, vals);
  out.log = std::string(errlog) + std::string(infolog);
  if (r != CUDA_SUCCESS) return r;
  r = cuModuleGetFunction(&out.function, out.module, "bench");
  if (r != CUDA_SUCCESS) {
    cuModuleUnload(out.module);
    out.module = nullptr;
  }
  return r;
}

void unload(Loaded& l) {
  if (l.module) cuModuleUnload(l.module);
  l.module = nullptr;
  l.function = nullptr;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options opt = parse_args(argc, argv);

    check(cuInit(0), "cuInit");
    CUdevice dev;
    check(cuDeviceGet(&dev, 0), "cuDeviceGet");

    char name[256] = {};
    int major = 0, minor = 0, sms = 0, clock_khz = 0;
    check(cuDeviceGetName(name, sizeof(name), dev), "cuDeviceGetName");
    check(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev), "get major");
    check(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev), "get minor");
    check(cuDeviceGetAttribute(&sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev), "get SM count");
    check(cuDeviceGetAttribute(&clock_khz, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, dev), "get clock");

    std::cout << "device=" << name << " cc=" << major << "." << minor
              << " sm_count=" << sms << " reported_clock_mhz=" << (clock_khz / 1000.0) << "\n";
    if (major != 12 || minor != 0) {
      std::cerr << "This repository's in-scope architectural target is SM120 (RTX 5090 / RTX PRO 6000 Blackwell).\n";
      return 2;
    }

    CUcontext ctx;
    check(cuDevicePrimaryCtxRetain(&ctx, dev), "cuDevicePrimaryCtxRetain");
    check(cuCtxSetCurrent(ctx), "cuCtxSetCurrent");

    const unsigned threads = 256;
    const unsigned warps_per_block = threads / 32;
    const unsigned blocks = static_cast<unsigned>(sms) * opt.blocks_per_sm;
    const size_t out_bytes = static_cast<size_t>(blocks) * warps_per_block * sizeof(uint32_t);
    CUdeviceptr out_ptr = 0;
    check(cuMemAlloc(&out_ptr, out_bytes), "cuMemAlloc");

    CUevent ev0, ev1;
    check(cuEventCreate(&ev0, CU_EVENT_DEFAULT), "cuEventCreate start");
    check(cuEventCreate(&ev1, CU_EVENT_DEFAULT), "cuEventCreate stop");

    auto manifest = build_manifest();
    size_t selected = 0;
    for (const auto& tc : manifest) {
      if (opt.documented_only && !tc.documented) continue;
      if (!opt.filter.empty() && tc.name.find(opt.filter) == std::string::npos) continue;
      ++selected;

      const std::string ptx = make_ptx(tc, opt.chains);
      Loaded l;
      const CUresult jr = jit_load(ptx, l);
      if (jr != CUDA_SUCCESS) {
        std::cout << (tc.documented ? "FAIL_DOCUMENTED" : "REJECTED_UNDOCUMENTED")
                  << " name=" << tc.name << " jit=" << cu_error(jr);
        if (!tc.note.empty()) std::cout << " note=\"" << tc.note << "\"";
        std::cout << "\n";
        if (opt.verbose_jit && !l.log.empty()) std::cout << l.log << "\n";
        continue;
      }

      if (!tc.documented) {
        std::cout << "ACCEPTED_UNDOCUMENTED name=" << tc.name;
        if (!tc.note.empty()) std::cout << " note=\"" << tc.note << "\"";
        std::cout << "\n";
        if (opt.verbose_jit && !l.log.empty()) std::cout << l.log << "\n";
        unload(l);
        continue;
      }

      if (opt.probes_only) {
        std::cout << "PASS_COMPILE name=" << tc.name << "\n";
        unload(l);
        continue;
      }

      unsigned iters = std::min(opt.iters, 16u);
      void* warm_args[] = {&out_ptr, &iters};
      check(cuLaunchKernel(l.function, blocks, 1, 1, threads, 1, 1, 0, nullptr, warm_args, nullptr), "warmup launch");
      check(cuCtxSynchronize(), "warmup sync");

      iters = opt.iters;
      void* args[] = {&out_ptr, &iters};
      check(cuEventRecord(ev0, nullptr), "event start");
      check(cuLaunchKernel(l.function, blocks, 1, 1, threads, 1, 1, 0, nullptr, args, nullptr), "timed launch");
      check(cuEventRecord(ev1, nullptr), "event stop");
      check(cuEventSynchronize(ev1), "event sync");
      float ms = 0.0f;
      check(cuEventElapsedTime(&ms, ev0, ev1), "elapsed time");

      int regs = 0;
      cuFuncGetAttribute(&regs, CU_FUNC_ATTRIBUTE_NUM_REGS, l.function);

      const long double warp_count = static_cast<long double>(blocks) * warps_per_block;
      const long double inst_count = warp_count * opt.iters * opt.chains;
      const long double flop_per_inst = 2.0L * tc.m * tc.n * tc.k;
      const long double seconds = static_cast<long double>(ms) / 1000.0L;
      const long double logical_tflops = inst_count * flop_per_inst / seconds / 1.0e12L;
      const long double nonzero_tflops = tc.sparse ? logical_tflops * 0.5L : logical_tflops;

      std::cout << std::fixed << std::setprecision(3)
                << "PASS name=" << tc.name
                << " ms=" << ms
                << " logical_tflops=" << static_cast<double>(logical_tflops);
      if (tc.sparse)
        std::cout << " nonzero_tflops=" << static_cast<double>(nonzero_tflops);
      std::cout << " regs_per_thread=" << regs
                << " blocks=" << blocks
                << " threads=" << threads
                << " chains=" << opt.chains
                << " iters=" << opt.iters
                << "\n";

      if (opt.verbose_jit && !l.log.empty()) std::cout << l.log << "\n";
      unload(l);
    }

    std::cout << "selected_cases=" << selected << " total_manifest_cases=" << manifest.size() << "\n";

    cuEventDestroy(ev0);
    cuEventDestroy(ev1);
    cuMemFree(out_ptr);
    cuDevicePrimaryCtxRelease(dev);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "fatal: " << e.what() << "\n";
    return 1;
  }
}
