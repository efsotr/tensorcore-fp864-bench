// Curated dense FP8/FP6/FP4 x FP6/FP6/FP4 microbenchmark for SM120.
//
// This intentionally reuses the canonical PTX manifest/generator from bench.cpp,
// but runs only a small practical subset. It excludes sparse instructions,
// F16 accumulation, duplicate default-scale spellings, selector expansion, and
// non-documented probes.

#include <chrono>
#include <cctype>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <map>

#define main tensorcore_fp864_full_main
#include "bench.cpp"
#undef main

namespace {

const std::vector<std::string>& curated_case_names() {
  static const std::vector<std::string> names = {
      // Dense unscaled kind::f8f6f4, F32 accumulation.
      "dense/f8f6f4/e4m3xe3m2/acc-f32",
      "dense/f8f6f4/e4m3xe2m3/acc-f32",
      "dense/f8f6f4/e4m3xe2m1/acc-f32",
      "dense/f8f6f4/e5m2xe3m2/acc-f32",
      "dense/f8f6f4/e5m2xe2m1/acc-f32",
      "dense/f8f6f4/e3m2xe3m2/acc-f32",
      "dense/f8f6f4/e2m3xe2m3/acc-f32",
      "dense/f8f6f4/e3m2xe2m1/acc-f32",
      "dense/f8f6f4/e2m3xe2m1/acc-f32",
      "dense/f8f6f4/e2m1xe2m1/acc-f32",

      // Dense MXF8/F6/F4, explicit UE8M0 1X spelling only.
      "dense/mxf8f6f4/e4m3xe3m2/ue8m0-1X",
      "dense/mxf8f6f4/e4m3xe2m3/ue8m0-1X",
      "dense/mxf8f6f4/e4m3xe2m1/ue8m0-1X",
      "dense/mxf8f6f4/e5m2xe2m1/ue8m0-1X",
      "dense/mxf8f6f4/e3m2xe3m2/ue8m0-1X",
      "dense/mxf8f6f4/e2m3xe2m1/ue8m0-1X",
      "dense/mxf8f6f4/e2m1xe2m1/ue8m0-1X",

      // Common FP4 block-scaled reference points.
      "dense/mxf4/e2m1xe2m1/ue8m0-2X",
      "dense/mxf4nvf4/e2m1xe2m1/ue4m3-4X",
  };
  return names;
}

bool is_curated_case(const Case& c) {
  const auto& names = curated_case_names();
  return std::find(names.begin(), names.end(), c.name) != names.end();
}

namespace fs = std::filesystem;

struct CuratedOptions {
  Options bench;
  std::string output_dir;
  bool quiet_cases = false;
};

struct CuratedResult {
  std::string status;
  std::string name;
  std::string opcode;
  std::string error;
  std::string ptx_version;
  std::string ptx_path;
  std::string cuda_repro_path;
  std::string jit_error_log_path;
  std::string jit_info_log_path;
  std::string jit_error_log;
  std::string jit_info_log;
  double best_ms = 0.0;
  double mean_ms = 0.0;
  double peak_logical_tflops = 0.0;
  double mean_logical_tflops = 0.0;
  int regs_per_thread = 0;
};

struct FamilyPeak {
  std::string case_name;
  double tflops = 0.0;
};

std::string json_escape(const std::string& in) {
  std::ostringstream out;
  for (unsigned char c : in) {
    switch (c) {
      case '\\': out << "\\\\"; break;
      case '"': out << "\\\""; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      default:
        if (c < 0x20) {
          out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
              << static_cast<int>(c) << std::dec << std::setfill(' ');
        } else {
          out << static_cast<char>(c);
        }
    }
  }
  return out.str();
}

std::string utc_stamp() {
  const std::time_t now = std::time(nullptr);
  std::tm tm{};
#if defined(_WIN32)
  gmtime_s(&tm, &now);
#else
  gmtime_r(&now, &tm);
#endif
  char buf[32] = {};
  std::strftime(buf, sizeof(buf), "%Y%m%dT%H%M%SZ", &tm);
  return buf;
}

std::string slug(std::string s) {
  for (char& c : s) {
    const unsigned char u = static_cast<unsigned char>(c);
    if (!std::isalnum(u) && c != '-' && c != '_') c = '_';
  }
  while (s.find("__") != std::string::npos) s.replace(s.find("__"), 2, "_");
  if (s.size() > 150) s.resize(150);
  return s;
}

std::string quote_arg(const std::string& s) {
  std::ostringstream out;
  out << '"';
  for (char ch : s) {
    if (ch == '\\' || ch == '"') out << '\\';
    out << ch;
  }
  out << '"';
  return out.str();
}

std::string command_line_string(int argc, char** argv) {
  std::ostringstream out;
  for (int i = 0; i < argc; ++i) {
    if (i) out << ' ';
    out << quote_arg(argv[i]);
  }
  return out.str();
}

std::string env_value(const char* name) {
  const char* v = std::getenv(name);
  return v ? v : "<unset>";
}

void write_text_file(const fs::path& path, const std::string& text) {
  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("cannot write " + path.string());
  out << text;
}

std::string double_percent(std::string s) {
  size_t pos = 0;
  while ((pos = s.find('%', pos)) != std::string::npos) {
    s.insert(pos, 1, '%');
    pos += 2;
  }
  return s;
}

std::string build_inline_ptx_cuda_repro(const Case& c) {
  const int acc_regs = c.acc_f16 ? 2 : 4;
  std::ostringstream body;
  body << "{\n"
       << "  .reg .b32 %a<" << c.a_regs << ">, %b<" << c.b_regs << ">;\n";
  if (c.acc_f16) body << "  .reg .b32 %d<" << acc_regs << ">;\n";
  else body << "  .reg .f32 %d<" << acc_regs << ">;\n";
  if (c.sparse) body << "  .reg .b32 %meta;\n";
  if (c.block_scale) body << "  .reg .b32 %sa, %sb;\n";

  for (int i = 0; i < c.a_regs; ++i)
    body << "  mov.b32 %a" << i << ", 0x14141414;\n";
  for (int i = 0; i < c.b_regs; ++i)
    body << "  mov.b32 %b" << i << ", 0x0c0c0c0c;\n";
  for (int i = 0; i < acc_regs; ++i) {
    if (c.acc_f16) body << "  mov.b32 %d" << i << ", 0;\n";
    else body << "  mov.f32 %d" << i << ", 0f00000000;\n";
  }
  if (c.sparse) body << "  mov.b32 %meta, 0x44444444;\n";
  if (c.block_scale) {
    body << "  mov.b32 %sa, 0x38383838;\n"
         << "  mov.b32 %sb, 0x38383838;\n";
  }

  const std::string d = reg_tuple("d", 0, acc_regs);
  body << "  " << c.opcode << "\n"
       << "    " << d << ", "
       << reg_tuple("a", 0, c.a_regs) << ", "
       << reg_tuple("b", 0, c.b_regs) << ", "
       << d;
  if (c.sparse) body << ", %meta, " << c.sparse_selector;
  if (c.block_scale) {
    body << ", %sa, {" << c.byte_a << ", " << c.thread_a << "}"
         << ", %sb, {" << c.byte_b << ", " << c.thread_b << "}";
  }
  body << ";\n}\n";

  std::ostringstream src;
  src << "// Auto-generated standalone CUDA reproduction for " << c.name << "\n"
      << "// PTX ISA requested by benchmark: " << ptx_version_for_case(c) << "\n"
      << "// Suggested compile: nvcc -arch=sm_120a -lineinfo inline_ptx_repro.cu -o repro\n"
      << "#include <cuda_runtime.h>\n"
      << "#include <cstdio>\n\n"
      << "__global__ void repro_kernel() {\n"
      << "  asm volatile(R\"PTX(\n"
      << double_percent(body.str())
      << ")PTX\");\n"
      << "}\n\n"
      << "int main() {\n"
      << "  repro_kernel<<<1, 32>>>();\n"
      << "  cudaError_t launch = cudaGetLastError();\n"
      << "  if (launch != cudaSuccess) {\n"
      << "    std::fprintf(stderr, \"launch: %s\\n\", cudaGetErrorString(launch));\n"
      << "    return 1;\n"
      << "  }\n"
      << "  cudaError_t sync = cudaDeviceSynchronize();\n"
      << "  if (sync != cudaSuccess) {\n"
      << "    std::fprintf(stderr, \"sync: %s\\n\", cudaGetErrorString(sync));\n"
      << "    return 2;\n"
      << "  }\n"
      << "  return 0;\n"
      << "}\n";
  return src.str();
}

std::string case_metadata(const Case& c, const Options& opt) {
  std::ostringstream out;
  out << "name=" << c.name << "\n"
      << "opcode=" << c.opcode << "\n"
      << "ptx_version=" << ptx_version_for_case(c) << "\n"
      << "target=sm_120a\n"
      << "m=" << c.m << "\n"
      << "n=" << c.n << "\n"
      << "k=" << c.k << "\n"
      << "a_regs=" << c.a_regs << "\n"
      << "b_regs=" << c.b_regs << "\n"
      << "acc_f16=" << (c.acc_f16 ? "yes" : "no") << "\n"
      << "sparse=" << (c.sparse ? "yes" : "no") << "\n"
      << "block_scale=" << (c.block_scale ? "yes" : "no") << "\n"
      << "scale_vec=" << c.scale_vec << "\n"
      << "byte_a=" << c.byte_a << "\n"
      << "thread_a=" << c.thread_a << "\n"
      << "byte_b=" << c.byte_b << "\n"
      << "thread_b=" << c.thread_b << "\n"
      << "chains=" << opt.chains << "\n"
      << "inner_unroll=" << kInnerUnroll << "\n"
      << "iters=" << opt.iters << "\n"
      << "blocks_per_sm=" << opt.blocks_per_sm << "\n"
      << "repeats=" << opt.repeats << "\n";
  return out.str();
}

CuratedOptions parse_curated_args(int argc, char** argv) {
  CuratedOptions o;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto value = [&](const char* opt) -> std::string {
      if (++i >= argc) throw std::runtime_error(std::string("missing value for ") + opt);
      return argv[i];
    };
    if (a == "--device") o.bench.device = std::stoi(value("--device"));
    else if (a == "--iters") o.bench.iters = static_cast<unsigned>(std::stoul(value("--iters")));
    else if (a == "--blocks-per-sm") o.bench.blocks_per_sm = static_cast<unsigned>(std::stoul(value("--blocks-per-sm")));
    else if (a == "--chains") o.bench.chains = static_cast<unsigned>(std::stoul(value("--chains")));
    else if (a == "--repeats") o.bench.repeats = static_cast<unsigned>(std::stoul(value("--repeats")));
    else if (a == "--filter") o.bench.filter = value("--filter");
    else if (a == "--list") o.bench.list_only = true;
    else if (a == "--verbose-jit") o.bench.verbose_jit = true;
    else if (a == "--output-dir") o.output_dir = value("--output-dir");
    else if (a == "--quiet-cases") o.quiet_cases = true;
    else if (a == "--include-probes" || a == "--probes-only" || a == "--all-operands") {
      throw std::runtime_error(
          "this curated benchmark excludes probes and selector expansion; "
          "do not use --include-probes, --probes-only, or --all-operands");
    } else if (a == "-h" || a == "--help") {
      std::cout
          << "--device N --iters N --blocks-per-sm N --chains {1,2,4,8} --repeats N\n"
          << "--filter TEXT --list --verbose-jit --output-dir DIR --quiet-cases\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown option: " + a);
    }
  }
  if (!(o.bench.chains == 1 || o.bench.chains == 2 ||
        o.bench.chains == 4 || o.bench.chains == 8)) {
    throw std::runtime_error("--chains must be one of 1,2,4,8");
  }
  if (!o.bench.iters || !o.bench.blocks_per_sm || !o.bench.repeats) {
    throw std::runtime_error("iteration/work counts must be non-zero");
  }
  return o;
}

void emit_line(std::ofstream& log, const std::string& line, bool quiet) {
  log << line << '\n';
  log.flush();
  if (!quiet) std::cout << line << '\n';
}

std::string family_name(const std::string& case_name) {
  if (case_name.rfind("dense/f8f6f4/", 0) == 0) return "unscaled_f8f6f4";
  if (case_name.rfind("dense/mxf8f6f4/", 0) == 0) return "mxf8f6f4";
  if (case_name.rfind("dense/mxf4nvf4/", 0) == 0) return "mxf4nvf4";
  if (case_name.rfind("dense/mxf4/", 0) == 0) return "mxf4";
  return "other";
}

std::map<std::string, FamilyPeak> extract_family_peaks(
    const std::vector<CuratedResult>& results) {
  std::map<std::string, FamilyPeak> peaks;
  for (const CuratedResult& r : results) {
    if (r.status != "PASS") continue;
    const std::string family = family_name(r.name);
    FamilyPeak& peak = peaks[family];
    if (r.peak_logical_tflops > peak.tflops) {
      peak.case_name = r.name;
      peak.tflops = r.peak_logical_tflops;
    }
  }
  return peaks;
}

void write_result_json(
    const fs::path& path,
    const std::string& stamp,
    const std::string& device_name,
    int major,
    int minor,
    int sms,
    int clock_khz,
    int driver_version,
    const Options& opt,
    const std::vector<CuratedResult>& results,
    const std::map<std::string, FamilyPeak>& peaks) {
  size_t passed = 0;
  size_t failed = 0;
  for (const auto& r : results) {
    if (r.status == "PASS") ++passed;
    else ++failed;
  }

  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("cannot write " + path.string());

  out << "{\n"
      << "  \"schema_version\": 1,\n"
      << "  \"suite\": \"fp864x-fp664-curated\",\n"
      << "  \"timestamp_utc\": \"" << json_escape(stamp) << "\",\n"
      << "  \"device\": {\n"
      << "    \"name\": \"" << json_escape(device_name) << "\",\n"
      << "    \"compute_capability\": \"" << major << "." << minor << "\",\n"
      << "    \"sm_count\": " << sms << ",\n"
      << "    \"reported_clock_mhz\": " << std::fixed << std::setprecision(3)
      << (clock_khz / 1000.0) << ",\n"
      << "    \"driver_api\": " << driver_version << ",\n"
      << "    \"ptx\": \"9.0\",\n"
      << "    \"target\": \"sm_120a\"\n"
      << "  },\n"
      << "  \"config\": {\n"
      << "    \"iters\": " << opt.iters << ",\n"
      << "    \"blocks_per_sm\": " << opt.blocks_per_sm << ",\n"
      << "    \"chains\": " << opt.chains << ",\n"
      << "    \"inner_unroll\": " << kInnerUnroll << ",\n"
      << "    \"repeats\": " << opt.repeats << ",\n"
      << "    \"filter\": \"" << json_escape(opt.filter) << "\"\n"
      << "  },\n"
      << "  \"summary\": {\n"
      << "    \"selected\": " << results.size() << ",\n"
      << "    \"passed\": " << passed << ",\n"
      << "    \"failed\": " << failed << ",\n"
      << "    \"peaks\": {\n";

  const std::vector<std::string> families = {
      "unscaled_f8f6f4", "mxf8f6f4", "mxf4", "mxf4nvf4"};
  for (size_t i = 0; i < families.size(); ++i) {
    const auto it = peaks.find(families[i]);
    out << "      \"" << families[i] << "\": ";
    if (it == peaks.end() || it->second.case_name.empty()) {
      out << "null";
    } else {
      out << "{\"tflops\": " << std::fixed << std::setprecision(6)
          << it->second.tflops << ", \"case_name\": \""
          << json_escape(it->second.case_name) << "\"}";
    }
    out << (i + 1 == families.size() ? "\n" : ",\n");
  }

  out << "    }\n"
      << "  },\n"
      << "  \"cases\": [\n";

  for (size_t i = 0; i < results.size(); ++i) {
    const CuratedResult& r = results[i];
    out << "    {\n"
        << "      \"status\": \"" << json_escape(r.status) << "\",\n"
        << "      \"name\": \"" << json_escape(r.name) << "\",\n"
        << "      \"opcode\": \"" << json_escape(r.opcode) << "\",\n"
        << "      \"error\": ";
    if (r.error.empty()) out << "null,\n";
    else out << "\"" << json_escape(r.error) << "\",\n";
    out << "      \"best_ms\": " << std::fixed << std::setprecision(6) << r.best_ms << ",\n"
        << "      \"mean_ms\": " << r.mean_ms << ",\n"
        << "      \"peak_logical_tflops\": " << r.peak_logical_tflops << ",\n"
        << "      \"mean_logical_tflops\": " << r.mean_logical_tflops << ",\n"
        << "      \"regs_per_thread\": " << r.regs_per_thread << "\n"
        << "    }" << (i + 1 == results.size() ? "\n" : ",\n");
  }

  out << "  ]\n"
      << "}\n";
}

void print_peak_table(const std::map<std::string, FamilyPeak>& peaks) {
  std::cout << "\nCurated dense peaks (TFLOP/s)\n"
            << std::left << std::setw(20) << "family"
            << std::right << std::setw(14) << "peak" << "  case\n";
  const std::vector<std::string> families = {
      "unscaled_f8f6f4", "mxf8f6f4", "mxf4", "mxf4nvf4"};
  for (const auto& family : families) {
    const auto it = peaks.find(family);
    std::cout << std::left << std::setw(20) << family;
    if (it == peaks.end() || it->second.case_name.empty()) {
      std::cout << std::right << std::setw(14) << "-" << "  -\n";
    } else {
      std::ostringstream value;
      value << std::fixed << std::setprecision(2) << it->second.tflops;
      std::cout << std::right << std::setw(14) << value.str()
                << "  " << it->second.case_name << '\n';
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const CuratedOptions ropt = parse_curated_args(argc, argv);
    const Options& opt = ropt.bench;
    const std::vector<Case> cases = base_manifest();

    if (opt.list_only) {
      size_t shown = 0;
      for (const Case& c : cases) {
        if (!is_curated_case(c)) continue;
        if (!opt.filter.empty() && c.name.find(opt.filter) == std::string::npos) continue;
        std::cout << c.name << " :: " << c.opcode << '\n';
        ++shown;
      }
      std::cout << "listed=" << shown
                << " curated_manifest=" << curated_case_names().size() << '\n';
      return shown ? 0 : 5;
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

    if (major != 12 || minor != 0) {
      std::cerr << "out-of-scope device: this benchmark is for SM120\n";
      return 2;
    }

    CUcontext ctx;
    check(cuDevicePrimaryCtxRetain(&ctx, dev), "cuDevicePrimaryCtxRetain");
    check(cuCtxSetCurrent(ctx), "cuCtxSetCurrent");

    const std::string stamp = utc_stamp();
    const fs::path run_dir = ropt.output_dir.empty()
        ? fs::path("results") / (stamp + "_" + slug(device_name) + "_fp864x_fp664")
        : fs::path(ropt.output_dir);
    fs::create_directories(run_dir);
    fs::create_directories(run_dir / "ptx");
    std::ofstream case_log(run_dir / "cases.log", std::ios::binary);
    if (!case_log) throw std::runtime_error("cannot write " + (run_dir / "cases.log").string());

    {
      std::ostringstream line;
      line << "suite=fp864x-fp664-curated"
           << " cases=" << curated_case_names().size()
           << " dense_only=yes sparse=no accumulator=f32"
           << " device=\"" << device_name << "\""
           << " cc=" << major << '.' << minor
           << " sm_count=" << sms
           << " reported_clock_mhz=" << (clock_khz / 1000.0)
           << " driver_api=" << driver_version
           << " ptx=9.0 target=sm_120a"
           << " inner_unroll=" << kInnerUnroll
           << " output_dir=\"" << run_dir.string() << "\"";
      emit_line(case_log, line.str(), false);
    }

    constexpr unsigned threads = 256;
    constexpr unsigned warps_per_block = threads / 32;
    const unsigned blocks = static_cast<unsigned>(sms) * opt.blocks_per_sm;
    CUdeviceptr output = 0;
    check(cuMemAlloc(
              &output,
              static_cast<size_t>(blocks) * warps_per_block * sizeof(uint32_t)),
          "cuMemAlloc");

    CUevent start, stop;
    check(cuEventCreate(&start, CU_EVENT_DEFAULT), "cuEventCreate(start)");
    check(cuEventCreate(&stop, CU_EVENT_DEFAULT), "cuEventCreate(stop)");

    std::vector<CuratedResult> results;
    size_t selected = 0;
    for (const Case& c : cases) {
      if (!is_curated_case(c)) continue;
      if (!opt.filter.empty() && c.name.find(opt.filter) == std::string::npos) continue;
      ++selected;

      CuratedResult result;
      result.name = c.name;
      result.opcode = c.opcode;

      const std::string ptx = build_ptx(c, opt.chains);
      const fs::path ptx_path = run_dir / "ptx" / (slug(c.name) + ".ptx");
      {
        std::ofstream ptx_out(ptx_path, std::ios::binary);
        if (!ptx_out) throw std::runtime_error("cannot write " + ptx_path.string());
        ptx_out << ptx;
      }

      LoadedModule mod;
      const CUresult jit_result = load_ptx(ptx, mod);
      if (jit_result != CUDA_SUCCESS) {
        result.status = "FAIL_DOCUMENTED";
        result.error = cuda_error(jit_result);
        emit_line(
            case_log,
            "FAIL_DOCUMENTED name=" + c.name + " error=\"" + result.error +
                "\" ptx=\"" + ptx_path.string() + "\"",
            ropt.quiet_cases);
        if (!mod.log.empty()) {
          emit_line(case_log, "JIT_LOG name=" + c.name + " " + mod.log, ropt.quiet_cases);
        }
        results.push_back(std::move(result));
        continue;
      }

      unsigned warm_iters = std::min(opt.iters, 256u);
      void* warm_args[] = {&output, &warm_iters};
      check(cuLaunchKernel(
                mod.function, blocks, 1, 1, threads, 1, 1,
                0, nullptr, warm_args, nullptr),
            "warmup launch");
      check(cuCtxSynchronize(), "warmup synchronize");

      float best_ms = std::numeric_limits<float>::infinity();
      float sum_ms = 0.0f;
      for (unsigned rep = 0; rep < opt.repeats; ++rep) {
        unsigned timed_iters = opt.iters;
        void* args[] = {&output, &timed_iters};
        check(cuEventRecord(start, nullptr), "event start");
        check(cuLaunchKernel(
                  mod.function, blocks, 1, 1, threads, 1, 1,
                  0, nullptr, args, nullptr),
              "timed launch");
        check(cuEventRecord(stop, nullptr), "event stop");
        check(cuEventSynchronize(stop), "event synchronize");
        float ms = 0.0f;
        check(cuEventElapsedTime(&ms, start, stop), "event elapsed");
        best_ms = std::min(best_ms, ms);
        sum_ms += ms;
      }

      check(cuFuncGetAttribute(
                &result.regs_per_thread,
                CU_FUNC_ATTRIBUTE_NUM_REGS,
                mod.function),
            "register count");

      const long double warp_count =
          static_cast<long double>(blocks) * warps_per_block;
      const long double instruction_count =
          warp_count * opt.iters * opt.chains * kInnerUnroll;
      const long double flops_per_instruction = 2.0L * c.m * c.n * c.k;
      result.best_ms = best_ms;
      result.mean_ms = sum_ms / opt.repeats;
      result.peak_logical_tflops = static_cast<double>(
          instruction_count * flops_per_instruction /
          (static_cast<long double>(result.best_ms) / 1000.0L) / 1.0e12L);
      result.mean_logical_tflops = static_cast<double>(
          instruction_count * flops_per_instruction /
          (static_cast<long double>(result.mean_ms) / 1000.0L) / 1.0e12L);
      result.status = "PASS";

      std::ostringstream line;
      line << std::fixed << std::setprecision(3)
           << "PASS name=" << c.name
           << " best_ms=" << result.best_ms
           << " peak_logical_tflops=" << result.peak_logical_tflops
           << " mean_logical_tflops=" << result.mean_logical_tflops
           << " regs_per_thread=" << result.regs_per_thread
           << " blocks=" << blocks
           << " threads=" << threads
           << " chains=" << opt.chains
           << " inner_unroll=" << kInnerUnroll
           << " iters=" << opt.iters
           << " repeats=" << opt.repeats;
      emit_line(case_log, line.str(), ropt.quiet_cases);

      if (opt.verbose_jit && !mod.log.empty()) {
        emit_line(case_log, mod.log, ropt.quiet_cases);
      }
      unload(mod);
      results.push_back(std::move(result));
    }

    const auto peaks = extract_family_peaks(results);
    const fs::path result_json = run_dir / "result.json";
    write_result_json(
        result_json, stamp, device_name, major, minor, sms, clock_khz,
        driver_version, opt, results, peaks);

    size_t passed = 0;
    for (const auto& r : results) if (r.status == "PASS") ++passed;
    const size_t failed = results.size() - passed;

    print_peak_table(peaks);
    std::cout << "\nselected=" << selected
              << " passed=" << passed
              << " failed=" << failed
              << " curated_manifest=" << curated_case_names().size() << '\n'
              << "Saved log: " << (run_dir / "cases.log") << '\n'
              << "Saved JSON: " << result_json << '\n';

    cuEventDestroy(start);
    cuEventDestroy(stop);
    cuMemFree(output);
    cuDevicePrimaryCtxRelease(dev);
    return selected ? 0 : 5;
  } catch (const std::exception& e) {
    std::cerr << "fatal: " << e.what() << '\n';
    return 1;
  }
}
