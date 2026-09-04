// Enhanced runner/reporting layer for tensorcore-fp864-bench.
//
// The audited PTX manifest and PTX generator live in bench.cpp.  We include
// that translation unit with its legacy main renamed so there is exactly one
// source of truth for instruction spellings while this file owns persistence,
// cubin capture, SASS verification, and headline peak extraction.

#include <chrono>
#include <cctype>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#define main tensorcore_fp864_legacy_main
#include "bench.cpp"
#undef main

namespace fs = std::filesystem;

namespace {

struct ReportOptions {
  Options bench;
  std::string output_dir;
  std::string nvdisasm = "nvdisasm";
  bool verify_sass = true;
  bool quiet_cases = false;
};

struct DeviceInfo {
  std::string name;
  int cc_major = 0;
  int cc_minor = 0;
  int sm_count = 0;
  int clock_khz = 0;
  int driver_version = 0;
};

struct CompiledModule {
  CUmodule module = nullptr;
  CUfunction function = nullptr;
  std::vector<unsigned char> cubin;
  std::string log;
};

struct SassAudit {
  std::string status = "NOT_CHECKED";
  std::string expectation;
  std::string family;
  std::string text_path;
  std::string json_path;
  int expected_mma_count = 0;
  int matched_mma_count = 0;
  int total_tensor_mma_count = 0;
};

struct Result {
  std::string status;
  std::string name;
  std::string doc_class;
  std::string opcode;
  std::string note;
  bool sparse = false;
  bool block_scale = false;
  bool acc_f16 = false;
  int m = 0, n = 0, k = 0;
  int regs_per_thread = 0;
  double best_ms = 0.0;
  double mean_ms = 0.0;
  double peak_logical_tflops = 0.0;
  double mean_logical_tflops = 0.0;
  double peak_nonzero_tflops = 0.0;
  std::string ptx_path;
  std::string cubin_path;
  std::string jit_log;
  SassAudit sass;
};

struct Peak {
  bool available = false;
  std::string mode;
  std::string pair;
  std::string format_a;
  std::string format_b;
  std::string direction;
  std::string accumulator;
  std::string case_name;
  std::string opcode;
  std::string sass_status;
  std::string sass_family;
  double tflops = 0.0;
};

std::string json_escape(const std::string& s) {
  std::ostringstream o;
  for (unsigned char ch : s) {
    switch (ch) {
      case '\\': o << "\\\\"; break;
      case '"': o << "\\\""; break;
      case '\n': o << "\\n"; break;
      case '\r': o << "\\r"; break;
      case '\t': o << "\\t"; break;
      default:
        if (ch < 0x20) {
          o << "\\u" << std::hex << std::setw(4) << std::setfill('0')
            << static_cast<int>(ch) << std::dec << std::setfill(' ');
        } else {
          o << static_cast<char>(ch);
        }
    }
  }
  return o.str();
}

std::string csv_escape(const std::string& s) {
  bool quote = false;
  for (char c : s) if (c == ',' || c == '"' || c == '\n' || c == '\r') quote = true;
  if (!quote) return s;
  std::string out = "\"";
  for (char c : s) {
    if (c == '"') out += "\"\"";
    else out += c;
  }
  out += '"';
  return out;
}

std::string upper(std::string s) {
  for (char& c : s) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
  return s;
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

uint64_t fnv1a64(const std::string& s) {
  uint64_t h = 1469598103934665603ULL;
  for (unsigned char c : s) {
    h ^= c;
    h *= 1099511628211ULL;
  }
  return h;
}

std::string case_stem(const Case& c) {
  std::ostringstream os;
  os << slug(c.name) << "_" << std::hex << std::setw(12) << std::setfill('0')
     << (fnv1a64(c.name + "|" + c.opcode) & 0xffffffffffffULL);
  return os.str();
}

std::string shell_quote(const std::string& s) {
#if defined(_WIN32)
  std::string out = "\"";
  for (char c : s) {
    if (c == '"') out += "\\\"";
    else out += c;
  }
  out += '"';
  return out;
#else
  std::string out = "'";
  for (char c : s) {
    if (c == '\'') out += "'\\''";
    else out += c;
  }
  out += '\'';
  return out;
#endif
}

std::string read_text(const fs::path& p) {
  std::ifstream f(p, std::ios::binary);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

void write_text(const fs::path& p, const std::string& s) {
  fs::create_directories(p.parent_path());
  std::ofstream f(p, std::ios::binary);
  if (!f) throw std::runtime_error("cannot write " + p.string());
  f << s;
}

void write_binary(const fs::path& p, const std::vector<unsigned char>& data) {
  fs::create_directories(p.parent_path());
  std::ofstream f(p, std::ios::binary);
  if (!f) throw std::runtime_error("cannot write " + p.string());
  f.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
}

ReportOptions parse_report_args(int argc, char** argv) {
  ReportOptions o;
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
    else if (a == "--include-probes") o.bench.include_probes = true;
    else if (a == "--probes-only") { o.bench.include_probes = true; o.bench.probes_only = true; }
    else if (a == "--all-operands") o.bench.all_operands = true;
    else if (a == "--list") o.bench.list_only = true;
    else if (a == "--verbose-jit") o.bench.verbose_jit = true;
    else if (a == "--output-dir") o.output_dir = value("--output-dir");
    else if (a == "--nvdisasm") o.nvdisasm = value("--nvdisasm");
    else if (a == "--no-sass-check") o.verify_sass = false;
    else if (a == "--quiet-cases") o.quiet_cases = true;
    else if (a == "-h" || a == "--help") {
      std::cout
          << "--device N --iters N --blocks-per-sm N --chains {1,2,4,8} --repeats N\n"
          << "--filter TEXT --include-probes --probes-only --all-operands --list --verbose-jit\n"
          << "--output-dir DIR --nvdisasm PATH --no-sass-check --quiet-cases\n\n"
          << "SASS verification is ON by default. A documented case is timed only after the\n"
          << "generated cubin passes structural nvdisasm checks.\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown option: " + a);
    }
  }
  if (!(o.bench.chains == 1 || o.bench.chains == 2 || o.bench.chains == 4 || o.bench.chains == 8)) {
    throw std::runtime_error("--chains must be one of 1,2,4,8");
  }
  if (!o.bench.iters || !o.bench.blocks_per_sm || !o.bench.repeats) {
    throw std::runtime_error("iteration/work counts must be non-zero");
  }
  return o;
}

CUresult compile_ptx_to_cubin(const std::string& ptx, CompiledModule& out) {
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

  CUlinkState link = nullptr;
  CUresult r = cuLinkCreate(6, options, values, &link);
  if (r != CUDA_SUCCESS) {
    out.log = std::string(error_log) + std::string(info_log);
    return r;
  }

  r = cuLinkAddData(link, CU_JIT_INPUT_PTX,
                    const_cast<char*>(ptx.c_str()), ptx.size() + 1,
                    "bench.ptx", 0, nullptr, nullptr);
  if (r == CUDA_SUCCESS) {
    void* image = nullptr;
    size_t size = 0;
    r = cuLinkComplete(link, &image, &size);
    if (r == CUDA_SUCCESS) {
      const auto* begin = static_cast<const unsigned char*>(image);
      out.cubin.assign(begin, begin + size);
      r = cuModuleLoadData(&out.module, image);
      if (r == CUDA_SUCCESS) r = cuModuleGetFunction(&out.function, out.module, "bench");
      if (r != CUDA_SUCCESS && out.module) {
        cuModuleUnload(out.module);
        out.module = nullptr;
      }
    }
  }
  out.log = std::string(error_log) + std::string(info_log);
  cuLinkDestroy(link);
  return r;
}

void unload_compiled(CompiledModule& m) {
  if (m.module) cuModuleUnload(m.module);
  m.module = nullptr;
  m.function = nullptr;
}

std::pair<std::string, std::string> formats_from_name(const std::string& name) {
  size_t pos = 0;
  while (pos < name.size()) {
    const size_t end = name.find('/', pos);
    const std::string part = name.substr(pos, end == std::string::npos ? std::string::npos : end - pos);
    const size_t x = part.find('x');
    if (x != std::string::npos && !part.empty() && part[0] == 'e') {
      return {part.substr(0, x), part.substr(x + 1)};
    }
    if (end == std::string::npos) break;
    pos = end + 1;
  }
  return {"", ""};
}

int format_bits(const std::string& f) {
  if (f == "e4m3" || f == "e5m2") return 8;
  if (f == "e3m2" || f == "e2m3") return 6;
  if (f == "e2m1") return 4;
  return 0;
}

std::string pair_class(const Case& c) {
  const auto [a, b] = formats_from_name(c.name);
  int x = format_bits(a), y = format_bits(b);
  if (!x || !y) return "";
  if (x < y) std::swap(x, y);
  if (x == y) return "fp" + std::to_string(x);
  return "fp" + std::to_string(x) + "xfp" + std::to_string(y);
}

std::string peak_mode(const Case& c) {
  if (c.sparse || c.name.rfind("dense/", 0) != 0) return "";
  if (c.name.find("dense/mxf4nvf4/") == 0) return "with_nvfp4";
  if (c.name.find("dense/mxf") == 0) return "with_mx";
  return "without_mx";
}

std::string expected_sass_family(const Case& c) {
  const bool omma = c.name.find("mxf4/") != std::string::npos ||
                    c.name.find("mxf4nvf4/") != std::string::npos;
  std::string f = omma ? "OMMA" : "QMMA";
  if (c.block_scale) f += ".SF";
  if (c.sparse) f += ".SP";
  return f;
}

bool line_has_tensor_mma(const std::string& line) {
  return line.find("HMMA") != std::string::npos ||
         line.find("QMMA") != std::string::npos ||
         line.find("OMMA") != std::string::npos ||
         line.find("IMMA") != std::string::npos ||
         line.find("DMMA") != std::string::npos;
}

SassAudit verify_sass(const Case& c, unsigned chains, const fs::path& cubin,
                      const fs::path& sass_dir, const std::string& nvdisasm,
                      bool tool_ok) {
  SassAudit a;
  a.family = expected_sass_family(c);
  a.expected_mma_count = static_cast<int>(chains * kInnerUnroll);
  const auto [fa, fb] = formats_from_name(c.name);
  const std::string shape = ".168" + std::to_string(c.k);
  const std::string accum = c.acc_f16 ? ".F16" : ".F32";
  a.expectation = a.family + shape + accum;
  if (!fa.empty()) a.expectation += "." + upper(fa) + "." + upper(fb);

  if (!tool_ok) {
    a.status = "SASS_TOOL_ERROR";
    return a;
  }

  const std::string stem = cubin.stem().string();
  const fs::path text = sass_dir / (stem + ".sass.txt");
  const fs::path json = sass_dir / (stem + ".sass.json");
  const fs::path err = sass_dir / (stem + ".nvdisasm.log");
  a.text_path = text.string();
  a.json_path = json.string();

  const std::string cmd_text = shell_quote(nvdisasm) + " -c -ndf " + shell_quote(cubin.string()) +
                               " > " + shell_quote(text.string()) + " 2> " + shell_quote(err.string());
  const int rc = std::system(cmd_text.c_str());
  if (rc != 0) {
    a.status = "SASS_DISASM_ERROR";
    return a;
  }

  // Also persist NVIDIA's native machine-readable disassembly. Failure of the
  // JSON sidecar does not invalidate the structural text check.
  const std::string cmd_json = shell_quote(nvdisasm) + " -json -ndf " + shell_quote(cubin.string()) +
                               " > " + shell_quote(json.string()) + " 2>> " + shell_quote(err.string());
  (void)std::system(cmd_json.c_str());

  std::istringstream in(read_text(text));
  std::string line;
  while (std::getline(in, line)) {
    if (!line_has_tensor_mma(line)) continue;
    ++a.total_tensor_mma_count;
    bool match = line.find(a.family) != std::string::npos &&
                 line.find(shape) != std::string::npos &&
                 line.find(accum) != std::string::npos;
    if (!fa.empty()) {
      match = match && line.find(upper(fa)) != std::string::npos &&
                       line.find(upper(fb)) != std::string::npos;
    }
    if (match) ++a.matched_mma_count;
  }

  if (a.matched_mma_count == a.expected_mma_count &&
      a.total_tensor_mma_count == a.expected_mma_count) {
    a.status = "SASS_OK";
  } else {
    a.status = "SASS_MISMATCH";
  }
  return a;
}

Peak make_peak(const Result& r, const Case& c) {
  Peak p;
  p.available = true;
  p.mode = peak_mode(c);
  p.pair = pair_class(c);
  const auto [a, b] = formats_from_name(c.name);
  p.format_a = a;
  p.format_b = b;
  p.direction = "fp" + std::to_string(format_bits(a)) + "xfp" + std::to_string(format_bits(b));
  p.accumulator = c.acc_f16 ? "f16" : "f32";
  p.case_name = c.name;
  p.opcode = c.opcode;
  p.sass_status = r.sass.status;
  p.sass_family = r.sass.family;
  p.tflops = r.peak_logical_tflops;
  return p;
}

std::map<std::pair<std::string, std::string>, Peak>
extract_peaks(const std::vector<std::pair<Case, Result>>& records, bool sass_required) {
  std::map<std::pair<std::string, std::string>, Peak> peaks;
  for (const auto& cr : records) {
    const Case& c = cr.first;
    const Result& r = cr.second;
    if (c.doc != DocClass::Documented || c.sparse || r.status != "PASS") continue;
    if (sass_required && r.sass.status != "SASS_OK") continue;
    const std::string mode = peak_mode(c);
    const std::string pair = pair_class(c);
    if (mode.empty() || pair.empty()) continue;
    const auto key = std::make_pair(mode, pair);
    auto it = peaks.find(key);
    if (it == peaks.end() || r.peak_logical_tflops > it->second.tflops) {
      peaks[key] = make_peak(r, c);
    }
  }
  return peaks;
}

void write_results_csv(const fs::path& path, const std::vector<std::pair<Case, Result>>& records) {
  std::ofstream f(path);
  f << "status,doc_class,name,sparse,block_scale,accumulator,m,n,k,best_ms,mean_ms,peak_logical_tflops,mean_logical_tflops,peak_nonzero_tflops,regs_per_thread,sass_status,sass_family,sass_expected_count,sass_matched_count,sass_total_tensor_count,opcode,ptx_path,cubin_path,sass_text_path,sass_json_path,note\n";
  for (const auto& cr : records) {
    const Result& r = cr.second;
    f << csv_escape(r.status) << ',' << csv_escape(r.doc_class) << ',' << csv_escape(r.name) << ','
      << (r.sparse ? 1 : 0) << ',' << (r.block_scale ? 1 : 0) << ','
      << (r.acc_f16 ? "f16" : "f32") << ',' << r.m << ',' << r.n << ',' << r.k << ','
      << std::setprecision(10) << r.best_ms << ',' << r.mean_ms << ',' << r.peak_logical_tflops << ','
      << r.mean_logical_tflops << ',' << r.peak_nonzero_tflops << ',' << r.regs_per_thread << ','
      << csv_escape(r.sass.status) << ',' << csv_escape(r.sass.family) << ','
      << r.sass.expected_mma_count << ',' << r.sass.matched_mma_count << ',' << r.sass.total_tensor_mma_count << ','
      << csv_escape(r.opcode) << ',' << csv_escape(r.ptx_path) << ',' << csv_escape(r.cubin_path) << ','
      << csv_escape(r.sass.text_path) << ',' << csv_escape(r.sass.json_path) << ',' << csv_escape(r.note) << '\n';
  }
}

void write_run_json(const fs::path& path, const DeviceInfo& d, const ReportOptions& opt,
                    const std::vector<std::pair<Case, Result>>& records,
                    bool sass_tool_ok, const std::string& run_stamp) {
  std::ofstream f(path);
  f << "{\n  \"schema_version\": 1,\n"
    << "  \"timestamp_utc\": \"" << json_escape(run_stamp) << "\",\n"
    << "  \"device\": {\"name\": \"" << json_escape(d.name) << "\", \"cc\": \""
    << d.cc_major << '.' << d.cc_minor << "\", \"sm_count\": " << d.sm_count
    << ", \"reported_clock_mhz\": " << (d.clock_khz / 1000.0)
    << ", \"driver_api\": " << d.driver_version << "},\n"
    << "  \"ptx\": {\"version\": \"9.1\", \"target\": \"sm_120a\"},\n"
    << "  \"sass\": {\"required\": " << (opt.verify_sass ? "true" : "false")
    << ", \"tool\": \"" << json_escape(opt.nvdisasm) << "\", \"tool_ok\": " << (sass_tool_ok ? "true" : "false") << "},\n"
    << "  \"benchmark\": {\"iters\": " << opt.bench.iters << ", \"blocks_per_sm\": " << opt.bench.blocks_per_sm
    << ", \"chains\": " << opt.bench.chains << ", \"inner_unroll\": " << kInnerUnroll
    << ", \"repeats\": " << opt.bench.repeats << "},\n"
    << "  \"cases\": [\n";
  for (size_t i = 0; i < records.size(); ++i) {
    const Result& r = records[i].second;
    f << "    {\"status\": \"" << json_escape(r.status) << "\", \"doc_class\": \"" << json_escape(r.doc_class)
      << "\", \"name\": \"" << json_escape(r.name) << "\", \"opcode\": \"" << json_escape(r.opcode)
      << "\", \"sparse\": " << (r.sparse ? "true" : "false") << ", \"block_scale\": " << (r.block_scale ? "true" : "false")
      << ", \"accumulator\": \"" << (r.acc_f16 ? "f16" : "f32") << "\", \"shape\": [" << r.m << ',' << r.n << ',' << r.k << ']'
      << ", \"best_ms\": " << std::setprecision(12) << r.best_ms << ", \"mean_ms\": " << r.mean_ms
      << ", \"peak_logical_tflops\": " << r.peak_logical_tflops << ", \"mean_logical_tflops\": " << r.mean_logical_tflops
      << ", \"peak_nonzero_tflops\": " << r.peak_nonzero_tflops << ", \"regs_per_thread\": " << r.regs_per_thread
      << ", \"sass\": {\"status\": \"" << json_escape(r.sass.status) << "\", \"family\": \"" << json_escape(r.sass.family)
      << "\", \"expectation\": \"" << json_escape(r.sass.expectation) << "\", \"expected_mma_count\": " << r.sass.expected_mma_count
      << ", \"matched_mma_count\": " << r.sass.matched_mma_count << ", \"total_tensor_mma_count\": " << r.sass.total_tensor_mma_count
      << ", \"text_path\": \"" << json_escape(r.sass.text_path) << "\", \"json_path\": \"" << json_escape(r.sass.json_path) << "\"}"
      << ", \"ptx_path\": \"" << json_escape(r.ptx_path) << "\", \"cubin_path\": \"" << json_escape(r.cubin_path)
      << "\", \"note\": \"" << json_escape(r.note) << "\"}" << (i + 1 == records.size() ? "\n" : ",\n");
  }
  f << "  ]\n}\n";
}

const std::vector<std::string>& headline_pairs() {
  static const std::vector<std::string> p = {"fp8", "fp6", "fp4", "fp8xfp6", "fp8xfp4", "fp6xfp4"};
  return p;
}

void write_peak_json(const fs::path& path, const DeviceInfo& d,
                     const std::map<std::pair<std::string, std::string>, Peak>& peaks,
                     bool sass_required) {
  const std::vector<std::string> modes = {"without_mx", "with_mx"};
  std::ofstream f(path);
  f << "{\n  \"schema_version\": 1,\n  \"device\": \"" << json_escape(d.name)
    << "\",\n  \"scope\": \"dense documented PTX only\",\n  \"sass_required\": " << (sass_required ? "true" : "false") << ",\n  \"headline\": [\n";
  bool first = true;
  for (const auto& mode : modes) for (const auto& pair : headline_pairs()) {
    if (!first) f << ",\n";
    first = false;
    const auto it = peaks.find({mode, pair});
    f << "    {\"mode\": \"" << mode << "\", \"pair\": \"" << pair << "\", \"available\": ";
    if (it == peaks.end()) {
      f << "false}";
    } else {
      const Peak& p = it->second;
      f << "true, \"peak_tflops\": " << std::setprecision(12) << p.tflops
        << ", \"format_a\": \"" << p.format_a << "\", \"format_b\": \"" << p.format_b
        << "\", \"direction\": \"" << p.direction << "\", \"accumulator\": \"" << p.accumulator
        << "\", \"case\": \"" << json_escape(p.case_name) << "\", \"ptx\": \"" << json_escape(p.opcode)
        << "\", \"sass_status\": \"" << p.sass_status << "\", \"sass_family\": \"" << p.sass_family << "\"}";
    }
  }
  // NVFP4 is intentionally separate: it is block-scaled FP4, but it is not MX.
  const auto nv = peaks.find({"with_nvfp4", "fp4"});
  f << "\n  ],\n  \"nvfp4_reference\": ";
  if (nv == peaks.end()) {
    f << "{\"available\": false}\n";
  } else {
    const Peak& p = nv->second;
    f << "{\"available\": true, \"peak_tflops\": " << p.tflops
      << ", \"format_a\": \"" << p.format_a << "\", \"format_b\": \"" << p.format_b
      << "\", \"case\": \"" << json_escape(p.case_name) << "\", \"ptx\": \"" << json_escape(p.opcode)
      << "\", \"sass_status\": \"" << p.sass_status << "\", \"sass_family\": \"" << p.sass_family << "\"}\n";
  }
  f << "}\n";
}

std::string peak_cell(const std::map<std::pair<std::string, std::string>, Peak>& peaks,
                      const std::string& mode, const std::string& pair) {
  const auto it = peaks.find({mode, pair});
  if (it == peaks.end()) return "—";
  std::ostringstream os;
  os << std::fixed << std::setprecision(2) << it->second.tflops;
  return os.str();
}

void write_summary_md(const fs::path& path, const DeviceInfo& d, const ReportOptions& opt,
                      const std::map<std::pair<std::string, std::string>, Peak>& peaks,
                      const std::vector<std::pair<Case, Result>>& records, bool sass_tool_ok) {
  size_t pass = 0, sass_ok = 0, failed = 0;
  for (const auto& cr : records) {
    if (cr.second.status == "PASS") ++pass;
    else if (cr.first.doc == DocClass::Documented) ++failed;
    if (cr.second.sass.status == "SASS_OK") ++sass_ok;
  }

  std::ofstream f(path);
  f << "# Tensor Core FP8/FP6/FP4 peak summary\n\n"
    << "Device: **" << d.name << "** · CC " << d.cc_major << '.' << d.cc_minor
    << " · " << d.sm_count << " SMs · PTX 9.1 / sm_120a\n\n"
    << "SASS gate: **" << (opt.verify_sass ? (sass_tool_ok ? "required and available" : "required but nvdisasm unavailable") : "disabled by user")
    << "**. A `PASS` case has been timed only after SASS validation when the gate is enabled.\n\n"
    << "## Headline dense peaks (TFLOP/s)\n\n"
    << "| Precision pair | Without MX | With MX |\n|---|---:|---:|\n";
  for (const auto& pair : headline_pairs()) {
    f << "| " << pair << " | " << peak_cell(peaks, "without_mx", pair)
      << " | " << peak_cell(peaks, "with_mx", pair) << " |\n";
  }

  f << "\n`with_mx` means the documented MX paths (`mxf8f6f4` / `mxf4`). `mxf4nvf4` is not folded into MX; it is reported separately below. Mixed categories take the maximum over both A×B orientations and all documented subformats; the winning orientation and exact PTX are shown in the detail table.\n\n"
    << "## Winning cases\n\n"
    << "| Mode | Pair | Peak TFLOP/s | A×B | Acc | SASS | PTX case |\n|---|---|---:|---|---|---|---|\n";
  for (const std::string mode : {std::string("without_mx"), std::string("with_mx")}) {
    for (const auto& pair : headline_pairs()) {
      const auto it = peaks.find({mode, pair});
      if (it == peaks.end()) continue;
      const Peak& p = it->second;
      f << "| " << mode << " | " << pair << " | " << std::fixed << std::setprecision(2) << p.tflops
        << " | " << p.format_a << " × " << p.format_b << " | " << p.accumulator
        << " | `" << p.sass_family << "` / " << p.sass_status << " | `" << p.case_name << "` |\n";
    }
  }
  const auto nv = peaks.find({"with_nvfp4", "fp4"});
  if (nv != peaks.end()) {
    const Peak& p = nv->second;
    f << "\n## NVFP4 reference\n\n"
      << "Best NVFP4-family FP4 result: **" << std::fixed << std::setprecision(2) << p.tflops
      << " TFLOP/s**, case `" << p.case_name << "`, SASS `" << p.sass_family << "` / " << p.sass_status << ".\n";
  }

  f << "\n## Run health\n\n"
    << "- Timed PASS cases: " << pass << "\n"
    << "- SASS_OK cases: " << sass_ok << "\n"
    << "- Documented cases not reaching PASS: " << failed << "\n"
    << "- Full machine-readable records: `run.json` and `results.csv`\n"
    << "- Headline machine-readable summary: `peak_summary.json`\n"
    << "- Per-case PTX/CUBIN/SASS: `ptx/`, `cubin/`, `sass/`\n";
}

void emit_line(std::ofstream& log, const std::string& line, bool quiet) {
  log << line << '\n';
  log.flush();
  if (!quiet) std::cout << line << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const ReportOptions ropt = parse_report_args(argc, argv);
    const Options& opt = ropt.bench;
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

    DeviceInfo di;
    char device_name[256] = {};
    check(cuDeviceGetName(device_name, sizeof(device_name), dev), "cuDeviceGetName");
    di.name = device_name;
    check(cuDeviceGetAttribute(&di.cc_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev), "CC major");
    check(cuDeviceGetAttribute(&di.cc_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev), "CC minor");
    check(cuDeviceGetAttribute(&di.sm_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev), "SM count");
    check(cuDeviceGetAttribute(&di.clock_khz, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, dev), "clock rate");
    check(cuDriverGetVersion(&di.driver_version), "driver version");

    if (di.cc_major != 12 || di.cc_minor != 0) {
      std::cerr << "out-of-scope device: this benchmark is for SM120\n";
      return 2;
    }

    CUcontext ctx;
    check(cuDevicePrimaryCtxRetain(&ctx, dev), "cuDevicePrimaryCtxRetain");
    check(cuCtxSetCurrent(ctx), "cuCtxSetCurrent");

    const std::string stamp = utc_stamp();
    fs::path run_dir = ropt.output_dir.empty()
        ? fs::path("results") / (stamp + "_" + slug(di.name))
        : fs::path(ropt.output_dir);
    fs::create_directories(run_dir / "ptx");
    fs::create_directories(run_dir / "cubin");
    fs::create_directories(run_dir / "sass");
    std::ofstream case_log(run_dir / "cases.log");

    bool sass_tool_ok = !ropt.verify_sass;
    if (ropt.verify_sass) {
      const fs::path version_file = run_dir / "sass" / "nvdisasm-version.txt";
      const std::string cmd = shell_quote(ropt.nvdisasm) + " --version > " + shell_quote(version_file.string()) + " 2>&1";
      sass_tool_ok = std::system(cmd.c_str()) == 0;
    }

    {
      std::ostringstream line;
      line << "device=\"" << di.name << "\" cc=" << di.cc_major << '.' << di.cc_minor
           << " sm_count=" << di.sm_count << " reported_clock_mhz=" << (di.clock_khz / 1000.0)
           << " driver_api=" << di.driver_version << " ptx=9.1 target=sm_120a"
           << " sass_required=" << (ropt.verify_sass ? "yes" : "no")
           << " sass_tool_ok=" << (sass_tool_ok ? "yes" : "no")
           << " output_dir=\"" << run_dir.string() << "\"";
      emit_line(case_log, line.str(), false);
    }

    constexpr unsigned threads = 256;
    constexpr unsigned warps_per_block = threads / 32;
    const unsigned blocks = static_cast<unsigned>(di.sm_count) * opt.blocks_per_sm;
    CUdeviceptr output = 0;
    check(cuMemAlloc(&output, static_cast<size_t>(blocks) * warps_per_block * sizeof(uint32_t)), "cuMemAlloc");
    CUevent start, stop;
    check(cuEventCreate(&start, CU_EVENT_DEFAULT), "cuEventCreate(start)");
    check(cuEventCreate(&stop, CU_EVENT_DEFAULT), "cuEventCreate(stop)");

    std::vector<std::pair<Case, Result>> records;
    size_t selected = 0;
    for (const Case& c : cases) {
      if (!opt.filter.empty() && c.name.find(opt.filter) == std::string::npos) continue;
      if (c.doc != DocClass::Documented && !opt.include_probes) continue;
      if (opt.probes_only && c.doc == DocClass::Documented) continue;
      ++selected;

      Result r;
      r.name = c.name;
      r.doc_class = doc_name(c.doc);
      r.opcode = c.custom_ptx ? "<custom PTX>" : c.opcode;
      r.note = c.note;
      r.sparse = c.sparse;
      r.block_scale = c.block_scale;
      r.acc_f16 = c.acc_f16;
      r.m = c.m; r.n = c.n; r.k = c.k;

      const std::string stem = case_stem(c);
      const fs::path ptx_path = run_dir / "ptx" / (stem + ".ptx");
      const fs::path cubin_path = run_dir / "cubin" / (stem + ".cubin");
      const std::string ptx = build_ptx(c, opt.chains);
      write_text(ptx_path, ptx);
      r.ptx_path = ptx_path.string();

      CompiledModule mod;
      const CUresult jit = compile_ptx_to_cubin(ptx, mod);
      r.jit_log = mod.log;
      if (jit != CUDA_SUCCESS) {
        r.status = c.doc == DocClass::Documented ? "FAIL_DOCUMENTED" :
                   c.doc == DocClass::DocAmbiguous ? "REJECTED_DOC_AMBIGUOUS" : "REJECTED_UNDOCUMENTED";
        std::ostringstream line;
        line << r.status << " name=" << c.name << " error=\"" << cuda_error(jit) << "\"";
        emit_line(case_log, line.str(), ropt.quiet_cases);
        records.push_back({c, r});
        continue;
      }

      write_binary(cubin_path, mod.cubin);
      r.cubin_path = cubin_path.string();

      // Probes are never executed. If nvdisasm is available, preserve their
      // observed SASS without claiming that the mapping is documented.
      if (c.doc != DocClass::Documented) {
        if (ropt.verify_sass && sass_tool_ok) {
          const fs::path text = run_dir / "sass" / (stem + ".sass.txt");
          const fs::path json = run_dir / "sass" / (stem + ".sass.json");
          const fs::path err = run_dir / "sass" / (stem + ".nvdisasm.log");
          const std::string cmd1 = shell_quote(ropt.nvdisasm) + " -c -ndf " + shell_quote(cubin_path.string()) +
                                   " > " + shell_quote(text.string()) + " 2> " + shell_quote(err.string());
          const int rc = std::system(cmd1.c_str());
          if (rc == 0) {
            r.sass.status = "SASS_OBSERVED_NON_DOCUMENTED";
            r.sass.text_path = text.string();
            r.sass.json_path = json.string();
            const std::string cmd2 = shell_quote(ropt.nvdisasm) + " -json -ndf " + shell_quote(cubin_path.string()) +
                                     " > " + shell_quote(json.string()) + " 2>> " + shell_quote(err.string());
            (void)std::system(cmd2.c_str());
          } else {
            r.sass.status = "SASS_DISASM_ERROR";
          }
        } else {
          r.sass.status = ropt.verify_sass ? "SASS_TOOL_ERROR" : "SASS_DISABLED";
        }
        r.status = c.doc == DocClass::DocAmbiguous ? "ACCEPTED_DOC_AMBIGUOUS" : "ACCEPTED_UNDOCUMENTED";
        emit_line(case_log, r.status + " name=" + c.name, ropt.quiet_cases);
        unload_compiled(mod);
        records.push_back({c, r});
        continue;
      }

      if (ropt.verify_sass) {
        r.sass = verify_sass(c, opt.chains, cubin_path, run_dir / "sass", ropt.nvdisasm, sass_tool_ok);
        if (r.sass.status != "SASS_OK") {
          r.status = r.sass.status;
          std::ostringstream line;
          line << r.status << " name=" << c.name
               << " expected=\"" << r.sass.expectation << "\""
               << " expected_count=" << r.sass.expected_mma_count
               << " matched=" << r.sass.matched_mma_count
               << " total_tensor_mma=" << r.sass.total_tensor_mma_count;
          emit_line(case_log, line.str(), ropt.quiet_cases);
          unload_compiled(mod);
          records.push_back({c, r});
          continue;
        }
      } else {
        r.sass.status = "SASS_DISABLED";
        r.sass.family = expected_sass_family(c);
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

      check(cuFuncGetAttribute(&r.regs_per_thread, CU_FUNC_ATTRIBUTE_NUM_REGS, mod.function), "register count");
      r.best_ms = best_ms;
      r.mean_ms = sum_ms / opt.repeats;
      const long double warp_count = static_cast<long double>(blocks) * warps_per_block;
      const long double instruction_count = warp_count * opt.iters * opt.chains * kInnerUnroll;
      const long double flops_per_instruction = 2.0L * c.m * c.n * c.k;
      r.peak_logical_tflops = static_cast<double>(instruction_count * flops_per_instruction /
                                                  (static_cast<long double>(r.best_ms) / 1000.0L) / 1.0e12L);
      r.mean_logical_tflops = static_cast<double>(instruction_count * flops_per_instruction /
                                                  (static_cast<long double>(r.mean_ms) / 1000.0L) / 1.0e12L);
      if (c.sparse) r.peak_nonzero_tflops = r.peak_logical_tflops * 0.5;
      r.status = "PASS";

      std::ostringstream line;
      line << std::fixed << std::setprecision(3)
           << "PASS name=" << c.name
           << " peak_logical_tflops=" << r.peak_logical_tflops
           << " best_ms=" << r.best_ms
           << " sass=" << r.sass.status
           << " sass_family=" << r.sass.family;
      emit_line(case_log, line.str(), ropt.quiet_cases);
      unload_compiled(mod);
      records.push_back({c, r});
    }

    const auto peaks = extract_peaks(records, ropt.verify_sass);
    write_results_csv(run_dir / "results.csv", records);
    write_run_json(run_dir / "run.json", di, ropt, records, sass_tool_ok, stamp);
    write_peak_json(run_dir / "peak_summary.json", di, peaks, ropt.verify_sass);
    write_summary_md(run_dir / "summary.md", di, ropt, peaks, records, sass_tool_ok);

    std::cout << "\nHeadline dense peaks (TFLOP/s)\n"
              << std::left << std::setw(12) << "pair"
              << std::right << std::setw(16) << "without MX"
              << std::setw(16) << "with MX" << '\n';
    for (const auto& pair : headline_pairs()) {
      std::cout << std::left << std::setw(12) << pair
                << std::right << std::setw(16) << peak_cell(peaks, "without_mx", pair)
                << std::setw(16) << peak_cell(peaks, "with_mx", pair) << '\n';
    }
    const auto nv = peaks.find({"with_nvfp4", "fp4"});
    if (nv != peaks.end()) {
      std::cout << "NVFP4 fp4 peak: " << std::fixed << std::setprecision(2) << nv->second.tflops << " TFLOP/s\n";
    }
    std::cout << "\nSaved report: " << (run_dir / "summary.md") << '\n'
              << "Machine JSON: " << (run_dir / "run.json") << '\n'
              << "Peak JSON: " << (run_dir / "peak_summary.json") << '\n'
              << "CSV: " << (run_dir / "results.csv") << '\n';

    cuEventDestroy(start);
    cuEventDestroy(stop);
    cuMemFree(output);
    cuDevicePrimaryCtxRelease(dev);

    // With the default SASS gate, missing nvdisasm or any documented SASS
    // mismatch is a run-integrity failure even though all partial reports exist.
    if (ropt.verify_sass) {
      if (!sass_tool_ok) return 3;
      for (const auto& cr : records) {
        if (cr.first.doc == DocClass::Documented &&
            (cr.second.status == "SASS_MISMATCH" || cr.second.status == "SASS_DISASM_ERROR" ||
             cr.second.status == "SASS_TOOL_ERROR")) return 4;
      }
    }
    return selected ? 0 : 5;
  } catch (const std::exception& e) {
    std::cerr << "fatal: " << e.what() << '\n';
    return 1;
  }
}
