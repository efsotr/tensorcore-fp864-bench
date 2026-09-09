// Curated dense FP8/FP6/FP4 x FP6/FP6/FP4 microbenchmark for SM120.
//
// This intentionally reuses the canonical PTX manifest/generator from bench.cpp,
// but runs only a small practical subset. It excludes sparse instructions,
// F16 accumulation, duplicate default-scale spellings, selector expansion, and
// non-documented probes.

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
      "dense/mxf4nvf4/e2m1xe2m1/ue8m0-4X",
      "dense/mxf4nvf4/e2m1xe2m1/ue4m3-4X",
  };
  return names;
}

bool is_curated_case(const Case& c) {
  const auto& names = curated_case_names();
  return std::find(names.begin(), names.end(), c.name) != names.end();
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options opt = parse_args(argc, argv);
    if (opt.include_probes || opt.probes_only || opt.all_operands) {
      throw std::runtime_error(
          "this curated benchmark excludes probes and selector expansion; "
          "do not use --include-probes, --probes-only, or --all-operands");
    }

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

    std::cout << "suite=fp864x-fp664-curated"
              << " cases=" << curated_case_names().size()
              << " dense_only=yes sparse=no accumulator=f32"
              << " device=\"" << device_name << "\""
              << " cc=" << major << '.' << minor
              << " sm_count=" << sms
              << " reported_clock_mhz=" << (clock_khz / 1000.0)
              << " driver_api=" << driver_version
              << " ptx=9.1 target=sm_120a"
              << " inner_unroll=" << kInnerUnroll << '\n';

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
    check(cuMemAlloc(&output,
                     static_cast<size_t>(blocks) * warps_per_block * sizeof(uint32_t)),
          "cuMemAlloc");

    CUevent start, stop;
    check(cuEventCreate(&start, CU_EVENT_DEFAULT), "cuEventCreate(start)");
    check(cuEventCreate(&stop, CU_EVENT_DEFAULT), "cuEventCreate(stop)");

    size_t selected = 0;
    for (const Case& c : cases) {
      if (!is_curated_case(c)) continue;
      if (!opt.filter.empty() && c.name.find(opt.filter) == std::string::npos) continue;
      ++selected;

      LoadedModule mod;
      const CUresult jit_result = load_ptx(build_ptx(c, opt.chains), mod);
      if (jit_result != CUDA_SUCCESS) {
        std::cout << "FAIL_DOCUMENTED name=" << c.name
                  << " error=\"" << cuda_error(jit_result) << "\"\n";
        if (opt.verbose_jit && !mod.log.empty()) std::cout << mod.log << '\n';
        continue;
      }

      unsigned warm_iters = std::min(opt.iters, 256u);
      void* warm_args[] = {&output, &warm_iters};
      check(cuLaunchKernel(mod.function, blocks, 1, 1, threads, 1, 1,
                           0, nullptr, warm_args, nullptr),
            "warmup launch");
      check(cuCtxSynchronize(), "warmup synchronize");

      float best_ms = std::numeric_limits<float>::infinity();
      float sum_ms = 0.0f;
      for (unsigned rep = 0; rep < opt.repeats; ++rep) {
        unsigned timed_iters = opt.iters;
        void* args[] = {&output, &timed_iters};
        check(cuEventRecord(start, nullptr), "event start");
        check(cuLaunchKernel(mod.function, blocks, 1, 1, threads, 1, 1,
                             0, nullptr, args, nullptr),
              "timed launch");
        check(cuEventRecord(stop, nullptr), "event stop");
        check(cuEventSynchronize(stop), "event synchronize");
        float ms = 0.0f;
        check(cuEventElapsedTime(&ms, start, stop), "event elapsed");
        best_ms = std::min(best_ms, ms);
        sum_ms += ms;
      }

      int regs_per_thread = 0;
      check(cuFuncGetAttribute(&regs_per_thread, CU_FUNC_ATTRIBUTE_NUM_REGS,
                               mod.function),
            "register count");

      const long double warp_count = static_cast<long double>(blocks) * warps_per_block;
      const long double instruction_count =
          warp_count * opt.iters * opt.chains * kInnerUnroll;
      const long double flops_per_instruction = 2.0L * c.m * c.n * c.k;
      const long double peak_logical =
          instruction_count * flops_per_instruction /
          (static_cast<long double>(best_ms) / 1000.0L) / 1.0e12L;
      const long double mean_logical =
          instruction_count * flops_per_instruction /
          ((static_cast<long double>(sum_ms) / opt.repeats) / 1000.0L) / 1.0e12L;

      std::cout << std::fixed << std::setprecision(3)
                << "PASS name=" << c.name
                << " best_ms=" << best_ms
                << " peak_logical_tflops=" << static_cast<double>(peak_logical)
                << " mean_logical_tflops=" << static_cast<double>(mean_logical)
                << " regs_per_thread=" << regs_per_thread
                << " blocks=" << blocks
                << " threads=" << threads
                << " chains=" << opt.chains
                << " inner_unroll=" << kInnerUnroll
                << " iters=" << opt.iters
                << " repeats=" << opt.repeats
                << '\n';

      if (opt.verbose_jit && !mod.log.empty()) std::cout << mod.log << '\n';
      unload(mod);
    }

    std::cout << "selected=" << selected
              << " curated_manifest=" << curated_case_names().size() << '\n';

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
