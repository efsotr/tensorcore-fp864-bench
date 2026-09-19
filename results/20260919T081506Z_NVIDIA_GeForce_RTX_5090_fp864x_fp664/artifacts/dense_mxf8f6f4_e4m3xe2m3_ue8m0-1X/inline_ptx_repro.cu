// Auto-generated standalone CUDA reproduction for dense/mxf8f6f4/e4m3xe2m3/ue8m0-1X
// PTX ISA requested by benchmark: 9.0
// Suggested compile: nvcc -arch=sm_120a -lineinfo inline_ptx_repro.cu -o repro
#include <cuda_runtime.h>
#include <cstdio>

__global__ void repro_kernel() {
  asm volatile(R"PTX(
{
  .reg .b32 %%a<4>, %%b<2>;
  .reg .f32 %%d<4>;
  .reg .b32 %%sa, %%sb;
  mov.b32 %%a0, 0x14141414;
  mov.b32 %%a1, 0x14141414;
  mov.b32 %%a2, 0x14141414;
  mov.b32 %%a3, 0x14141414;
  mov.b32 %%b0, 0x0c0c0c0c;
  mov.b32 %%b1, 0x0c0c0c0c;
  mov.f32 %%d0, 0f00000000;
  mov.f32 %%d1, 0f00000000;
  mov.f32 %%d2, 0f00000000;
  mov.f32 %%d3, 0f00000000;
  mov.b32 %%sa, 0x38383838;
  mov.b32 %%sb, 0x38383838;
  mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e2m3.f32.ue8m0
    {%%d0, %%d1, %%d2, %%d3}, {%%a0, %%a1, %%a2, %%a3}, {%%b0, %%b1}, {%%d0, %%d1, %%d2, %%d3}, %%sa, {0, 0}, %%sb, {0, 0};
}
)PTX");
}

int main() {
  repro_kernel<<<1, 32>>>();
  cudaError_t launch = cudaGetLastError();
  if (launch != cudaSuccess) {
    std::fprintf(stderr, "launch: %s\n", cudaGetErrorString(launch));
    return 1;
  }
  cudaError_t sync = cudaDeviceSynchronize();
  if (sync != cudaSuccess) {
    std::fprintf(stderr, "sync: %s\n", cudaGetErrorString(sync));
    return 2;
  }
  return 0;
}
