// Auto-generated standalone CUDA reproduction for dense/f8f6f4/e2m3xe2m1/acc-f32
// PTX ISA requested by benchmark: 9.0
// Suggested compile: nvcc -arch=sm_120a -lineinfo inline_ptx_repro.cu -o repro
#include <cuda_runtime.h>
#include <cstdio>

__global__ void repro_kernel() {
  asm volatile(R"PTX(
{
  .reg .b32 %%a<4>, %%b<2>;
  .reg .f32 %%d<4>;
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
  mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e2m3.e2m1.f32
    {%%d0, %%d1, %%d2, %%d3}, {%%a0, %%a1, %%a2, %%a3}, {%%b0, %%b1}, {%%d0, %%d1, %%d2, %%d3};
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
