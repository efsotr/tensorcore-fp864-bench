cmake_minimum_required(VERSION 3.20)
project(tensorcore_fp864_bench LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(CUDAToolkit REQUIRED)

add_executable(tensorcore-fp864-bench
    src/runner.cpp
)

target_link_libraries(tensorcore-fp864-bench PRIVATE CUDA::cuda_driver)

target_compile_options(tensorcore-fp864-bench PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-O2 -Wall -Wextra -Wpedantic>
)
