cmake_minimum_required(VERSION 3.15)

project(CUDA_Test_Clion)

enable_language(CUDA)
find_package(CUDA REQUIRED)
include(FindCUDA)

#include_directories(/opt/cuda/include ${MPI_INCLUDE_PATH})

set(CMAKE_CXX_STANDARD 17)
file(GLOB SOURCES *.cu *.cpp *.cuh)
cuda_add_executable(CUDA_Test_Clion ${SOURCES})

list(APPEND CMAKE_CXX_FLAGS "-std=c++0x -O3 -ffast-math -Wall")

list(APPEND CUDA_NVCC_FLAGS --compiler-options -fno-strict-aliasing -lineinfo -use_fast_math -Xptxas -dlcm=cg)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_20,code=sm_20)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_35,code=sm_35)

#target_link_libraries(CUDA_Test_Clion /opt/cuda/lib/libcudart.so)
