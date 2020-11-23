cmake_minimum_required(VERSION 3.15)

project(CUDA_Test_Clion)

enable_language(CUDA)
find_package(CUDA REQUIRED)
include(FindCUDA)

#include_directories(/opt/cuda/include ${MPI_INCLUDE_PATH})

set(CMAKE_CXX_STANDARD 17)
file(GLOB SOURCES *.cu *.cpp *.cuh)
file(GLOB MEMORY_PACKAGE package/sharedMemory/*.cu package/sharedMemory/*.cuh)
cuda_add_executable(CUDA_Test_Clion ${SOURCES} ${MEMORY_PACKAGE})
message(SOURCES: ${SOURCES})

list(APPEND CMAKE_CXX_FLAGS "-std=c++0x -O3 -ffast-math -Wall")

list(APPEND CUDA_NVCC_FLAGS --compiler-options -m64 -fno-strict-aliasing -lineinfo -use_fast_math -Xptxas -dlcm=cg)
list(APPEND CUDA_NVCC_FLAGS -gencodearch=compute_61,code=compute_61,sm_61)

#target_link_libraries(CUDA_Test_Clion /opt/cuda/lib/libcudart.so)
