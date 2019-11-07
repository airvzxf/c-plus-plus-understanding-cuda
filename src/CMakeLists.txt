CMAKE_MINIMUM_REQUIRED(VERSION 3.15)

PROJECT(CUDA_Test_Clion)

ENABLE_LANGUAGE(CUDA)
FIND_PACKAGE(CUDA REQUIRED)
INCLUDE(FindCUDA)

#INCLUDE_DIRECTORIES(/usr/local/cuda/include ${MPI_INCLUDE_PATH})

SET(CMAKE_CXX_STANDARD 17)
#SET(CMAKE_C_STANDARD 11)

FILE(GLOB SOURCES *.cu *.cpp *.h)
#FILE(GLOB SOURCES *.cu *.c *.h)
CUDA_ADD_EXECUTABLE(CUDA_Test_Clion ${SOURCES})

LIST(APPEND CMAKE_CXX_FLAGS "-std=c++0x -O3 -ffast-math -Wall")
#LIST(APPEND CMAKE_CXX_FLAGS "-std=c++0x -O3 -ffast-math -Wall")

LIST(APPEND CUDA_NVCC_FLAGS --compiler-options -fno-strict-aliasing -lineinfo -use_fast_math -Xptxas -dlcm=cg)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_20,code=sm_20)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_35,code=sm_35)

#TARGET_INCLUDE_DIRECTORIES(CUDA_Test_Clion PRIVATE /opt/cuda/samples/common/inc/)

TARGET_LINK_LIBRARIES(CUDA_Test_Clion /opt/cuda/targets/x86_64-linux/lib/libcudart.so)
