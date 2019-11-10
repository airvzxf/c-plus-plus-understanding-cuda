#include <cmath>
#include <chrono>
#include <cstdio>
#include <ctime>

#include <device_launch_parameters.h>
#include <cuda_runtime.h>

#include "cuda.cuh"

__global__
void cudaVectorAdd(const float *a, const float *b, float *c, int numElements) {
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements) {
        c[i] = a[i] + b[i];
    }
}

uint64_t getTimerNow() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

void getTime(uint64_t &startTime) {
    uint64_t timeStopped = getTimerNow();
    uint64_t timeElapsed = timeStopped - startTime;

    printf("timeElapsed: %.3f seconds\n", (double) timeElapsed / 1000.0);
    printf("\n");

    startTime = timeStopped;
}

int cuda() {
    uint64_t timeRecorded = getTimerNow();

    cudaDeviceProp property{};
    cudaGetDeviceProperties(&property, 0);

    // Error code to check return values for CUDA calls
    cudaError_t err;

    printf("Hello, World! From CUDA.\n");
    getTime(timeRecorded);

    // Print the vector length to be used, and compute its size
    int numElements = 50000;
//    int numElements = 625000000;
    size_t size = numElements * sizeof(float);
    printf("[Vector addition of %d elements]\n", numElements);

    // Allocate the host input vector A
    auto *hA = (float *) malloc(size);

    // Allocate the host input vector B
    auto *hB = (float *) malloc(size);

    // Allocate the host output vector C
    auto *hC = (float *) malloc(size);

    // Verify that allocations succeeded
    if (hA == nullptr || hB == nullptr || hC == nullptr) {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    getTime(timeRecorded);
    printf("Start generating the random numbers\n");
    // Initialize the host input vectors
    for (int i = 0; i < numElements; ++i) {
        hA[i] = (float) rand() / (float) RAND_MAX;
        hB[i] = (float) rand() / (float) RAND_MAX;
    }
    getTime(timeRecorded);

    printf("Allocate the device input.\n");
    // Allocate the device input vector A
    float *dA = nullptr;
    err = cudaMalloc((void **) &dA, size);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector B
    float *dB = nullptr;
    err = cudaMalloc((void **) &dB, size);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device output vector C
    float *dC = nullptr;
    err = cudaMalloc((void **) &dC, size);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    getTime(timeRecorded);
    // Copy the host input vectors A and B in host memory to the device input vectors in
    // device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(dA, hA, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(dB, hB, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    getTime(timeRecorded);
    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = property.maxThreadsPerBlock;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    cudaVectorAdd << < blocksPerGrid, threadsPerBlock >> > (dA, dB, dC, numElements);
    err = cudaGetLastError();

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch cudaVectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    getTime(timeRecorded);
    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(hC, dC, size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    getTime(timeRecorded);
    // Verify that the result vector is correct
    printf("Verify that the result vector is correct\n");
    for (int i = 0; i < numElements; ++i) {
        if (std::fabs(hA[i] + hB[i] - hC[i]) > 1e-5) {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    getTime(timeRecorded);
    printf("Test PASSED\n");

    // Free device global memory
    err = cudaFree(dA);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(dB);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(dC);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free host memory
    free(hC);

    // Allocate the host output vector C
    auto *hC2 = (float *) malloc(size);


    getTime(timeRecorded);
    printf("Add vectors using the CPU.\n");
    for (int index = 0; index < numElements; index++) {
        hC2[index] = hA[index] + hB[index];
    }

    getTime(timeRecorded);
    printf("Verify that the result vector is correct\n");
    for (int i = 0; i < numElements; ++i) {
        if (std::fabs(hA[i] + hB[i] - hC[i]) > 1e-5) {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    // Free host memory
    free(hA);
    free(hB);
    free(hC);

    getTime(timeRecorded);
    printf("Done\n");
    return 0;
}
