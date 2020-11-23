#include <cmath>
#include <chrono>
#include <cstdio>
#include <ctime>

#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>

#include "vectorAdd.cuh"

// Device code
__global__
void cudaVectorAdd(const float *a, const float *b, float *c, int numElements) {
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements) {
        c[i] = a[i] + b[i];
    }
}

// Host code
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

int vectorAdd() {
    uint64_t timeRecorded = getTimerNow();

    cudaDeviceProp property{};
    cudaGetDeviceProperties(&property, 0);

    // Error code to check return values for CUDA calls
    cudaError_t err;

    printf("Hello, World! From CUDA.\n");
    getTime(timeRecorded);

    // Print the vector length to be used, and compute its size
    int numElements = 50000;
//    int numElements = 725000000;
//    int numElements = 625000000;
    size_t size = numElements * sizeof(float);
    printf("[Vector addition of %d elements]\n", numElements);

    // Allocate the host input vector A
    auto *hostA = (float *) malloc(size);

    // Allocate the host input vector B
    auto *hostB = (float *) malloc(size);

    // Allocate the host output vector C
    auto *hostC = (float *) malloc(size);

    // Verify that allocations succeeded
    if (hostA == nullptr || hostB == nullptr || hostC == nullptr) {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    getTime(timeRecorded);
    printf("Start generating the random numbers\n");
    // Initialize the host input vectors
    for (int i = 0; i < numElements; ++i) {
        hostA[i] = (float) rand() / (float) RAND_MAX;
        hostB[i] = (float) rand() / (float) RAND_MAX;
    }
    getTime(timeRecorded);

    printf("Allocate the device input.\n");
    // Allocate the device input vector A
    float *deviceA = nullptr;
    err = cudaMalloc((void **) &deviceA, size);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector B
    float *deviceB = nullptr;
    err = cudaMalloc((void **) &deviceB, size);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device output vector C
    float *deviceC = nullptr;
    err = cudaMalloc((void **) &deviceC, size);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    getTime(timeRecorded);
    // Copy the host input vectors A and B in host memory to the device input vectors in
    // device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(deviceA, hostA, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(deviceB, hostB, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    getTime(timeRecorded);
    // Launch the Vector Add CUDA Kernel
//    int threadsPerBlock = 32;
    int threadsPerBlock = property.maxThreadsPerBlock;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    cudaVectorAdd << < blocksPerGrid, threadsPerBlock >> > (deviceA, deviceB, deviceC, numElements);
    err = cudaGetLastError();

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch cuda kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    getTime(timeRecorded);
    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(hostC, deviceC, size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    getTime(timeRecorded);
    // Verify that the result vector is correct
    printf("Verify that the result vector is correct\n");
    for (int i = 0; i < numElements; ++i) {
        if (std::fabs(hostA[i] + hostB[i] - hostC[i]) > 1e-5) {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    getTime(timeRecorded);
    printf("Test PASSED\n");

    // Free device global memory
    err = cudaFree(deviceA);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(deviceB);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(deviceC);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free host memory
    free(hostC);

    // Allocate the host output vector C
    auto *hC2 = (float *) malloc(size);


    getTime(timeRecorded);
    printf("Add vectors using the CPU.\n");
    for (int index = 0; index < numElements; index++) {
        hC2[index] = hostA[index] + hostB[index];
    }

    getTime(timeRecorded);
    printf("Verify that the result vector is correct\n");
    for (int i = 0; i < numElements; ++i) {
        if (std::fabs(hostA[i] + hostB[i] - hostC[i]) > 1e-5) {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    // Free host memory
    free(hostA);
    free(hostB);
    free(hostC);

    getTime(timeRecorded);
    printf("Done\n");
    return 0;
}
