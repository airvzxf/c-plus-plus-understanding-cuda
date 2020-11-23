#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>

#include "NotSharedMemory.cuh"
#include "MatrixStructure.cuh"

// Thread block size
#define BLOCK_SIZE 4

// Forward declaration of the matrix multiplication kernel
__global__ void matMulKernel(matrixStructure a, matrixStructure b, matrixStructure c);

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void matMul(const matrixStructure hostA, const matrixStructure hostB, matrixStructure hostC) {
    cudaSetDevice(0);

    // Load hostA and hostB to device memory
    matrixStructure deviceA;
    deviceA.width = hostA.width;
    deviceA.height = hostA.height;
    size_t size = hostA.width * hostA.height * sizeof(float);
    cudaMalloc(&deviceA.elements, size);
    cudaMemcpy(deviceA.elements, hostA.elements, size, cudaMemcpyHostToDevice);

    matrixStructure deviceB;
    deviceB.width = hostB.width;
    deviceB.height = hostB.height;
    size = hostB.width * hostB.height * sizeof(float);
    cudaMalloc(&deviceB.elements, size);
    cudaMemcpy(deviceB.elements, hostB.elements, size, cudaMemcpyHostToDevice);

    // Allocate hostC in device memory
    matrixStructure deviceC;
    deviceC.width = hostC.width;
    deviceC.height = hostC.height;
    size = hostC.width * hostC.height * sizeof(float);
    cudaMalloc(&deviceC.elements, size);

    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(hostB.width / dimBlock.x, hostA.height / dimBlock.y);
    matMulKernel << < dimGrid, dimBlock >> > (deviceA, deviceB, deviceC);
    cudaDeviceSynchronize();

    double x = hostB.width / dimBlock.x;
    double y = hostA.height / dimBlock.y;
    printf("x: %f  |  y: %f\n\n", x, y);
//    matMulKernel << < (x, y), dimBlock >> > (deviceA, deviceB, deviceC);

    // Read hostC from device memory
    cudaMemcpy(hostC.elements, deviceC.elements, size, cudaMemcpyDeviceToHost);

    verifyMathMul(hostA, hostB, hostC);

    // Free device memory
    cudaFree(deviceA.elements);
    cudaFree(deviceB.elements);
    cudaFree(deviceC.elements);
}

// Matrix multiplication kernel called by matMul()
__global__ void matMulKernel(const matrixStructure a, const matrixStructure b, matrixStructure c) {
    // Each thread computes one element of C
    // by accumulating results into cValue
    float cValue = 0;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int index = row * c.width + col;
    for (int e = 0; e < a.width; ++e) {
        cValue += a.elements[row * a.width + e] * b.elements[e * b.width + col];
//        if (index == 6 || index == 8 || index == 9) {
            printf(
                    "index: %d | "
                    "row: %d | "
                    "col: %d | "
                    "e: %d | "
                    "size: %d | "
                    "a: %d | "
                    "a: %.12f | "
                    "b: %d | "
                    "b: %.12f | "
                    "cValue: %.12f | "
                    "GPU\n",
                    index, row, col, e, a.width,
                    row * a.width + e, a.elements[row * a.width + e],
                    e * b.width + col, b.elements[e * b.width + col], cValue
            );
//        }
    }
    c.elements[index] = cValue;
}

void notSharedMemory() {
    static int size = BLOCK_SIZE;

    matrixStructure a;
    setMatrixStructure(a, size);
    setMatrixValues(a);

    matrixStructure b;
    setMatrixStructure(b, size);
    setMatrixValuesInverse(b);

    matrixStructure c;
    setMatrixStructure(c, size);

    matMul(a, b, c);

//    matrixStructure verifyC;
//    setMatrixStructure(verifyC, size);
////    mathMulHost(a, b, verifyC);
////    verifyMathMul(a, b, verifyC);
//
//    matMul(a, b, verifyC);
//
//    for (int index = 0; index < size * size; index++) {
//        printf("c[%d] %.12f != %.12f\n", index,
//               verifyC.elements[index], c.elements[index]);
//        if (c.elements[index] != verifyC.elements[index]) {
//            fprintf(stderr, "c[%d] %.12f != %.12f\n", index,
//                    verifyC.elements[index], c.elements[index]);
//        }
//    }
}
