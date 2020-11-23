#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>

#include "SharedMemory.cuh"
#include "MatrixExtendStructure.cuh"

// Thread block size
#define BLOCK_SIZE 16

// Get a matrix element
__device__ float getElement(const matrixExtendStructure matrix, unsigned int row, unsigned int col) {
    return matrix.elements[row * matrix.stride + col];
}

// Set a matrix element
__device__ void setElement(matrixExtendStructure matrix, unsigned int row, unsigned int col, float value) {
    matrix.elements[row * matrix.stride + col] = value;
}

// Get the BLOCK_SIZExBLOCK_SIZE sub-matrix Asub of A that is
// located col sub-matrices to the right and row sub-matrices down
// from the upper-left corner of A
__device__ matrixExtendStructure getSubMatrix(matrixExtendStructure matrix, unsigned int row, unsigned int col) {
    matrixExtendStructure aSub;
    aSub.width = BLOCK_SIZE;
    aSub.height = BLOCK_SIZE;
    aSub.stride = matrix.stride;
    aSub.elements = &matrix.elements[matrix.stride * BLOCK_SIZE * row + BLOCK_SIZE * col];
    return aSub;
}

// Forward declaration of the matrix multiplication kernel
__global__ void matMulKernel(matrixExtendStructure a, matrixExtendStructure b, matrixExtendStructure c);

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void matMul(const matrixExtendStructure a, const matrixExtendStructure b, matrixExtendStructure c) {
    // Load A and B to device memory
    matrixExtendStructure deviceA;
    deviceA.width = deviceA.stride = a.width;
    deviceA.height = a.height;
    size_t size = a.width * a.height * sizeof(float);
    cudaMalloc(&deviceA.elements, size);
    cudaMemcpy(deviceA.elements, a.elements, size, cudaMemcpyHostToDevice);

    matrixExtendStructure deviceB;
    deviceB.width = deviceB.stride = b.width;
    deviceB.height = b.height;
    size = b.width * b.height * sizeof(float);
    cudaMalloc(&deviceB.elements, size);
    cudaMemcpy(deviceB.elements, b.elements, size, cudaMemcpyHostToDevice);

    // Allocate C in device memory
    matrixExtendStructure deviceC;
    deviceC.width = deviceC.stride = c.width;
    deviceC.height = c.height;
    size = c.width * c.height * sizeof(float);
    cudaMalloc(&deviceC.elements, size);

    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(b.width / dimBlock.x, a.height / dimBlock.y);
    matMulKernel << < dimGrid, dimBlock >> > (deviceA, deviceB, deviceC);

    // Read C from device memory
    cudaMemcpy(c.elements, deviceC.elements, size, cudaMemcpyDeviceToHost);

    verifyMathMulExtend(a, b, c);

    // Free device memory
    cudaFree(deviceA.elements);
    cudaFree(deviceB.elements);
    cudaFree(deviceC.elements);
}

// Matrix multiplication kernel called by matMul()
__global__ void matMulKernel(const matrixExtendStructure a, const matrixExtendStructure b, matrixExtendStructure c) {
    // Block row and column
    unsigned int blockRow = blockIdx.y;
    unsigned int blockCol = blockIdx.x;

    // Each thread block computes one sub-matrix cSub of C
    matrixExtendStructure cSub = getSubMatrix(c, blockRow, blockCol);

    // Each thread computes one element of cSub
    // by accumulating results into cValue
    float cValue = 0;

    // Thread row and column within cSub
    unsigned int row = threadIdx.y;
    unsigned int col = threadIdx.x;

    // Loop over all the sub-matrices of A and B that are
    // required to compute cSub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int m = 0; m < (a.width / BLOCK_SIZE); ++m) {

        // Get sub-matrix aSub of A
        matrixExtendStructure aSub = getSubMatrix(a, blockRow, m);

        // Get sub-matrix bSub of B
        matrixExtendStructure bSub = getSubMatrix(b, m, blockCol);

        // Shared memory used to store aSub and bSub respectively
        __shared__ float aShared[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float bShared[BLOCK_SIZE][BLOCK_SIZE];

        // Load aSub and bSub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        aShared[row][col] = getElement(aSub, row, col);
        bShared[row][col] = getElement(bSub, row, col);

        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();
        // Multiply aSub and bSub together
        for (int e = 0; e < BLOCK_SIZE; ++e)
            cValue += aShared[row][e] * bShared[e][col];

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write cSub to device memory
    // Each thread writes one element
    setElement(cSub, row, col, cValue);
}

void sharedMemory() {
    static int size = 2;

    matrixExtendStructure a;
    setMatrixExtendStructure(a, size);
    setMatrixExtendValues(a);

    matrixExtendStructure b;
    setMatrixExtendStructure(b, size);
    setMatrixExtendValuesInverse(b);

    matrixExtendStructure c;
    setMatrixExtendStructure(c, size);

    matrixExtendStructure verifyC;
    setMatrixExtendStructure(verifyC, size);
    mathMulExtendHost(a, b, verifyC);
    verifyMathMulExtend(a, b, verifyC);

    matMul(a, b, c);
}
