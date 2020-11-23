//#include <cuda_runtime_api.h>
//
//#include "mathAdd.cuh"
//
//// Kernel definition
//__global__ void CudaMatAdd(float A[N][N], float B[N][N], float C[N][N]) {
//    int i = threadIdx.x;
//    int j = threadIdx.y;
//    C[i][j] = A[i][j] + B[i][j];
//}
//
//int MathAdd() {
//    // Kernel invocation with one block of N * N * 1 threads
//    int numBlocks = 1;
//    dim3 threadsPerBlock(N, N);
//    CudaMatAdd << < numBlocks, threadsPerBlock >> > (A, B, C);
//}