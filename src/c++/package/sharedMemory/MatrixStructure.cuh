#ifndef CUDA_TEST_CLION_MATRIXSTRUCTURE_H
#define CUDA_TEST_CLION_MATRIXSTRUCTURE_H

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
typedef struct {
    int width;
    int height;
    float *elements;
} matrixStructure;

void setMatrixStructure(matrixStructure &matrix, int size);

void setMatrixValues(matrixStructure &matrix);

void setMatrixValuesInverse(matrixStructure &matrix);

void mathMulHost(matrixStructure a, matrixStructure b, matrixStructure &c);

void verifyMathMul(matrixStructure a, matrixStructure b, matrixStructure c);

#endif //CUDA_TEST_CLION_MATRIXSTRUCTURE_H
