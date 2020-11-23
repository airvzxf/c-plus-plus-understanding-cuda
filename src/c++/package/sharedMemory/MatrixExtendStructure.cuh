#ifndef CUDA_TEST_CLION_MATRIXEXTENDSTRUCTURE_H
#define CUDA_TEST_CLION_MATRIXEXTENDSTRUCTURE_H

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.stride + col)
typedef struct {
    int width;
    int height;
    int stride;
    float *elements;
} matrixExtendStructure;

void setMatrixExtendStructure(matrixExtendStructure &matrix, int size);

void setMatrixExtendValues(matrixExtendStructure &matrix);

void setMatrixExtendValuesInverse(matrixExtendStructure &matrix);

void mathMulExtendHost(matrixExtendStructure a, matrixExtendStructure b, matrixExtendStructure &c);

void verifyMathMulExtend(matrixExtendStructure a, matrixExtendStructure b, matrixExtendStructure c);


#endif //CUDA_TEST_CLION_MATRIXEXTENDSTRUCTURE_H
