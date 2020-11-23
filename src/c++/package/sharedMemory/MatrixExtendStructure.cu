#include <cstdlib>
#include <cstdio>

#include "MatrixExtendStructure.cuh"

void setMatrixExtendStructure(matrixExtendStructure &matrix, int size) {
    matrix.width = size;
    matrix.height = size;
    matrix.elements = (float *) malloc(sizeof(*matrix.elements) * size * size);
}

void setMatrixExtendValues(matrixExtendStructure &matrix) {
    int size = matrix.width * matrix.height;
    for (int column = 0; column < size; column++) {
        matrix.elements[column] = float(column / 100.0);
    }
}

void setMatrixExtendValuesInverse(matrixExtendStructure &matrix) {
    int inverseColumn;
    int size = matrix.width * matrix.height;
    for (int column = 0; column < size; column++) {
        inverseColumn = size - 1 - column;
        matrix.elements[inverseColumn] = float(column / 100.0);
    }
}

void mathMulExtendHost(matrixExtendStructure a, matrixExtendStructure b, matrixExtendStructure &c) {
    for (int column = 0; column < a.width * a.height; column++) {
        c.elements[column] = a.elements[column] * b.elements[column];
    }
}

void verifyMathMulExtend(matrixExtendStructure a, matrixExtendStructure b, matrixExtendStructure c) {
    float times;
    bool successful = true;
    for (int column = 0; column < a.width * a.height; column++) {
        times = a.elements[column] * b.elements[column];
        if (c.elements[column] != times) {
            successful = false;
            fprintf(stderr, "FAILED: C element at column: %d\n", column);
            fprintf(stderr, "  A * B: = C\n");
            fprintf(stderr, "  %f * %f = %f\n", a.elements[column], b.elements[column], times);
            fprintf(stderr, "  VS\n");
            fprintf(stderr, "  %f * %f = %f\n", a.elements[column], b.elements[column], c.elements[column]);
        }
    }

    if (successful) {
        printf("GOOD: Verified the Matrix C without errors!\n");
    }    else {
        fprintf(stderr, "FAILED: The validation of the result is wrong.\n");
    }

    printf("\n");
}
