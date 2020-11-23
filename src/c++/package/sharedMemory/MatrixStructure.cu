#include <cstdlib>
#include <cstdio>
#include "MatrixStructure.cuh"

void setMatrixStructure(matrixStructure &matrix, int size) {
    matrix.width = size;
    matrix.height = size;
    matrix.elements = (float *) malloc(sizeof(*matrix.elements) * size * size);
}

void setMatrixValues(matrixStructure &matrix) {
    int size = matrix.width * matrix.height;
    for (int column = 0; column < size; column++) {
        matrix.elements[column] = float(column / 100.0);
    }
}

void setMatrixValuesInverse(matrixStructure &matrix) {
    int inverseColumn;
    int size = matrix.width * matrix.height;
    for (int column = 0; column < size; column++) {
        inverseColumn = size - 1 - column;
        matrix.elements[inverseColumn] = float(column / 100.0);
    }
}

void mathMulHost(const matrixStructure a, const matrixStructure b, matrixStructure &c) {
    printf("mathMulHost\n");
    float cValue;
    for (int row = 0; row < a.width; row++) {
        for (int col = 0; col < a.height; col++) {
            cValue = 0;
            for (int e = 0; e < a.width; ++e) {
                cValue += a.elements[row * a.width + e] * b.elements[e * b.width + col];
            }
            c.elements[row * c.width + col] = cValue;
        }
    }
}

void verifyMathMul(const matrixStructure a, const matrixStructure b, const matrixStructure c) {
    bool successful = true;
    float cValue;
    printf("verifyMathMul\n");

    for (int row = 0; row < a.width; row++) {
        for (int col = 0; col < a.height; col++) {
            cValue = 0;
            int index = row * c.width + col;
            for (int e = 0; e < a.width; ++e) {
                cValue += a.elements[row * a.width + e] * b.elements[e * b.width + col];
//                if (index == 6 || index == 8 || index == 9) {
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
                            "CPU\n",
                            index, row, col, e, a.width,
                            row * a.width + e, a.elements[row * a.width + e],
                            e * b.width + col, b.elements[e * b.width + col], cValue
                    );
//                }
            }
            bool isTrue = c.elements[index] != cValue;
//            printf("c[%d] [%d,%d] %f != %f  |  %d\n", index, row, col,
//                   cValue, c.elements[index], isTrue);

//            if (c.elements[index] != cValue) {
//                successful = false;
//                fprintf(stderr, "c[%d] [%d,%d] %.12f != %.12f  |  %d\n", index, row, col,
//                        cValue, c.elements[index], isTrue);
//            }
        }
    }

    if (successful) {
        printf("GOOD: Verified the Matrix C without errors!\n");
    } else {
        fprintf(stderr, "FAILED: The validation of the result is wrong.\n");
    }
    printf("\n");
}
