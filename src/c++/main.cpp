#include <cstdio>
//#include "vectorAdd.cuh"
//#include "mathAdd.cuh"
#include "package/sharedMemory/NotSharedMemory.cuh"
#include "package/sharedMemory/SharedMemory.cuh"

int main() {
    printf("Hello, World! From C++.\n\n");
//    vectorAdd();
//    MathAdd();
    notSharedMemory();
//    sharedMemory();
    return 0;
}
