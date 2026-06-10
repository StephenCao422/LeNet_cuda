#ifndef LAYER_CUSTOM_MATMUL_H
#define LAYER_CUSTOM_MATMUL_H

#include <cuda_runtime.h>

#define MATMUL_TILE_WIDTH 16

// Tiled matrix multiplication kernel. Computes C = AB
// You don't need to modify this kernel.
__global__ void matrixMultiplyShared(const float *A, const float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns);

#endif  // LAYER_CUSTOM_MATMUL_H