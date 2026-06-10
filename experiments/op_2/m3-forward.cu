#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16

__global__ void matmul_conv_fused(const float *mask, const float *input, float *output,
                                  int Batch, int Map_out, int Channel, int Height, int Width, int K)
{

    const int Height_out = Height - K + 1;
    const int Width_out  = Width  - K + 1;
    const int image_size = Height_out * Width_out;
    const int Height_unrolled = Channel * K * K; // rows of unrolled B
    const size_t Width_unrolled = (size_t)Batch * image_size; // cols of unrolled B

    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH]; // tile from mask
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH]; // tile from (virtual) unrolled input

    int ty = threadIdx.y, tx = threadIdx.x;

    int row = blockIdx.y * TILE_WIDTH + ty;         // output map index m
    size_t col = (size_t)blockIdx.x * TILE_WIDTH + tx; // unrolled column index

    int b_img = (int)(col / image_size);
    int hw = (int)(col % image_size);
    int h_out = hw / Width_out;
    int w_out = hw % Width_out;

    float val = 0.0f;

    int numTiles = (Height_unrolled + TILE_WIDTH - 1) / TILE_WIDTH;

    // #pragma unroll 1
    for (int tileId = 0; tileId < numTiles; tileId++) {

        int a_col = tileId * TILE_WIDTH + tx;
        if (row < Map_out && a_col < Height_unrolled) {
            tileA[ty][tx] = mask[(size_t)row * Height_unrolled + a_col];
        } else {
            tileA[ty][tx] = 0.0f;
        }

        int b_row = tileId * TILE_WIDTH + ty;
        if (b_row < Height_unrolled && col < Width_unrolled) {
            int c = b_row / (K * K);
            int pq = b_row % (K * K);
            int p = pq / K;
            int q = pq % K;

            tileB[ty][tx] = input[(size_t)b_img * (Channel * Height * Width)
                                 + (size_t)c * (Height * Width)
                                 + (size_t)(h_out + p) * Width
                                 + (w_out + q)];
        } else {
            tileB[ty][tx] = 0.0f;
        }

        __syncthreads();

        if (row < Map_out && col < Width_unrolled) {
            // TILE_WIDTH=16, 4 iterations of 4 FMAs each
            for (int i = 0; i < TILE_WIDTH; i += 4) {
                val += tileA[ty][i]* tileB[i][tx];
                val += tileA[ty][i + 1] * tileB[i + 1][tx];
                val += tileA[ty][i + 2] * tileB[i + 2][tx];
                val += tileA[ty][i + 3] * tileB[i + 3][tx];
            }
        }

        __syncthreads();
    }


    if (row < Map_out && col < Width_unrolled) {
        output[(size_t)b_img * (Map_out * image_size)
             + (size_t)row * image_size
             + (size_t)h_out * Width_out
             + w_out] = val;
    }
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    cudaMalloc((void**)device_output_ptr, (size_t)Batch * Map_out * Height_out * Width_out * sizeof(float));
    cudaMalloc((void**)device_input_ptr, (size_t)Batch * Channel * Height * Width * sizeof(float));
    cudaMalloc((void**)device_mask_ptr, (size_t)Map_out * Channel * K * K * sizeof(float));

    cudaMemcpy(*device_input_ptr, host_input, (size_t)Batch * Channel * Height * Width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask, (size_t)Map_out * Channel * K * K * sizeof(float), cudaMemcpyHostToDevice);
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out  = Width  - K + 1;
    const size_t Width_unrolled = (size_t)Batch * Height_out * Width_out;

    dim3 block(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 grid((unsigned int)((Width_unrolled + TILE_WIDTH - 1) / TILE_WIDTH), (unsigned int)((Map_out + TILE_WIDTH - 1) / TILE_WIDTH), 1);

    matmul_conv_fused<<<grid, block>>>(device_mask, device_input, device_output, Batch, Map_out, Channel, Height, Width, K);
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out  = Width  - K + 1;

    cudaMemcpy(host_output, device_output, (size_t)Batch * Map_out * Height_out * Width_out * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(device_output);
    cudaFree(device_input);
    cudaFree(device_mask);
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
