#include <cmath>
#include <iostream>
#include <mma.h>
#include <cuda_fp16.h>
#include "gpu-new-forward.h"

using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

__global__ void matmul_conv_fused_wmma(const float *mask, const float *input, float *output,
                                       int Batch, int Map_out, int Channel, int Height, int Width, int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    const int image_size = Height_out * Width_out;
    const int Height_unrolled = Channel * K * K;
    const size_t Width_unrolled = (size_t)Batch * image_size;

    __shared__ __half tileA[WMMA_M * WMMA_K];
    __shared__ __half tileB[WMMA_K * WMMA_N];
    __shared__ float tileC[WMMA_M * WMMA_N];

    const int warpM = blockIdx.y;
    const int warpN = blockIdx.x;
    const int tid = threadIdx.x;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    const int numTiles = (Height_unrolled + WMMA_K - 1) / WMMA_K;

    for (int tileId = 0; tileId < numTiles; tileId++) {

        // load A tile 
        #pragma unroll
        for (int idx = tid; idx < WMMA_M*WMMA_K; idx += 32) {
            int i = idx / WMMA_K;
            int j = idx % WMMA_K;
            int row = warpM * WMMA_M + i;
            int a_col = tileId * WMMA_K + j;
            float v = 0.0f;
            if (row < Map_out && a_col < Height_unrolled){
                v = mask[(size_t)row * Height_unrolled + a_col];
            }
            tileA[i * WMMA_K + j] = __float2half(v);
        }

        // load B tile
        #pragma unroll
        for (int idx = tid; idx < WMMA_K * WMMA_N; idx += 32) {
            int i = idx / WMMA_N;
            int j = idx % WMMA_N;
            int b_row = tileId * WMMA_K + i;
            size_t b_col = (size_t)warpN * WMMA_N + j;
            float v = 0.0f;
            if (b_row < Height_unrolled && b_col < Width_unrolled) {
                int b_img = (int)(b_col / image_size);
                int hw = (int)(b_col % image_size);
                int h_out = hw / Width_out;
                int w_out = hw % Width_out;
                int c = b_row / (K * K);
                int pq = b_row % (K * K);
                int p = pq / K;
                int q = pq % K;
                v = input[(size_t)b_img * (Channel * Height * Width)
                        + (size_t)c * (Height * Width)
                        + (size_t)(h_out + p) * Width
                        + (w_out + q)];
            }
            tileB[i * WMMA_N + j] = __float2half(v);
        }

        __syncwarp();

        // tensor-core MMA
        wmma::load_matrix_sync(a_frag, tileA, WMMA_K);
        wmma::load_matrix_sync(b_frag, tileB, WMMA_N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        __syncwarp();
    }

    // store C tile to shared memory
    wmma::store_matrix_sync(tileC, c_frag, WMMA_N, wmma::mem_row_major);
    __syncwarp();

    #pragma unroll
    for (int idx = tid; idx < WMMA_M * WMMA_N; idx += 32) {
        int i = idx / WMMA_N;
        int j = idx % WMMA_N;
        int row = warpM * WMMA_M + i;
        size_t col = (size_t)warpN * WMMA_N + j;
        if (row < Map_out && col < Width_unrolled) {
            int b_img = (int)(col / image_size);
            int hw = (int)(col % image_size);
            int h_out = hw / Width_out;
            int w_out = hw % Width_out;
            output[(size_t)b_img * (Map_out * image_size)
                 + (size_t)row * image_size
                 + (size_t)h_out * Width_out
                 + w_out] = tileC[i * WMMA_N + j];
        }
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
    const int Width_out = Width - K + 1;
    const size_t Width_unrolled = (size_t)Batch * Height_out * Width_out;

    dim3 block(32, 1, 1); // one warp per block
    dim3 grid((unsigned int)((Width_unrolled + WMMA_N - 1) / WMMA_N), (unsigned int)((Map_out + WMMA_M - 1) / WMMA_M), 1);

    matmul_conv_fused_wmma<<<grid, block>>>(device_mask, device_input, device_output, Batch, Map_out, Channel, Height, Width, K);
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