#include <cmath>
#include <iostream>
#include <cuda_fp16.h>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16

static __half *d_mask_half  = nullptr;
static __half *d_input_half = nullptr;

__global__ void float_to_half_kernel(const float *__restrict__ in, __half *__restrict__ out, size_t n)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = __float2half(in[i]);
    }
}

__global__ void matmul_conv_fused_half(const __half *__restrict__ mask,
                                       const __half *__restrict__ input,
                                       float *__restrict__ output,
                                       int Batch, int Map_out, int Channel, int Height, int Width, int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    const int image_size = Height_out * Width_out;
    const int Height_unrolled = Channel * K * K;
    const size_t Width_unrolled = (size_t)Batch * image_size;

    __shared__ __half tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ __half tileB[TILE_WIDTH][TILE_WIDTH];

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int row = blockIdx.y * TILE_WIDTH + ty;
    size_t col = (size_t)blockIdx.x * TILE_WIDTH + tx;

    int b_img = (int)(col / image_size);
    int hw = (int)(col % image_size);
    int h_out = hw / Width_out;
    int w_out = hw % Width_out;

    __half2 val2 = __float2half2_rn(0.0f);
    const __half zero_h = __float2half(0.0f);

    int numTiles = (Height_unrolled + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int tileId = 0; tileId < numTiles; tileId++) {

        int a_col = tileId * TILE_WIDTH + tx;
        if (row < Map_out && a_col < Height_unrolled) {
            tileA[ty][tx] = mask[(size_t)row * Height_unrolled + a_col];
        } else {
            tileA[ty][tx] = zero_h;
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
            tileB[ty][tx] = zero_h;
        }

        __syncthreads();

        if (row < Map_out && col < Width_unrolled) {
            //__half2
            for (int i = 0; i < TILE_WIDTH; i += 2) {
                __half2 a2 = __halves2half2(tileA[ty][i], tileA[ty][i + 1]);
                __half2 b2 = __halves2half2(tileB[i][tx], tileB[i + 1][tx]);
                val2 = __hadd2(val2, __hmul2(a2, b2));
            }
        }

        __syncthreads();
    }

    if (row < Map_out && col < Width_unrolled) {
        __half lo = __low2half(val2);
        __half hi = __high2half(val2);
        output[(size_t)b_img * (Map_out * image_size) + (size_t)row * image_size + (size_t)h_out * Width_out + w_out] = __half2float(__hadd(lo, hi));
    }
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    const size_t input_elems = (size_t)Batch * Channel * Height * Width;
    const size_t mask_elems = (size_t)Map_out * Channel * K * K;

    cudaMalloc((void**)device_output_ptr, (size_t)Batch * Map_out * Height_out * Width_out * sizeof(float));
    cudaMalloc((void**)device_input_ptr, input_elems * sizeof(float));
    cudaMalloc((void**)device_mask_ptr, mask_elems * sizeof(float));

    cudaMemcpy(*device_input_ptr, host_input, input_elems * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask, mask_elems * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_input_half, input_elems * sizeof(__half));
    cudaMalloc((void**)&d_mask_half, mask_elems * sizeof(__half));

    const int conv_block = 256;
    int input_grid = (int)((input_elems + conv_block - 1) / conv_block);
    int mask_grid = (int)((mask_elems + conv_block - 1) / conv_block);

    float_to_half_kernel<<<input_grid, conv_block>>>(*device_input_ptr, d_input_half, input_elems);
    float_to_half_kernel<<<mask_grid, conv_block>>>(*device_mask_ptr, d_mask_half,  mask_elems);
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    const size_t Width_unrolled = (size_t)Batch * Height_out * Width_out;

    dim3 block(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 grid((unsigned int)((Width_unrolled + TILE_WIDTH - 1) / TILE_WIDTH), (unsigned int)((Map_out + TILE_WIDTH - 1) / TILE_WIDTH), 1);

    matmul_conv_fused_half<<<grid, block>>>(d_mask_half, d_input_half, device_output, Batch, Map_out, Channel, Height, Width, K);
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    cudaMemcpy(host_output, device_output, (size_t)Batch * Map_out * Height_out * Width_out * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(device_output);
    cudaFree(device_input);
    cudaFree(device_mask);

    if (d_input_half){
        cudaFree(d_input_half);
        d_input_half = nullptr;
    }
    if (d_mask_half){
        cudaFree(d_mask_half);
        d_mask_half = nullptr;
    }
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
