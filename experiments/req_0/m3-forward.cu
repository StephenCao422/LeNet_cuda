#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#include "matmul.h"

#define PERMUTE_BLOCK_SIZE 256
#define NUM_STREAMS 4

static cudaStream_t streams[4];
static float *in_chunk[4];
static float *out_chunk[4];
static float *unrolled[4];
static float *matmul[4];
static const float *pinned_input = nullptr;
static float *pinned_output = nullptr;
static int chunk_batch = 0;

__global__ void matrix_unrolling_kernel(const float *input, float *output,
                                        const int Batch, const int Channel,
                                        const int Height, const int Width,
                                        const int K) {
    const int Height_out = Height - K + 1;
    const int Width_out  = Width  - K + 1;

    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]

    size_t col = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = (size_t)blockIdx.y * blockDim.y + threadIdx.y;

    const size_t Width_unrolled  = (size_t)Batch * Height_out * Width_out;
    const size_t Height_unrolled = (size_t)Channel * K * K;

    if (row < Height_unrolled && col < Width_unrolled) {
        int c = row / (K * K);
        int pq = row % (K * K);
        int p = pq / K;
        int q = pq % K;

        int b = col / (Height_out * Width_out);
        int hw = col % (Height_out * Width_out);
        int h_out = hw / Width_out;
        int w_out = hw % Width_out;

        output[row * Width_unrolled + col] = in_4d(b, c, h_out + p, w_out + q);
    }

    #undef in_4d
}

__global__ void matrix_permute_kernel(const float *input, float *output, int Map_out,
                                      int Batch, int image_size) {
    int b = blockIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < image_size) {
        for (int m = 0; m < Map_out; m++) {
            output[b * Map_out * image_size + m * image_size + x] =
                    input[m * Batch * image_size + b * image_size + x];
        }
    }
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    const int image_size = Height_out * Width_out;
    const int Height_unrolled = Channel * K * K;

    *device_input_ptr  = nullptr;
    *device_output_ptr = nullptr;

    const size_t mask_bytes = (size_t)Map_out * Channel * K * K * sizeof(float);
    cudaMalloc((void**)device_mask_ptr, mask_bytes);
    cudaMemcpy(*device_mask_ptr, host_mask, mask_bytes, cudaMemcpyHostToDevice);

    const size_t in_bytes = (size_t)Batch * Channel * Height * Width * sizeof(float);
    const size_t out_bytes = (size_t)Batch * Map_out * image_size * sizeof(float);
    cudaHostRegister((void*)host_input, in_bytes, cudaHostRegisterDefault);
    cudaHostRegister((void*)host_output, out_bytes, cudaHostRegisterDefault);
    pinned_input = host_input;
    pinned_output = (float*)host_output;

    chunk_batch = (Batch + NUM_STREAMS - 1) / NUM_STREAMS;
    const size_t in_per_img = (size_t)Channel * Height * Width;
    const size_t out_per_img = (size_t)Map_out * image_size;
    const size_t Width_unrolled_max = (size_t)chunk_batch * image_size;

    for (int s = 0; s < NUM_STREAMS; s++) {
        cudaStreamCreate(&streams[s]);
        cudaMalloc((void**)&in_chunk[s], (size_t)chunk_batch * in_per_img  * sizeof(float));
        cudaMalloc((void**)&out_chunk[s], (size_t)chunk_batch * out_per_img * sizeof(float));
        cudaMalloc((void**)&unrolled[s], (size_t)Height_unrolled * Width_unrolled_max * sizeof(float));
        cudaMalloc((void**)&matmul[s], (size_t)Map_out * Width_unrolled_max * sizeof(float));
    }
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    const int image_size = Height_out * Width_out;
    const int Height_unrolled = Channel * K * K;

    const size_t in_per_img = (size_t)Channel * Height * Width;
    const size_t out_per_img = (size_t)Map_out * image_size;

    for (int s = 0; s < NUM_STREAMS; s++) {
        const int b_start = s * chunk_batch;
        if (b_start >= Batch) {
            break;
        }
        const int b_end = (b_start + chunk_batch < Batch) ? (b_start + chunk_batch) : Batch;
        const int chunk = b_end - b_start;

        const int Width_unrolled = chunk * image_size;

        cudaStream_t stream = streams[s];

        cudaMemcpyAsync(in_chunk[s], pinned_input + b_start * in_per_img, (size_t)chunk * in_per_img * sizeof(float), cudaMemcpyHostToDevice, stream);

        dim3 unroll_block(16, 16, 1);
        dim3 unroll_grid((Width_unrolled + 16 - 1) / 16, (Height_unrolled + 16 - 1) / 16, 1);
        matrix_unrolling_kernel<<<unroll_grid, unroll_block, 0, stream>>>(in_chunk[s], unrolled[s], chunk, Channel, Height, Width, K);

        dim3 matmul_grid((Width_unrolled - 1) / MATMUL_TILE_WIDTH + 1, (Map_out - 1) / MATMUL_TILE_WIDTH + 1, 1);
        dim3 matmul_block(MATMUL_TILE_WIDTH, MATMUL_TILE_WIDTH, 1);
        matrixMultiplyShared<<<matmul_grid, matmul_block, 0, stream>>>(device_mask, unrolled[s], matmul[s], Map_out, Height_unrolled, Height_unrolled, Width_unrolled, Map_out, Width_unrolled);

        dim3 permute_grid((image_size - 1) / PERMUTE_BLOCK_SIZE + 1, chunk, 1);
        matrix_permute_kernel<<<permute_grid, PERMUTE_BLOCK_SIZE, 0, stream>>>(matmul[s], out_chunk[s], Map_out, chunk, image_size);

        cudaMemcpyAsync(pinned_output + b_start * out_per_img, out_chunk[s], (size_t)chunk * out_per_img * sizeof(float), cudaMemcpyDeviceToHost, stream);
    }
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    for (int s = 0; s < NUM_STREAMS; s++) {
        cudaStreamSynchronize(streams[s]);
        cudaFree(in_chunk[s]);
        cudaFree(out_chunk[s]);
        cudaFree(unrolled[s]);
        cudaFree(matmul[s]);
        cudaStreamDestroy(streams[s]);
    }
    // double check free ptr
    if (pinned_input){
        cudaHostUnregister((void*)pinned_input);
    }
    if (pinned_output) {
        cudaHostUnregister((void*)pinned_output);
    }
    pinned_input = nullptr;
    pinned_output = nullptr;

    if (device_output){
        cudaFree(device_output);
    }
    if (device_input){
        cudaFree(device_input);
    }
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
