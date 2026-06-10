#include <cmath>
#include <cstdint>
#include <iostream>
#include <cuda_fp16.h>
#include <mma.h>
#include "gpu-new-forward.h"

#define MAX_MASK_SIZE 4096
__constant__ float  const_mask[MAX_MASK_SIZE];
__constant__ __half const_mask_h[MAX_MASK_SIZE];

__device__ __half global_mask_padded[16 * 208];

template<int MAP_OUT, int CHANNEL, int KK_SIZE, int HEIGHT, int WIDTH, int ROWS_PER_BLOCK>
__global__ void direct_conv_kernel(const float * __restrict__ input, float * __restrict__ output, int Batch)
{
    constexpr int HEIGHT_OUT = HEIGHT - KK_SIZE + 1;
    constexpr int WIDTH_OUT = WIDTH  - KK_SIZE + 1;
    constexpr int IMAGE_SIZE = HEIGHT_OUT * WIDTH_OUT;
    constexpr int HW = HEIGHT * WIDTH;
    constexpr int CHW = CHANNEL * HW;
    constexpr int OUT_IMG_ST = MAP_OUT * IMAGE_SIZE;
    constexpr int STRIP_H = ROWS_PER_BLOCK + KK_SIZE - 1;
    constexpr int NTHREADS = ROWS_PER_BLOCK * WIDTH_OUT;
    constexpr int STRIP_ELEMS = CHANNEL * STRIP_H * WIDTH;

    const int b_img = blockIdx.y;
    const int h_row_base = blockIdx.x * ROWS_PER_BLOCK;
    const int tid = threadIdx.x;

    __shared__ float in_strip[STRIP_ELEMS];

    const size_t img_base = (size_t)b_img * CHW + (size_t)h_row_base * WIDTH;
    #pragma unroll
    for (int c = 0; c < CHANNEL; c++) {
        const size_t src = img_base + (size_t)c * HW;
        const int dst_off = c * STRIP_H * WIDTH;
        for (int i = tid; i < STRIP_H * WIDTH; i += NTHREADS){
            in_strip[dst_off + i] = input[src + i];
        }
    }
    __syncthreads();

    const int h_out_local = tid / WIDTH_OUT;
    const int w_out = tid - h_out_local * WIDTH_OUT;
    const int h_out = h_row_base + h_out_local;

    float acc[MAP_OUT];
    #pragma unroll
    for (int m = 0; m < MAP_OUT; m++){
        acc[m] = 0.0f;
    }

    #pragma unroll
    for (int c = 0; c < CHANNEL; c++) {
        const int c_off = c * STRIP_H * WIDTH;
        #pragma unroll
        for (int p = 0; p < KK_SIZE; p++) {
            const int row_off = c_off + (h_out_local + p) * WIDTH;
            #pragma unroll
            for (int q = 0; q < KK_SIZE; q++) {
                float v = in_strip[row_off + w_out + q];
                #pragma unroll
                for (int m = 0; m < MAP_OUT; m++) {
                    acc[m] += const_mask[((m * CHANNEL + c) * KK_SIZE + p) * KK_SIZE + q] * v;
                }
            }
        }
    }

    const size_t out_base = (size_t)b_img * OUT_IMG_ST + (size_t)h_out * WIDTH_OUT + (size_t)w_out;
    #pragma unroll
    for (int m = 0; m < MAP_OUT; m++) {
        output[out_base + (size_t)m * IMAGE_SIZE] = acc[m];
    }
}

template<int ROWS_PER_BLOCK>
__global__ void wmma_strip_conv_kernel(const float * __restrict__ input, float * __restrict__ output, int Batch)
{
    using namespace nvcuda::wmma;

    constexpr int CHANNEL = 4;
    constexpr int KK_SIZE = 7;
    constexpr int HEIGHT = 40;
    constexpr int WIDTH = 40;
    constexpr int MAP_OUT = 16;
    constexpr int HEIGHT_OUT = HEIGHT - KK_SIZE + 1; // 34
    constexpr int WIDTH_OUT = WIDTH  - KK_SIZE + 1;// 34
    constexpr int IMAGE_SIZE = HEIGHT_OUT * WIDTH_OUT; // 1156
    constexpr int KK = KK_SIZE * KK_SIZE; // 49
    constexpr int H_UNROLLED = CHANNEL * KK; // 196
    constexpr int H_UNROLL_PAD = ((H_UNROLLED + 15) / 16) * 16; // 208
    constexpr int CHW = CHANNEL * HEIGHT * WIDTH; // 6400
    constexpr int HW = HEIGHT * WIDTH; // 1600
    constexpr int OUT_IMG_ST = MAP_OUT * IMAGE_SIZE; // 18496
    constexpr int STRIP_H = ROWS_PER_BLOCK + KK_SIZE - 1;
    constexpr int STRIP_ELEMS = CHANNEL * STRIP_H * WIDTH;

    constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;
    constexpr int BN = ROWS_PER_BLOCK * WIDTH_OUT;
    constexpr int BN_PAD = ((BN + WMMA_N - 1) / WMMA_N) * WMMA_N;
    constexpr int N_WARPS = BN_PAD / WMMA_N;
    constexpr int NTHREADS = 32 * N_WARPS;
    constexpr int numTiles = H_UNROLL_PAD / WMMA_K; // 13

    const int b_img = blockIdx.y;
    const int h_row_base = blockIdx.x * ROWS_PER_BLOCK;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    // mask read directly from global_mask_padded
    __shared__ float  in_strip[STRIP_ELEMS];
    //B reads before any warp writes C
    __shared__ __align__(16) float Cs[WMMA_M * BN_PAD];
    __half * Bs = reinterpret_cast<__half*>(Cs);

    //load input strip
    const size_t img_base = (size_t)b_img * CHW + (size_t)h_row_base * WIDTH;
    #pragma unroll
    for (int c = 0; c < CHANNEL; c++) {
        const size_t src_base = img_base + (size_t)c * HW;
        const int dst_off  = c * STRIP_H * WIDTH;
        for (int i = tid; i < STRIP_H * WIDTH; i += NTHREADS) {
            int row = i / WIDTH;
            int abs_row = h_row_base + row;
            in_strip[dst_off + i] = (abs_row < HEIGHT) ? input[src_base + i] : 0.0f;
        }
    }
    __syncthreads();

    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    fill_fragment(c_frag, 0.0f);

    //each warp 16col slab of B
    #pragma unroll
    for (int t = 0; t < numTiles; t++) {
        const int k_base = t * WMMA_K;

        #pragma unroll
        for (int iter = 0; iter < (WMMA_K * WMMA_N) / 32; iter++){//8 iters(256 elems/32 lanes)
            int lin= iter * 32 + lane;
            int blocal = lin >> 4; // row within 16-row B-tile
            int col_in_warp = lin & 15; // col within warp's 16-col slab
            int bcol = warp_id * WMMA_N + col_in_warp;
            int brow = k_base + blocal;

            __half bv = __float2half(0.0f);
            if (brow < H_UNROLLED && bcol < BN) {
                int c = brow / KK;
                int pq = brow - c * KK;
                int p = pq / KK_SIZE;
                int q = pq - p * KK_SIZE;
                int h_out_local = bcol / WIDTH_OUT;
                int w_out = bcol - h_out_local * WIDTH_OUT;

                //strip address=c-channel offset + (h_out_local + p) row + (w_out + q) col
                int strip_off = c * STRIP_H * WIDTH + (h_out_local + p) * WIDTH + (w_out + q);
                bv = __float2half(in_strip[strip_off]);
            }
            Bs[blocal * BN_PAD + bcol] = bv;
        }
        __syncwarp();

        fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, row_major> a_frag;
        fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, row_major> b_frag;
        load_matrix_sync(a_frag, global_mask_padded + k_base, H_UNROLL_PAD);
        load_matrix_sync(b_frag, Bs + warp_id * WMMA_N, BN_PAD);
        mma_sync(c_frag, a_frag, b_frag, c_frag);
        //next iter overwrites this warp slab only
    }

    //cross warp barrier before any warp overwrites B memory via store_matrix_sync
    __syncthreads();
    store_matrix_sync(Cs + warp_id * WMMA_N, c_frag, BN_PAD, mem_row_major);
    __syncthreads();

    //scatter to output
    for (int idx = tid; idx < MAP_OUT * BN; idx += NTHREADS) {
        int row = idx / BN;
        int bcol = idx - row * BN;
        int h_out_local = bcol / WIDTH_OUT;
        int w_out = bcol - h_out_local * WIDTH_OUT;
        int h_out = h_row_base + h_out_local;
        if (h_out < HEIGHT_OUT) {
            size_t out_idx = (size_t)b_img * OUT_IMG_ST + (size_t)row * IMAGE_SIZE + (size_t)h_out * WIDTH_OUT + (size_t)w_out;
            output[out_idx] = Cs[row * BN_PAD + bcol];
        }
    }
}


__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    const size_t mask_count = (size_t)Map_out * Channel * K * K;

    cudaMalloc((void**)device_output_ptr, (size_t)Batch * Map_out * Height_out * Width_out * sizeof(float));
    cudaMalloc((void**)device_input_ptr, (size_t)Batch * Channel * Height * Width * sizeof(float));
    *device_mask_ptr = nullptr;

    cudaMemcpy(*device_input_ptr, host_input, (size_t)Batch * Channel * Height * Width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(const_mask, host_mask, mask_count * sizeof(float));

    static __half host_mask_h[MAX_MASK_SIZE];
    for (size_t i = 0; i < mask_count; i++) {
        host_mask_h[i] = __float2half_rn(host_mask[i]);
    }
    cudaMemcpyToSymbol(const_mask_h, host_mask_h, mask_count * sizeof(__half));

    //padded mask (16 rows*208 cols, zero-pad cols from 196 to 207)for strip kernel
    if (Map_out == 16 && Channel == 4 && K == 7) {
        constexpr int H_UNROLLED = 4 * 7 * 7;// 196
        constexpr int H_UNROLL_PAD = 208;
        static __half host_mask_padded[16 * H_UNROLL_PAD];
        for (int r = 0; r < 16; r++) {
            for (int c = 0; c < H_UNROLL_PAD; c++) {
                host_mask_padded[r * H_UNROLL_PAD + c] =
                    (c < H_UNROLLED) ? __float2half_rn(host_mask[r * H_UNROLLED + c]) : __float2half_rn(0.0f);
            }
        }
        cudaMemcpyToSymbol(global_mask_padded, host_mask_padded, sizeof(host_mask_padded));
    }
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out  = Width  - K + 1;

    if (Map_out == 4 && Channel == 1 && K == 7 && Height == 86 && Width == 86) {
        //conv1 direct_conv_kernel
        constexpr int ROWS_PER_BLOCK = 2;
        constexpr int NTHREADS = ROWS_PER_BLOCK * 80;
        dim3 block(NTHREADS, 1, 1);
        dim3 grid((unsigned int)(80 / ROWS_PER_BLOCK), (unsigned int)Batch, 1);
        direct_conv_kernel<4, 1, 7, 86, 86, ROWS_PER_BLOCK><<<grid, block>>>(device_input, device_output, Batch);
    } 
    else if (Map_out == 16 && Channel == 4 && K == 7 && Height == 40 && Width == 40) {
        //conv2 strip-loaded WMMA
        //RPB=4; BN=136, BN_PAD=144, 9 warps, 288 threads
        //ceil(34/4)=9 row group per image
        constexpr int ROWS_PER_BLOCK = 4;
        constexpr int BN_PAD = ((ROWS_PER_BLOCK * 34 + 15) / 16) * 16;
        constexpr int N_WARPS = BN_PAD / 16;
        constexpr int NTHREADS = 32 * N_WARPS;
        dim3 block(NTHREADS, 1, 1);
        dim3 grid((unsigned int)((34 + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK), (unsigned int)Batch, 1);
        wmma_strip_conv_kernel<ROWS_PER_BLOCK><<<grid, block>>>(device_input, device_output, Batch);
    }
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out  = Width  - K + 1;

    cudaMemcpy(host_output, device_output, (size_t)Batch * Map_out * Height_out * Width_out * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(device_output);
    cudaFree(device_input);
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
