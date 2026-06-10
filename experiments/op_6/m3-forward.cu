#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

template<int BM, int BN, int BK, int TM, int TN>

__global__ void matmul_conv_fused_t(const float *mask, const float *input, float *output,
                                  int Batch, int Map_out, int Channel, int Height, int Width, int K)
{
    constexpr int NTHREADS = (BM / TM) * (BN / TN);
    constexpr int THREADS_X = BN / TN;

    const int Height_out = Height - K + 1;
    const int Width_out  = Width  - K + 1;
    const int image_size = Height_out * Width_out;
    const int Height_unrolled = Channel * K * K;
    const size_t Width_unrolled = (size_t)Batch * image_size;

    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];
    // hoisted column decomposition for the BN cols
    __shared__ int col_bimg[BN];
    __shared__ int col_hout[BN];
    __shared__ int col_wout[BN];

    const int tid = threadIdx.x;
    const int ty_th = tid / THREADS_X;
    const int tx_th = tid - ty_th * THREADS_X;

    const int row_base = blockIdx.y * BM;
    const size_t col_base = (size_t)blockIdx.x * BN;

    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++){
        #pragma unroll
        for (int j = 0; j < TN; j++){
            acc[i][j] = 0.0f;
        }
    }
    // precompute column decomposition
    for (int i = tid; i < BN; i += NTHREADS) {
        size_t gc = col_base + (size_t)i;
        if (gc < Width_unrolled) {
            int b_img = (int)(gc / image_size);
            int hw    = (int)(gc - (size_t)b_img * image_size);
            int h_out = hw / Width_out;
            int w_out = hw - h_out * Width_out;
            col_bimg[i] = b_img;
            col_hout[i] = h_out;
            col_wout[i] = w_out;
        } else {
            col_bimg[i] = -1;
        }
    }
    __syncthreads();

    constexpr int A_ELEMS = BM * BK;
    constexpr int B_ELEMS = BK * BN;
    constexpr int A_ITERS = (A_ELEMS + NTHREADS - 1) / NTHREADS;
    constexpr int B_ITERS = (B_ELEMS + NTHREADS - 1) / NTHREADS;
    constexpr bool A_EXACT = (A_ITERS * NTHREADS == A_ELEMS);
    constexpr bool B_EXACT = (B_ITERS * NTHREADS == B_ELEMS);

    const int KK = K * K;
    const int numTiles = (Height_unrolled + BK - 1) / BK;

    for (int t = 0; t < numTiles; t++) {
        const int k_base = t * BK;

        // load A tile(mask rows x BK)
        #pragma unroll
        for (int i = 0; i < A_ITERS; i++) {
            int lin = i * NTHREADS + tid;
            if (A_EXACT || lin < A_ELEMS) {
                int arow = lin / BK;
                int acol = lin - arow * BK;
                int grow = row_base + arow;
                int gcol = k_base + acol;
                As[arow][acol] = (grow < Map_out && gcol < Height_unrolled)
                    ? mask[(size_t)grow * Height_unrolled + gcol]
                    : 0.0f;
            }
        }

        // load B tile (BK x BN)
        #pragma unroll
        for (int i = 0; i < B_ITERS; i++) {
            int lin = i * NTHREADS + tid;
            if (B_EXACT || lin < B_ELEMS) {
                int brow = lin / BN;
                int bcol = lin - brow * BN;
                int grow = k_base + brow;

                float bv = 0.0f;
                int b_img = col_bimg[bcol];
                if (grow < Height_unrolled && b_img >= 0) {
                    int c  = grow / KK;
                    int pq = grow - c * KK;
                    int p  = pq / K;
                    int q  = pq - p * K;
                    int h_out = col_hout[bcol];
                    int w_out = col_wout[bcol];
                    bv = input[(size_t)b_img * (Channel * Height * Width)
                             + (size_t)c     * (Height * Width)
                             + (size_t)(h_out + p) * Width
                             + (w_out + q)];
                }
                Bs[brow][bcol] = bv;
            }
        }
        __syncthreads();

        // TM x TN FMAs per k
        float reg_a[TM];
        float reg_b[TN];
        #pragma unroll
        for (int k = 0; k < BK; k++) {
            #pragma unroll
            for (int i = 0; i < TM; i++)
                reg_a[i] = As[ty_th * TM + i][k];
            #pragma unroll
            for (int j = 0; j < TN; j++)
                reg_b[j] = Bs[k][tx_th * TN + j];
            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TN; j++)
                    acc[i][j] += reg_a[i] * reg_b[j];
        }
        __syncthreads();
    }
    
    // write back
    const int row_start = row_base + ty_th * TM;
    const int bcol_start = tx_th * TN;
    #pragma unroll
    for (int j = 0; j < TN; j++) {
        int bcol = bcol_start + j;
        int b_img = col_bimg[bcol];
        if (b_img < 0) continue;
        int h_out = col_hout[bcol];
        int w_out = col_wout[bcol];
        size_t out_bm_base = (size_t)b_img * (Map_out * image_size);
        size_t hw_off      = (size_t)h_out * Width_out + w_out;
        #pragma unroll
        for (int i = 0; i < TM; i++) {
            int row = row_start + i;
            if (row >= Map_out) continue;
            output[out_bm_base + (size_t)row * image_size + hw_off] = acc[i][j];
        }
    }
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out  = Width  - K + 1;

    cudaMalloc((void**)device_output_ptr, (size_t)Batch * Map_out * Height_out * Width_out * sizeof(float));
    cudaMalloc((void**)device_input_ptr,  (size_t)Batch * Channel * Height * Width * sizeof(float));
    cudaMalloc((void**)device_mask_ptr,   (size_t)Map_out * Channel * K * K * sizeof(float));

    cudaMemcpy(*device_input_ptr, host_input, (size_t)Batch * Channel * Height * Width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr,  host_mask,  (size_t)Map_out * Channel * K * K * sizeof(float),        cudaMemcpyHostToDevice);
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out  = Width  - K + 1;
    const size_t Width_unrolled = (size_t)Batch * Height_out * Width_out;

    //layer-1 (Map_out=4): BM=4, BN=128, BK=32, TM=1, TN=4, 128 threads, no padded row waste
    //layer-2 (Map_out=16): BM=16, BN=64, BK=16, TM=2, TN=4, 128 threads, same as baseline
    if (Map_out <= 4) {
        constexpr int BM = 4;
        constexpr int BN = 128;
        constexpr int BK = 32;
        constexpr int TM = 1;
        constexpr int TN = 4;

        constexpr int NTHREADS = (BM / TM) * (BN / TN);
        dim3 block(NTHREADS, 1, 1);
        dim3 grid((unsigned int)((Width_unrolled + BN - 1) / BN), (unsigned int)((Map_out + BM - 1) / BM), 1);
        matmul_conv_fused_t<BM, BN, BK, TM, TN><<<grid, block>>>(device_mask, device_input, device_output, Batch, Map_out, Channel, Height, Width, K);
    } else {
        constexpr int BM = 16;
        constexpr int BN = 64;
        constexpr int BK = 16;
        constexpr int TM = 2;
        constexpr int TN = 4;
        constexpr int NTHREADS = (BM / TM) * (BN / TN);
        dim3 block(NTHREADS, 1, 1);
        dim3 grid((unsigned int)((Width_unrolled + BN - 1) / BN), (unsigned int)((Map_out + BM - 1) / BM), 1);
        matmul_conv_fused_t<BM, BN, BK, TM, TN><<<grid, block>>>(device_mask, device_input, device_output, Batch, Map_out, Channel, Height, Width, K);
    }
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
