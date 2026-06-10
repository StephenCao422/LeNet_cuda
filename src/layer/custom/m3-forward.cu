#include <cmath>
#include <cstdint>
#include <iostream>
#include <cuda_fp16.h>
#include <mma.h>
#include "gpu-new-forward.h"

using namespace nvcuda::wmma;

__constant__ float  const_mask[4096];
__device__ __half global_mask_padded[16 * 208];

template<int ROWS_PER_BLOCK, int WIDTH_OUT, int HEIGHT_OUT, int IMAGE_SIZE>
__device__ __inline__ void write_c2_pair(float * __restrict__ output, size_t img_out_base, int h_row_base, int m, int bcol_pair, float v0, float v1)
{
    const int BN = ROWS_PER_BLOCK * WIDTH_OUT;
    if (bcol_pair >= BN){
        return;
    }

    int h_out_local = bcol_pair / WIDTH_OUT;
    int w_out = bcol_pair - h_out_local * WIDTH_OUT;
    int h_out = h_row_base + h_out_local;

    if (h_out >= HEIGHT_OUT){
        return;
    }

    float2 *dst = reinterpret_cast<float2*>(&output[img_out_base + (size_t)m * IMAGE_SIZE + (size_t)h_out * WIDTH_OUT + (size_t)w_out]);
    *dst = make_float2(v0, v1);
}

template<int ROWS_PER_BLOCK>
__global__ void layer_1_kernel(const float * __restrict__ input, float * __restrict__ output, int Batch)
{
    const int CHANNEL = 1;
    const int k_size = 7;
    const int HEIGHT = 86;
    const int WIDTH = 86;
    const int MAP_OUT = 4;
    const int HEIGHT_OUT = HEIGHT - k_size + 1;
    const int WIDTH_OUT = WIDTH - k_size + 1;
    const int IMAGE_SIZE = HEIGHT_OUT * WIDTH_OUT;
    const int strip_h = ROWS_PER_BLOCK + k_size - 1;
    const int BN = ROWS_PER_BLOCK * WIDTH_OUT;
    const int COARSEN_N = 4;// each thread has 4 adjacent output cols
    const int NTHREADS = BN / COARSEN_N;

    const int K_DIM = CHANNEL * k_size * k_size;
    const int TILE_K = k_size;
    const int NUM_K_PHASES = K_DIM / TILE_K;
    const int B_PREFETCH = TILE_K + COARSEN_N - 1;//10 B val per phase

    const int b_img = blockIdx.y;
    const int h_row_base = blockIdx.x * ROWS_PER_BLOCK;
    const int tid = threadIdx.x;

    //B tile source in shared mem
    __shared__ float Bs_strip[CHANNEL * strip_h * WIDTH];

    const size_t img_base = (size_t)b_img * (CHANNEL * HEIGHT * WIDTH) + (size_t)h_row_base * WIDTH;
    #pragma unroll
    for (int c = 0; c < CHANNEL; c++) {
        const size_t src = img_base + (size_t)c * (HEIGHT * WIDTH);
        const int dst_off = c * strip_h * WIDTH;
        for (int i = tid; i < strip_h * WIDTH; i += NTHREADS) {
            Bs_strip[dst_off + i] = input[src + i];
        }
    }
    __syncthreads();

    //thread owns COARSEN_N adjacent cols
    const int n0 = tid * COARSEN_N;
    const int h_out_local = n0 / WIDTH_OUT;
    const int w0 = n0 - h_out_local * WIDTH_OUT;
    const int h_out = h_row_base + h_out_local;

    // C tile in reg(4 colsxM out)
    float acc0[MAP_OUT];
    float acc1[MAP_OUT];
    float acc2[MAP_OUT];
    float acc3[MAP_OUT];
    #pragma unroll
    for (int m = 0; m < MAP_OUT; m++) {
        acc0[m] = 0.0f; acc1[m] = 0.0f; acc2[m] = 0.0f; acc3[m] = 0.0f;
    }

    #pragma unroll
    for (int phase = 0; phase < NUM_K_PHASES; phase++) {
        const int c = phase / k_size;
        const int p = phase - c * k_size;
        const int b_row_off = c * strip_h * WIDTH + (h_out_local + p) * WIDTH;
        const int a_phase_off = phase * TILE_K;

        //prefetch B val across all q
        float Breg[B_PREFETCH];
        #pragma unroll
        for (int i = 0; i < B_PREFETCH; i++) {
            Breg[i] = Bs_strip[b_row_off + w0 + i];
        }

        //inner K tile
        #pragma unroll
        for (int q = 0; q < TILE_K; q++) {
            const float bv0 = Breg[q];
            const float bv1 = Breg[q + 1];
            const float bv2 = Breg[q + 2];
            const float bv3 = Breg[q + 3];
            const int k_idx = a_phase_off + q;
            #pragma unroll
            for (int m = 0; m < MAP_OUT; m++) {
                const float av = const_mask[m * K_DIM + k_idx];//const cache broadcast
                acc0[m] += av * bv0;
                acc1[m] += av * bv1;
                acc2[m] += av * bv2;
                acc3[m] += av * bv3;
            }
        }
    }

    //store
    if (h_out < HEIGHT_OUT) {
        const size_t out_base = (size_t)b_img * (MAP_OUT * IMAGE_SIZE) + (size_t)h_out * WIDTH_OUT + (size_t)w0;
        #pragma unroll
        for (int m = 0; m < MAP_OUT; m++) {
            float4 *dst = reinterpret_cast<float4*>(&output[out_base + (size_t)m * IMAGE_SIZE]);
            *dst = make_float4(acc0[m], acc1[m], acc2[m], acc3[m]);
        }
    }
}

template<int ROWS_PER_BLOCK>
__global__ void layer_2_kernel(const float * __restrict__ input, float * __restrict__ output, int Batch)
{
    const int CHANNEL = 4;
    const int k_size = 7;
    const int HEIGHT = 40;
    const int WIDTH = 40;
    const int MAP_OUT = 16;
    const int HEIGHT_OUT = HEIGHT - k_size + 1;
    const int WIDTH_OUT = WIDTH - k_size + 1;
    const int IMAGE_SIZE = HEIGHT_OUT * WIDTH_OUT;

    //unroll
    const int h_unrolled = CHANNEL * (k_size*k_size);
    const int unrolled_pad = ((h_unrolled + 15) / 16) * 16;
    const int strip_h = ROWS_PER_BLOCK + k_size - 1;
    const int strip_elems = CHANNEL * strip_h * WIDTH;
    //wmma
    const int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;
    const int BN = ROWS_PER_BLOCK * WIDTH_OUT;
    const int LDM = ((BN + WMMA_N - 1) / WMMA_N) * WMMA_N;
    const int N_WARPS = LDM / WMMA_N;
    const int NTHREADS = 32 * N_WARPS;
    const int numTiles = unrolled_pad / WMMA_K;

    const int b_pair = blockIdx.y * 2;
    const int h_row_base = blockIdx.x * ROWS_PER_BLOCK;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    __shared__ __align__(16) __half in_strip[2 * strip_elems];
    __shared__ __align__(16) __half Bs[WMMA_K * LDM];

    //load input strip to shared mem(2imgs)
    #pragma unroll
    for (int img_slot = 0; img_slot < 2; img_slot++) {
        const int b_img = b_pair + img_slot;
        const bool valid_img = b_img < Batch;
        const size_t img_base = (size_t)b_img * (CHANNEL * HEIGHT * WIDTH) + (size_t)h_row_base * WIDTH;
        const int strip_slot_off = img_slot * strip_elems;
        #pragma unroll
        for (int c = 0; c < CHANNEL; c++) {
            const size_t src_base = img_base + (size_t)c * (HEIGHT * WIDTH);
            const int dst_off = strip_slot_off + c * strip_h * WIDTH;
            for (int i = tid; i < strip_h * WIDTH; i += NTHREADS) {
                int row = i / WIDTH;
                int abs_row = h_row_base + row;
                float v = (valid_img && abs_row < HEIGHT) ? input[src_base + i] : 0.0f;
                in_strip[dst_off + i] = __float2half_rn(v);
            }
        }
    }
    __syncthreads();

    //store adjacent output columns as half2
    const int my_col_pair_idx = lane & 7;
    const int my_bcol_a = warp_id * WMMA_N + 2 * my_col_pair_idx;
    const bool my_pair_valid = (my_bcol_a + 1) < BN;
    const int my_h_out_local = my_pair_valid ? (my_bcol_a / WIDTH_OUT) : 0;
    const int my_w_out_a = my_pair_valid ? (my_bcol_a - my_h_out_local * WIDTH_OUT) : 0;
    const int my_strip_col_off = my_h_out_local * WIDTH + my_w_out_a;
    const __half hzero = __float2half_rn(0.0f);

    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag0;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag1;
    fill_fragment(c_frag0, 0.0f);
    fill_fragment(c_frag1, 0.0f);

    #pragma unroll
    for (int t = 0; t < numTiles; t++) {
        const int k_base = t * WMMA_K;

        //compute one row offset, broadcast in warp
        int my_br_off = -1;
        if (lane < 16) {
            int brow_local = k_base + lane;
            if (brow_local < h_unrolled) {
                int c = brow_local / (k_size*k_size);
                int pq = brow_local - c * (k_size*k_size);
                int p = pq / k_size;
                int q = pq - p * k_size;
                my_br_off = c * strip_h * WIDTH + p * WIDTH + q;
            }
        }

        fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, row_major> a_frag;
        load_matrix_sync(a_frag, global_mask_padded + k_base, unrolled_pad);

        #pragma unroll
        for (int img_slot = 0; img_slot < 2; img_slot++) {
            const int strip_slot_off = img_slot * strip_elems;
            //pack input pairs into B
            #pragma unroll
            for (int iter = 0; iter < 4; iter++) {
                int pair_idx = iter * 32 + lane;
                int blocal = pair_idx >> 3;
                int br_off = __shfl_sync(0xFFFFFFFF, my_br_off, blocal);

                __half ha = hzero;
                __half hb = hzero;
                if (br_off >= 0 && my_pair_valid) {
                    int off_a = strip_slot_off + br_off + my_strip_col_off;
                    ha = in_strip[off_a];
                    hb = in_strip[off_a + 1];
                }
                __half2 bv2 = __halves2half2(ha, hb);
                *reinterpret_cast<__half2*>(&Bs[blocal * LDM + my_bcol_a]) = bv2;
            }
            __syncwarp();

            //wmma
            fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, row_major> b_frag;
            load_matrix_sync(b_frag, Bs + warp_id * WMMA_N, LDM);
            if (img_slot == 0) {
                mma_sync(c_frag0, a_frag, b_frag, c_frag0);
            } else {
                mma_sync(c_frag1, a_frag, b_frag, c_frag1);
            }
        }
    }

    //write output
    const int groupId = lane >> 2;
    const int tg = lane & 3;
    const int col_lo = warp_id * WMMA_N + tg * 2;
    const int col_hi = warp_id * WMMA_N + tg * 2 + 8;

    const size_t img0_out_base = (size_t)b_pair * (MAP_OUT * IMAGE_SIZE);
    write_c2_pair<ROWS_PER_BLOCK, WIDTH_OUT, HEIGHT_OUT, IMAGE_SIZE>(output, img0_out_base, h_row_base, groupId,     col_lo, c_frag0.x[0], c_frag0.x[1]);
    write_c2_pair<ROWS_PER_BLOCK, WIDTH_OUT, HEIGHT_OUT, IMAGE_SIZE>(output, img0_out_base, h_row_base, groupId + 8, col_lo, c_frag0.x[2], c_frag0.x[3]);
    write_c2_pair<ROWS_PER_BLOCK, WIDTH_OUT, HEIGHT_OUT, IMAGE_SIZE>(output, img0_out_base, h_row_base, groupId,     col_hi, c_frag0.x[4], c_frag0.x[5]);
    write_c2_pair<ROWS_PER_BLOCK, WIDTH_OUT, HEIGHT_OUT, IMAGE_SIZE>(output, img0_out_base, h_row_base, groupId + 8, col_hi, c_frag0.x[6], c_frag0.x[7]);

    if (b_pair + 1 < Batch) {
        const size_t img1_out_base = (size_t)(b_pair + 1) * (MAP_OUT * IMAGE_SIZE);
        write_c2_pair<ROWS_PER_BLOCK, WIDTH_OUT, HEIGHT_OUT, IMAGE_SIZE>(output, img1_out_base, h_row_base, groupId,     col_lo, c_frag1.x[0], c_frag1.x[1]);
        write_c2_pair<ROWS_PER_BLOCK, WIDTH_OUT, HEIGHT_OUT, IMAGE_SIZE>(output, img1_out_base, h_row_base, groupId + 8, col_lo, c_frag1.x[2], c_frag1.x[3]);
        write_c2_pair<ROWS_PER_BLOCK, WIDTH_OUT, HEIGHT_OUT, IMAGE_SIZE>(output, img1_out_base, h_row_base, groupId,     col_hi, c_frag1.x[4], c_frag1.x[5]);
        write_c2_pair<ROWS_PER_BLOCK, WIDTH_OUT, HEIGHT_OUT, IMAGE_SIZE>(output, img1_out_base, h_row_base, groupId + 8, col_hi, c_frag1.x[6], c_frag1.x[7]);
    }
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    const size_t mask_count = (size_t)Map_out * Channel * K * K;

    cudaMalloc((void**)device_output_ptr, (size_t)Batch * Map_out * Height_out * Width_out * sizeof(float));
    cudaMalloc((void**)device_input_ptr, (size_t)Batch * Channel * Height * Width * sizeof(float));
    cudaMalloc((void**)device_mask_ptr, mask_count * sizeof(float));

    cudaMemcpy(*device_input_ptr, host_input, (size_t)Batch * Channel * Height * Width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask, mask_count * sizeof(float), cudaMemcpyHostToDevice);
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    if (Map_out == 4 && Channel == 1 && K == 7 && Height == 86 && Width == 86) {
        cudaMemcpyToSymbol(const_mask, device_mask, (size_t)Map_out * Channel * K * K * sizeof(float), 0, cudaMemcpyDeviceToDevice);

        const int ROWS_PER_BLOCK = 4;
        const int NTHREADS = (ROWS_PER_BLOCK * 80) / 4;//COARSEN_N=4
        dim3 block(NTHREADS, 1, 1);
        dim3 grid((unsigned int)(80 / ROWS_PER_BLOCK), (unsigned int)Batch, 1);
        layer_1_kernel<ROWS_PER_BLOCK><<<grid, block>>>(device_input, device_output, Batch);
    }
    else if (Map_out == 16 && Channel == 4 && K == 7 && Height == 40 && Width == 40) {
        const int h_unrolled = 4 * 7 * 7;
        const int unrolled_pad = 208;
        static float host_mask[16 * h_unrolled];
        static __half host_mask_padded[16 * unrolled_pad];
        cudaMemcpy(host_mask, device_mask, sizeof(host_mask), cudaMemcpyDeviceToHost);
        for (int r = 0; r < 16; r++) {
            for (int c = 0; c < unrolled_pad; c++) {
                const float v = (c < h_unrolled) ? host_mask[r * h_unrolled + c] : 0.0f;
                host_mask_padded[r * unrolled_pad + c] = __float2half_rn(v);
            }
        }
        cudaMemcpyToSymbol(global_mask_padded, host_mask_padded, sizeof(host_mask_padded));

        //batch pair wmma path,RPB=5
        const int ROWS_PER_BLOCK = 5;
        const int LDM = ((ROWS_PER_BLOCK * 34 + 15) / 16) * 16;
        const int N_WARPS = LDM / 16;
        const int NTHREADS = 32 * N_WARPS;
        dim3 block(NTHREADS, 1, 1);
        dim3 grid((unsigned int)((34 + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK), (unsigned int)((Batch + 1) / 2), 1);
        layer_2_kernel<ROWS_PER_BLOCK><<<grid, block>>>(device_input, device_output, Batch);
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
