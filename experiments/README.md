# Optimization Experiment Catalog

This directory preserves isolated CUDA convolution experiments. Each numbered
variant focuses on one method so its behavior can be profiled separately.
These files are reference implementations and are not compiled by the default
CMake targets.

## Baselines

Two baselines are used because not every optimization fits the same execution
model:

- **Fused implicit GEMM:** reads convolution windows directly from the input
  while performing tiled matrix multiplication. Used by `op_0`, `op_1`,
  `op_2`, `op_5`, `op_6`, and `req_1`.
- **Explicit unrolling:** launches input-unrolling, matrix-multiplication, and
  output-permutation steps separately. Used by `req_0` and `op_4`.

## Required Experiments

### `req_0` - Multi-Stream Pipeline

**File:** [`req_0/m3-forward.cu`](req_0/m3-forward.cu)

Divides a batch into four chunks. Each CUDA stream asynchronously copies a
pinned host chunk to the GPU, runs unrolling, tiled GEMM, and permutation, then
copies its output chunk back. The experiment studies overlap between data
transfer and computation.

### `req_1` - WMMA Tensor Cores

**File:** [`req_1/m3-forward.cu`](req_1/m3-forward.cu)

Builds 16x16 FP16 matrix tiles in shared memory and uses CUDA WMMA
`load_matrix_sync`, `mma_sync`, and `store_matrix_sync`. Accumulation remains
FP32. The convolution input is unrolled implicitly while each WMMA tile is
assembled.

## Optional Experiments

### `op_0` - Constant Weights

**File:** [`op_0/m3-forward.cu`](op_0/m3-forward.cu)

Copies the convolution mask to `__constant__` memory. Threads in a warp often
read the same weight, allowing the constant cache to broadcast that value
instead of repeatedly reading global memory.

### `op_1` - Restricted Pointers

**File:** [`op_1/m3-forward.cu`](op_1/m3-forward.cu)

Adds `__restrict__` to mask, input, and output pointers in the fused kernel and
host interface. This tells the compiler that the buffers do not alias and may
enable more aggressive load scheduling and register reuse.

### `op_2` - Unrolled FMA Loop

**File:** [`op_2/m3-forward.cu`](op_2/m3-forward.cu)

Manually expands the inner 16-element tiled dot product into four groups of
four multiply-add operations. This reduces loop-control work and exposes
independent arithmetic to the compiler.

### `op_4` - cuBLAS SGEMM

**File:** [`op_4/m3-forward.cu`](op_4/m3-forward.cu)

Materializes the unrolled input matrix, calls `cublasSgemm` for FP32 matrix
multiplication, and launches a permutation kernel to restore
batch-major output layout. This compares a library GEMM with the custom
matrix-multiplication kernel.

### `op_5` - FP16 Half2

**File:** [`op_5/m3-forward.cu`](op_5/m3-forward.cu)

Converts input and mask buffers from FP32 to FP16, stores FP16 tiles in shared
memory, and uses `__half2` to process two products at a time. The paired result
is reduced and converted back to FP32 for output.

### `op_6` - Register/Shared Tiling

**File:** [`op_6/m3-forward.cu`](op_6/m3-forward.cu)

Uses configurable block tiles (`BM`, `BN`, `BK`) in shared memory and
per-thread output tiles (`TM`, `TN`) in registers. Threads cooperatively load
matrix tiles, reuse those values across multiple outputs, and accumulate a
small output matrix per thread.

## Combined Kernel

### `competition.cu` - Hybrid Layer Specialization

**File:** [`competition.cu`](competition.cu)

Uses different algorithms for the two LeNet convolution layers:

- The first layer uses an FP32 tiled implicit-GEMM kernel with shared input
  strips and constant-memory weights.
- The second layer processes two images per block with FP16 input strips,
  WMMA Tensor Core multiplication, and FP32 accumulation.

The production version evolved from this combined experiment and is located
at [`../src/layer/custom/m3-forward.cu`](../src/layer/custom/m3-forward.cu).

## Archived Development Kernels

| File | Short name | Description |
| --- | --- | --- |
| [`archive/m3-strip.cu`](archive/m3-strip.cu) | **Single-Image Strip WMMA** | Earlier layer-specialized version using direct strip convolution for layer 1 and one-image WMMA strip processing for layer 2 |
| [`archive/m3-forward-prev.cu`](archive/m3-forward-prev.cu) | **Batch-Paired WMMA Draft** | Later draft that processes two layer-2 images per block before the final kernel refinements |

## Using a Variant

The experiment files implement the same `GPUInterface` methods as the
production kernel. To build one manually, replace the production source path
in a temporary CMake target with the selected experiment file. Keep the
production source unchanged when comparing variants so each result remains
reproducible.
