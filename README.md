# LeNet CUDA

CUDA-accelerated inference for a modified LeNet-style convolutional neural
network on 86x86 Fashion-MNIST images. The project compares a CPU convolution,
a basic CUDA implementation, input-unrolling/GEMM approaches, kernel fusion,
and a final architecture-specific optimized implementation.

> **Repository status:** This is ECE 408 coursework and must remain private
> unless the course staff provides explicit written permission to publish it.

![Modified LeNet architecture](assets/lenet.png)

## Highlights

- Direct CUDA convolution baseline
- Input unrolling with tiled matrix multiplication
- Fused implicit-GEMM convolution
- Constant-memory weights and loop unrolling
- Joint register/shared-memory tiling
- FP16 and WMMA Tensor Core acceleration
- Optional benchmark mode with warm-up and median kernel timing

The final implementation specializes the two convolution layers separately:

- Layer 1 uses FP32 direct convolution with strip tiling and register coarsening.
- Layer 2 uses FP16 input tiles and WMMA operations with FP32 accumulation.

## Repository Layout

```text
.
|-- src/                         Mini-DNN layers and CUDA support
|   `-- layer/custom/            Convolution kernels
|-- experiments/                 Source-only optimization variants
|-- scripts/                     Delta/Slurm job examples
|-- third_party/eigen/           Vendored Eigen dependency
|-- assets/                      Architecture diagram
|-- CMakeLists.txt
|-- ece408net.cc                 Network definition and model loading
|-- m1_cpu.cc                    CPU baseline entry point
|-- m1_gpu.cc                    CUDA inference and benchmark entry point
`-- weights-86.bin               Pretrained model parameters
```

Generated logs, profiler databases, reports, and build products are excluded
from version control.

## Requirements

- Linux
- CMake 3.20+
- CUDA Toolkit with `nvcc`
- NVIDIA GPU with compute capability 8.0 or 8.6 by default

The optimized kernels target Ampere Tensor Cores. Override the CUDA
architectures at configure time when needed.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

For a different GPU architecture:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=86
```

## Dataset

The executables expect the resized Fashion-MNIST IDX files:

```text
t10k-86-images-idx3-ubyte
t10k-86-labels-idx1-ubyte
```

Set the dataset directory before running outside the course cluster:

```bash
export LENET_DATA_DIR=/path/to/fmnist-86
```

The default path is `/projects/bche/project/data/fmnist-86/`. The pretrained
weights default to `weights-86.bin`; override that with
`LENET_WEIGHTS_PATH=/path/to/weights-86.bin`.

## Run

Run inference for a selected batch size:

```bash
./build/lenet_cuda 10000
```

Run repeated kernel timing:

```bash
./build/lenet_cuda 10000 --competition
```

Other build targets preserve the implementation stages:

```text
m1_cpu    CPU convolution baseline
m1_gpu    basic CUDA convolution
m2_unroll explicit input unrolling plus GEMM
m2_fused  fused implicit-GEMM convolution
m3        final optimized implementation
viz       feature-map export utility
```

## Profiling

Examples for NVIDIA Nsight Systems and Nsight Compute:

```bash
nsys profile --stats=true ./build/lenet_cuda 10000
ncu --set full ./build/lenet_cuda 10000
```

Cluster-specific Slurm examples are under `scripts/`.

## Acknowledgments

The network framework is based on
[mini-dnn-cpp](https://github.com/iamhankai/mini-dnn-cpp). See `LICENSE` for
the supplied framework license.
