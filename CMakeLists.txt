cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

project(LeNetCUDA LANGUAGES CUDA CXX)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "80;86")
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")

include_directories("${PROJECT_SOURCE_DIR}/third_party/eigen")

aux_source_directory(. DIR_SRCS)

add_subdirectory(src)

add_library(ece408net ece408net.cc ece408net.h)

# new-forward.cu is unused in CPU convolution, but we need to include it to make compiler happy
add_executable(m1_cpu m1_cpu.cc "${PROJECT_SOURCE_DIR}/src/layer/custom/new-forward.cu")
target_link_libraries(m1_cpu ece408net MiniDNNLib)
# Set compiler flags for debug builds
target_compile_options(m1_cpu PRIVATE "$<$<CONFIG:DEBUG>:-gdwarf-5>")

add_executable(m1_gpu m1_gpu.cc "${PROJECT_SOURCE_DIR}/src/layer/custom/new-forward.cu")
target_link_libraries(m1_gpu ece408net MiniDNNLib)

add_executable(m2_unroll m1_gpu.cc
                         "${PROJECT_SOURCE_DIR}/src/layer/custom/unroll-new-forward.cu"
                         "${PROJECT_SOURCE_DIR}/src/layer/custom/matmul.cu")
target_link_libraries(m2_unroll ece408net MiniDNNLib)

add_executable(m2_fused m1_gpu.cc
                         "${PROJECT_SOURCE_DIR}/src/layer/custom/kernel-fusion-forward.cu")
target_link_libraries(m2_fused ece408net MiniDNNLib)
add_executable(m3 m1_gpu.cc "${PROJECT_SOURCE_DIR}/src/layer/custom/m3-forward.cu"
                            "${PROJECT_SOURCE_DIR}/src/layer/custom/matmul.cu")
target_link_libraries(m3 ece408net MiniDNNLib cublas)

add_executable(lenet_cuda m1_gpu.cc
                          "${PROJECT_SOURCE_DIR}/src/layer/custom/m3-forward.cu"
                          "${PROJECT_SOURCE_DIR}/src/layer/custom/matmul.cu")
target_link_libraries(lenet_cuda ece408net MiniDNNLib cublas)

add_executable(viz viz.cc
                   "${PROJECT_SOURCE_DIR}/src/layer/custom/m3-forward.cu"
                   "${PROJECT_SOURCE_DIR}/src/layer/custom/matmul.cu")
target_link_libraries(viz ece408net MiniDNNLib cublas)
