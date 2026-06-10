#include "conv_cust.h"
#include <algorithm>
#include <cstdlib>
#include <math.h>
#include <iostream>

bool Conv_Custom::benchmark_enabled_ = false;
int Conv_Custom::benchmark_warmup_rounds_ = 5;
int Conv_Custom::benchmark_measure_rounds_ = 10;

void Conv_Custom::init() {
  height_out = (1 + (height_in - height_kernel + 2 * pad_h) / stride);
  width_out =   (1 + (width_in - width_kernel + 2 * pad_w) / stride);
  dim_out = height_out * width_out * channel_out;

  weight.resize(channel_in * height_kernel * width_kernel, channel_out);
  bias.resize(channel_out);
  grad_weight.resize(channel_in * height_kernel * width_kernel, channel_out);
  grad_bias.resize(channel_out);
  set_normal_random(weight.data(), weight.size(), 0, 0.01);
  set_normal_random(bias.data(), bias.size(), 0, 0.01);
  //std::cout << weight.colwise().sum() << std::endl;
  //std::cout << weight.colwise().sum() + bias.transpose() << std::endl;
  // gpuInterface.get_device_properties();
}

void Conv_Custom::configure_benchmark(bool enabled, int warmup_rounds,
                                      int measure_rounds) {
  benchmark_enabled_ = enabled;
  benchmark_warmup_rounds_ = std::max(0, warmup_rounds);
  benchmark_measure_rounds_ = std::max(1, measure_rounds);
}

float Conv_Custom::run_competition_forward(
    float* y, const float* x, const float* k, int B, int M, int C, int K) {
  float *x_d;
  float *y_d;
  float *k_d;

  std::vector<float> kernel_times;
  kernel_times.reserve(benchmark_measure_rounds_);

  gpuUtils.insert_pre_barrier_kernel();

  gpuInterface.conv_forward_gpu_prolog(y, x, k, &y_d, &x_d, &k_d, B, M, C, height_in, width_in, K);

  for (int i = 0; i < benchmark_warmup_rounds_; ++i) {
    auto start_time_kernel = std::chrono::high_resolution_clock::now();
    gpuInterface.conv_forward_gpu(y_d, x_d, k_d, B, M, C, height_in, width_in, K);
    cudaDeviceSynchronize();

    if (i == 0) {
      auto end_time_kernel = std::chrono::high_resolution_clock::now();
      std::chrono::duration<float, std::milli> duration_kernel = (end_time_kernel - start_time_kernel);
      if (duration_kernel.count() > kCompetitionWarmupAbortThresholdMs) {
        std::cerr << "Competition warm-up exceeded "
                  << kCompetitionWarmupAbortThresholdMs
                  << " ms; aborting." << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }
  }

  for (int i = 0; i < benchmark_measure_rounds_; ++i) {
    auto start_time_kernel = std::chrono::high_resolution_clock::now();
    gpuInterface.conv_forward_gpu(y_d, x_d, k_d, B, M, C, height_in, width_in, K);
    cudaDeviceSynchronize();
    auto end_time_kernel = std::chrono::high_resolution_clock::now();

    std::chrono::duration<float, std::milli> duration_kernel = (end_time_kernel - start_time_kernel);
    kernel_times.push_back(duration_kernel.count());
  }

  gpuInterface.conv_forward_gpu_epilog(y, y_d, x_d, k_d, B, M, C, height_in, width_in, K);

  gpuUtils.insert_post_barrier_kernel();
  return compute_median(kernel_times);
}

float Conv_Custom::compute_median(std::vector<float> values) {
  std::sort(values.begin(), values.end());
  const size_t mid = values.size() / 2;
  if (values.size() % 2 == 0) {
    return (values[mid - 1] + values[mid]) / 2.0f;
  }
  return values[mid];
}

void Conv_Custom::forward(const Matrix& bottom) {
  int n_sample = bottom.cols();
  top.resize(height_out * width_out * channel_out, n_sample);
  float *x = (float*)bottom.data();
  float *y = (float*)top.data();
  float *k = (float*)weight.data();
  float *b = (float*)bias.data();

  const int B = n_sample;
  const int M = channel_out;
  const int C = channel_in;
  const int K = height_kernel; // Assuming width_kernel is also K

  float *x_d;
  float *y_d;
  float *k_d;

  static int conv_layer_id = 0;
  conv_layer_id++;
  std::cout<<"Conv-GPU (Conv"<<conv_layer_id<<": "<<C<<" -> "<<M<<" channels, "<<K<<"x"<<K<<")=="<<std::endl;

  if (benchmark_enabled_) {
    std::cout<<"Competition Mode: warmup="<<benchmark_warmup_rounds_
             <<", measured="<<benchmark_measure_rounds_<<std::endl;
    float median_kernel_ms = run_competition_forward(y, x, k, B, M, C, K);
    std::cout<<"Op Time: " << median_kernel_ms << " ms"<<std::endl;
    std::cout<<"Throughput: " << n_sample / (median_kernel_ms / 1000.0)
             << " images/sec"<<std::endl;
    return;
  }

  // Launch marker kernel to aid with student function timing
  gpuUtils.insert_pre_barrier_kernel();
  
  // Start layer timer
  auto start_time_layer = std::chrono::high_resolution_clock::now();
  // Data transfer CPU to GPU
  gpuInterface.conv_forward_gpu_prolog(y, x, k, &y_d, &x_d, &k_d, B, M, C, height_in, width_in, K);
  
  // Start kernel timer
  auto start_time_kernel = std::chrono::high_resolution_clock::now();
  // Hand off to GPU for computation
  gpuInterface.conv_forward_gpu(y_d, x_d, k_d, B, M, C, height_in, width_in, K);
  cudaDeviceSynchronize();
  // Stop kernel timer
  auto end_time_kernel = std::chrono::high_resolution_clock::now();
  
  // Data transfer GPU to CPU
  gpuInterface.conv_forward_gpu_epilog(y, y_d, x_d, k_d, B, M, C, height_in, width_in, K);

  // Stop layer timer
  auto end_time_layer = std::chrono::high_resolution_clock::now();

  // Launch barrier kernel to aid with timing with nsight-compute
  gpuUtils.insert_post_barrier_kernel();

  std::chrono::duration<float, std::milli> duration_layer = (end_time_layer-start_time_layer);
  std::cout<<"Layer Time: " << duration_layer.count() << " ms"<<std::endl;
  
  std::chrono::duration<float, std::milli> duration_kernel = (end_time_kernel-start_time_kernel);
  std::cout<<"Op Time: " << duration_kernel.count() << " ms"<<std::endl;
  std::cout<<"Throughput: " << n_sample / (duration_kernel.count() / 1000.0) << " images/sec"<<std::endl;
}

void Conv_Custom::backward(const Matrix& bottom, const Matrix& grad_top) {

}

void Conv_Custom::update(Optimizer& opt) {

}

std::vector<float> Conv_Custom::get_parameters() const {
  std::vector<float> res(weight.size() + bias.size());
  // Copy the data of weights and bias to a long vector
  std::copy(weight.data(), weight.data() + weight.size(), res.begin());
  std::copy(bias.data(), bias.data() + bias.size(), res.begin() + weight.size());
  return res;
}

void Conv_Custom::set_parameters(const std::vector<float>& param) {
  if(static_cast<int>(param.size()) != weight.size() + bias.size())
      throw std::invalid_argument("Parameter size does not match");
  std::copy(param.begin(), param.begin() + weight.size(), weight.data());
  std::copy(param.begin() + weight.size(), param.end(), bias.data());
}

std::vector<float> Conv_Custom::get_derivatives() const {
  std::vector<float> res(grad_weight.size() + grad_bias.size());
  // Copy the data of weights and bias to a long vector
  std::copy(grad_weight.data(), grad_weight.data() + grad_weight.size(), res.begin());
  std::copy(grad_bias.data(), grad_bias.data() + grad_bias.size(),
            res.begin() + grad_weight.size());
  return res;
}
