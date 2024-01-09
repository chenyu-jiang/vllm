/*
    MIT License

    Copyright (c) Microsoft Corporation.
    Copyright (c) Amazon.com, Inc. or its affiliates. All Rights Reserved.

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE
*/

/*!
 * \file src/op/dispatch/cuda/kernels/tutel_moe_dispatch.cu
 * \brief tutel moe fast dispatcher cuda kernel, modified from 
          https://github.com/microsoft/tutel/blob/v0.1.x/tutel/jit_kernels/sparse.py
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/Atomic.cuh>


#include "cuda_compat.h"
#include "dispatch_utils.h"

#define CUDA_CALL(func)                                           \
  do {                                                            \
    cudaError_t e = (func);                                       \
    CHECK(e == cudaSuccess) << "CUDA: " << cudaGetErrorString(e); \
  } while (false)

namespace vllm {

template <typename fp_t, typename int_t>
__global__ __launch_bounds__(1024) void moe_dispatch_kernel(
    fp_t* __restrict__ dispatched_input,
    const int_t* __restrict__ expert_indices,
    const int_t* __restrict__ locations,
    const fp_t* __restrict__ input_hidden_states,
    const int k,
    const int tokens,
    const int hidden,
    const int capacity) {
    // dispatched_input: [experts, capacity, hidden]
    for (int_t i = blockIdx.x; i < tokens * k; i += gridDim.x) {
        int_t token_id = i / k;
        _Pragma("unroll")
        for (int j = threadIdx.x; j < hidden; j += 1024) {
            gpuAtomicAdd(&dispatched_input[
                        (expert_indices[i] * capacity + locations[i]) * (hidden) + j
                      ],
                      input_hidden_states[token_id * (hidden) + j]);
        }
    }
}

template <typename fp_t, typename int_t>
__global__ __launch_bounds__(1024) void moe_gather_kernel(
    fp_t* __restrict__ decoded_output,
    const fp_t* __restrict__ expert_output,
    const fp_t* __restrict__ routing_weights,
    const int_t* __restrict__ expert_indices,
    const int_t* __restrict__ locations,
    const int k,
    const int tokens,
    const int hidden,
    const int capacity) {
    // expert_output: [experts, capacity, hidden]
    for (int i = blockIdx.x; i < tokens * k; i += gridDim.x) {
        int_t token_id = i / k;
        _Pragma("unroll")
        for (int j = threadIdx.x; j < hidden; j += 1024) {
            gpuAtomicAdd(
                &decoded_output[token_id * hidden + j],
                routing_weights[i] * expert_output[(expert_indices[i] * capacity + locations[i]) * (hidden) + j]
            );
        }
    }
}

#define thread_num  1024
template <typename int_t>
__global__ void moe_gen_location_kernel(
        int_t* output_locations,
        const int_t* expert_indices,
        int k,
        int tokens) {
    // [thread_extent] blockIdx.x = dim_E
    // [thread_extent] threadIdx.x = 1024
    __shared__ int temp[thread_num + 1];
    int thid = threadIdx.x, bid = blockIdx.x;
    int last_sum = -1;
    // each block checks for one expert
    for (int S = 0; S < tokens * k; S += thread_num) {
        int offset = 1;
        if (S + thid < tokens * k) {
            temp[thid] = expert_indices[S + thid] == bid ? 1 : 0;
        } else {
            temp[thid] = 0;
        }
        for (int d = thread_num >> 1; d > 0; d >>= 1) {
                __syncthreads();
                if (thid < d)
                        temp[offset * (2 * thid + 2) - 1] += temp[offset * (2 * thid + 1) - 1];
                offset *= 2;
        }
        if (thid == 0)
                temp[thread_num] = temp[thread_num - 1], temp[thread_num - 1] = 0;
        for (int d = 1; d < thread_num; d *= 2) {
                offset >>= 1;
                __syncthreads();
                if (thid < d) {
                        int ai = offset * (2 * thid + 1) - 1;
                        int bi = offset * (2 * thid + 2) - 1;
                        int t = temp[ai];
                        temp[ai] = temp[bi];
                        temp[bi] += t;
                }
        }
        __syncthreads();
        if (S + thid < tokens * k && expert_indices[S+thid] == bid) {
            output_locations[S+thid] = temp[thid + 1] + last_sum;
        }
        __syncthreads();
        last_sum += temp[thread_num];
    }
}

}  // namespace vllm

void moe_dispatch(torch::Tensor& dispatched_input,
                    torch::Tensor& expert_indices,
                    torch::Tensor& locations,
                    torch::Tensor& input_hidden_states,
                    const int k,
                    const int tokens,
                    const int hidden,
                    const int num_experts) {
    dim3 grid(512);
    dim3 block(1024);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input_hidden_states));
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    int capacity = k * tokens;
    VLLM_DISPATCH_FLOATING_TYPES(
        input_hidden_states.scalar_type(),
        "moe_dispatch_kernel",
        [&]{
            // to be safe, zero out dispatched_input
            // dispatched_input is of shape (experts, capacity, hidden)
            CUDA_CALL(cudaMemsetAsync(dispatched_input.data_ptr<scalar_t>(),
                                        0,
                                        num_experts * capacity * hidden * sizeof(scalar_t),
                                        stream));
            switch (expert_indices.scalar_type()) {
                case torch::ScalarType::Int:
                    vllm::moe_dispatch_kernel<scalar_t, int><<<grid, block, 0, stream>>>(
                        dispatched_input.data_ptr<scalar_t>(),
                        expert_indices.data_ptr<int>(),
                        locations.data_ptr<int>(),
                        input_hidden_states.data_ptr<scalar_t>(),
                        k,
                        tokens,
                        hidden,
                        capacity);
                    break;
                case torch::ScalarType::Long:
                    vllm::moe_dispatch_kernel<scalar_t, int64_t><<<grid, block, 0, stream>>>(
                        dispatched_input.data_ptr<scalar_t>(),
                        expert_indices.data_ptr<int64_t>(),
                        locations.data_ptr<int64_t>(),
                        input_hidden_states.data_ptr<scalar_t>(),
                        k,
                        tokens,
                        hidden,
                        capacity);
                    break;
                default:
                    TORCH_CHECK(false, "expert_indices must be of type Int or Long");
            }
        }
    );
}

void moe_gather(torch::Tensor& decoded_output,
                torch::Tensor& expert_output,
                torch::Tensor& routing_weights,
                torch::Tensor& expert_indices,
                torch::Tensor& locations,
                const int k,
                const int tokens,
                const int hidden) {
    dim3 grid(512);
    dim3 block(1024);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(expert_output));
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    int capacity = k * tokens;
    VLLM_DISPATCH_FLOATING_TYPES(
        expert_output.scalar_type(),
        "moe_gather_kernel",
        [&] {
            switch (expert_indices.scalar_type()) {
                case torch::ScalarType::Int:
                    vllm::moe_gather_kernel<scalar_t, int><<<grid, block, 0, stream>>>(
                        decoded_output.data_ptr<scalar_t>(),
                        expert_output.data_ptr<scalar_t>(),
                        routing_weights.data_ptr<scalar_t>(),
                        expert_indices.data_ptr<int>(),
                        locations.data_ptr<int>(),
                        k,
                        tokens,
                        hidden,
                        capacity);
                    break;
                case torch::ScalarType::Long:
                    vllm::moe_gather_kernel<scalar_t, int64_t><<<grid, block, 0, stream>>>(
                        decoded_output.data_ptr<scalar_t>(),
                        expert_output.data_ptr<scalar_t>(),
                        routing_weights.data_ptr<scalar_t>(),
                        expert_indices.data_ptr<int64_t>(),
                        locations.data_ptr<int64_t>(),
                        k,
                        tokens,
                        hidden,
                        capacity);
                    break;
                default:
                    TORCH_CHECK(false, "expert_indices must be of type Int or Long");
            }
        }
    );
}


void moe_gen_location(torch::Tensor& output_locations,
                      torch::Tensor& expert_indices,
                      const int k,
                      const int tokens,
                      const int num_experts) {
    dim3 grid(num_experts);
    dim3 block(1024);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(expert_indices));
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    VLLM_DISPATCH_INTEGRAL_TYPES(
        expert_indices.scalar_type(),
        "moe_gen_location_kernel",
        [&] {
            vllm::moe_gen_location_kernel<scalar_t><<<grid, block, 0, stream>>>(
                output_locations.data_ptr<scalar_t>(),
                expert_indices.data_ptr<scalar_t>(),
                k,
                tokens
            );
        }
    );
}

