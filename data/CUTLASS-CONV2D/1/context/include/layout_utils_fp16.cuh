#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "cutlass/half.h"

/// NCHW (= WHCN col-major) → NHWC   (__half)
__global__ void nchw_to_nhwc_fp16_kernel(
    const __half* __restrict__ src,
    cutlass::half_t* __restrict__ dst,
    int N, int C, int H, int W)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N*C*H*W) return;
    int w = idx % W;
    int h = (idx / W) % H;
    int c = (idx / (W*H)) % C;
    int n = idx / (W*H*C);
    dst[n*(H*W*C) + h*(W*C) + w*C + c] = cutlass::half_t(src[idx]);
}

/// NHWC → NCHW (= WHCN col-major)   (__half)
__global__ void nhwc_to_nchw_fp16_kernel(
    const cutlass::half_t* __restrict__ src,
    __half* __restrict__ dst,
    int N, int C, int H, int W)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N*C*H*W) return;
    int c = idx % C;
    int w = (idx / C) % W;
    int h = (idx / (C*W)) % H;
    int n = idx / (C*W*H);
    dst[n*(C*H*W) + c*(H*W) + h*W + w] = ((__half)src[n*(H*W*C) + h*(W*C) + w*C + c]);
}

inline void nchw_to_nhwc_fp16(const __half* src, cutlass::half_t* dst,
                               int N, int C, int H, int W,
                               cudaStream_t stream = nullptr)
{
    int total = N*C*H*W;
    nchw_to_nhwc_fp16_kernel<<<(total+255)/256, 256, 0, stream>>>(src, dst, N, C, H, W);
}

inline void nhwc_to_nchw_fp16(const cutlass::half_t* src, __half* dst,
                               int N, int C, int H, int W,
                               cudaStream_t stream = nullptr)
{
    int total = N*C*H*W;
    nhwc_to_nchw_fp16_kernel<<<(total+255)/256, 256, 0, stream>>>(src, dst, N, C, H, W);
}
