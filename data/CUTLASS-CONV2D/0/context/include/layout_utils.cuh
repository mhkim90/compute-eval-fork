#pragma once
#include <cuda_runtime.h>

/// NCHW → NHWC  (also used for filter: treat K as N, C_in as C, R as H, S as W)
__global__ void nchw_to_nhwc_kernel(
    const float* __restrict__ src,   // [N, C, H, W]
    float*       __restrict__ dst,   // [N, H, W, C]
    int N, int C, int H, int W)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C * H * W) return;
    int w = idx % W;
    int h = (idx / W) % H;
    int c = (idx / (W * H)) % C;
    int n = idx / (W * H * C);
    dst[n*(H*W*C) + h*(W*C) + w*C + c] = src[idx];
}

/// NHWC → NCHW
__global__ void nhwc_to_nchw_kernel(
    const float* __restrict__ src,   // [N, H, W, C]
    float*       __restrict__ dst,   // [N, C, H, W]
    int N, int C, int H, int W)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C * H * W) return;
    int c = idx % C;
    int w = (idx / C) % W;
    int h = (idx / (C * W)) % H;
    int n = idx / (C * W * H);
    dst[n*(C*H*W) + c*(H*W) + h*W + w] = src[idx];
}

inline void nchw_to_nhwc(const float* src, float* dst,
                          int N, int C, int H, int W, cudaStream_t stream)
{
    int total = N * C * H * W;
    nchw_to_nhwc_kernel<<<(total + 255) / 256, 256, 0, stream>>>(src, dst, N, C, H, W);
}

inline void nhwc_to_nchw(const float* src, float* dst,
                          int N, int C, int H, int W, cudaStream_t stream)
{
    int total = N * C * H * W;
    nhwc_to_nchw_kernel<<<(total + 255) / 256, 256, 0, stream>>>(src, dst, N, C, H, W);
}
