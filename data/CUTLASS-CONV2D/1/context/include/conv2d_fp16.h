#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

/// Half-precision (FP16) forward 2D convolution using CUTLASS.
/// ALL tensors use WHCN column-major layout (== NCHW row-major in memory).
///
/// Memory layout (col-major indexing):
///   input:   [W, H, C_in, N]  →  offset = w + W*h + W*H*c + W*H*C_in*n
///   filter:  [S, R, C_in, K]  →  offset = s + S*r + S*R*c  + S*R*C_in*k
///   output:  [Q, P, K,    N]  →  offset = q + Q*p + Q*P*k  + Q*P*K*n
///     P = (H + 2*pad_h - R) / stride_h + 1
///     Q = (W + 2*pad_w - S) / stride_w + 1
///
/// FP16 I/O, float accumulation.
/// Implementation must convert NCHW→NHWC before calling CUTLASS and
/// NHWC→NCHW on the output before returning.
///
/// @param input    Device pointer (__half), WHCN col-major (= NCHW row-major)
/// @param filter   Device pointer (__half), WHCN col-major (= NCHW row-major)
/// @param output   Device pointer (__half), WHCN col-major (= NCHW row-major)
/// @param N        Batch size          [1, 2048]
/// @param H        Input height        [3, 32768]
/// @param W        Input width         [3, 32768]
/// @param C_in     Input channels      [1, 32]
/// @param K        Output channels     [1, 32]
/// @param R        Filter height       [3, 255], always odd
/// @param S        Filter width        [3, 255], always odd
/// @param pad_h    Zero-padding height
/// @param pad_w    Zero-padding width
/// @param stride_h Convolution stride height
/// @param stride_w Convolution stride width
/// @param stream   CUDA stream (may be nullptr)
void conv2d_cutlass_fp16_whcn(
    const __half* input,
    const __half* filter,
    __half*       output,
    int N, int H, int W, int C_in,
    int K, int R, int S,
    int pad_h,    int pad_w,
    int stride_h, int stride_w,
    cudaStream_t stream = nullptr);
