#pragma once
#include <cuda_runtime.h>

/// Forward 2D convolution using CUTLASS.
/// ALL tensors use WHCN column-major layout (== NCHW row-major in memory).
///
/// Memory layout (col-major indexing):
///   input:   [W, H, C_in, N]  →  offset = w + W*h + W*H*c + W*H*C_in*n
///   filter:  [S, R, C_in, K]  →  offset = s + S*r + S*R*c  + S*R*C_in*k
///   output:  [Q, P, K,    N]  →  offset = q + Q*p + Q*P*k  + Q*P*K*n
///     P = (H + 2*pad_h - R) / stride_h + 1
///     Q = (W + 2*pad_w - S) / stride_w + 1
///
/// The implementation must convert NCHW→NHWC before calling CUTLASS and
/// NHWC→NCHW on the output before returning.
///
/// @param input    Device pointer, WHCN col-major (= NCHW row-major)
/// @param filter   Device pointer, WHCN col-major (= NCHW row-major)
/// @param output   Device pointer, WHCN col-major (= NCHW row-major), written by this function
/// @param N        Batch size
/// @param H        Input height
/// @param W        Input width
/// @param C_in     Input channels
/// @param K        Output channels (number of filters)
/// @param R        Filter height
/// @param S        Filter width
/// @param pad_h    Zero-padding on height dimension
/// @param pad_w    Zero-padding on width dimension
/// @param stride_h Convolution stride on height dimension
/// @param stride_w Convolution stride on width dimension
/// @param stream   CUDA stream (may be nullptr for default stream)
void conv2d_cutlass_whcn(
    const float* input,
    const float* filter,
    float*       output,
    int N, int H, int W, int C_in,
    int K, int R, int S,
    int pad_h,    int pad_w,
    int stride_h, int stride_w,
    cudaStream_t stream = nullptr);
