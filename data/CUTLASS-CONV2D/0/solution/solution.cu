#include "include/conv2d.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(_e));                \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

// CUTLASS headers
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm.h"

// ─────────────────────────────────────────────────────────────────────────────
// Layout conversion kernels: NCHW (== WHCN col-major) <-> NHWC
// ─────────────────────────────────────────────────────────────────────────────

/// NCHW → NHWC
__global__ void nchw_to_nhwc_kernel(
    const float* __restrict__ src,   // [N, C, H, W]
    float*       __restrict__ dst,   // [N, H, W, C]
    int N, int C, int H, int W)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    if (idx >= total) return;

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
    int total = N * C * H * W;
    if (idx >= total) return;

    int c = idx % C;
    int w = (idx / C) % W;
    int h = (idx / (C * W)) % H;
    int n = idx / (C * W * H);

    dst[n*(C*H*W) + c*(H*W) + h*W + w] = src[idx];
}

static void nchw_to_nhwc(const float* src, float* dst,
                          int N, int C, int H, int W, cudaStream_t stream)
{
    int total = N * C * H * W;
    nchw_to_nhwc_kernel<<<(total + 255) / 256, 256, 0, stream>>>(src, dst, N, C, H, W);
}

static void nhwc_to_nchw(const float* src, float* dst,
                          int N, int C, int H, int W, cudaStream_t stream)
{
    int total = N * C * H * W;
    nhwc_to_nchw_kernel<<<(total + 255) / 256, 256, 0, stream>>>(src, dst, N, C, H, W);
}

// ─────────────────────────────────────────────────────────────────────────────
// CUTLASS Conv2d type definitions (sm80, fp32, NHWC)
// ─────────────────────────────────────────────────────────────────────────────

using ElementInput    = float;
using ElementFilter   = float;
using ElementOutput   = float;
using ElementAccum    = float;

using LayoutInput   = cutlass::layout::TensorNHWC;
using LayoutFilter  = cutlass::layout::TensorNHWC;
using LayoutOutput  = cutlass::layout::TensorNHWC;

using MMAOp = cutlass::arch::OpClassSimt;
using SmArch = cutlass::arch::Sm80;

using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 8>;
using WarpShape        = cutlass::gemm::GemmShape<32, 64, 8>;
using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput, 1, ElementAccum, ElementAccum>;

using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
    ElementInput,  LayoutInput,
    ElementFilter, LayoutFilter,
    ElementOutput, LayoutOutput,
    ElementAccum,
    MMAOp,
    SmArch,
    ThreadblockShape, WarpShape, InstructionShape,
    EpilogueOp,
    cutlass::conv::threadblock::GemmIdentityThreadblockSwizzle<>,
    2,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kOptimized
>::Kernel;

using ImplicitGemm = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

void conv2d_cutlass_whcn(
    const float* input,
    const float* filter,
    float*       output,
    int N, int H, int W, int C_in,
    int K, int R, int S,
    int pad_h,    int pad_w,
    int stride_h, int stride_w,
    cudaStream_t stream)
{
    int P = (H + 2*pad_h - R) / stride_h + 1;
    int Q = (W + 2*pad_w - S) / stride_w + 1;

    // Allocate temporary NHWC buffers
    float *d_input_nhwc, *d_filter_nhwc, *d_output_nhwc;
    CHECK_CUDA(cudaMalloc(&d_input_nhwc,  (size_t)N * H * W * C_in * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_filter_nhwc, (size_t)K * R * S * C_in * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output_nhwc, (size_t)N * P * Q * K     * sizeof(float)));

    // Convert input and filter: NCHW (WHCN col-major) → NHWC
    nchw_to_nhwc(input,  d_input_nhwc,  N, C_in, H, W, stream);
    nchw_to_nhwc(filter, d_filter_nhwc, K, C_in, R, S, stream);

    // CUTLASS problem size
    cutlass::conv::Conv2dProblemSize problem_size(
        {N, H, W, C_in},                          // input  NHWC
        {K, R, S, C_in},                          // filter KRSC
        {pad_h, pad_w, pad_h, pad_w},             // padding (top, left, bottom, right)
        {stride_h, stride_w},                     // stride
        {1, 1},                                   // dilation
        {N, P, Q, K},                             // output NHWC
        cutlass::conv::Mode::kCrossCorrelation,
        /*split_k_slices=*/1);

    typename ImplicitGemm::Arguments args(
        problem_size,
        {d_input_nhwc,  {C_in, C_in * W, C_in * W * H}},
        {d_filter_nhwc, {C_in, C_in * S, C_in * S * R}},
        {d_output_nhwc, {K,    K    * Q, K    * Q * P}},
        {d_output_nhwc, {K,    K    * Q, K    * Q * P}},
        {/*alpha=*/1.f, /*beta=*/0.f});

    ImplicitGemm implicit_gemm;

    size_t workspace_size = implicit_gemm.get_workspace_size(args);
    void* workspace = nullptr;
    if (workspace_size > 0) CHECK_CUDA(cudaMalloc(&workspace, workspace_size));

    cutlass::Status status = implicit_gemm.initialize(args, workspace, stream);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "CUTLASS conv2d initialize failed: %d\n", (int)status);
        exit(1);
    }

    status = implicit_gemm(stream);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "CUTLASS conv2d run failed: %d\n", (int)status);
        exit(1);
    }

    // Convert output: NHWC → NCHW (WHCN col-major)
    nhwc_to_nchw(d_output_nhwc, output, N, K, P, Q, stream);

    if (stream) cudaStreamSynchronize(stream);

    if (workspace) CHECK_CUDA(cudaFree(workspace));
    CHECK_CUDA(cudaFree(d_input_nhwc));
    CHECK_CUDA(cudaFree(d_filter_nhwc));
    CHECK_CUDA(cudaFree(d_output_nhwc));
}
