#include "include/conv2d_fp16.h"
#include "include/layout_utils_fp16.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>

// CUTLASS headers
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/epilogue/thread/linear_combination.h"

// ─────────────────────────────────────────────────────────────────────────────
// CUTLASS FP16 type definitions (sm80, FP16 I/O, float accum, NHWC, Tensor Core)
// ─────────────────────────────────────────────────────────────────────────────

using ElementInput  = cutlass::half_t;
using ElementFilter = cutlass::half_t;
using ElementOutput = cutlass::half_t;
using ElementAccum  = float;
using LayoutNHWC    = cutlass::layout::TensorNHWC;

// Use Tensor Core (OpClassTensorOp) for FP16 on sm80
using MMAOp    = cutlass::arch::OpClassTensorOp;
using SmArch   = cutlass::arch::Sm80;

using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
using WarpShape        = cutlass::gemm::GemmShape<64,  64,  32>;
using InstructionShape = cutlass::gemm::GemmShape<16,  8,   16>;

using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementAccum, ElementAccum>;

using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
    ElementInput,  LayoutNHWC,
    ElementFilter, LayoutNHWC,
    ElementOutput, LayoutNHWC,
    ElementAccum,
    MMAOp, SmArch,
    ThreadblockShape, WarpShape, InstructionShape,
    EpilogueOp,
    cutlass::conv::threadblock::GemmIdentityThreadblockSwizzle<>,
    /*Stages=*/3,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kOptimized
>::Kernel;

using ImplicitGemm = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

void conv2d_cutlass_fp16_whcn(
    const __half* input,
    const __half* filter,
    __half*       output,
    int N, int H, int W, int C_in,
    int K, int R, int S,
    int pad_h,    int pad_w,
    int stride_h, int stride_w,
    cudaStream_t stream)
{
    int P = (H + 2*pad_h - R) / stride_h + 1;
    int Q = (W + 2*pad_w - S) / stride_w + 1;

    // Allocate temporary NHWC __half buffers
    cutlass::half_t *d_in_nhwc, *d_flt_nhwc, *d_out_nhwc;
    cudaMalloc(&d_in_nhwc,  (size_t)N*H*W*C_in * sizeof(cutlass::half_t));
    cudaMalloc(&d_flt_nhwc, (size_t)K*R*S*C_in * sizeof(cutlass::half_t));
    cudaMalloc(&d_out_nhwc, (size_t)N*P*Q*K     * sizeof(cutlass::half_t));

    // Convert NCHW (WHCN col-major) → NHWC
    nchw_to_nhwc_fp16(input,  d_in_nhwc,  N, C_in, H, W, stream);
    nchw_to_nhwc_fp16(filter, d_flt_nhwc, K, C_in, R, S, stream);

    // CUTLASS conv2d
    cutlass::conv::Conv2dProblemSize problem_size(
        {N, H, W, C_in},
        {K, R, S, C_in},
        {pad_h, pad_w, pad_h, pad_w},
        {stride_h, stride_w},
        {1, 1},
        {N, P, Q, K},
        cutlass::conv::Mode::kCrossCorrelation,
        /*split_k_slices=*/1);

    typename ImplicitGemm::Arguments args(
        problem_size,
        {d_in_nhwc,  {C_in, C_in*W,   C_in*W*H  }},
        {d_flt_nhwc, {C_in, C_in*S,   C_in*S*R  }},
        {d_out_nhwc, {K,    K*Q,       K*Q*P     }},
        {d_out_nhwc, {K,    K*Q,       K*Q*P     }},
        {/*alpha=*/ElementAccum(1), /*beta=*/ElementAccum(0)});

    ImplicitGemm implicit_gemm;
    size_t ws_size = implicit_gemm.get_workspace_size(args);
    void* ws = nullptr;
    if (ws_size > 0) cudaMalloc(&ws, ws_size);

    cutlass::Status status = implicit_gemm.initialize(args, ws, stream);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "CUTLASS FP16 conv2d initialize failed: %d\n", (int)status);
        exit(1);
    }
    status = implicit_gemm(stream);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "CUTLASS FP16 conv2d run failed: %d\n", (int)status);
        exit(1);
    }

    // Ensure deterministic completion before output layout conversion
    if (stream) {
        cudaStreamSynchronize(stream);
    } else {
        cudaDeviceSynchronize();
    }

    // Convert NHWC → NCHW (WHCN col-major)
    nhwc_to_nchw_fp16(d_out_nhwc, output, N, K, P, Q, stream);

    if (ws) cudaFree(ws);
    cudaFree(d_in_nhwc);
    cudaFree(d_flt_nhwc);
    cudaFree(d_out_nhwc);
}
