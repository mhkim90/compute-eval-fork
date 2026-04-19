#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cudnn.h>
#include "include/conv2d.h"

#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t e = (call);                                                 \
        if (e != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(e));                 \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

#define CHECK_CUDNN(call)                                                       \
    do {                                                                        \
        cudnnStatus_t s = (call);                                               \
        if (s != CUDNN_STATUS_SUCCESS) {                                        \
            fprintf(stderr, "cuDNN error at %s:%d: %s\n",                      \
                    __FILE__, __LINE__, cudnnGetErrorString(s));                \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

/// cuDNN reference convolution.
/// Uses CUDNN_TENSOR_NCHW which is identical in memory to WHCN col-major.
/// No layout conversion required — data is passed directly.
static void conv2d_cudnn_nchw_reference(
    const float* d_input,    // WHCN col-major == NCHW row-major [N,C,H,W]
    const float* d_filter,   // WHCN col-major == NCHW row-major [K,C,R,S]
    float*       d_output,   // WHCN col-major == NCHW row-major [N,K,P,Q]
    int N, int H, int W, int C, int K, int R, int S,
    int pad_h, int pad_w, int stride_h, int stride_w)
{
    int P = (H + 2*pad_h - R) / stride_h + 1;
    int Q = (W + 2*pad_w - S) / stride_w + 1;

    cudnnHandle_t handle;
    CHECK_CUDNN(cudnnCreate(&handle));

    cudnnTensorDescriptor_t xDesc, yDesc;
    cudnnFilterDescriptor_t wDesc;
    cudnnConvolutionDescriptor_t convDesc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&xDesc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&yDesc));
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&wDesc));
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&convDesc));

    // NCHW == WHCN col-major: no conversion needed
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, K, P, Q));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(wDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, K, C, R, S));
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(convDesc,
        pad_h, pad_w, stride_h, stride_w,
        /*dilation_h=*/1, /*dilation_w=*/1,
        CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;

    size_t wsSize = 0;
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(
        handle, xDesc, wDesc, convDesc, yDesc, algo, &wsSize));
    void* workspace = nullptr;
    if (wsSize > 0) CHECK_CUDA(cudaMalloc(&workspace, wsSize));

    float alpha = 1.f, beta = 0.f;
    CHECK_CUDNN(cudnnConvolutionForward(handle,
        &alpha, xDesc, d_input, wDesc, d_filter,
        convDesc, algo, workspace, wsSize,
        &beta, yDesc, d_output));

    CHECK_CUDA(cudaDeviceSynchronize());

    if (workspace) cudaFree(workspace);
    cudnnDestroyTensorDescriptor(xDesc);
    cudnnDestroyTensorDescriptor(yDesc);
    cudnnDestroyFilterDescriptor(wDesc);
    cudnnDestroyConvolutionDescriptor(convDesc);
    cudnnDestroy(handle);
}

static void run_test(
    int N, int H, int W, int C, int K, int R, int S,
    int pad_h, int pad_w, int stride_h, int stride_w)
{
    int P = (H + 2*pad_h - R) / stride_h + 1;
    int Q = (W + 2*pad_w - S) / stride_w + 1;

    size_t in_sz  = (size_t)N * C * H * W;
    size_t flt_sz = (size_t)K * C * R * S;
    size_t out_sz = (size_t)N * K * P * Q;

    // Initialise host data in WHCN col-major (== NCHW row-major) order
    std::vector<float> h_in(in_sz), h_flt(flt_sz);
    for (size_t i = 0; i < in_sz;  i++) h_in[i]  = (float)(i % 7) * 0.1f - 0.3f;
    for (size_t i = 0; i < flt_sz; i++) h_flt[i] = (float)(i % 5) * 0.05f - 0.1f;

    float *d_in, *d_flt, *d_cutlass, *d_cudnn;
    CHECK_CUDA(cudaMalloc(&d_in,      in_sz  * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_flt,     flt_sz * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_cutlass, out_sz * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_cudnn,   out_sz * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_in,  h_in.data(),  in_sz  * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_flt, h_flt.data(), flt_sz * sizeof(float), cudaMemcpyHostToDevice));

    // CUTLASS solution (accepts WHCN col-major, handles NCHW->NHWC conversion internally)
    conv2d_cutlass_whcn(d_in, d_flt, d_cutlass,
                        N, H, W, C, K, R, S,
                        pad_h, pad_w, stride_h, stride_w);

    // cuDNN reference (NCHW == WHCN col-major, no conversion needed)
    conv2d_cudnn_nchw_reference(d_in, d_flt, d_cudnn,
                                N, H, W, C, K, R, S,
                                pad_h, pad_w, stride_h, stride_w);

    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> h_cut(out_sz), h_dnn(out_sz);
    CHECK_CUDA(cudaMemcpy(h_cut.data(), d_cutlass, out_sz * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_dnn.data(), d_cudnn,   out_sz * sizeof(float), cudaMemcpyDeviceToHost));

    float max_err = 0.f;
    size_t max_idx = 0;
    for (size_t i = 0; i < out_sz; i++) {
        float err = fabsf(h_cut[i] - h_dnn[i]);
        if (err > max_err) { max_err = err; max_idx = i; }
    }

    printf("  N=%d H=%d W=%d C=%d K=%d R=%d S=%d pad=(%d,%d) stride=(%d,%d) "
           "-> max_err=%.6f (idx=%zu)\n",
           N, H, W, C, K, R, S, pad_h, pad_w, stride_h, stride_w,
           max_err, max_idx);

    const float kTol = 1e-3f;
    if (max_err > kTol) {
        fprintf(stderr, "FAIL: max error %.6f exceeds tolerance %.6f\n", max_err, kTol);
        cudaFree(d_in); cudaFree(d_flt); cudaFree(d_cutlass); cudaFree(d_cudnn);
        exit(1);
    }

    cudaFree(d_in); cudaFree(d_flt); cudaFree(d_cutlass); cudaFree(d_cudnn);
}

int main() {
    printf("Running CUTLASS conv2d vs cuDNN (WHCN col-major / NCHW) tests...\n");

    // Test 1: basic 3x3 conv, same padding
    run_test(/*N*/2, /*H*/8,  /*W*/8,  /*C*/16, /*K*/32, /*R*/3, /*S*/3, 1, 1, 1, 1);
    // Test 2: 1x1 conv
    run_test(/*N*/1, /*H*/16, /*W*/16, /*C*/32, /*K*/64, /*R*/1, /*S*/1, 0, 0, 1, 1);
    // Test 3: stride 2
    run_test(/*N*/2, /*H*/16, /*W*/16, /*C*/8,  /*K*/16, /*R*/3, /*S*/3, 1, 1, 2, 2);
    // Test 4: asymmetric spatial dims
    run_test(/*N*/1, /*H*/12, /*W*/20, /*C*/4,  /*K*/8,  /*R*/3, /*S*/5, 1, 2, 1, 1);

    printf("PASS\n");
    return 0;
}
