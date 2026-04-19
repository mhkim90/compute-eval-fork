#include <cstdio>
#include <cstdlib>
#include <cfloat>
#include <cmath>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cudnn.h>
#include "include/conv2d_fp16.h"

#define CHECK_CUDA(call) do { \
    cudaError_t e=(call); \
    if(e!=cudaSuccess){fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} \
} while(0)

#define CHECK_CUDNN(call) do { \
    cudnnStatus_t s=(call); \
    if(s!=CUDNN_STATUS_SUCCESS){fprintf(stderr,"cuDNN error %s:%d: %s\n",__FILE__,__LINE__,cudnnGetErrorString(s));exit(1);} \
} while(0)

// ─────────────────────────────────────────────────────────────────────────────
// cuDNN FP16 reference — NCHW == WHCN col-major, no conversion needed
// ─────────────────────────────────────────────────────────────────────────────
static void cudnn_fp16_reference(
    const __half* d_in, const __half* d_flt, __half* d_out,
    int N, int H, int W, int C, int K, int R, int S,
    int pad_h, int pad_w, int stride_h, int stride_w)
{
    int P = (H + 2*pad_h - R)/stride_h + 1;
    int Q = (W + 2*pad_w - S)/stride_w + 1;

    cudnnHandle_t handle; CHECK_CUDNN(cudnnCreate(&handle));

    cudnnTensorDescriptor_t xD, yD;
    cudnnFilterDescriptor_t wD;
    cudnnConvolutionDescriptor_t cD;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&xD));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&yD));
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&wD));
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&cD));

    // NCHW == our native WHCN col-major
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(xD, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, N, C, H, W));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(yD, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, N, K, P, Q));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(wD, CUDNN_DATA_HALF, CUDNN_TENSOR_NCHW, K, C, R, S));
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(cD,
        pad_h, pad_w, stride_h, stride_w, 1, 1,
        CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT)); // float math for FP16
    CHECK_CUDNN(cudnnSetConvolutionMathType(cD, CUDNN_TENSOR_OP_MATH));

    // Benchmark all supported algorithms and pick the fastest
    const int kMaxAlgos = 8;
    int retN = 0;
    cudnnConvolutionFwdAlgoPerf_t perfs[kMaxAlgos];
    CHECK_CUDNN(cudnnFindConvolutionForwardAlgorithm(
        handle, xD, wD, cD, yD, kMaxAlgos, &retN, perfs));

    if (retN == 0) {
        fprintf(stderr, "cuDNN error: no convolution forward algorithms returned\n");
        exit(1);
    }

    cudnnConvolutionFwdAlgo_t algo;
    size_t wsSize = 0;
    float bestTime = FLT_MAX;
    bool foundAlgo = false;
    for (int i = 0; i < retN; i++) {
        if (perfs[i].status == CUDNN_STATUS_SUCCESS && perfs[i].time < bestTime) {
            algo      = perfs[i].algo;
            wsSize    = perfs[i].memory;
            bestTime  = perfs[i].time;
            foundAlgo = true;
        }
    }
    if (!foundAlgo) {
        fprintf(stderr, "cuDNN error: no successful convolution forward algorithm found\n");
        exit(1);
    }

    void* ws = nullptr;
    if (wsSize > 0) CHECK_CUDA(cudaMalloc(&ws, wsSize));

    float alpha = 1.f, beta = 0.f;
    CHECK_CUDNN(cudnnConvolutionForward(handle,
        &alpha, xD, d_in, wD, d_flt, cD, algo, ws, wsSize,
        &beta, yD, d_out));
    CHECK_CUDA(cudaDeviceSynchronize());

    if (ws) cudaFree(ws);
    cudnnDestroyTensorDescriptor(xD);
    cudnnDestroyTensorDescriptor(yD);
    cudnnDestroyFilterDescriptor(wD);
    cudnnDestroyConvolutionDescriptor(cD);
    cudnnDestroy(handle);
}

// ─────────────────────────────────────────────────────────────────────────────
// Run one test config — returns true on PASS, false on FAIL
// ─────────────────────────────────────────────────────────────────────────────
static bool run_test(
    int N, int H, int W, int C, int K, int R, int S,
    int pad_h, int pad_w, int stride_h, int stride_w,
    float rel_tol = 1e-3f)  // 0.1% relative tolerance
{
    int P = (H + 2*pad_h - R)/stride_h + 1;
    int Q = (W + 2*pad_w - S)/stride_w + 1;

    if (P <= 0 || Q <= 0) {
        printf("  SKIP N=%d H=%d W=%d C=%d K=%d R=%d S=%d (P=%d Q=%d invalid)\n",
               N,H,W,C,K,R,S,P,Q);
        return true; // skip invalid configs
    }

    size_t in_sz  = (size_t)N*C*H*W;
    size_t flt_sz = (size_t)K*C*R*S;
    size_t out_sz = (size_t)N*K*P*Q;

    // Check memory — skip if would exceed 4 GB
    size_t total_bytes = (in_sz + flt_sz + 2*out_sz) * sizeof(__half);
    if (total_bytes > (size_t)4*1024*1024*1024ULL) {
        printf("  SKIP N=%d H=%d W=%d C=%d K=%d R=%d S=%d (memory %.1f GB > 4GB limit)\n",
               N,H,W,C,K,R,S, total_bytes/(1024.0*1024.0*1024.0));
        return true;
    }

    // Host data — small values to avoid FP16 overflow
    std::vector<__half> h_in(in_sz), h_flt(flt_sz);
    for (size_t i = 0; i < in_sz;  i++) h_in[i]  = __float2half((float)(i % 7) * 0.02f - 0.06f);
    for (size_t i = 0; i < flt_sz; i++) h_flt[i] = __float2half((float)(i % 5) * 0.01f - 0.02f);

    __half *d_in, *d_flt, *d_cutlass, *d_cudnn;
    CHECK_CUDA(cudaMalloc(&d_in,      in_sz  * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_flt,     flt_sz * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_cutlass, out_sz * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_cudnn,   out_sz * sizeof(__half)));

    CHECK_CUDA(cudaMemcpy(d_in,  h_in.data(),  in_sz  * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_flt, h_flt.data(), flt_sz * sizeof(__half), cudaMemcpyHostToDevice));

    // CUTLASS FP16 solution
    conv2d_cutlass_fp16_whcn(d_in, d_flt, d_cutlass,
                             N, H, W, C, K, R, S,
                             pad_h, pad_w, stride_h, stride_w);

    // cuDNN FP16 reference (NCHW == WHCN col-major)
    cudnn_fp16_reference(d_in, d_flt, d_cudnn,
                         N, H, W, C, K, R, S,
                         pad_h, pad_w, stride_h, stride_w);

    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<__half> h_cut(out_sz), h_dnn(out_sz);
    CHECK_CUDA(cudaMemcpy(h_cut.data(), d_cutlass, out_sz * sizeof(__half), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_dnn.data(), d_cudnn,   out_sz * sizeof(__half), cudaMemcpyDeviceToHost));

    cudaFree(d_in); cudaFree(d_flt); cudaFree(d_cutlass); cudaFree(d_cudnn);

    // Relative error check: |cutlass - cudnn| / (|cudnn| + epsilon)
    float max_rel_err = 0.f;
    size_t max_idx = 0;
    float max_cut_val = 0.f, max_dnn_val = 0.f;
    for (size_t i = 0; i < out_sz; i++) {
        float cut = __half2float(h_cut[i]);
        float dnn = __half2float(h_dnn[i]);
        float rel = fabsf(cut - dnn) / (fabsf(dnn) + 1e-6f);
        if (rel > max_rel_err) {
            max_rel_err = rel;
            max_idx = i;
            max_cut_val = cut;
            max_dnn_val = dnn;
        }
    }

    bool passed = max_rel_err <= rel_tol;
    printf("  %s N=%d H=%d W=%d C=%d K=%d R=%d S=%d pad=(%d,%d) stride=(%d,%d)"
           " -> max_rel_err=%.4f%% (idx=%zu cut=%.4f dnn=%.4f)\n",
           passed ? "PASS" : "FAIL",
           N, H, W, C, K, R, S, pad_h, pad_w, stride_h, stride_w,
           max_rel_err * 100.f, max_idx, max_cut_val, max_dnn_val);

    return passed;
}

int main() {
    printf("=== CUTLASS FP16 Conv2d vs cuDNN (WHCN col-major / NCHW), 0.1%% relative tolerance ===\n\n");

    // ─────────────────────────────────────────────────────────────────────
    // Progressive test sweep — starts small, grows to large.
    // Each group tests one dimension growing while others stay moderate.
    // Filter sizes are always odd (3,5,7,...).
    // ─────────────────────────────────────────────────────────────────────

    struct TestConfig {
        int N, H, W, C, K, R, S, pad_h, pad_w, stride_h, stride_w;
        const char* label;
    };

    std::vector<TestConfig> configs = {
        // ── Tier 1: Tiny (smoke tests) ──────────────────────────────────
        {1,   3,   3,  1,  1,  3,  3, 1, 1, 1, 1, "tiny-1x1ch-3x3f"},
        {1,   4,   4,  1,  1,  3,  3, 1, 1, 1, 1, "tiny-1x1ch-3x3f-4sp"},
        {1,   8,   8,  4,  4,  3,  3, 1, 1, 1, 1, "small-4ch-3x3f"},
        {1,   8,   8,  4,  4,  5,  5, 2, 2, 1, 1, "small-4ch-5x5f"},
        {2,   8,   8,  8,  8,  3,  3, 1, 1, 1, 1, "small-8ch-3x3f"},

        // ── Tier 2: Small ───────────────────────────────────────────────
        {1,  16,  16,  8,  8,  3,  3, 1, 1, 1, 1, "16sp-8ch-3x3f"},
        {2,  16,  16, 16, 16,  3,  3, 1, 1, 1, 1, "16sp-16ch-3x3f"},
        {1,  16,  16,  4,  8,  5,  5, 2, 2, 1, 1, "16sp-4-8ch-5x5f"},
        {4,  16,  16,  8, 16,  7,  7, 3, 3, 1, 1, "16sp-8-16ch-7x7f"},
        {1,  32,  32, 16, 16,  3,  3, 1, 1, 1, 1, "32sp-16ch-3x3f"},
        {2,  32,  32, 16, 32,  5,  5, 2, 2, 1, 1, "32sp-16-32ch-5x5f"},
        {1,  32,  32,  8, 16,  7,  7, 3, 3, 1, 1, "32sp-8-16ch-7x7f"},

        // ── Tier 3: Medium spatial ──────────────────────────────────────
        {1,  64,  64,  8,  8,  3,  3, 1, 1, 1, 1, "64sp-8ch-3x3f"},
        {2,  64,  64, 16, 16,  5,  5, 2, 2, 1, 1, "64sp-16ch-5x5f"},
        {4,  64,  64, 32, 32,  3,  3, 1, 1, 1, 1, "64sp-32ch-3x3f"},
        {1, 128, 128, 16, 16,  3,  3, 1, 1, 1, 1, "128sp-16ch-3x3f"},
        {2, 128, 128, 32, 32,  5,  5, 2, 2, 1, 1, "128sp-32ch-5x5f"},
        {1, 256, 256,  8,  8,  3,  3, 1, 1, 1, 1, "256sp-8ch-3x3f"},
        {2, 256, 256, 16, 16,  3,  3, 1, 1, 1, 1, "256sp-16ch-3x3f"},
        {1, 256, 256, 32, 32,  7,  7, 3, 3, 1, 1, "256sp-32ch-7x7f"},

        // ── Tier 4: Large filter sizes ──────────────────────────────────
        {1,  64,  64,  8,  8, 11, 11, 5, 5, 1, 1, "64sp-8ch-11x11f"},
        {1, 128, 128,  4,  8, 15, 15, 7, 7, 1, 1, "128sp-4-8ch-15x15f"},
        {1, 256, 256,  4,  4, 31, 31,15,15, 1, 1, "256sp-4ch-31x31f"},
        {1, 128, 128,  8,  8, 63, 63,31,31, 1, 1, "128sp-8ch-63x63f"},
        {1, 512, 512,  4,  4,127,127,63,63, 1, 1, "512sp-4ch-127x127f"},
        {1, 512, 512,  2,  2,255,255,127,127, 1, 1, "512sp-2ch-255x255f"},

        // ── Tier 5: Large spatial ────────────────────────────────────────
        {1,  512,  512, 16, 16,  3,  3, 1, 1, 1, 1, "512sp-16ch-3x3f"},
        {2,  512,  512, 32, 32,  3,  3, 1, 1, 1, 1, "512sp-32ch-3x3f"},
        {1, 1024, 1024,  8,  8,  3,  3, 1, 1, 1, 1, "1024sp-8ch-3x3f"},
        {1, 1024, 1024, 16, 16,  5,  5, 2, 2, 1, 1, "1024sp-16ch-5x5f"},
        {1, 2048, 2048,  4,  4,  3,  3, 1, 1, 1, 1, "2048sp-4ch-3x3f"},
        {2, 2048, 2048,  8,  8,  3,  3, 1, 1, 1, 1, "2048sp-8ch-3x3f"},
        {1, 4096, 4096,  4,  4,  3,  3, 1, 1, 1, 1, "4096sp-4ch-3x3f"},
        {1, 8192, 8192,  2,  2,  3,  3, 1, 1, 1, 1, "8192sp-2ch-3x3f"},

        // ── Tier 6: Large batch ──────────────────────────────────────────
        {  16,  32,  32, 16, 16,  3,  3, 1, 1, 1, 1, "batch16-32sp-16ch"},
        {  64,  32,  32,  8,  8,  3,  3, 1, 1, 1, 1, "batch64-32sp-8ch"},
        { 256,  16,  16,  8,  8,  3,  3, 1, 1, 1, 1, "batch256-16sp-8ch"},
        { 512,  16,  16,  4,  4,  3,  3, 1, 1, 1, 1, "batch512-16sp-4ch"},
        {1024,   8,   8,  4,  4,  3,  3, 1, 1, 1, 1, "batch1024-8sp-4ch"},
        {2048,   4,   4,  2,  2,  3,  3, 1, 1, 1, 1, "batch2048-4sp-2ch"},

        // ── Tier 7: Stride > 1 ───────────────────────────────────────────
        {1,  64,  64, 16, 16,  3,  3, 1, 1, 2, 2, "64sp-16ch-3x3f-s2"},
        {2, 128, 128, 16, 32,  5,  5, 2, 2, 2, 2, "128sp-16-32ch-5x5f-s2"},
        {1, 256, 256,  8, 16,  7,  7, 3, 3, 2, 2, "256sp-8-16ch-7x7f-s2"},

        // ── Tier 8: Extreme spatial (boundary check) ─────────────────────
        {1, 16384, 16384, 1, 1,  3,  3, 1, 1, 1, 1, "16384sp-1ch-3x3f"},
        {1, 32768, 32768, 1, 1,  3,  3, 1, 1, 1, 1, "32768sp-1ch-3x3f"},
    };

    int total = 0, passed = 0, failed = 0;
    for (auto& c : configs) {
        total++;
        printf("[%d/%zu] %s\n", total, configs.size(), c.label);
        bool ok = run_test(c.N, c.H, c.W, c.C, c.K, c.R, c.S,
                           c.pad_h, c.pad_w, c.stride_h, c.stride_w,
                           /*rel_tol=*/1e-3f);
        if (ok) {
            passed++;
        } else {
            failed++;
            fprintf(stderr, "\nFAIL at config: %s (N=%d H=%d W=%d C=%d K=%d R=%d S=%d)\n",
                    c.label, c.N, c.H, c.W, c.C, c.K, c.R, c.S);
            // Print summary before exiting
            printf("\n=== Summary: %d passed, %d failed (stopped at first failure) ===\n",
                   passed, failed);
            return 1;
        }
    }

    printf("\n=== Summary: %d/%d configs passed ===\n", passed, total);
    printf("PASS\n");
    return 0;
}
