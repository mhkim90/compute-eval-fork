# Project Conventions

## Toolchain
- **CUDA**: 12.x (`nvcc`, pre-installed)
- **Host compiler**: GCC 13 (`g++-13`, `-ccbin g++-13`)
- **C++ standard**: C++20 (`-std=c++20`)
- **Compute capability**: sm_80 (`-arch=sm_80`)

## Tensor Layout
- **Native format: WHCN column-major** (identical in memory to NCHW row-major).
  - `element[n,c,h,w]` → offset `w + W*h + W*H*c + W*H*C*n`
- CUTLASS conv2d requires **NHWC** layout internally.
- cuDNN works directly with our native layout using `CUDNN_TENSOR_NCHW` — no conversion needed on the cuDNN side.

## CUTLASS Conv2d Rules (ALWAYS follow)
1. Allocate temporary NHWC device buffers for input, filter, and output.
2. Convert input and filter **NCHW → NHWC** before calling CUTLASS.
3. Call CUTLASS implicit GEMM conv2d.
4. Convert output **NHWC → NCHW** before returning to caller.
5. Free temporary buffers after use.

## cuDNN Reference Rules (ALWAYS follow)
1. Use `CUDNN_TENSOR_NCHW` — matches native WHCN col-major layout directly.
2. Use `cudnnGetConvolutionForwardAlgorithm_v7` with `CUDNN_DETERMINISTIC` filter for
   deterministic algorithm selection — never use `cudnnFindConvolutionForwardAlgorithm`
   (timing-based, non-deterministic) and never hardcode an algorithm.
3. Use `CUDNN_CROSS_CORRELATION` (not CONVOLUTION).

## Multi-Agent Workflow
When asked to implement a CUTLASS conv2d kernel, use the **Manager → Coder → Tester** pipeline:
- Invoke the `manager` agent — it will orchestrate the full pipeline.
- Do NOT attempt to write CUTLASS + test code in a single agent turn.

## Shared Utilities
- Layout conversion kernels live in `include/layout_utils.cuh`.
- Always `#include "layout_utils.cuh"` instead of reimplementing conversions.
