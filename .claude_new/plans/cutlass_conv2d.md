# Plan: CUTLASS Conv2d — compute-eval-fork

This plan file contains all repository-specific knowledge for the CUTLASS conv2d task.
The generic Manager / Coder / Tester agents in `.claude_new/agents/` read this file at
runtime; they contain no repo-specific rules themselves.

---

## Task description

Implement the function `conv2d_cutlass_whcn()` for CUTLASS implicit-GEMM conv2d evaluation
in the compute-eval-fork repository. The function receives WHCN col-major (= NCHW row-major)
tensors, converts them to NHWC internally for CUTLASS, calls the CUTLASS implicit GEMM
forward convolution kernel, and converts the output back to NCHW before returning.

---

## Parameter schema

The manager must extract the following parameters from the user's request:

| Parameter  | Type | Description |
|------------|------|-------------|
| `N`        | int  | Batch size |
| `H`        | int  | Input height |
| `W`        | int  | Input width |
| `C_in`     | int  | Input channels |
| `K`        | int  | Output channels (number of filters) |
| `R`        | int  | Filter height |
| `S`        | int  | Filter width |
| `pad_h`    | int  | Padding along height |
| `pad_w`    | int  | Padding along width |
| `stride_h` | int  | Stride along height |
| `stride_w` | int  | Stride along width |

---

## Toolchain

- **CUDA**: 12.x (`nvcc` pre-installed)
- **Host compiler**: GCC 13 (`g++-13`), flag `-ccbin g++-13`
- **C++ standard**: C++20, flag `-std=c++20`
- **Compute capability**: sm_80, flag `-arch=sm_80`
- **Optimization**: `-O2`
- **Environment setup** (run before any build or run step):
  ```bash
  source scripts/cutlass.env
  ```
  This sets `CUDA_HOME`, `CUTLASS_HOME`, `PATH`, `LD_LIBRARY_PATH`, `CUDAHOSTCXX`, etc.
- **CUTLASS include paths**: `$CUTLASS_HOME/include` and `$CUTLASS_HOME/tools/util/include`
- **Project include path**: `include/` (relative to `data/CUTLASS-CONV2D/0/`)

---

## Tensor layout rules

- **Native format**: WHCN column-major, which is identical in memory to NCHW row-major.
  - `element[n,c,h,w]` → linear offset `w + W*h + W*H*c + W*H*C*n`
- **CUTLASS requirement**: tensors must be in NHWC layout before being passed to the kernel.
- **cuDNN reference**: uses `CUDNN_TENSOR_NCHW`, which matches the native WHCN format directly —
  no conversion is needed on the cuDNN side.
- Shared layout conversion utilities live in `include/layout_utils.cuh` (relative to the
  `data/CUTLASS-CONV2D/0/` directory).

---

## Implementation rules

Follow these rules exactly, in order:

1. `#include "layout_utils.cuh"` at the top — **never** reimplement layout conversions inline.
2. Allocate temporary NHWC device buffers for input, filter, and output using `cudaMalloc`.
3. Convert input and filter from **NCHW → NHWC** using the utilities from `layout_utils.cuh`.
4. Call the CUTLASS implicit GEMM forward convolution kernel using the type configuration
   specified in the **Type configuration** section below.
5. Convert the output from **NHWC → NCHW** using the utilities from `layout_utils.cuh`.
6. Free all temporary device buffers with `cudaFree` after use.

---

## Function signature

The coder must implement exactly the following signature (no changes to parameter names or types):

```cpp
void conv2d_cutlass_whcn(
    const float* input,   // device ptr, WHCN col-major == NCHW row-major [N, C_in, H, W]
    const float* filter,  // device ptr, WHCN col-major == NCHW row-major [K, C_in, R, S]
    float*       output,  // device ptr, WHCN col-major == NCHW row-major [N, K, P, Q]
    int N, int H, int W, int C_in,
    int K, int R, int S,
    int pad_h,    int pad_w,
    int stride_h, int stride_w,
    cudaStream_t stream);
```

---

## Required headers

The coder must include exactly these headers (in addition to any standard CUDA headers):

```cpp
#include "layout_utils.cuh"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/epilogue/thread/linear_combination.h"
```

---

## Type configuration

Use the following CUTLASS template configuration:

```cpp
// Element types
using ElementInput  = float;
using ElementFilter = float;
using ElementOutput = float;
using ElementAccum  = float;

// Layouts (all NHWC for CUTLASS)
using LayoutInput  = cutlass::layout::TensorNHWC;
using LayoutFilter = cutlass::layout::TensorNHWC;
using LayoutOutput = cutlass::layout::TensorNHWC;

// MMA and architecture
using MMAOp  = cutlass::arch::OpClassSimt;
using SmArch = cutlass::arch::Sm80;

// Tile shapes
using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 8>;
using WarpShape        = cutlass::gemm::GemmShape<32, 64, 8>;
using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

// Epilogue
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput, 1, ElementAccum, ElementAccum>;

// Conv mode and split-K
constexpr cutlass::conv::Mode kMode = cutlass::conv::Mode::kCrossCorrelation;
constexpr int kSplitKSlices = 1;
```

---

## Output file

The coder must write the solution to:

```
data/CUTLASS-CONV2D/0/solution/solution.cu
```

(Path is relative to the repository root.)

---

## Build command

The tester must run the following commands exactly (in a single shell invocation):

```bash
source scripts/cutlass.env && \
cd data/CUTLASS-CONV2D/0 && \
nvcc -std=c++20 -O2 -ccbin g++-13 \
  -I include \
  -I $CUTLASS_HOME/include \
  -I $CUTLASS_HOME/tools/util/include \
  -arch=sm_80 \
  solution/solution.cu test/test_main.cu \
  -lcudnn \
  -o /tmp/conv2d_test
```

The commands must be run from the **repository root** so that `source scripts/cutlass.env`
resolves correctly.

---

## Run command

```bash
source scripts/cutlass.env && /tmp/conv2d_test
```

---

## Correctness criteria

- **PASS**: the last line of stdout is exactly `PASS` **and** the process exits with code 0.
- **FAIL**: any other output or non-zero exit code.
- **Tolerance**: the test internally checks `|cutlass - cudnn| / (|cudnn| + 1e-6) <= 1e-3`
  (max relative error 0.1%) for all output elements.
- **Test cases covered** (hard-coded in the test harness):
  1. 3×3 conv, same padding, stride 1
  2. 1×1 conv, no padding, stride 1
  3. 3×3 conv, stride 2
  4. Asymmetric spatial dimensions

---

## cuDNN reference rules (informational — for Tester context only)

The test harness uses cuDNN as the reference implementation:

- Tensor format: `CUDNN_TENSOR_NCHW` (matches native WHCN layout directly)
- Algorithm selection: `cudnnGetConvolutionForwardAlgorithm_v7` with `CUDNN_DETERMINISTIC`
  (never `cudnnFindConvolutionForwardAlgorithm` — non-deterministic)
- Convolution mode: `CUDNN_CROSS_CORRELATION`
