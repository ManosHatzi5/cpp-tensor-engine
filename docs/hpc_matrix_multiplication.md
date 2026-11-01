# High-Performance Matrix Multiplication: Hardware Optimization Journey

In neural network engines, Matrix Multiplication (GEMM) is the primary computational bottleneck. This document outlines the progression of optimizing the `Tensor2D` GEMM kernels from a naive mathematical implementation to a hardware-aware, cache-optimized architecture.

## 1. The Baseline: Naive CPU Matmul ($O(N^3)$)
The mathematical definition of matrix multiplication uses an `i -> j -> k` loop permutation.
While mathematically correct, this is disastrous for modern hardware.
* **The Memory Wall:** CPUs process data in sub-nanoseconds, but fetching from RAM takes ~100ns. Data is fetched in 64-byte "Cache Lines" (16 floats).
* **Cache Thrashing:** In the naive `j` loop, reading down a column of matrix B means jumping 4KB in memory on every step. The CPU loads 16 floats, reads 1, throws the rest away, and stalls waiting for RAM.

## 2. Milestone 1: Loop Permutation ($i \to k \to j$)
By simply swapping the inner loops to `i -> k -> j`, we change the access pattern from a "Dot Product" to a "Streaming Accumulation".
* **Spatial Locality:** Both matrix B and the result matrix C are read/written strictly horizontally.
* **Unit-Stride:** Every step advances exactly 4 bytes. The hardware stream prefetcher recognizes this pattern and pipelines data into the L1 cache before the CPU even asks for it.
* **Auto-Vectorization:** Because the innermost loop writes to independent cells of C, the compiler easily maps this to 256-bit AVX SIMD instructions (processing 8 floats per clock cycle).

## 3. Milestone 2: Cache Tiling (Blocking)
Even with unit-stride reads, a large matrix (e.g., 1MB) will flush the 32KB L1 cache before the CPU can reuse the data.
* **Temporal Locality:** We partitioned the matrices into `32x32` blocks.
* A single tile of A, B, and C takes exactly $12\text{ KB}$, fitting perfectly inside the $32\text{ KB}$ L1 Data Cache.
* The CPU performs 32,768 arithmetic operations strictly within the L1 cache before requesting new data from RAM, dropping memory bus traffic by orders of magnitude.
* **Result:** Our `matmul_tiled` achieved a **~6x to 10x speedup** over the naive implementation in Release mode.

## 4. Milestone 3: Transposed B & Instruction-Level Parallelism (ILP)
In Deep Learning, backpropagation naturally provides transposed weight matrices. We implemented `matmul_transposed_b_tiled` to test dual-unit-stride dot products.
* **The Bottleneck:** A standard inner dot product requires adding into a single `sum` variable. Modern Fused-Multiply-Add (FMA) instructions take 4 clock cycles. The CPU pipeline stalled because every addition waited for the previous one to finish.
* **The Solution (Loop Unrolling):** We manually unrolled the inner loop to use 4 independent accumulators (`sum0` to `sum3`). This breaks the dependency chain, allowing the CPU to issue 4 simultaneous FMA instructions per cycle, keeping the hardware pipeline saturated.

## Conclusion
While the manually unrolled Transposed-B kernel achieved excellent speeds, the `i-k-j` tiled approach remains the overall champion for pure C++ implementations. It provides the compiler with a mathematically perfect streaming structure, allowing it to natively emit optimal SIMD assembly without manual unrolling.
```eof