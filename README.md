# ML Systems Engine

A lightweight, from-scratch deep learning engine built in pure C++. 

This project serves as a foundational exploration of Machine Learning Systems Engineering. It implements core neural network primitives—forward and backward passes, optimizers, and loss functions—without relying on external math or tensor libraries (like Eigen or BLAS). 

A major focus of this project is **High-Performance Computing (HPC) and Hardware-Aware Optimization**, specifically addressing memory hierarchies, L1/L2 cache locality, and Instruction-Level Parallelism (ILP) to accelerate Matrix Multiplication (GEMM).

## 🚀 Key Features

* **Deep Learning Primitives:**
  * **Layers:** `LinearLayer`, `ReLULayer`, `SigmoidLayer`.
  * **Optimizers:** `SGD`, `Adam` (with momentum and RMSProp mechanics).
  * **Containers:** `Sequential` model architecture for streamlined forward/backward chaining.
  * **Loss Functions:** Mean Squared Error (MSE), Cross-Entropy.
* **Hardware-Optimized Tensor Math:**
  * Custom `Tensor2D` class with manual memory management and bounds checking.
  * Extensively profiled and optimized Matrix Multiplication kernels.
* **Comprehensive Test Suite:** Includes mathematical correctness equivalence tests and high-resolution latency/throughput benchmarks.

## ⚡ Hardware-Aware Optimizations (HPC)

The computational bottleneck of neural network training is Matrix Multiplication. This engine includes a highly optimized GEMM pipeline that respects modern CPU memory architecture:

1. **Loop Permutation ($i \to k \to j$):** Transitioned from the naive dot-product loop order to a streaming accumulation pattern, ensuring unit-stride memory access and enabling compiler auto-vectorization (AVX-256).
2. **L1 Cache Tiling (Blocking):** Partitioned matrices into $32 \times 32$ tiles to pin the active working set ($12\text{ KB}$) strictly inside the L1 Data Cache, practically eliminating main memory RAM fetches during the micro-kernel execution.
3. **Instruction-Level Parallelism (ILP):** Explored manual loop unrolling with independent accumulators to break Fused-Multiply-Add (FMA) pipeline latency stalls.

### Performance Gains
On a standard CPU (single-threaded, `-O3` Release mode), the cache-tiled implementation yields massive performance improvements over the naive mathematical implementation:
* **Isolated Kernel Throughput:** Up to **$5\times - 6\times$ faster** (jumping from ~2.5 GFLOPS to ~15.9 GFLOPS).
* **End-to-End Network Training:** Achieves a **$2.45\times$ overall wall-clock speedup** during multi-layer backpropagation and weight updates.

*(Read the full hardware optimization journey in `docs/matmul_optimization_journey.md`)*

## 🧠 API Example

Building and training a network follows an intuitive, object-oriented API similar to PyTorch:

```cpp
// 1. Build the architecture
Sequential model;
model.add(std::make_unique<LinearLayer>(64, 128));
model.add(std::make_unique<ReLULayer>());
model.add(std::make_unique<LinearLayer>(128, 16));

// 2. Initialize Optimizer
SGD optimizer(0.001f);

// 3. Training Loop
for (int epoch = 0; epoch < 10; ++epoch) {
    Tensor2D pred = model.forward(X_batch);
    Tensor2D grad = Loss::mse_derivative(pred, Y_batch);
    model.backward(grad);
    optimizer.step(model.get_layers());
}
