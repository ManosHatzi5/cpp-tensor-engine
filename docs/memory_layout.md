# HPC Memory Layout & Buffer Access Rationale

## 1. Raw Heap Pointers vs. Container Overhead
`Tensor2D` stores its payload in a private `std::vector<float> data_`. Exposing `float* data()` returns the heap address of index `[0]` (the internal `begin_ptr`). This design intentionally bypasses the 24-byte stack container metadata (`begin`, `end`, `capacity` pointers).

* **Indirection Avoidance**: Calling methods on a container pointer (`std::vector<float>*`) requires the CPU to resolve the stack pointer before fetching the heap address. A raw `float*` loads directly into CPU base registers, enabling direct indexed addressing: `MOVSS xmm0, [base + index * 4]`.
* **Stack vs. Heap Targets**: `std::vector<float>*` points to the container structure on the stack. `float*` points directly to the numerical payload on the heap.

## 2. Contiguous Memory & Pointer Arithmetic
Heap memory in `std::vector` is strictly contiguous. In C/C++, array indexing is fundamentally defined as pointer arithmetic:
`ptr[i] == *(ptr + i)`

Because the compiler knows `ptr` points to 4-byte `float` blocks, the CPU scales the memory offset automatically:
* `ptr[0]` -> `*(base_address + 0 * 4)`
* `ptr[i]` -> `*(base_address + i * 4)`

This allows seamless, flat iteration across multi-million element tensors without passing the heavier `std::vector` object to optimization kernels.

## 3. Optimizer Iteration vs. 2D Tensor Indexing
Standard matrix coordinate access via `operator()(r, c)` evaluates `r * cols + c` per element. This consumes redundant Arithmetic Logic Unit (ALU) cycles during element-wise updates.

Element-wise optimizers (Adam, SGD with Momentum) apply identical scalar equations to every parameter and do not depend on 2D topology. Exposing contiguous 1D pointers allows:
1. **Flat linear iteration** bounded strictly by `tensor.size()`.
2. **Auto-vectorization** across SIMD registers (AVX2/AVX-512) to process 4 to 8 floats per instruction cycle.

## 4. Const Overloading & Memory Safety
To satisfy strict mutation contracts across the execution graph:
* `float* data()` allows write access on mutable tensors (e.g., weights, biases).
* `const float* data() const` guarantees read-only access on immutable tensors (e.g., gradient buffers, input streams).

**Safety invariant**: Because raw pointers lack bounds checking, all kernel loops utilizing `.data()` must be strictly bounded by `.size()` to prevent heap out-of-bounds corruption.