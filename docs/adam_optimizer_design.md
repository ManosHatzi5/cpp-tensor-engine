# Optimizer Architecture & Memory Management

## 1. Physical Memory Addresses as Identifiers (`void*`)
The optimizer operates completely decoupled from the neural network topology. It does not know what a "Linear Layer" or a "Convolutional Layer" is. To track the historical gradients (moments) of specific parameters, it uses raw memory addresses.
* By casting a tensor's reference to a `void*` (e.g., `0x7ffee4a10`), we use its literal physical location in RAM as its unique ID.
* This guarantees that even if multiple layers have tensors of the exact same shape, the optimizer perfectly distinguishes them.

## 2. $O(1)$ Buffer Retrieval (`std::unordered_map`)
Because Adam requires tracking the 1st moment ($m_t$) and 2nd moment ($v_t$) for every single parameter across time, we allocate `Tensor2D` "buffers" to persist this historical data.
* `std::unordered_map<void*, Tensor2D>` creates a hash table.
* During the `.step()` loop, the optimizer hashes the physical address of the weight matrix and performs an $O(1)$ constant-time lookup to instantly retrieve its exact $m$ and $v$ history.
### The Key-Value Payload Distinction
* **The Key (`void*`):** The physical memory address of the target `Tensor2D` object instance (e.g., Layer 1's weight object).
* **The Value (`Tensor2D`):** A completely separate, dynamically allocated `Tensor2D` instantiated at $t=1$. It is initialized to zeros and acts exclusively as the historical memory buffer ($m_t$ or $v_t$) for that specific parameter. It does not duplicate the weights.

## 3. Strict Type Safety (`explicit`)
The `explicit` keyword on the constructor prevents the C++ compiler from performing hidden, implicit type conversions.
* Without it, passing a float to a function expecting an optimizer pointer (e.g., `train(0.01f)`) might silently construct a temporary Adam object.
* `explicit` forces deliberate instantiation (`Adam(0.01f)`), preventing subtle runtime bugs.

## 4. Execution Flow Separation
* **`step()` (The Global Coordinator):** Called once per batch. It increments the global timestep $t$ (for bias correction) and loops polymorphically over the `std::vector<std::unique_ptr<Layer>>`, extracting parameter references.
* **`update_tensor()` (The Math Engine):** The low-level mutation kernel. It takes one target tensor and its gradient, executes the scalar precomputations, applies the SIMD-ready memory loop, and physically mutates the target tensor's floats in place.