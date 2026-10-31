# Broadcasting and Reduction Mechanisms

## 1. Forward Pass: Virtual Broadcasting (`add_bias`)
In mini-batch Gradient Descent, the affine transformation for a layer is defined as:
`Z = W * X + b`

* `W * X` produces a matrix of shape `(D_out x B)`, where `B` is the batch size.
* The bias `b` is a learned parameter vector of shape `(D_out x 1)`.

Mathematical matrix addition requires strictly matching dimensions. A naive implementation would allocate a new `(D_out x B)` matrix and physically copy `b` across all `B` columns.

**HPC Memory Optimization:**
`add_bias` acts as a *virtual broadcasting kernel*. It iterates over the columns of the `(D_out x B)` matrix, repeatedly reading from the single `(D_out x 1)` bias vector in cache. This avoids an $O(D_{out} \times B)$ memory allocation and prevents saturating the memory bus with redundant data copies, ensuring maximum throughput during the forward pass.

## 2. Backward Pass: Gradient Reduction (`sum_along_columns`)
During backpropagation, we compute the gradient of the loss with respect to the bias: `dL/db`.

The upstream gradient `dZ` (the gradient of the loss with respect to the pre-activations) arrives with shape `(D_out x B)` because it contains the individual errors for every sample in the mini-batch. However, the bias `b` itself only has shape `(D_out x 1)`.

By the multivariate chain rule, the total gradient for a broadcasted parameter is the sum of the gradients across all paths it influenced. Because `b` was broadcast to every sample in the batch, its gradient is the sum of the errors across the batch dimension:
`dL/db = Σ (dZ_i) for i = 1 to B`

**HPC Implementation:**
`sum_along_columns` serves as the exact inverse of broadcasting. It reduces the `(D_out x B)` gradient matrix along axis 1 (the columns) down to a `(D_out x 1)` vector. This ensures the output `dL/db` identically matches the shape of `b`, allowing element-wise optimizers (like Adam or SGD) to safely update the parameter buffer.