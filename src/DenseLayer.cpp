#include "DenseLayer.h"

DenseLayer::DenseLayer(size_t input_size, size_t output_size)
    : weights_(output_size, input_size),
      biases_(output_size, 1) {
    // Biases typically stay 0.0f, but weights must break symmetry:
    weights_.randomize(-0.5f, 0.5f);
}

Tensor2D DenseLayer::forward(const Tensor2D& input) const {
    // Z = W * X + B
    Tensor2D z = weights_.matmul(input) + biases_;
    // Return non-linear activation
    return z.relu();
}