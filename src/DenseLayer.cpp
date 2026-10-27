#include "DenseLayer.h"

DenseLayer::DenseLayer(size_t input_size, size_t output_size)
    : weights_(output_size, input_size),
      biases_(output_size, 1) {}

Tensor2D DenseLayer::forward(const Tensor2D& input) const {
    // Z = W * X + B
    Tensor2D z = weights_.matmul(input) + biases_;
    // Return non-linear activation
    return z.relu();
}