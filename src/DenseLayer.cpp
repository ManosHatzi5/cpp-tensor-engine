#include "DenseLayer.h"

DenseLayer::DenseLayer(size_t input_size, size_t output_size)
    : weights_(output_size, input_size),
      biases_(output_size, 1),
      d_weights_(output_size, input_size),
      d_biases_(output_size, 1),
      last_input_(input_size, 1),
      last_z_(output_size, 1) {
    weights_.randomize(-0.5f, 0.5f);
}

Tensor2D DenseLayer::forward(const Tensor2D& input) {
    // Cache the input X for dW calculation in backward
    last_input_ = input;

    // Z = W * X + B
    last_z_ = weights_.matmul(input) + biases_;

    // A = ReLU(Z)
    return last_z_.relu();
}

Tensor2D DenseLayer::backward(const Tensor2D& d_out) {
    // 1. delta = (dL / dA) hadamard ReLU'(Z)
    Tensor2D delta = d_out.multiply(last_z_.relu_derivative());

    // 2. dL / dW = delta * X^T
    d_weights_ = delta.matmul(last_input_.transpose());

    // 3. dL / dB = delta
    d_biases_ = delta;

    // 4. dL / dX = W^T * delta (to be consumed by the previous layer)

    // any transpose operation is done to geometrically match either the input of the previous layer
    // or to match the dimensions so matmul can work
    return weights_.transpose().matmul(delta);
}