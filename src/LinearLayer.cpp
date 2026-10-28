#include "LinearLayer.h"

// Linear Layer acts simply as the linear transformation z = wx + b
// in general for this project each layer acts as a mathematical box/transformation

LinearLayer::LinearLayer(size_t input_size, size_t output_size)
    : weights_(output_size, input_size),
      biases_(output_size, 1),
      d_weights_(output_size, input_size),
      d_biases_(output_size, 1),
      last_input_(input_size, 1) {
    weights_.randomize(-0.5f, 0.5f);
}

Tensor2D LinearLayer::forward(const Tensor2D& input) {
    last_input_ = input;
    // Z = W * X + B  (No activation applied here)
    return weights_.matmul(input) + biases_;
}

Tensor2D LinearLayer::backward(const Tensor2D& d_out) {
    // d_out is dZ (or delta).
    // dW = dZ * X^T
    d_weights_ = d_out.matmul(last_input_.transpose());
    // dB = dZ
    d_biases_ = d_out;

    // Return dX = W^T * dZ
    return weights_.transpose().matmul(d_out);
}

void LinearLayer::update_parameters(float learning_rate) {
    // implementing SGD
    weights_ = weights_ + (d_weights_ * (-learning_rate));
    biases_  = biases_  + (d_biases_  * (-learning_rate));
}