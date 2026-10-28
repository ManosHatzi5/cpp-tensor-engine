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
    //(No activation applied here)
    last_input_ = input;
    // Z = (W * X) + B with virtual broadcasting across batch columns
    return weights_.matmul(input).add_bias(biases_);
}

Tensor2D LinearLayer::backward(const Tensor2D& d_out) {
    // d_out is dZ (or delta).
    // 1. dW = dZ * X^T
    // Shape check: (D_out x B) * (B x D_in) -> (D_out x D_in)
    d_weights_ = d_out.matmul(last_input_.transpose());

    // 2. dB = sum along columns of dZ
    // Shape check: (D_out x B) -> (D_out x 1)
    d_biases_ = d_out.sum_along_columns();

    // 3. dX = W^T * dZ (passed to preceding layer)
    // Shape check: (D_in x D_out) * (D_out x B) -> (D_in x B)
    return weights_.transpose().matmul(d_out);
}

void LinearLayer::update_parameters(float learning_rate) {
    // implementing SGD
    weights_ = weights_ + (d_weights_ * (-learning_rate));
    biases_  = biases_  + (d_biases_  * (-learning_rate));
}


















