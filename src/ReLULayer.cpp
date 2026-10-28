#include "ReLULayer.h"

Tensor2D ReLULayer::forward(const Tensor2D& input) {
    last_z_ = input; // caching z
    return last_z_.relu();
}

Tensor2D ReLULayer::backward(const Tensor2D& d_out) {
    // d_out is dA.
    // Return dZ = dA hadamard ReLU'(Z)
    return d_out.multiply(last_z_.relu_derivative());
}