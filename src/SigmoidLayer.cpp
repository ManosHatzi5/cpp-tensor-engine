#include "SigmoidLayer.h"
#include <stdexcept>

Tensor2D SigmoidLayer::forward(const Tensor2D& input) {
    // Tensor2D already has a primitive sigmoid() kernel
    // So delegate to the Tensor2D kernel and cache the result A
    last_output_ = input.sigmoid();
    return last_output_;
}

Tensor2D SigmoidLayer::backward(const Tensor2D& d_out) {
    // geometric consistency
    if (d_out.rows() != last_output_.rows() || d_out.cols() != last_output_.cols()) {
        throw std::invalid_argument("Sigmoid backward gradient dimension mismatch.");
    }
    // dZ = dA hadamard_product sigma'(Z)
    // We compute sigma'(Z) instantly using the cached A (last_output_)
    return d_out.multiply(last_output_.sigmoid_derivative_from_output());
}