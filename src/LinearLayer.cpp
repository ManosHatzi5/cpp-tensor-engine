#include "LinearLayer.h"
#include <cmath>
#include <random>

// Linear Layer acts simply as the linear transformation z = wx + b
// in general for this project each layer acts as a mathematical box/transformation

LinearLayer::LinearLayer(size_t in_features, size_t out_features, InitScheme scheme)
    : in_features_(in_features),
      out_features_(out_features),
      weights_(out_features, in_features),
      biases_(out_features, 1),
      d_weights_(out_features, in_features),
      d_biases_(out_features, 1),
      last_input_(0, 0) {
    initialize_parameters(scheme);
}

void LinearLayer::initialize_parameters(InitScheme scheme) {
    if (scheme == InitScheme::Zeros) {
        return;
    }

    // Deterministic seed for reproducible tests
    std::mt19937 gen(42);

    float stddev = 0.0f;
    if (scheme == InitScheme::He) {
        // Var(W) = 2 / n_in  ==>  stddev = sqrt(2 / n_in)
        stddev = std::sqrt(2.0f / static_cast<float>(in_features_));
    } else if (scheme == InitScheme::Xavier) {
        // Var(W) = 2 / (n_in + n_out)  ==>  stddev = sqrt(2 / (n_in + n_out))
        stddev = std::sqrt(2.0f / static_cast<float>(in_features_ + out_features_));
    }

    std::normal_distribution<float> dist(0.0f, stddev);

    for (size_t r = 0; r < out_features_; ++r) {
        for (size_t c = 0; c < in_features_; ++c) {
            weights_(r, c) = dist(gen);
        }
    }

    // Biases are initialized to 0.0f
    for (size_t r = 0; r < out_features_; ++r) {
        biases_(r, 0) = 0.0f;
    }
}

Tensor2D LinearLayer::forward(const Tensor2D& input) {
    if (input.rows() != in_features_) {
        throw std::invalid_argument("Input row count must match layer input features.");
    }
    //(No activation applied here)
    last_input_ = input; // Cache input X for backward pass dW calculation

    // Z = W * X + b using cache-tiled GEMM kernel
    Tensor2D z = weights_.matmul_tiled(input, 32);
    // add the bias with virtual broadcasting across batch columns
    return z.add_bias(biases_);
}

Tensor2D LinearLayer::backward(const Tensor2D& grad_output) {
    if (grad_output.rows() != out_features_) {
        throw std::invalid_argument("grad_output row count must match layer output features.");
    }

    // 1. dW = grad_output (dZ) * X^T
    // Shape check: (D_out x B) * (B x D_in) -> (D_out x D_in)
    d_weights_ = grad_output.matmul_tiled(last_input_.transpose(), 32);

    // 2. dB = sum along columns across batch dimension
    // Shape check: (D_out x B) -> (D_out x 1)
    d_biases_ = grad_output.sum_along_columns();

    // 3. dX = W^T * dZ (passed to preceding layer)
    // Shape check: (D_in x D_out) * (D_out x B) -> (D_in x B)
    Tensor2D d_input = weights_.transpose().matmul_tiled(grad_output, 32);

    return d_input;
}


















