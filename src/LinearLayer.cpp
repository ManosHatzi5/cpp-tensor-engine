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


















