#include "Loss.h"
#include <stdexcept>

namespace Loss {

float mse(const Tensor2D& y_pred, const Tensor2D& y_true) {
    if (y_pred.rows() != y_true.rows() || y_pred.cols() != y_true.cols()) {
        throw std::invalid_argument("MSE dimension mismatch: shapes must be identical.");
    }

    float total_loss = 0.0f;

    // We treat the inputs as generic 2D Tensors rather than 1D vectors to seamlessly
    // support Batch Processing. If we pass a batch of N inputs, y_pred is (K x N).
    // The total_elements accounts for the full grid (Outputs * BatchSize).
    const size_t total_elements = y_pred.rows() * y_pred.cols();

    for (size_t r = 0; r < y_pred.rows(); ++r) {
        for (size_t c = 0; c < y_pred.cols(); ++c) {
            // Compute the raw error vector component
            float diff = y_pred(r, c) - y_true(r, c);

            // BARE-METAL OPTIMIZATION:
            // We use `diff * diff` instead of `std::pow(diff, 2)`.
            // std::pow is a heavy standard library call built for fractional/negative bounds.
            // diff * diff compiles down to a single FMUL (floating-point multiply) CPU instruction.
            total_loss += diff * diff;
        }
    }

    // Explicitly static_cast total_elements (size_t) to float to ensure strict type
    // compliance and prevent implicit conversion warnings in systems compilers.
    // Dividing by total_elements gives us the "Mean" to keep gradients stable across batch sizes.
    return total_loss / static_cast<float>(total_elements);
}

Tensor2D mse_derivative(const Tensor2D& y_pred, const Tensor2D& y_true) {
    if (y_pred.rows() != y_true.rows() || y_pred.cols() != y_true.cols()) {
        throw std::invalid_argument("MSE derivative dimension mismatch: shapes must be identical.");
    }

    Tensor2D grad(y_pred.rows(), y_pred.cols());
    const size_t total_elements = y_pred.rows() * y_pred.cols();

    // Mathematical derivative of (1/K) * sum((y_pred - y_true)^2) is (2/K) * (y_pred - y_true).
    // We compute the (2/K) scaling factor ONCE outside the loop to avoid doing
    // heavy division arithmetic repeatedly in the hot path.
    const float scale = 2.0f / static_cast<float>(total_elements);

    for (size_t r = 0; r < y_pred.rows(); ++r) {
        for (size_t c = 0; c < y_pred.cols(); ++c) {
            // This populates the Gradient Matrix (Jacobian vector).
            // It tells backpropagation the exact direction and magnitude
            // required to push this specific prediction toward the ground truth.
            grad(r, c) = scale * (y_pred(r, c) - y_true(r, c));
        }
    }

    return grad;
}

} // namespace Loss