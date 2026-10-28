#ifndef ML_SYSTEMS_ENGINE_DENSELAYER_H
#define ML_SYSTEMS_ENGINE_DENSELAYER_H

#include "Tensor2D.h"

class DenseLayer {
private:
    Tensor2D weights_;
    Tensor2D biases_;

    // Gradients accumulated during backprop
    Tensor2D d_weights_;
    Tensor2D d_biases_;

    // Cached states from the forward pass, required for the backpropagation
    Tensor2D last_input_;
    Tensor2D last_z_;

public:
    DenseLayer(size_t input_size, size_t output_size);

    // Forward pass: Z = W * X + B, returns A = ReLU(Z)
    [[nodiscard]] Tensor2D forward(const Tensor2D& input);

    // Backward pass: takes d_out = dL/dA, returns d_in = dL/dX
    [[nodiscard]] Tensor2D backward(const Tensor2D& d_out);

    // Getters for inspection and testing
    [[nodiscard]] const Tensor2D& get_weights() const { return weights_; }
    [[nodiscard]] const Tensor2D& get_biases() const { return biases_; }
    [[nodiscard]] const Tensor2D& get_d_weights() const { return d_weights_; }
    [[nodiscard]] const Tensor2D& get_d_biases() const { return d_biases_; }
};

#endif // ML_SYSTEMS_ENGINE_DENSELAYER_H