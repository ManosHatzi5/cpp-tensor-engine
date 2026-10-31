#ifndef ML_SYSTEMS_ENGINE_LINEARLAYER_H
#define ML_SYSTEMS_ENGINE_LINEARLAYER_H

#include "Layer.h"

enum class InitScheme {
    Zeros,
    Xavier, // Standard for Sigmoid / Tanh
    He      // Standard for ReLU
};

class LinearLayer : public Layer {
private:
    size_t in_features_;
    size_t out_features_;

    Tensor2D weights_;
    Tensor2D biases_;

    Tensor2D d_weights_;
    Tensor2D d_biases_;

    Tensor2D last_input_;

    void initialize_parameters(InitScheme scheme);

public:
    // Defaults to He initialization
    LinearLayer(size_t in_features, size_t out_features, InitScheme scheme = InitScheme::He);

    [[nodiscard]] Tensor2D forward(const Tensor2D& input) override;
    [[nodiscard]] Tensor2D backward(const Tensor2D& d_out) override;
    [[nodiscard]] bool has_parameters() const override { return true; }

    Tensor2D& get_weights() override { return weights_; }
    Tensor2D& get_biases() override { return biases_; }
    [[nodiscard]] const Tensor2D& get_d_weights() const override { return d_weights_; }
    [[nodiscard]] const Tensor2D& get_d_biases() const override { return d_biases_; }
};

#endif // ML_SYSTEMS_ENGINE_LINEARLAYER_H



















