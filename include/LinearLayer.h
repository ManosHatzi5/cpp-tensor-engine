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
    void update_parameters(float learning_rate) override;

    [[nodiscard]] const Tensor2D& weights() const { return weights_; }
    [[nodiscard]] const Tensor2D& biases() const { return biases_; }
};

#endif // ML_SYSTEMS_ENGINE_LINEARLAYER_H



















