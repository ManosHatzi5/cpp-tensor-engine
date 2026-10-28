#ifndef ML_SYSTEMS_ENGINE_LINEARLAYER_H
#define ML_SYSTEMS_ENGINE_LINEARLAYER_H

#include "Layer.h"

class LinearLayer : public Layer {
private:
    Tensor2D weights_;
    Tensor2D biases_;
    Tensor2D d_weights_;
    Tensor2D d_biases_;
    Tensor2D last_input_;

public:
    LinearLayer(size_t input_size, size_t output_size);

    [[nodiscard]] Tensor2D forward(const Tensor2D& input) override;
    [[nodiscard]] Tensor2D backward(const Tensor2D& d_out) override;
    void update_parameters(float learning_rate) override;

    [[nodiscard]] const Tensor2D& get_weights() const { return weights_; }
    [[nodiscard]] const Tensor2D& get_d_weights() const { return d_weights_; }
};

#endif // ML_SYSTEMS_ENGINE_LINEARLAYER_H