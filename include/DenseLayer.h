#ifndef ML_SYSTEMS_ENGINE_DENSELAYER_H
#define ML_SYSTEMS_ENGINE_DENSELAYER_H
#include "Tensor2D.h"


class DenseLayer {
private:
    Tensor2D weights_;
    Tensor2D biases_;

public:
    // input_size (N) and output_size (K)
    DenseLayer(size_t input_size, size_t output_size);

    // Computes Y = ReLU(W * X + B)
    [[nodiscard]] Tensor2D forward(const Tensor2D& input) const;

    // Getters for testing
    [[nodiscard]] const Tensor2D& get_weights() const { return weights_; }
    [[nodiscard]] const Tensor2D& get_biases() const { return biases_; }
};


#endif //ML_SYSTEMS_ENGINE_DENSELAYER_H
