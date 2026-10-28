#ifndef ML_SYSTEMS_ENGINE_LAYER_H
#define ML_SYSTEMS_ENGINE_LAYER_H

#include "Tensor2D.h"

class Layer {
public:
    virtual ~Layer() = default;

    // Takes an input, returns the output, caches necessary states
    [[nodiscard]] virtual Tensor2D forward(const Tensor2D& input) = 0;

    // Takes the incoming gradient, computes parameter gradients, returns dX
    [[nodiscard]] virtual Tensor2D backward(const Tensor2D& d_out) = 0;

    // Virtual but not pure, because activation layers don't have parameters to update
    virtual void update_parameters(float learning_rate) {}
};

#endif // ML_SYSTEMS_ENGINE_LAYER_H