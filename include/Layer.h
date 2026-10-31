#ifndef ML_SYSTEMS_ENGINE_LAYER_H
#define ML_SYSTEMS_ENGINE_LAYER_H

#include "Tensor2D.h"
#include <stdexcept>

class Layer {
public:
    virtual ~Layer() = default;

    // Takes an input, returns the output, caches necessary states
    [[nodiscard]] virtual Tensor2D forward(const Tensor2D& input) = 0;

    // Takes the incoming gradient, computes parameter gradients, returns dX
    [[nodiscard]] virtual Tensor2D backward(const Tensor2D& d_out) = 0;

    // Identify if the layer has learnable parameters
    virtual bool has_parameters() const { return false; }

    // Expose parameters and gradients by reference for the Optimizer
    virtual Tensor2D& get_weights() { throw std::logic_error("No weights in this layer"); }
    virtual Tensor2D& get_biases() { throw std::logic_error("No biases in this layer"); }
    virtual const Tensor2D& get_d_weights() const { throw std::logic_error("No d_weights"); }
    virtual const Tensor2D& get_d_biases() const { throw std::logic_error("No d_biases"); }

};

#endif // ML_SYSTEMS_ENGINE_LAYER_H