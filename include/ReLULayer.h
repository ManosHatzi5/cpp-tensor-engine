#ifndef ML_SYSTEMS_ENGINE_RELULAYER_H
#define ML_SYSTEMS_ENGINE_RELULAYER_H

#include "Layer.h"

class ReLULayer : public Layer {
private:
    // Caches Z (the output of the previous LinearLayer)
    Tensor2D last_z_;

public:
    // Default constructor is sufficient; no dimensions needed for element-wise ops
    ReLULayer() : last_z_(0, 0) {}

    [[nodiscard]] Tensor2D forward(const Tensor2D& input) override;
    [[nodiscard]] Tensor2D backward(const Tensor2D& d_out) override;
    // Inherits the empty update_parameters from Layer
};

#endif // ML_SYSTEMS_ENGINE_RELULAYER_H




