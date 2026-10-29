#ifndef ML_SYSTEMS_ENGINE_SIGMOIDLAYER_H
#define ML_SYSTEMS_ENGINE_SIGMOIDLAYER_H

#include "Layer.h"

class SigmoidLayer : public Layer {
private:
    // Caches the activated output A = sigma(Z) to avoid re-evaluating exp(-Z) in backward
    Tensor2D last_output_;

public:
    SigmoidLayer() : last_output_(0, 0) {}

    [[nodiscard]] Tensor2D forward(const Tensor2D& input) override;
    [[nodiscard]] Tensor2D backward(const Tensor2D& d_out) override;
};

#endif // ML_SYSTEMS_ENGINE_SIGMOIDLAYER_H