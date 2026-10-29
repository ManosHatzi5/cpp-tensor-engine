#ifndef ML_SYSTEMS_ENGINE_LOSS_H
#define ML_SYSTEMS_ENGINE_LOSS_H

#include "Tensor2D.h"

namespace Loss {
    // MEAN SQUARED ERROR (Regression) /------------/
    // Computes scalar MSE loss: (1/K) * sum((y_pred - y_true)^2)
    [[nodiscard]] float mse(const Tensor2D& y_pred, const Tensor2D& y_true);
    // Computes gradient: (2/K) * (y_pred - y_true)
    [[nodiscard]] Tensor2D mse_derivative(const Tensor2D& y_pred, const Tensor2D& y_true);

    //BINARY CROSS ENTROPY (Binary Classification) /----------/
    [[nodiscard]] float bce(const Tensor2D& y_pred, const Tensor2D& y_true);
    [[nodiscard]] Tensor2D bce_derivative(const Tensor2D& y_pred, const Tensor2D& y_true);
}

#endif // ML_SYSTEMS_ENGINE_LOSS_H