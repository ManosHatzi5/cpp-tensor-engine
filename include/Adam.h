#ifndef ML_SYSTEMS_ENGINE_ADAM_H
#define ML_SYSTEMS_ENGINE_ADAM_H

#include "Optimizer.h"
#include <unordered_map>

/**
 * @brief Adaptive Moment Estimation (Adam) Optimizer.
 * @note Decoupled memory manager that tracks 1st and 2nd gradient moments
 *       using physical memory addresses (void*) as unique tensor identifiers.
 */
class Adam : public Optimizer {
private:
    float lr_;
    float beta1_;
    float beta2_;
    float epsilon_;
    int t_;

    // Hash maps binding a tensor's physical RAM address to its historical moment buffers
    std::unordered_map<void*, Tensor2D> m_buffers_;
    std::unordered_map<void*, Tensor2D> v_buffers_;

    /**
     * @brief Core mutation kernel applying the Adam update rule in-place.
     */
    void update_tensor(Tensor2D& param, const Tensor2D& grad);

public:
    /**
     * @brief Constructs the Adam optimizer.
     * @note Marked explicit to prevent accidental implicit conversions from floats.
     */
    explicit Adam(float lr = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f);

    /**
     * @brief Executes a single optimization step across all learnable parameters.
     */
    void step(std::vector<std::unique_ptr<Layer>>& layers) override;
};

#endif














