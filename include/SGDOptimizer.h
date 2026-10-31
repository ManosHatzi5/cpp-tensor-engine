#ifndef ML_SYSTEMS_ENGINE_SGD_H
#define ML_SYSTEMS_ENGINE_SGD_H

#include "Optimizer.h"

class SGD : public Optimizer {
private:
    float lr_;
public:
    explicit SGD(float lr = 0.01f) : lr_(lr) {}

    void step(std::vector<std::unique_ptr<Layer>>& layers) override {
        for (auto& layer : layers) {
            if (layer->has_parameters()) {
                // Update Weights
                Tensor2D& w = layer->get_weights();
                const Tensor2D& dw = layer->get_d_weights();
                float* w_data = w.data();
                const float* dw_data = dw.data();
                for (size_t i = 0; i < w.size(); ++i) {
                    w_data[i] -= lr_ * dw_data[i];
                }

                // Update Biases
                Tensor2D& b = layer->get_biases();
                const Tensor2D& db = layer->get_d_biases();
                float* b_data = b.data();
                const float* db_data = db.data();
                for (size_t i = 0; i < b.size(); ++i) {
                    b_data[i] -= lr_ * db_data[i];
                }
            }
        }
    }
};

#endif