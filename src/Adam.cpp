#include "Adam.h"
#include <cmath>

Adam::Adam(float lr, float beta1, float beta2, float epsilon)
    : lr_(lr), beta1_(beta1), beta2_(beta2), epsilon_(epsilon), t_(0) {}

void Adam::step(std::vector<std::unique_ptr<Layer>>& layers) {
    t_++; // Increment timestep t

    for (auto& layer : layers) {
        if (layer->has_parameters()) {
            update_tensor(layer->get_weights(), layer->get_d_weights());
            update_tensor(layer->get_biases(), layer->get_d_biases());
        }
    }
}

void Adam::update_tensor(Tensor2D& param, const Tensor2D& grad) {
    // KEY DISTINCTION: We hash the address of the Tensor2D *object* (&param),
    // not the address of its raw float array (param.data()). The object address
    // serves as the immutable, unique ID for these specific weights.
    void* param_ptr = static_cast<void*>(&param);

    // 1. Initialize buffers to zero on the very first step
    if (m_buffers_.find(param_ptr) == m_buffers_.end()) {
        m_buffers_.emplace(param_ptr, Tensor2D(param.rows(), param.cols()));
        v_buffers_.emplace(param_ptr, Tensor2D(param.rows(), param.cols()));
    }

    Tensor2D& m = m_buffers_.at(param_ptr);
    Tensor2D& v = v_buffers_.at(param_ptr);

    // 2. HPC Scalar Precomputation
    // Computed exactly once per tensor update, avoiding division inside the loop
    const float bias_correction1 = 1.0f - std::pow(beta1_, t_);
    const float bias_correction2 = 1.0f - std::pow(beta2_, t_);
    const float step_size = lr_ * (std::sqrt(bias_correction2) / bias_correction1);
    const float eps_tilde = epsilon_ * std::sqrt(bias_correction2);

    // 3. Raw SIMD-ready Memory Loop
    float* p_data = param.data();
    const float* g_data = grad.data();
    float* m_data = m.data();
    float* v_data = v.data();
    const size_t N = param.size();

    // TEMPORAL EVOLUTION VIA IN-PLACE MUTATION:
    // Index `i` represents spatial coordinate in the parameter buffer, NOT timestep `t`.
    // Prior to this line, m_data[i] stores the accumulated moment from step (t-1).
    // By evaluating the right-hand expression before assigning to m_data[i], we read
    // m_(t-1) and commit m_t to memory in-place, eliminating redundant buffer allocations.
    for (size_t i = 0; i < N; ++i) {
        // Step A: Update biased 1st moment
        m_data[i] = beta1_ * m_data[i] + (1.0f - beta1_) * g_data[i];

        // Step B: Update biased 2nd moment
        v_data[i] = beta2_ * v_data[i] + (1.0f - beta2_) * (g_data[i] * g_data[i]);

        // Step C & D: Parameter update using the precomputed scalar identity
        p_data[i] -= step_size * (m_data[i] / (std::sqrt(v_data[i]) + eps_tilde));
    }
}