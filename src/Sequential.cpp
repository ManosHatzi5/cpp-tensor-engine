#include "Sequential.h"

#include <ranges>

void Sequential::add(std::unique_ptr<Layer> layer) {
    layers_.push_back(std::move(layer));
}

Tensor2D Sequential::forward(const Tensor2D& input) {
    Tensor2D current_activation = input;
    for (const auto& layer : layers_) {
        current_activation = layer->forward(current_activation);
    }
    return current_activation;
}

Tensor2D Sequential::backward(const Tensor2D& d_out) {
    Tensor2D current_grad = d_out;
    // Walk backward through the graph using reverse iterators
    for (auto & layer : std::views::reverse(layers_)) {
        current_grad = layer->backward(current_grad);
    }
    // the code before clang-tidy:
    // for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
    //     current_grad = (*it)->backward(current_grad);}
    return current_grad;
}

void Sequential::update_parameters(float learning_rate) {
    for (const auto& layer : layers_) {
        layer->update_parameters(learning_rate);
    }
}