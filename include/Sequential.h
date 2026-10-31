#ifndef ML_SYSTEMS_ENGINE_SEQUENTIAL_H
#define ML_SYSTEMS_ENGINE_SEQUENTIAL_H

#include <vector>
#include <memory>
#include "Layer.h"

class Sequential {
private:
    std::vector<std::unique_ptr<Layer>> layers_;

public:
    Sequential() = default;

    // Move semantics to transfer ownership of layers into the container
    void add(std::unique_ptr<Layer> layer);

    // Iterates forward: Input -> Layer 0 -> Layer 1 -> ... -> Output
    [[nodiscard]] Tensor2D forward(const Tensor2D& input);

    // Iterates in reverse: d_out -> Layer N-1 -> Layer N-2 -> ... -> d_in
    [[nodiscard]] Tensor2D backward(const Tensor2D& d_out);

    [[nodiscard]] size_t size() const { return layers_.size(); }

    // Exposes layer references so external optimizers can apply parameter updates
    std::vector<std::unique_ptr<Layer>>& get_layers() noexcept { return layers_; }
};

#endif // ML_SYSTEMS_ENGINE_SEQUENTIAL_H