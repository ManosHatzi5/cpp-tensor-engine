#ifndef ML_SYSTEMS_ENGINE_OPTIMIZER_H
#define ML_SYSTEMS_ENGINE_OPTIMIZER_H

#include "Layer.h"
#include <vector>
#include <memory>

class Optimizer {
public:
    virtual ~Optimizer() = default;
    
    // Executes one optimization step across all parameters in the network
    virtual void step(std::vector<std::unique_ptr<Layer>>& layers) = 0;
};

#endif