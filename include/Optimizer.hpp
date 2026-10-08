#pragma once

# include "Layer.hpp"
# include <vector>

class Optimizer{
public:
    float lr; // learning rate
    Optimizer(float learning_rate) : lr(learning_rate){};
    virtual ~Optimizer() = default;

    // Pure virtual function that each specific optimizer must implement
    virtual void step(Network& net) = 0;
};

class SGDOptimizer : public Optimizer {
public:
    // Constructor passes the learning rate directly to base Optimizer constructor
    SGDOptimizer(float learning_rate) : Optimizer(learning_rate) {}

    // Overriding the step function to implement the SGD loop
    void step(Network& net) override;
};

class AdamOptimizer : public Optimizer {
private:
    float beta1;
    float beta2;
    float epsilon;
    int time_step;

    // State accumulators: Parallel vectors mirroring the layers' parameter shapes
    // We need a 2D vector structure: [layer_index][parameter_index]
    std::vector<std::vector<Tensor>> m_params;
    std::vector<std::vector<Tensor>> v_params;

    // Lazy-initialize states when step() runs for the first time
    bool is_initialized;

public:
    AdamOptimizer(float learning_rate, float b1 = 0.9f, float b2 = 0.999f, float eps = 1e-8f)
        : Optimizer(learning_rate), beta1(b1), beta2(b2), epsilon(eps), time_step(0), is_initialized(false) {}

    void step(Network& net) override;
};