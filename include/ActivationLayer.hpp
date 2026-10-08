#pragma once

#include "Layer.hpp"
#include "Tensor.hpp"
#include "Tensor4D.hpp"
#include "Layer4D.hpp"
#include <vector>

class ReLULayer: public Layer{
private:
    Tensor input_cache;

public:
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& output_gradient) override;

    void ReLULayer::save(std::ofstream& out) const {}
    void ReLULayer::load(std::ifstream& in) {}
};

class SoftmaxLayer: public Layer{
private:
    Tensor cached_output;

public:
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& output_gradient) override;

    void SoftmaxLayer::save(std::ofstream& out) const {}
    void SoftmaxLayer::load(std::ifstream& in) {}  
};

class ReLU4DLayer : public Layer4D{
private:
    Tensor4D input_cache;

public:
    ReLU4DLayer():input_cache(0,0,0,0){};

    Tensor4D forward(const Tensor4D& input) override;
    Tensor4D backward(const Tensor4D& d_output) override;

    void ReLU4DLayer::save(std::ofstream& out) const {}
    void ReLU4DLayer::load(std::ifstream& in) {}
};