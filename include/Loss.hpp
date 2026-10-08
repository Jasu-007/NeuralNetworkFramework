#pragma once

# include <vector>
# include "Tensor.hpp"

class Loss{
public:
    virtual ~Loss() = default;

    virtual float forward(const Tensor& pred,const Tensor& true_vals) = 0;

    virtual Tensor backward(const Tensor& pred,const Tensor& true_vals) = 0;
};

class MSELoss: public Loss{
public:
    float forward(const Tensor& pred,const Tensor& true_vals) override;

    Tensor backward(const Tensor& pred,const Tensor& true_vals) override;
};

class CrossEntropyLoss: public Loss{
public:
    float forward(const Tensor& pred,const Tensor& true_vals) override;

    Tensor backward(const Tensor& pred,const Tensor& true_vals) override;
};