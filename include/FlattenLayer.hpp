# pragma once

# include "Tensor4D.hpp"
# include "Tensor.hpp"

class FlattenLayer{
private:
    size_t cached_C =0;
    size_t cached_H =0;
    size_t cached_W =0;

public:
    FlattenLayer() = default;

    Tensor forward(const Tensor4D& input);
    Tensor4D backward(const Tensor& d_ouput);
};