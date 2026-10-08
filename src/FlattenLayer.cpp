# include "FlattenLayer.hpp"

Tensor FlattenLayer::forward(const Tensor4D& input){
    cached_C = input.C;
    cached_H = input.H;
    cached_W = input.W;

    size_t features = cached_C*cached_H*cached_W;
    
    Tensor output(input.N,features);

    output.data = input.data;

    return output;
}

Tensor4D FlattenLayer::backward(const Tensor& d_output){
    Tensor4D d_input(d_output.rows,cached_C,cached_H,cached_W);

    d_input.data = d_output.data;

    return d_input;
}
