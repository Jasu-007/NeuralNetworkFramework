# include "DenseLayer.hpp"
# include "Tensor.hpp"
# include <random>
# include <cmath>
# include <stdexcept>

// 1. Constructor with Random Xavier/Glorot Initialization
DenseLayer::DenseLayer(int input_dim, int output_dim) 
    : input_dim(input_dim),output_dim(output_dim),weights(input_dim, output_dim), biases(1, output_dim) {
    
    // Calculate Xavier/Glorot standard deviation and initializing weights
    float stddev = std::sqrt(2.0f / (input_dim + output_dim));
    weights.randomize_normal(0.0f, stddev);
    
}

//defining forward and backward passes

Tensor DenseLayer::forward(const Tensor& input){
    input_cache = input;
    Tensor result = input.multiply(weights);
    for (int r = 0; r < result.rows; ++r) {
        for (int c = 0; c < result.cols; ++c) {
            result(r, c) += biases(0, c);
        }
    }
    return result;
}

Tensor DenseLayer::backward(const Tensor& output_gradient){
    Tensor input_gradient = output_gradient.multiply(weights.transpose());
    weight_gradient = (input_cache.transpose()).multiply(output_gradient);

    // Calculate bias_gradient by summing columns across all batch rows
    bias_gradient = Tensor(1, biases.cols); 
    for (int r = 0; r < output_gradient.rows; ++r) {
        for (int c = 0; c < output_gradient.cols; ++c) {
            bias_gradient(0, c) += output_gradient(r, c);
        }
    }

    return input_gradient;
}

std::vector<Tensor*> DenseLayer::get_parameters(){
    return {&weights,&biases};
}

std::vector<Tensor*> DenseLayer::get_gradients(){
    return {&weight_gradient,&bias_gradient};
}

void DenseLayer::save(std::ofstream& out) const{
    // 1. Writing the dimensions first so we can validate on load
    out.write(reinterpret_cast<const char*>(&input_dim), sizeof(input_dim));
    out.write(reinterpret_cast<const char*>(&output_dim), sizeof(output_dim));

    out.write(reinterpret_cast<const char*>(weights.data.data()), weights.data.size() * sizeof(float));
    
    out.write(reinterpret_cast<const char*>(biases.data.data()), biases.data.size() * sizeof(float));
}

void DenseLayer::load(std::ifstream& in) {
    size_t in_dim, out_dim;
    // Read the metadata header
    in.read(reinterpret_cast<char*>(&in_dim), sizeof(in_dim));
    in.read(reinterpret_cast<char*>(&out_dim), sizeof(out_dim));

    // Structural Validation
    if (in_dim != input_dim || out_dim != output_dim) {
        throw std::runtime_error("Dimension mismatch while loading DenseLayer weights!");
    }

    in.read(reinterpret_cast<char*>(weights.data.data()), weights.data.size() * sizeof(float));
    in.read(reinterpret_cast<char*>(biases.data.data()), biases.data.size() * sizeof(float));
}
