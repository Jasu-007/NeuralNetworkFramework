# include "ActivationLayer.hpp"
# include <algorithm>
# include <cmath>

Tensor ReLULayer::forward(const Tensor& input){
    input_cache = input;

    Tensor result(input.rows,input.cols);

    for (int i=0;i<input.rows;i++){
        for (int j=0;j<input.cols;j++){
            result(i,j) = (input(i,j)>0)?input(i,j):0.0f;
        }
    }
    return result;
}

Tensor ReLULayer::backward(const Tensor& output_gradient){
    Tensor input_gradient(output_gradient.rows,output_gradient.cols);

    for (int i=0;i<output_gradient.rows;i++){
        for (int j=0;j<output_gradient.cols;j++){
            if (input_cache(i,j)>0){
                input_gradient(i,j)=output_gradient(i,j);
            }
        }
    }

    return input_gradient;
}

Tensor SoftmaxLayer::forward(const Tensor& input){
    Tensor result(input.rows,input.cols);

    for (size_t i=0;i<input.rows;i++){
        float maxi = input(i,0);
        for (size_t j=1;j<input.cols;j++){
            maxi = std::max(maxi,input(i,j));
        }
        float sum = 0;
        for (size_t j=0;j<input.cols;j++){
            result(i,j) = std::exp(input(i,j)-maxi);
            sum+=result(i,j);
        }
        for (size_t j=0;j<input.cols;j++){
            result(i,j)/=sum;
        }
    }

    cached_output = result;

    return result;
}

Tensor SoftmaxLayer::backward(const Tensor& output_gradient){
    Tensor grad_input (output_gradient.rows,output_gradient.cols);

    for (size_t i=0;i<output_gradient.rows;i++){
        float dot_product = 0;
        for (size_t k=0;k<output_gradient.cols;k++){
            dot_product+= (output_gradient(i,k)*cached_output(i,k));
        }
        for (size_t j=0;j<output_gradient.cols;j++){
            grad_input(i, j) = cached_output(i, j) * (output_gradient(i, j) - dot_product);
        }
    }
    return grad_input;
}

Tensor4D ReLU4DLayer::forward(const Tensor4D& input){
    input_cache = input;

    Tensor4D output(input.N,input.C,input.H,input.W);
    size_t total_elements = input.data.size();
    for (size_t i=0;i<total_elements;i++){
        if (input.data[i]>0){
            output.data[i]=input.data[i];
        }
    }

    return output;
}

Tensor4D ReLU4DLayer::backward(const Tensor4D& d_output){
    Tensor4D d_input(d_output.N,d_output.C,d_output.H,d_output.W);

    for (size_t i=0;i<input_cache.data.size();i++){
        if (input_cache.data[i]>0){
            d_input.data[i] = d_output.data[i];
        }
    }

    return d_input;
}
