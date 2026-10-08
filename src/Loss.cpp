# include "Loss.hpp"
# include <cmath>

float MSELoss::forward(const Tensor& pred,const Tensor& true_vals){
    float total_sum = 0.0f;
    int total_elements = pred.rows * pred.cols;

    // Flattening to a single loop maximizes cache efficiency
    for (int i = 0; i < total_elements; i++) {
        float diff = pred.data[i] - true_vals.data[i]; 
        total_sum += diff * diff;
    }

    return total_sum / static_cast<float>(total_elements);
}

Tensor MSELoss::backward(const Tensor& pred,const Tensor& true_vals){
    Tensor gradient(pred.rows,pred.cols);
    int total_elements = pred.rows*pred.cols;

    for (int i=0;i<total_elements;i++){
        gradient.data[i] = (2.0f/static_cast<float>(total_elements))*(pred.data[i]-true_vals.data[i]);
    }
    
    return gradient;
}

float CrossEntropyLoss::forward(const Tensor& pred,const Tensor& true_vals){
    float total_loss = 0.0f;
    for (size_t i=0;i<pred.rows;i++){
        for (size_t j=0;j<pred.cols;j++){
            total_loss+= (true_vals(i,j)*std::log(pred(i,j)+1e-15f));
        }
    }
    return (-1.0f*total_loss)/pred.rows;
}

Tensor CrossEntropyLoss::backward(const Tensor& pred,const Tensor& true_vals){
    Tensor grad_input(pred.rows,pred.cols);

    for (size_t i=0;i<pred.rows;i++){
        for (size_t j=0;j<pred.cols;j++){
            grad_input(i,j) = ((-1.0f*true_vals(i,j))/((pred(i,j)+1e-15f)*pred.rows));
        }
    }

    return grad_input;
}