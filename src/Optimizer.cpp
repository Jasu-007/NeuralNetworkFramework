# include "Optimizer.hpp"
# include <vector>
# include <cmath>

void SGDOptimizer::step(Network& net){
    std::vector<Layer*> layers = net.get_layers();
    for (const auto& it:layers){
        std::vector<Tensor*> params = it->get_parameters();
        std::vector<Tensor*> grads = it->get_gradients();
        for (size_t i=0;i<params.size();i++){
            Tensor* p = params[i];
            Tensor* g = grads[i];
            int total_elements = p->rows*p->cols;
            for (int j=0;j<total_elements;j++){
                p->data[j] -= lr * g->data[j];
            }
        }
    }
}

void AdamOptimizer::step(Network& net){
    std::vector<Layer*> layers = net.get_layers();
    if (!is_initialized){
        m_params.resize(layers.size());
        v_params.resize(layers.size());

        for (size_t i=0;i<layers.size();i++){
            std::vector<Tensor*> params = layers[i]->get_parameters();
            m_params[i].reserve(params.size());
            v_params[i].reserve(params.size());

            for (size_t j=0;j<params.size();j++){
                m_params[i].push_back(Tensor (params[j]->rows,params[j]->cols));
                v_params[i].push_back(Tensor (params[j]->rows,params[j]->cols));
            }
        }
        
        is_initialized = true;
    }

    time_step++;

    float correction1 = 1.0f - std::pow(beta1,time_step);
    float correction2 = 1.0f - std::pow(beta2,time_step);

    for (size_t i=0;i<layers.size();i++){
        std::vector<Tensor*> params = layers[i]->get_parameters();
        std::vector<Tensor*> grads = layers[i]->get_gradients();
        for (size_t k=0;k<params.size();k++){
            Tensor* param_tensor = params[k];
            Tensor* grad_tensor = grads[k];

            Tensor& m_tensor = m_params[i][k];
            Tensor& v_tensor = v_params[i][k];

            int total_elements = param_tensor->rows*param_tensor->cols;

            for (int j=0;j<total_elements;j++){
                float g = grad_tensor->data[j];

                m_tensor.data[j] = (beta1*m_tensor.data[j])+(1.0f-beta1)*g;
                v_tensor.data[j] = (beta2*v_tensor.data[j])+(1.0f-beta2)*g*g;

                float m_hat = m_tensor.data[j]/correction1;
                float v_hat = v_tensor.data[j]/correction2;

                param_tensor->data[j] -= (lr*m_hat)/(epsilon+std::sqrt(v_hat));
            }
        }
    }
}