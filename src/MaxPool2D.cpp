# include "MaxPool2D.hpp"
# include <limits>
# include <cmath>

MaxPool2D::MaxPool2D(size_t p_size,size_t s)
:   pool_size(p_size),stride(s),input_cache(0,0,0,0)
{}

Tensor4D MaxPool2D::forward(const Tensor4D& input){
    input_cache = input;  //caching for backward pass

    size_t N=input.N,C=input.C,H=input.H,W=input.W;
    size_t H_out = (H-pool_size)/(stride)+1;
    size_t W_out = (W-pool_size)/(stride)+1;

    Tensor4D output(N,C,H_out,W_out);

    for (size_t i=0;i<N;i++){
        for (size_t j=0;j<C;j++){
            for (size_t k=0;k<H_out;k++){
                for (size_t l=0;l<W_out;l++){
                    size_t start_h = k*stride;
                    size_t start_w = l*stride;
                    float max_val = -std::numeric_limits<float>::infinity();
                    for (size_t m=0;m<pool_size;m++){
                        for (size_t n=0;n<pool_size;n++){
                            max_val = std::max(max_val,input(i,j,start_h+m,start_w+n));
                        }
                    }
                    output(i,j,k,l) = max_val;
                }
            }
        }
    }
    
    return output;
}

Tensor4D MaxPool2D::backward(const Tensor4D& d_output){
    Tensor4D d_input(input_cache.N,input_cache.C,input_cache.H,input_cache.W);
    for (size_t i=0;i<d_output.N;i++){
        for (size_t j=0;j<d_output.C;j++){
            for (size_t k=0;k<d_output.H;k++){
                for (size_t l=0;l<d_output.W;l++){
                    size_t start_h = k*stride;
                    size_t start_w = l*stride;

                    size_t max_h = start_h;
                    size_t max_w = start_w;
                    float max_val = -std::numeric_limits<float>::infinity();

                    for (size_t m = 0; m < pool_size; m++) {
                        for (size_t n = 0; n < pool_size; n++) {
                            size_t curr_h = start_h + m;
                            size_t curr_w = start_w + n;
                            
                            if (input_cache(i, j, curr_h, curr_w) > max_val) {
                                max_val = input_cache(i, j, curr_h, curr_w);
                                max_h = curr_h;
                                max_w = curr_w;
                            }
                        }
                    }

                    d_input(i, j, max_h, max_w) += d_output(i, j, k, l);
                }
            }
        }
    }
    return d_input;
}