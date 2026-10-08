# include "Conv2DLayer.hpp"
#include <random>
#include <cmath>
# include <stdexcept>
# include <algorithm>
# include <immintrin.h>

Conv2DLayer::Conv2DLayer(size_t in_c, size_t out_c, size_t k_size, size_t s, size_t p)
:   in_channels(in_c),
    out_channels(out_c),
    kernel_size(k_size),
    stride(s),
    padding(p),
    // Constructing 4D tensors (initially default or zero-allocated)
    weights(out_c, in_c, k_size, k_size),
    d_weights(out_c, in_c, k_size, k_size),
    // Sizing bias vectors
    biases(out_c, 0.0f),
    d_biases(out_c, 0.0f),
    input_cache(0, 0, 0, 0)
{
    // 1. Calculate fan_in for He Initialization 
    // fan_in = input_channels * kernel_height * kernel_width
    float fan_in = static_cast<float>(in_c * k_size * k_size);
    float std_dev = std::sqrt(2.0f / fan_in);

    // 2. Setup standard normal distribution random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, std_dev);

    // 3. Populate the weights Tensor4D data vector with random values
    // Accessing your underlying weight data vector directly
    for (size_t i = 0; i < weights.data.size(); ++i) {
        weights.data[i] = dist(gen);
    }
}


Tensor4D Conv2DLayer::forward(const Tensor4D& input){
    input_cache = input;
    
    size_t N = input.N,H=input.H,W=input.W;
    size_t H_out = H - kernel_size +1;
    size_t W_out = W - kernel_size +1;

    Tensor4D output(N,out_channels,H_out,W_out);

    // 1. Initialize the output tensor with bias values upfront
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < out_channels; j++) {
            float b = biases[j];
            __m256 v_bias = _mm256_set1_ps(b);
            size_t m = 0;
            
            // Fast bias initialization via SIMD
            for (; m <= H_out * W_out - 8; m += 8) {
                _mm256_storeu_ps(&output(i, j, 0, 0) + m, v_bias);
            }
            for (; m < H_out * W_out; m++) {
                (&output(i, j, 0, 0))[m] = b;
            }
        }
    }

    // 2. Optimized Convolution Loops
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < out_channels; j++) {
            for (size_t l = 0; l < H_out; l++) {
                for (size_t c_in = 0; c_in < in_channels; c_in++) {
                    for (size_t kh = 0; kh < kernel_size; kh++) {
                        for (size_t kw = 0; kw < kernel_size; kw++) {
                            
                            // Cache our kernel weight scalar and broadcast it to 8 slots
                            float w_val = weights(j, c_in, kh, kw);
                            __m256 v_W = _mm256_set1_ps(w_val);
                            
                            size_t m = 0;
                            // 3. Inner Vector Engine: Sweep across output image row in blocks of 8
                            for (; m <= W_out - 8; m += 8) {
                                // Direct raw memory pointer addresses
                                float* out_ptr = &output(i, j, l, m);
                                const float* in_ptr = &input(i, c_in, l + kh, m + kw);

                                // Load current states
                                __m256 v_out = _mm256_loadu_ps(out_ptr);
                                __m256 v_in  = _mm256_loadu_ps(in_ptr);

                                // Accumulate: output = output + (input * weight)
                                v_out = _mm256_fmadd_ps(v_in, v_W, v_out);

                                // Store back
                                _mm256_storeu_ps(out_ptr, v_out);
                            }

                            // 4. Scalar Cleanup Loop
                            for (; m < W_out; m++) {
                                output(i, j, l, m) += input(i, c_in, l + kh, m + kw) * w_val;
                            }
                            
                        }
                    }
                }
            }
        }
    }
    return output;
}

Tensor4D Conv2DLayer::backward(const Tensor4D& d_output) {
    size_t N = input_cache.N, H = input_cache.H, W = input_cache.W;
    size_t H_out = H - kernel_size + 1;
    size_t W_out = W - kernel_size + 1;

    Tensor4D d_input(N, in_channels, H, W);
    
    d_weights.fill_zero();
    std::fill(d_biases.begin(), d_biases.end(), 0.0f);

    // 1. Clean Bias Isolation Pass (No 'if' conditions inside the main convolution)
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < out_channels; j++) {
            float bias_grad = 0.0f;
            for (size_t l = 0; l < H_out; l++) {
                size_t m = 0;
                // SIMD accumulate bias gradients along the row
                __m256 v_bias_sum = _mm256_setzero_ps();
                for (; m <= W_out - 8; m += 8) {
                    __m256 v_dout = _mm256_loadu_ps(&d_output(i, j, l, m));
                    v_bias_sum = _mm256_add_ps(v_bias_sum, v_dout);
                }
                // Horizontal add of the vector register lanes
                float align_buf[8];
                _mm256_storeu_ps(align_buf, v_bias_sum);
                for (int b = 0; b < 8; b++) bias_grad += align_buf[b];
                
                // Cleanup tail for bias
                for (; m < W_out; m++) {
                    bias_grad += d_output(i, j, l, m);
                }
            }
            d_biases[j] += bias_grad;
        }
    }

    // 2. Optimized Gradient Computation Loops
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < out_channels; j++) {
            for (size_t l = 0; l < H_out; l++) {
                for (size_t c_in = 0; c_in < in_channels; c_in++) {
                    for (size_t kh = 0; kh < kernel_size; kh++) {
                        for (size_t kw = 0; kw < kernel_size; kw++) {
                            
                            // Cache our static weight factor for d_input calculation
                            float w_val = weights(j, c_in, kh, kw);
                            __m256 v_W = _mm256_set1_ps(w_val);
                            
                            size_t m = 0;
                            __m256 v_dw_acc = _mm256_setzero_ps();

                            // 3. Inner SIMD Engine: Processing columns along W_out
                            for (; m <= W_out - 8; m += 8) {
                                // Load d_output and input cache segments
                                __m256 v_dout = _mm256_loadu_ps(&d_output(i, j, l, m));
                                __m256 v_in   = _mm256_loadu_ps(&input_cache(i, c_in, l + kh, m + kw));
                                
                                // Accumulate d_weights update into vector register lanes: v_dw_acc += v_dout * v_in
                                v_dw_acc = _mm256_fmadd_ps(v_dout, v_in, v_dw_acc);

                                // Compute and update d_input memory values: d_input += v_dout * v_W
                                float* din_ptr = &d_input(i, c_in, l + kh, m + kw);
                                __m256 v_din = _mm256_loadu_ps(din_ptr);
                                v_din = _mm256_fmadd_ps(v_dout, v_W, v_din);
                                _mm256_storeu_ps(din_ptr, v_din);
                            }

                            // Horizontal reduce the vector register lanes into a single scalar value for d_weights
                            float dw_buf[8];
                            _mm256_storeu_ps(dw_buf, v_dw_acc);
                            for (int b = 0; b < 8; b++) {
                                d_weights(j, c_in, kh, kw) += dw_buf[b];
                            }

                            // 4. Scalar Cleanup Tail
                            for (; m < W_out; m++) {
                                float dout_val = d_output(i, j, l, m);
                                d_weights(j, c_in, kh, kw) += dout_val * input_cache(i, c_in, l + kh, m + kw);
                                d_input(i, c_in, l + kh, m + kw)  += dout_val * w_val;
                            }

                        }
                    }
                }
            }
        }
    }
    return d_input;
}

void Conv2DLayer::update_parameters(float learning_rate){
    for (size_t o=0;o<out_channels;o++){
        for (size_t i=0;i<in_channels;i++){
            for (size_t kh=0;kh<kernel_size;kh++){
                for (size_t kw=0;kw<kernel_size;kw++){
                    weights(o,i,kh,kw) -= learning_rate*d_weights(o,i,kh,kw);
                }
            }
        }
    }

    for (size_t o=0;o<out_channels;o++){
        biases[o] -= learning_rate*d_biases[o];
    }
}

void Conv2DLayer::save(std::ofstream& out) const {
    // Writing the metadata header for structural validation
    out.write(reinterpret_cast<const char*>(&out_channels), sizeof(out_channels));
    out.write(reinterpret_cast<const char*>(&kernel_size), sizeof(kernel_size));

    out.write(reinterpret_cast<const char*>(weights.data.data()), weights.data.size() * sizeof(float));
    
    out.write(reinterpret_cast<const char*>(biases.data()), biases.size() * sizeof(float));
}

void Conv2DLayer::load(std::ifstream& in) {
    size_t in_filters, in_kernel_size;
    
    // 1. Reading metadata from file
    in.read(reinterpret_cast<char*>(&in_filters), sizeof(in_filters));
    in.read(reinterpret_cast<char*>(&in_kernel_size), sizeof(in_kernel_size));

    // 2. Structural safety verification
    if (in_filters != out_channels || in_kernel_size != kernel_size) {
        throw std::runtime_error("Dimension mismatch while loading Conv2DLayer weights!");
    }

    in.read(reinterpret_cast<char*>(weights.data.data()), weights.data.size() * sizeof(float));
    in.read(reinterpret_cast<char*>(biases.data()), biases.size() * sizeof(float));
}