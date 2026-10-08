# include "Tensor.hpp"
# include <iostream>
# include <cassert>
# include <random>
# include <immintrin.h> // Required header for AVX2 intrinsics
# include <cassert>


//defining constructors
Tensor::Tensor(): rows(0),cols(0){}

Tensor::Tensor(int r,int c): rows(r),cols(c),data(r*c,0.0f){}

Tensor::Tensor(int r,int c,const std::vector<float>& values): rows(r),cols(c),data(values){
    assert(r*c==values.size());
}


//defining transpose
Tensor Tensor::transpose()const{
    Tensor result(cols,rows);
    for (int i=0;i<rows;i++){
        for (int j=0;j<cols;j++){
            result(j,i) = (*this)(i,j);
        }
    }
    return result;
}


//defining print function
void Tensor::print()const{
    std::cout<<"["<<rows<<"*"<<cols<<"] tensor:\n";
    for (int i=0;i<rows;i++){
        std::cout<<"[";
        for (int j=0;j<cols;j++){
            std::cout<<(*this)(i,j)<<" ";
        }
        std::cout<<"]\n";
    }
    std::cout<<std::endl;
}


//defining multiply function and safegaurding using assert
Tensor Tensor::multiply(const Tensor& B) const {
    assert(cols == B.rows);
    
    Tensor result(rows, B.cols);

    for (int i = 0; i < rows; i++) {
        for (int k = 0; k < cols; k++) {
            // Cache our scalar multiplier from Matrix A
            float val_A = (*this)(i, k);
            
            // 1. Broadcast val_A into all 8 slots of a 256-bit vector register
            // Register state: [val_A, val_A, val_A, val_A, val_A, val_A, val_A, val_A]
            __m256 v_A = _mm256_set1_ps(val_A);
            
            int j = 0;
            
            // 2. The SIMD Vector Engine
            // Step through columns in blocks of 8 elements
            for (; j <= B.cols - 8; j += 8) {
                // Get direct pointers to the 1D memory array locations
                float* res_ptr = &result(i, j);
                const float* B_ptr = &B(k, j);

                // Load 8 contiguous floats from memory into registers
                __m256 v_res = _mm256_loadu_ps(res_ptr); // 'u' handles unaligned memory safely
                __m256 v_B   = _mm256_loadu_ps(B_ptr);

                // Fused Multiply-Add: v_res = v_res + (v_A * v_B)
                v_res = _mm256_fmadd_ps(v_A, v_B, v_res);

                // Store the updated 8 parallel results straight back to memory
                _mm256_storeu_ps(res_ptr, v_res);
            }

            // 3. The Scalar Cleanup Tail
            // If B.cols isn't a multiple of 8, process remaining elements one-by-one
            for (; j < B.cols; j++) {
                result(i, j) += val_A * B(k, j);
            }
        }
    }
    return result;
}

//defining random initializer
void Tensor::randomize_normal(float mean, float stddev) {
    // Using a fixed seed (e.g., 1337) ensures the network behaves identically 
    // every time we run it.
    static std::default_random_engine generator(1337);
    std::normal_distribution<float> distribution(mean, stddev);

    for (int i = 0; i < rows * cols; ++i) {
        data[i] = distribution(generator);
    }
}