#include "../include/Tensor.hpp" // Ensures it can see your Tensor class definition
#include <iostream>
#include <chrono>
#include <cassert>
#include <cmath>

void run_correctness_test() {
    std::cout << "--- Running Correctness Test ---" << std::endl;
    
    // Matrix dimensions: 10 columns means 8 elements hit SIMD, 2 hit cleanup!
    Tensor A(2, 3);
    Tensor B(3, 10);
    
    // Fill Matrix A with simple values
    int count = 1;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            A(i, j) = (float)(count++);
        }
    }
    
    // Fill Matrix B with simple repeating values
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 10; ++j) {
            B(i, j) = (float)((i + j) % 3 + 1);
        }
    }
    
    // Compute using your new AVX2 multiplication
    Tensor C = A.multiply(B);
    
    // Visually verify a couple of positions
    std::cout << "C(0,0) [Expected: 14]: " << C(0, 0) << std::endl;
    std::cout << "C(1,9) [Expected: 32]: " << C(1, 9) << std::endl;
    std::cout << "✅ If the values match expectations, your pointer offsets are perfect!" << std::endl;
}

void run_performance_benchmark() {
    std::cout << "\n--- Running Performance Benchmark ---" << std::endl;
    
    size_t dim = 500;
    Tensor A(dim, dim);
    Tensor B(dim, dim);
    
    // Initialize matrices
    for (size_t i = 0; i < dim; ++i) {
        for (size_t j = 0; j < dim; ++j) {
            A(i, j) = 1.1f;
            B(i, j) = 0.9f;
        }
    }
    
    // Warm-up run to prime CPU cache
    Tensor warm_up = A.multiply(B);
    
    // Benchmark timed run
    auto start = std::chrono::high_resolution_clock::now();
    
    Tensor C = A.multiply(B);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    
    std::cout << "Time taken to multiply " << dim << "x" << dim << " matrices: " 
              << duration.count() << " ms" << std::endl;
}

int main() {
    run_correctness_test();
    run_performance_benchmark();
    return 0;
}