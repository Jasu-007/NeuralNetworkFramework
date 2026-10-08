cmake_minimum_required(VERSION 3.10)
project(CppMLFramework VERSION 1.0 LANGUAGES CXX)

# Force modern C++ standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable aggressive hardware-level optimization flags
if(MSVC)
    add_compile_options(/O2 /arch:AVX2)
else()
    add_compile_options(-O3 -march=native -Wall -Wextra)
endif()

# Identify include paths
include_directories(include)

# Create a library from your core framework source files
add_library(ml_framework
    src/Tensor.cpp
    src/Layer.cpp
    src/DenseLayer.cpp
    src/ActivationLayer.cpp
    src/Loss.cpp
    src/Optimizer.cpp
)

# Compile the final application and link it with your framework library

add_executable(xor_engine apps/xor_main.cpp)
target_link_libraries(xor_engine ml_framework)
target_compile_options(xor_engine apps/xor_main.cpp PRIVATE -mavx2 -mfma)

# 1. Create the executable target (include your tensor source files)
add_executable(multi_test multi_test.cpp tensor.cpp)

# 2. Force the compiler to unlock AVX2 and Fused Multiply-Add hardware lines
if(MSVC)
    target_compile_options(multi_test PRIVATE /arch:AVX2)
else()
    target_compile_options(multi_test PRIVATE -mavx2 -mfma -O3)
endif()