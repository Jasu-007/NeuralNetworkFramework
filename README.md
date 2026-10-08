# Custom C++ Deep Learning & Computer Vision Framework

A high-performance, modular deep learning framework engineered completely from scratch in C++17—without relying on external linear algebra, matrix, or computer vision libraries. Built with a focus on low-level memory layout control, spatial tensor transformations, SIMD hardware acceleration, and modular execution layers.

---

## 🚀 Framework Architecture & Key Capabilities

* **Hardware-Accelerated SIMD Engine (AVX2 & FMA):** Low-level x86 vectorization integrated into core matrix operations (`Tensor::multiply`), accelerating heavy compute bottlenecks (Conv2D, Dense, and Recurrent layers) across 256-bit SIMD register lanes.
* **Native 4D Tensor Engine ($N \times C \times H \times W$):** Extended contiguous memory operations beyond 2D tabular matrices to natively process 4D spatial data blocks, multi-channel feature maps, and spatial mini-batches.
* **Modular Polymorphic Architecture:** Object-oriented base routing (`Layer`, `Optimizer`, `Loss`) allowing decoupled, plug-and-play assembly of hybrid 4D, 2D, and sequential execution graphs inside an abstract `Network` container wrapper.
* **Computer Vision & CNN Core:**
  * **Convolutional Layer (`Conv2DLayer`):** Supports multi-channel sliding kernels, spatial padding, stride configurations, and SIMD-accelerated patch extractions.
  * **Pooling Layer (`MaxPool2DLayer`):** Implemented spatial downsampling with max-index tracking for backpropagation through non-overlapping sub-regions.
  * **Bridge Layer (`FlattenLayer`):** Zero-copy structural tensor reshaping that transitions 4D spatial maps ($N \times C \times H \times W$) straight into flat feature vectors for fully connected classification.
* **Recurrent Sequential Core:**
  * **Recurrent Layer (`RNNLayer`):** Vanilla RNN architecture featuring Backpropagation Through Time (BPTT), internal hidden memory state caching ($h_t$), and custom temporal vector sequence processing.
* **Regularization & Weight Serialization:**
  * **Weight Initialization:** He (Kaiming) Normal and Xavier (Glorot) Normal initialization for variance-scaled gradient control across ReLU and Tanh layers.
  * **Binary Serialization:** Native binary export/import (`save`/`load` routines via `std::ofstream`/`std::ifstream`) for parameter checkpointing and model persistence.
  * **Adaptive Learning Rate Scheduler:** Exponential decay routines integrated into optimizer loops to balance fast initial convergence with fine-grained loss minimization.
* **Optimizers & Loss Functions:**
  * **Optimizers:** Stochastic Gradient Descent (SGD) and adaptive Adam Optimization.
  * **Loss Engines:** Categorical Cross-Entropy and Mean Squared Error (MSE).

---

## 📊 Framework Benchmarks & Verification

To verify mathematical validity, spatial backward pass accuracy, and numerical stability, the custom CNN pipeline was benchmarked against the full MNIST dataset.

### 🔢 Spatial Vision Benchmark: MNIST Handwritten Digit Classification

* **Dataset Scale:** 60,000 Training Samples | 10,000 Independent Testing Samples
* **Pipeline Topology:** 
  $$\text{Input } (1 \times 28 \times 28) \longrightarrow \text{Conv2D } (16 \text{ filters}, 3 \times 3) \longrightarrow \text{ReLU} \longrightarrow \text{MaxPool2D } (2 \times 2) \longrightarrow \text{Flatten} \longrightarrow \text{Dense } (2704 \to 64) \longrightarrow \text{ReLU} \longrightarrow \text{Dense } (64 \to 10) \longrightarrow \text{Softmax}$$
* **Optimizer:** Adam ($\alpha_0 = 0.003$ with Exponential Alpha Decay every 2 epochs)
* **Loss Function:** Categorical Cross-Entropy
* **Training Strategy:** Mini-Batch Gradient Descent (Batch Size: 32 / 64)

| Metric | 2D Tabular Baseline | Custom 4D CNN Engine |
| :--- | :--- | :--- |
| **Input Tensor Format** | Flattened Array ($1 \times 784$) | Native 4D Spatial Tensor ($N \times 1 \times 28 \times 28$) |
| **Feature Extraction** | Fully Connected | 16 Spatial Feature Maps + Max Pooling |
| **Vector Acceleration** | Scalar Fallback | **AVX2 / FMA 256-bit SIMD Lanes** |
| **Training Epochs** | 10 Epochs | 10 Epochs |
| **Final Training Loss** | 0.0268 | **0.0019** |
| **Final Training Accuracy** | 97.50% | **99.99%** |
| **Out-of-Sample Test Accuracy** | 97.48% | **98.56%** |

---

## 📁 Repository Blueprint

```text
.
├── include/                   # Interface Definitions & Header Blueprints
│   ├── Tensor.hpp             # 2D & 4D Dynamic Tensor Storage Engine (AVX2 Enabled)
│   ├── Layer.hpp              # Polymorphic Layer Base Blueprint
│   ├── Conv2DLayer.hpp        # 2D Spatial Convolution Layer
│   ├── MaxPool2DLayer.hpp     # Spatial Sub-sampling & Pooling
│   ├── FlattenLayer.hpp       # 4D-to-2D Dimension Reshaping Bridge
│   ├── DenseLayer.hpp         # Fully Connected Classification Layer
│   ├── RNNLayer.hpp           # Recurrent Layer with BPTT Memory Pipeline
│   ├── ActivationLayer.hpp    # Non-linear Activations (ReLU / Tanh)
│   ├── SoftmaxLayer.hpp       # Categorical Probabilistic Output Layer
│   ├── Optimizer.hpp          # Polymorphic Optimization Interface
│   ├── AdamOptimizer.hpp      # Adaptive Moment Tracking Optimizer
│   ├── Loss.hpp               # Objective Function Interface
│   ├── CrossEntropyLoss.hpp   # Multi-class Classification Loss
│   └── Utils.hpp              # Dataset Stream Parsing & Normalization
│
├── src/                       # Core Framework Logic Implementation
│   ├── Tensor.cpp             # SIMD Accelerated Vector Instructions
│   ├── Conv2DLayer.cpp
│   ├── MaxPool2DLayer.cpp
│   ├── FlattenLayer.cpp
│   ├── DenseLayer.cpp
│   ├── RNNLayer.cpp           # Backpropagation Through Time Implementation
│   ├── SoftmaxLayer.cpp
│   ├── CrossEntropyLoss.cpp
│   └── Utils.cpp
│
├── app/                       # Execution Routines & Application Drivers
│   ├── mnist_test_CNN.cpp     # Full 4D Convolutional Training Loop
│   ├── rnn_test.cpp           # Recurrent Sequence Verification Routine
│   ├── rnn_data_test.cpp      # Character Transformation & Memory Inspection
│   ├── multi_test.cpp         # SIMD Benchmark & Execution Test
│   └── xor_main.cpp           # Non-linear Logic Gate Verification
│
└── data/                      # Local Datasets Directory
    └── mnist/                 # Normalized MNIST Image Matrices
```
---
## 🛠️ Build & Execution Instructions

### Prerequisites
* GCC/Clang with C++17 support and **AVX2/FMA compiler flags**.
* CMake (v3.12 or higher).

### Compiling with CMake
```powershell
# Clone the repository
git clone [https://github.com/naitikjoshi188/NeuralNetworkFramework.git](https://github.com/naitikjoshi188/NeuralNetworkFramework.git)
cd NeuralNetworkFramework

# Create a build directory
mkdir build
cd build

# Configure and compile
cmake ..
cmake --build .
```
---
## 🛣️ Development Roadmap

* [x] Native 2D Tabular Dense Engine & Basic Optimizers
* [x] Multi-Channel 4D Tensor Engine ($N \times C \times H \times W$)
* [x] Conv2D, MaxPool2D, and Flatten Layer Integration
* [x] He & Xavier Weight Initialization + Exponential Learning Rate Decay
* [x] **AVX2 / FMA SIMD Hardware Vectorization** for Core Matrix Operations
* [x] **Recurrent Neural Network (`RNNLayer`) Engine** with Backpropagation Through Time (BPTT)
* [x] **Weight Serialization:** Binary exporting (`std::ofstream`/`std::ifstream`) for model saving and loading
* [ ] **Element-wise SIMD Extensions:** Vectorizing ReLU, Tanh, and MaxPool operations using `_mm256_max_ps`
* [ ] **Gated Recurrent Architectures:** Extending sequential processing to LSTM / GRU cells to mitigate vanishing gradients over long sequences
