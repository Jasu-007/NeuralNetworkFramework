#include <iostream>
#include "Tensor.hpp"
#include "Layer.hpp"
#include "DenseLayer.hpp"
#include "ActivationLayer.hpp"
#include "Optimizer.hpp"
#include "Loss.hpp"
#include "Utils.hpp"

int main() {
    std::cout << "==================================================\n";
    std::cout << "Starting Multi-Class Medical Diagnosis Simulation\n";
    std::cout << "==================================================\n";

    // 1. Setup Data Preprocessing
    Tensor X_raw, Y_raw;
    
    // Load dataset from the data folder (Label column index is 5)
    DataUtils::load_csv("data/medical_data.csv", X_raw, Y_raw, 5);
    
    std::cout << "Dataset Shape: " << X_raw.rows << " Samples, " << X_raw.cols << " Biometric Features.\n";

    // Apply normalization & conversion utilities
    DataUtils::normalize_features(X_raw);
    Tensor Y_one_hot = DataUtils::one_hot_encoder(Y_raw, 3); // 3 Unique diagnosis classes

    // 2. Building the Network Container
    Network net;

    // Input (5 features) -> Hidden Layer (16 nodes) -> ReLU
    net.add(new DenseLayer(5, 16));
    net.add(new ReLULayer());
    // Hidden Layer (16 nodes) -> Output Layer (3 classes) -> Softmax
    net.add(new DenseLayer(16, 3));
    net.add(new SoftmaxLayer());

    // 3. Initialize Loss and your new Adam Optimizer
    CrossEntropyLoss loss_fn;
    AdamOptimizer optimizer(0.01f); // Plugs directly into the Optimizer base class pointer

    std::cout << "Framework components locked via Virtual Base Classes. Training...\n\n";

    // 4. The Core Training Loop (Exactly like XOR)
    int total_epochs = 600;
    for (int epoch = 1; epoch <= total_epochs; ++epoch) {
        // Step A: Forward Pass
        Tensor Y_pred = net.forward(X_raw);

        // Step B: Compute Loss Cost
        float loss = loss_fn.forward(Y_pred, Y_one_hot);

        // Step C: Backward Pass (Backpropagation)
        Tensor loss_grad = loss_fn.backward(Y_pred, Y_one_hot);
        net.backward(loss_grad);

        // Step D: Update Parameters (Adam)
        optimizer.step(net);

        // Print progress every 100 epochs
        if (epoch % 100 == 0 || epoch == 1) {
            std::cout << "Epoch " << epoch << " | Cross-Entropy Loss: " << loss << "\n";
        }
    }

    // 5. Verify and Validate the Final Categorical Predictions
    std::cout << "\n==================================================\n";
    std::cout << "Training Complete! Validating Class Predictions:\n";
    std::cout << "==================================================\n";
    
    Tensor final_preds = net.forward(X_raw);
    for (size_t i = 0; i < 5; ++i) { // Display first 5 patients for verification
        int true_class = static_cast<int>(Y_raw(i, 0));
        
        // Find argmax column to decode the prediction
        int predicted_class = 0;
        float max_prob = final_preds(i, 0);
        for (size_t c = 1; c < final_preds.cols; ++c) {
            if (final_preds(i, c) > max_prob) {
                max_prob = final_preds(i, c);
                predicted_class = c;
            }
        }

        std::cout << "Patient " << i << " | True Diagnosis Class: " << true_class 
                  << " | Engine Output Prediction: " << predicted_class 
                  << " (Confidence: " << max_prob * 100.0f << "%)\n";
    }

    return 0;
}