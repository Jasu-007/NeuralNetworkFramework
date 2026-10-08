#include <iostream>
#include <cmath>
#include "Tensor.hpp"
#include "Layer.hpp"
#include "DenseLayer.hpp"
#include "ActivationLayer.hpp"
#include "Optimizer.hpp"
#include "Loss.hpp"
#include "Utils.hpp"

int main() {
    std::cout << "==================================================\n";
    std::cout << "Starting Full MNIST Deep Learning Vision Pipeline\n";
    std::cout << "==================================================\n";

    // 1. Setup Data Preprocessing Space
    Tensor X_train, Y_train;
    Tensor X_test, Y_test;
    
    // Load Training Dataset (Label sits at column index 0)
    std::cout << "Loading MNIST Training Set [data/mnist_train.csv]..." << std::endl;
    DataUtils::load_csv("data/mnist_train.csv", X_train, Y_train, 0);
    std::cout << "» Loaded " << X_train.rows << " training samples.\n\n";

    // Load Testing Dataset
    std::cout << "Loading MNIST Testing Set [data/mnist_test.csv]..." << std::endl;
    DataUtils::load_csv("data/mnist_test.csv", X_test, Y_test, 0);
    std::cout << "» Loaded " << X_test.rows << " independent testing samples.\n\n";

    // Normalize pixel intensity maps (0-255 -> 0.0-1.0)
    std::cout << "Executing feature scaling normalization..." << std::endl;
    DataUtils::normalize_features(X_train);
    DataUtils::normalize_features(X_test);

    // One-Hot Encode labels for categorical cross-entropy tracking
    Tensor Y_train_hot = DataUtils::one_hot_encoder(Y_train, 10);
    Tensor Y_test_hot  = DataUtils::one_hot_encoder(Y_test, 10);

    // 2. Building the Network Topology
    Network net;
    net.add(new DenseLayer(784, 128));
    net.add(new ReLULayer());
    net.add(new DenseLayer(128, 64));
    net.add(new ReLULayer());
    net.add(new DenseLayer(64, 10));
    net.add(new SoftmaxLayer());

    CrossEntropyLoss loss_fn;
    AdamOptimizer optimizer(0.003f);

    std::cout << "\nArchitecture loaded. Commencing network optimization...\n";

    // Hyperparameters for Mini-Batching
    int batch_size = 64; 
    int total_epochs = 10; // Updates happen multiple times per epoch!
    int num_batches = std::ceil(static_cast<float>(X_train.rows) / batch_size);

    std::cout << "\nStarting Mini-Batch execution loop..." << std::endl;
    std::cout << "Total Samples: " << X_train.rows << " | Batch Size: " << batch_size << " | Steps per Epoch: " << num_batches << "\n\n";

    // 3. The Core Optimization Loop
    for (int epoch = 1; epoch <= total_epochs; ++epoch) {
        float epoch_loss = 0.0f;

        for (int b = 0; b < num_batches; ++b) {
            // Slice out the current mini-batch tensors cleanly using the new tool
            Tensor X_batch = DataUtils::get_batch(X_train, b, batch_size);
            Tensor Y_batch = DataUtils::get_batch(Y_train_hot, b, batch_size);

            // Forward Pass on mini-batch step
            Tensor Y_pred = net.forward(X_batch);
            epoch_loss += loss_fn.forward(Y_pred, Y_batch);

            // Backpropagation & Weight Update per batch step
            Tensor loss_grad = loss_fn.backward(Y_pred, Y_batch);
            net.backward(loss_grad);
            optimizer.step(net);
        }

        // Average loss over all processed steps
        epoch_loss /= num_batches;
        std::cout << "Epoch " << epoch << "/" << total_epochs << " | Avg Training Loss: " << epoch_loss << "\n";
    }

    // Run Testing Generalization Evaluation on Unseen Test File
    std::cout << "\n==================================================\n";
    std::cout << "Evaluating Performance on Full Out-of-Sample Test Set...\n";
    std::cout << "==================================================\n";
    
    Tensor test_preds = net.forward(X_test);
    int correct_test_predictions = 0;

    for (size_t i = 0; i < X_test.rows; ++i) {
        int true_digit = static_cast<int>(Y_test(i, 0));
        
        int predicted_digit = 0;
        float max_prob = test_preds(i, 0);
        for (size_t c = 1; c < test_preds.cols; ++c) {
            if (test_preds(i, c) > max_prob) {
                max_prob = test_preds(i, c);
                predicted_digit = c;
            }
        }

        if (predicted_digit == true_digit) {
            correct_test_predictions++;
        }
    }

    float test_accuracy = (static_cast<float>(correct_test_predictions) / X_test.rows) * 100.0f;
    std::cout << ">> Mini-Batched Test Accuracy: " << test_accuracy << "%\n";

    return 0;
}