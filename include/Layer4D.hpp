#pragma once

#include <vector>
#include "Tensor4D.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>

class Layer4D {
public:
    virtual ~Layer4D() = default;

    virtual Tensor4D forward(const Tensor4D& input) = 0;
    virtual Tensor4D backward(const Tensor4D& output_gradient) = 0;

     //defining virtual methods to save and load weights
    virtual void save(std::ofstream& out) const = 0;
    virtual void load(std::ifstream& in) = 0;
};

class Network4D {
private:
    std::vector<Layer4D*> layers;

public:
    Network4D() = default;
    
    ~Network4D() {
        for (auto &it : layers) {
            delete it;
            it = nullptr;
        }
    }

    void add(Layer4D* layer) {
        layers.push_back(layer);
    }

    Tensor4D forward(Tensor4D input) {
        for (auto* layer : layers) {
            input = layer->forward(input);
        }
        return input;
    }

    Tensor4D backward(Tensor4D output_gradient) {

        for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
            output_gradient = (*it)->backward(output_gradient);
        }
        return output_gradient;
    }

    void saveModel(const std::string& filename) const {
        std::ofstream out(filename, std::ios::binary);
        if (!out.is_open()) {
            throw std::runtime_error("Error: Could not open file " + filename + " for saving 4D weights!");
        }

        // Looping through and serialize each 4D layer sequentially
        for (const auto* layer : layers) {
            layer->save(out);
        }

        out.close();
        std::cout << "4D Model weights successfully saved to: " << filename << "\n";
    }

    void loadModel(const std::string& filename) {
        std::ifstream in(filename, std::ios::binary);
        if (!in.is_open()) {
            throw std::runtime_error("Error: Could not open file " + filename + " for loading 4D weights!");
        }

        // Reading the parameters back into each 4D layer in the exact same sequence
        for (auto* layer : layers) {
            layer->load(in);
        }

        in.close();
        std::cout << "4D Model weights successfully loaded from: " << filename << "\n";
    }
};