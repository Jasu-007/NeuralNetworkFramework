# include <Layer.hpp>

Network::Network(){
}

//forward and backward pass combining all layers

Tensor Network::forward(Tensor input){
    for (const auto& it:layers){
        input = it->forward(input);
    }
    return input;
}

Tensor Network::backward(const Tensor& output_gradient){
    Tensor output = output_gradient;
    for (int i=layers.size()-1;i>=0;i--){
        output = layers[i]->backward(output);
    }
    return output;
}

void Network::saveModel(const std::string& filename) const {
    // Opening file in binary mode for writing
    std::ofstream out(filename, std::ios::binary);
    if (!out.is_open()) {
        throw std::runtime_error("Error: Could not open file " + filename + " for saving weights!");
    }

    // Sequentially askning every layer to dump its parameters to the stream
    for (const auto* layer : layers) {
        layer->save(out);
    }

    out.close();
    std::cout << "Model weights successfully saved to: " << filename << "\n";
}

void Network::loadModel(const std::string& filename) {
    // Opening file in binary mode for reading
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Error: Could not open file " + filename + " for loading weights!");
    }

    // Sequentially reading parameters back into the layers in the exact same order
    for (auto* layer : layers) {
        layer->load(in);
    }

    in.close();
    std::cout << "Model weights successfully loaded from: " << filename << "\n";
}