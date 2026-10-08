# include "Utils.hpp"
# include <cassert>
# include <cmath>
# include <algorithm>

void DataUtils::load_csv(const std::string& filepath,Tensor& X_out,Tensor& Y_out,int label_column_idx){
    
    std::ifstream file(filepath);
    assert(file.is_open());

    std::vector<std::vector<float>> features;
    std::vector<float> labels;
    std::string line;

    while (std::getline(file,line)){
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string token;
        std::vector<float> feature_row;
        float label_val = 0.0f;

        int col_idx = 0;
        while (std::getline(ss,token,',')){
            if (token.empty()) continue;

            float val = std::stof(token);

            if (col_idx==label_column_idx){
                label_val = val;
            }
            else{
                feature_row.push_back(val);
            }
            col_idx++;
        }
        features.push_back(feature_row);
        labels.push_back(label_val);
    }
    Tensor feature_tensor(features.size(),features[0].size());
    Tensor label_tensor(labels.size(),1);

    for (int i=0;i<feature_tensor.rows;i++){
        label_tensor(i,0) = labels[i];
        for (int j=0;j<feature_tensor.cols;j++){
            feature_tensor(i,j) = features[i][j];
        }
    }
    X_out = feature_tensor;
    Y_out = label_tensor;
    return;
}

void DataUtils::normalize_features(Tensor& X){
    float maxi,mini;
    for (int j=0;j<X.cols;j++){
        maxi = X(0,j);
        mini = X(0,j);
        for (int i=1;i<X.rows;i++){
            maxi = std::max(maxi,X(i,j));
            mini = std::min(mini,X(i,j));
        }
        for (int i=0;i<X.rows;i++){
            X(i,j) = (X(i,j)-mini)/(maxi-mini+1e-15);
        }
    }
}

Tensor DataUtils::one_hot_encoder(const Tensor& Y_true,int num_classes){
    Tensor encoded(Y_true.rows,num_classes);
    for (size_t i=0;i<Y_true.rows;i++){
        int class_idx = static_cast<int> (Y_true(i,0));
        encoded(i,class_idx) = 1.0f;
    }
    return encoded;
}

Tensor DataUtils::get_batch(const Tensor& source, int batch_idx, int batch_size) {
    size_t start_row = static_cast<size_t>(batch_idx * batch_size);
    size_t end_row = std::min(static_cast<int>(start_row + static_cast<size_t>(batch_size)), (source.rows));
    
    size_t actual_batch_size = end_row - start_row;
    
    // Create a sub-tensor with the exact slice dimensions
    Tensor batch(actual_batch_size, source.cols);
    
    // Copy the matrix data values across row blocks
    for (size_t i = 0; i < actual_batch_size; ++i) {
        for (size_t j = 0; j < source.cols; ++j) {
            batch(i, j) = source(start_row + i, j);
        }
    }
    
    return batch;
}

Tensor4D DataUtils::get_spatial_batch(const Tensor& source, int batch_idx, int batch_size, size_t channels, size_t height, size_t width) {
    size_t start_row = static_cast<size_t>(batch_idx * batch_size);
    size_t end_row = std::min(static_cast<size_t>(start_row + batch_size), static_cast<size_t>(source.rows));
    size_t actual_batch_size = end_row - start_row;

    // Instantiate your 4D Tensor matching current mini-batch capacity
    Tensor4D spatial_batch(actual_batch_size, channels, height, width);

    // Flat size of a single sample's features (e.g., 1 * 28 * 28 = 784)
    size_t features_per_sample = channels * height * width;

    // Direct memory block assignment
    for (size_t i = 0; i < actual_batch_size; ++i) {
        size_t source_row = start_row + i;
        for (size_t j = 0; j < features_per_sample; ++j) {
            // Since source matrix is flat features, map directly to the 4D vector
            spatial_batch.data[i * features_per_sample + j] = source(source_row, j);
        }
    }

    return spatial_batch;
}