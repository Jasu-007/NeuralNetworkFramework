#pragma once

# include <vector>
# include <iostream>

class Tensor{
public:
    //variables
    std::vector<float> data;
    int rows;
    int cols;

    //constructors
    Tensor();
    Tensor(int r,int c);
    Tensor(int r,int c,const std::vector<float>& values);
    
    //accessing elements
    inline float& operator()(int r,int c){
        return data[r*cols+c];  
    }
    inline const float& operator()(int r,int c)const{
        return data[r*cols+c];
    }

    //common functions
    Tensor transpose() const;
    void print() const;

    Tensor multiply(const Tensor&B) const;

    void randomize_normal(float mean, float stddev);
};