# pragma once

# include <vector>
# include <cstddef>
# include <stdexcept>

class Tensor4D{
public:
    size_t N,C,H,W;
    std::vector<float> data;

    Tensor4D(size_t n,size_t c,size_t h,size_t w,float init_val = 0.0f)
    : N(n), C(c), H(h), W(w), data(n*c*h*w,init_val) {}

    // Clean inline operator overloading for 4D coordinate indexing
    inline float& operator()(size_t n, size_t c, size_t h, size_t w) {
        //checking for indices to be in range
        if (n>=N||c>=C||h>=H||w>=W){
            throw std::out_of_range("Tensor4D index out of bounds!");
        }

        return data[(n * C * H * W) + (c * H * W) + (h * W) + w];
    }

    inline const float& operator()(size_t n, size_t c, size_t h, size_t w) const {
        //checking for indices to be in range
        if (n>=N||c>=C||h>=H||w>=W){
            throw std::out_of_range("Tensor4D index out of bounds!");
        }

        return data[(n * C * H * W) + (c * H * W) + (h * W) + w];
    }
    
    void fill_zero() {
        std::fill(data.begin(), data.end(), 0.0f);
    }
};