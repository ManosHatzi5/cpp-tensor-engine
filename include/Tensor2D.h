#pragma once
#include <vector>
#include <cstddef> // for size_t

class Tensor2D {
private:
    size_t rows_;
    size_t cols_;
    std::vector<float> data_; // The single, flat contiguous block of memory

public:
    // Constructor
    Tensor2D(size_t rows, size_t cols);

    // Getters
    [[nodiscard]] size_t rows() const { return rows_; }
    [[nodiscard]] size_t cols() const { return cols_; }

    // Operator Overloads for 2D indexing (r = row, c = col)
    float& operator()(size_t r, size_t c);
    const float& operator()(size_t r, size_t c) const;

    // Matrix Addition (Element-wise)
    Tensor2D operator+(const Tensor2D& other) const;

    // Scalar Multiplication (Broadcasting a single float to all elements)
    Tensor2D operator*(float scalar) const;

    // Element-wise Multiplication (Hadamard Product)
    [[nodiscard]] Tensor2D multiply(const Tensor2D& other) const;

    // Standard Matrix Multiplication (Dot Product / GEMM)
    // C = A.matmul(B) or C = A * B
    [[nodiscard]] Tensor2D matmul(const Tensor2D& other) const;

    // Matrix Transposition
    [[nodiscard]] Tensor2D transpose() const;

    // Activation Functions
    [[nodiscard]] Tensor2D relu() const;
    [[nodiscard]] Tensor2D sigmoid() const;

    // Derivative of Functions
    [[nodiscard]] Tensor2D relu_derivative() const;

    // Kernel: Adds a column vector (rows x 1) across all columns of this matrix (rows x cols)
    // to make batches possible and not going one by one we need to broadcast our bias vector
    // from d_outx1 to d_outxB where B the batch number. But in order to save memory we dont
    // just copy paste the vector B times, we keep the d_outx1 and "virtually" go through
    // the d_outxB Tensor. See implementation
    [[nodiscard]] Tensor2D add_bias(const Tensor2D& bias) const;

    // Kernel: Sums across columns (axis 1), reducing a (rows x cols) matrix to a (rows x 1) column vector
    // as a result of the above kernel, bias remains a d_outx1 tensor so we need the dB to also have the same
    // dimensions so it can pass safely to each layer as input during backpropagation
    // so keeping the same logic that dB = dZ, now we have dB = [dZ1, dZ2..., dZB], we just take the sum
    // of those number resulting into one scalar number, dB = dZk, dZk = sum(dZ1, dZ2..., dZB)
    [[nodiscard]] Tensor2D sum_along_columns() const;

    // In-place uniform random initialization in [min_val, max_val]
    void randomize(float min_val = -1.0f, float max_val = 1.0f);

};