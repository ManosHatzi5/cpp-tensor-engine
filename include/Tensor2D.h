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

    // In-place uniform random initialization in [min_val, max_val]
    void randomize(float min_val = -1.0f, float max_val = 1.0f);

};