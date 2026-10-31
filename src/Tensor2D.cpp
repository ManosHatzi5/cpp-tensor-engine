#include "Tensor2D.h"
#include <stdexcept>
#include <cmath>
#include <algorithm>

Tensor2D::Tensor2D(size_t rows, size_t cols)
    :rows_(rows), cols_(cols), data_(rows*cols, 0.0f) {}

float& Tensor2D::operator()(size_t r, size_t c) {
    // C++ doesnt safely halt an out of bounds reach
    // so we have to write our own code to throw an exception if that happens
    // otherwise the system will just go and overwrite data in that place
    if (r >= rows_ || c >= cols_) {
        throw std::out_of_range("Index out of bounds.");
    }
    return data_[r*cols_+c];
}

const float& Tensor2D::operator()(size_t r, size_t c) const {
    if (r >= rows_ || c >= cols_) {
        throw std::out_of_range("Index out of bounds.");
    }
    return data_[r*cols_+c];
}

Tensor2D Tensor2D::operator+(const Tensor2D& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("The matrices are not of the same dimensions.");
    }
    Tensor2D result(rows_, cols_);
    for (size_t i = 0; i < rows_*cols_; ++i) {
        result.data_[i] = data_[i] + other.data_[i];
    }
    return result;
}

Tensor2D Tensor2D::operator*(float scalar) const {
    Tensor2D result(rows_, cols_);
    for (size_t i = 0; i < rows_ * cols_; ++i) {
        result.data_[i] = data_[i] * scalar;
    }
    return result;
}

Tensor2D Tensor2D::multiply(const Tensor2D& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("The matrices are not of the same dimensions for Hadamard product.");
    }
    Tensor2D result(rows_, cols_);
    for (size_t i = 0; i < rows_ * cols_; ++i) {
        result.data_[i] = data_[i] * other.data_[i];
    }
    return result;
}

Tensor2D Tensor2D::matmul(const Tensor2D& other) const {
    if (cols_ != other.rows_) {
        throw std::invalid_argument("Incompatible dimensions for matrix multiplication");
    }

    Tensor2D result(rows_, other.cols_);

    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < other.cols_; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < cols_; ++k) {
                sum += data_[i*cols_ + k] * other.data_[k*other.cols_ + j];
            }
            result(i, j) = sum;
        }
    }

    return result;
}

Tensor2D Tensor2D::transpose() const {
    Tensor2D transposed(cols_, rows_);
    for (size_t j = 0; j < cols_; ++j) {
        for (size_t i = 0; i < rows_; ++i) {
            // we already fixed the () operator so we can just:
            transposed(j, i) = (*this)(i, j);
            //transposed.data_[j * rows_ + i] = data_[i * cols_ + j];
        }
    }
    return transposed;
}

Tensor2D Tensor2D::relu() const {
    Tensor2D result(rows_, cols_);
    for (size_t i = 0; i < rows_ * cols_; ++i) {
        result.data_[i] = std::max(0.0f, data_[i]);
    }
    return result;
}

#include <random>

void Tensor2D::randomize(float min_val, float max_val) {
    // Standard mersenne twister seeded with a non-deterministic device
    static std::mt19937 gen(42); // fixed seed 42 for reproducible unit tests
    std::uniform_real_distribution<float> dist(min_val, max_val);

    for (size_t i = 0; i < rows_ * cols_; ++i) {
        data_[i] = dist(gen);
    }
}

Tensor2D Tensor2D::relu_derivative() const {
    Tensor2D result(rows_, cols_);
    for (size_t i = 0; i < rows_ * cols_; ++i) {
        result.data_[i] = (data_[i] > 0.0f) ? 1.0f : 0.0f;
    }
    return result;
}

// for information abt this function check the header file
Tensor2D Tensor2D::add_bias(const Tensor2D& bias) const {
    if (bias.cols() != 1 || bias.rows() != rows_) {
        throw std::invalid_argument("Bias must be a column vector matching the matrix row count.");
    }

    Tensor2D result(rows_, cols_);

    // Virtual broadcasting: loop over row-major storage
    for (size_t r = 0; r < rows_; ++r) {
        float b = bias(r, 0); // Read the bias scalar once per row into a register
        const size_t row_offset = r * cols_;
        for (size_t c = 0; c < cols_; ++c) {
            result.data_[row_offset + c] = data_[row_offset + c] + b;
        }
    }

    return result;
}

// for information abt this function check the header file
Tensor2D Tensor2D::sum_along_columns() const {
    Tensor2D result(rows_, 1);

    for (size_t r = 0; r < rows_; ++r) {
        float sum = 0.0f;
        const size_t row_offset = r * cols_;
        for (size_t c = 0; c < cols_; ++c) {
            sum += data_[row_offset + c];
        }
        result(r, 0) = sum;
    }

    return result;
}

#include <cmath>

Tensor2D Tensor2D::sigmoid() const {
    Tensor2D result(rows_, cols_);
    const size_t total_elements = rows_ * cols_;

    for (size_t i = 0; i < total_elements; ++i) {
        // Evaluate sigma(z) = 1.0 / (1.0 + e^-z)
        result.data_[i] = 1.0f / (1.0f + std::exp(-data_[i]));
    }

    return result;
}

Tensor2D Tensor2D::sigmoid_derivative_from_output() const {
    Tensor2D result(rows_, cols_);
    const size_t total_elements = rows_ * cols_;

    for (size_t i = 0; i < total_elements; ++i) {
        const float a = data_[i];
        // sigma'(z) = a * (1 - a)
        // This is O(1) pure arithmetic with no transcendent function calls
        result.data_[i] = a * (1.0f - a); 
    }

    return result;
}


Tensor2D Tensor2D::matmul_tiled(const Tensor2D& other, size_t block_size) const {
    if (cols_ != other.rows_) {
        throw std::invalid_argument("Matrix inner dimensions must match for multiplication.");
    }

    const size_t M = rows_;
    const size_t K = cols_;
    const size_t N = other.cols_;

    // Initialize accumulator matrix C to zero
    Tensor2D result(M, N);

    const float* a_data = this->data();
    const float* b_data = other.data();
    float* c_data = result.data();

    // -------------------------------------------------------------------------
    // OUTER 3 LOOPS: Block / Tile Coordinates
    // Jump across memory in strides of `block_size` (default = 32x32)
    // -------------------------------------------------------------------------
    for (size_t i0 = 0; i0 < M; i0 += block_size) {
        const size_t i_max = std::min(i0 + block_size, M);

        for (size_t j0 = 0; j0 < N; j0 += block_size) {
            const size_t j_max = std::min(j0 + block_size, N);

            for (size_t k0 = 0; k0 < K; k0 += block_size) {
                const size_t k_max = std::min(k0 + block_size, K);

                // -------------------------------------------------------------
                // INNER 3 LOOPS: Micro-Kernel within L1/L2 Cache Boundaries
                // Loop order: i -> k -> j (enables horizontal, unit-stride streaming)
                // -------------------------------------------------------------
                for (size_t i = i0; i < i_max; ++i) {
                    const size_t a_row_offset = i * K;
                    const size_t c_row_offset = i * N;

                    for (size_t k = k0; k < k_max; ++k) {
                        const float a_val = a_data[a_row_offset + k];
                        const size_t b_row_offset = k * N;

                        // Innermost loop: Contiguous row-major read of B and write to C
                        for (size_t j = j0; j < j_max; ++j) {
                            c_data[c_row_offset + j] += a_val * b_data[b_row_offset + j];
                        }
                    }
                }
                // -------------------------------------------------------------

            }
        }
    }

    return result;
}

















