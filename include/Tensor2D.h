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

    // ACTIVATION FUNCTIONS     /-------------/
    [[nodiscard]] Tensor2D relu() const;
    // Computes the element-wise sigmoid activation: 1 / (1 + exp(-z))
    [[nodiscard]] Tensor2D sigmoid() const;

    // DERIVATIVES OF FUNCTIONS /-------------/
    [[nodiscard]] Tensor2D relu_derivative() const;
    // Computes the derivative assuming the caller tensor is the ALREADY ACTIVATED output A.
    // Evaluates a * (1.0f - a) element-wise to avoid recomputing exp(-z).
    [[nodiscard]] Tensor2D sigmoid_derivative_from_output() const;

    /**
     * @brief Broadcasts a column vector across all columns of this matrix (In-place/Out-of-place).
     * @note Memory Optimization: Virtually projects a (D_out x 1) bias vector across a
     *       (D_out x Batch) tensor without allocating memory for a duplicated matrix.
     * @param bias A (rows x 1) column vector to broadcast.
     * @return A new Tensor2D resulting from the broadcasted addition.
     */
    [[nodiscard]] Tensor2D add_bias(const Tensor2D& bias) const;

    /**
     * @brief Reduces the matrix along the column dimension (axis 1) via summation.
     * @note Backpropagation Kernel: Collapses a (D_out x Batch) gradient tensor into a
     *       (D_out x 1) gradient vector to match the dimensions of the bias parameters.
     * @return A (rows x 1) column vector containing the row-wise sums.
     */
    [[nodiscard]] Tensor2D sum_along_columns() const;

    // In-place uniform random initialization in [min_val, max_val]
    void randomize(float min_val = -1.0f, float max_val = 1.0f);

    /**
     * @brief Direct pointer to the underlying contiguous row-major buffer.
     * @note Bypasses container indirection for SIMD vectorization and cache-line streaming.
     *       Callers must bound access using size() to avoid out-of-bounds heap faults.
     * @return Raw pointer to the first element (float*).
     */
    [[nodiscard]] float* data() noexcept { return data_.data(); }
    [[nodiscard]] const float* data() const noexcept { return data_.data(); }

    /**
     * @brief Total number of elements in the tensor (rows * cols).
     */
    [[nodiscard]] size_t size() const noexcept { return data_.size(); }

    /**
     * @brief Blocked matrix multiplication using cache tiling.
     * @details Partitions matrices into sub-blocks of size (block_size x block_size)
     *          to ensure working sets fit within L1/L2 data cache boundaries.
     * @param other Matrix B (K x N).
     * @param block_size Edge length of the tile (default: 32).
     * @return Result matrix C = A * B of shape (M x N).
     */
    [[nodiscard]] Tensor2D matmul_tiled(const Tensor2D& other, size_t block_size = 32) const;
};



























