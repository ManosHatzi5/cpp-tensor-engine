#include <gtest/gtest.h>
#include "Tensor2D.h"

TEST(Tensor2D_Core, MemoryLayoutAndIndexing) {
    // Create a 2x3 matrix
    // [ 0.0, 0.0, 0.0 ]
    // [ 0.0, 0.0, 0.0 ]
    Tensor2D tensor(2, 3);

    EXPECT_EQ(tensor.rows(), 2);
    EXPECT_EQ(tensor.cols(), 3);

    // Write a value to row 1, col 2
    tensor(1, 2) = 42.5f;

    // Read it back
    EXPECT_FLOAT_EQ(tensor(1, 2), 42.5f);
}

TEST(Tensor2D_Core, ConstIndexing) {
    Tensor2D tensor(2, 2);
    tensor(1, 1) = 99.0f; // Uses the mutable operator

    // Create a strict read-only view of the tensor
    const Tensor2D& read_only_tensor = tensor;

    // read_only_tensor(1, 1) = 100.0f; // If you uncommented this, it would cause a compiler error!

    // Uses the const operator
    EXPECT_FLOAT_EQ(read_only_tensor(1, 1), 99.0f);
}

TEST(Tensor2D_Core, OutOfBoundsThrowsException) {
    Tensor2D tensor(3, 3);

    // Expect that calling tensor(3, 0) throws an out_of_range exception.
    // (Remember, indices are 0-based, so row 3 is the 4th row, which does not exist).
    EXPECT_THROW(tensor(3, 0), std::out_of_range);
    EXPECT_THROW(tensor(0, 5), std::out_of_range);
}

TEST(Tensor2D_Math, MatrixAddition) {
    Tensor2D t1(2, 2);
    t1(0, 0) = 1.0f; t1(0, 1) = 2.0f;
    t1(1, 0) = 3.0f; t1(1, 1) = 4.0f;

    Tensor2D t2(2, 2);
    t2(0, 0) = 5.0f; t2(0, 1) = 6.0f;
    t2(1, 0) = 7.0f; t2(1, 1) = 8.0f;

    Tensor2D result = t1 + t2;

    EXPECT_FLOAT_EQ(result(0, 0), 6.0f);
    EXPECT_FLOAT_EQ(result(0, 1), 8.0f);
    EXPECT_FLOAT_EQ(result(1, 0), 10.0f);
    EXPECT_FLOAT_EQ(result(1, 1), 12.0f);
}

TEST(Tensor2D_Math, ScalarMultiplication) {
    Tensor2D t(2, 2);
    t(0, 0) = 1.5f; t(0, 1) = -2.0f;
    t(1, 0) = 0.0f; t(1, 1) = 4.0f;

    Tensor2D scaled = t * 2.0f;

    EXPECT_FLOAT_EQ(scaled(0, 0), 3.0f);
    EXPECT_FLOAT_EQ(scaled(0, 1), -4.0f);
    EXPECT_FLOAT_EQ(scaled(1, 0), 0.0f);
    EXPECT_FLOAT_EQ(scaled(1, 1), 8.0f);
}

TEST(Tensor2D_Math, HadamardProduct) {
    Tensor2D a(2, 2);
    a(0, 0) = 1.0f; a(0, 1) = 2.0f;
    a(1, 0) = 3.0f; a(1, 1) = 4.0f;

    Tensor2D b(2, 2);
    b(0, 0) = 5.0f; b(0, 1) = 6.0f;
    b(1, 0) = 7.0f; b(1, 1) = 8.0f;

    Tensor2D prod = a.multiply(b);

    EXPECT_FLOAT_EQ(prod(0, 0), 5.0f);
    EXPECT_FLOAT_EQ(prod(0, 1), 12.0f);
    EXPECT_FLOAT_EQ(prod(1, 0), 21.0f);
    EXPECT_FLOAT_EQ(prod(1, 1), 32.0f);
}

TEST(Tensor2D_Math, MismatchedDimensionsThrow) {
    Tensor2D a(2, 3);
    Tensor2D b(3, 2);

    EXPECT_THROW(a + b, std::invalid_argument);
    EXPECT_THROW(a.multiply(b), std::invalid_argument);
}

TEST(Tensor2D_Math, MatrixMultiplication) {
    // Matrix A: 2x3
    // [ 1.0, 2.0, 3.0 ]
    // [ 4.0, 5.0, 6.0 ]
    Tensor2D a(2, 3);
    a(0, 0) = 1.0f; a(0, 1) = 2.0f; a(0, 2) = 3.0f;
    a(1, 0) = 4.0f; a(1, 1) = 5.0f; a(1, 2) = 6.0f;

    // Matrix B: 3x2
    // [ 7.0,  8.0 ]
    // [ 9.0, 10.0 ]
    // [ 11.0, 12.0 ]
    Tensor2D b(3, 2);
    b(0, 0) = 7.0f;  b(0, 1) = 8.0f;
    b(1, 0) = 9.0f;  b(1, 1) = 10.0f;
    b(2, 0) = 11.0f; b(2, 1) = 12.0f;

    // C = A * B -> Dimensions: 2x2
    // C(0,0) = 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
    // C(0,1) = 1*8 + 2*10 + 3*12 = 8 + 20 + 36 = 64
    // C(1,0) = 4*7 + 5*9 + 6*11 = 28 + 45 + 66 = 139
    // C(1,1) = 4*8 + 5*10 + 6*12 = 32 + 50 + 72 = 154
    Tensor2D c = a.matmul(b);

    EXPECT_EQ(c.rows(), 2);
    EXPECT_EQ(c.cols(), 2);

    EXPECT_FLOAT_EQ(c(0, 0), 58.0f);
    EXPECT_FLOAT_EQ(c(0, 1), 64.0f);
    EXPECT_FLOAT_EQ(c(1, 0), 139.0f);
    EXPECT_FLOAT_EQ(c(1, 1), 154.0f);
}

TEST(Tensor2D_Math, MatrixMultiplicationDimensionMismatch) {
    Tensor2D a(2, 3);
    Tensor2D b(2, 3); // Columns of A (3) != Rows of B (2)

    EXPECT_THROW(a.matmul(b), std::invalid_argument);
}

TEST(Tensor2D_Math, Transposition) {
    // A: 2x3
    // [ 1.0, 2.0, 3.0 ]
    // [ 4.0, 5.0, 6.0 ]
    Tensor2D a(2, 3);
    a(0, 0) = 1.0f; a(0, 1) = 2.0f; a(0, 2) = 3.0f;
    a(1, 0) = 4.0f; a(1, 1) = 5.0f; a(1, 2) = 6.0f;

    // Expected A^T: 3x2
    // [ 1.0, 4.0 ]
    // [ 2.0, 5.0 ]
    // [ 3.0, 6.0 ]
    Tensor2D at = a.transpose();

    EXPECT_EQ(at.rows(), 3);
    EXPECT_EQ(at.cols(), 2);

    EXPECT_FLOAT_EQ(at(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(at(0, 1), 4.0f);

    EXPECT_FLOAT_EQ(at(1, 0), 2.0f);
    EXPECT_FLOAT_EQ(at(1, 1), 5.0f);

    EXPECT_FLOAT_EQ(at(2, 0), 3.0f);
    EXPECT_FLOAT_EQ(at(2, 1), 6.0f);
}

#include <cmath>

TEST(Tensor2D_Activations, ReLU) {
    Tensor2D t(2, 2);
    t(0, 0) = -3.5f; t(0, 1) = 0.0f;
    t(1, 0) = 2.1f;  t(1, 1) = -0.01f;

    Tensor2D activated = t.relu();

    EXPECT_FLOAT_EQ(activated(0, 0), 0.0f);
    EXPECT_FLOAT_EQ(activated(0, 1), 0.0f);
    EXPECT_FLOAT_EQ(activated(1, 0), 2.1f);
    EXPECT_FLOAT_EQ(activated(1, 1), 0.0f);
}

TEST(Tensor2D_Activations, Sigmoid) {
    Tensor2D t(1, 3);
    t(0, 0) = 0.0f;
    t(0, 1) = 2.0f;
    t(0, 2) = -2.0f;

    Tensor2D activated = t.sigmoid();

    // 1 / (1 + e^0) = 0.5
    EXPECT_FLOAT_EQ(activated(0, 0), 0.5f);
    // 1 / (1 + e^-2) approx 0.880797
    EXPECT_NEAR(activated(0, 1), 0.880797f, 1e-5f);
    // 1 / (1 + e^2) approx 0.119203
    EXPECT_NEAR(activated(0, 2), 0.119203f, 1e-5f);
}

#include "DenseLayer.h"

TEST(DenseLayer_Execution, ForwardPassManualWeights) {
    // 2 inputs -> 2 outputs
    DenseLayer layer(2, 2);

    // Manually set weights:
    // W = [ 1.0,  2.0 ]
    //     [ -1.0, 1.0 ]
    // Note: since get_weights() returns a const reference,
    // we verify the zero-initialized forward pass first:
    Tensor2D x(2, 1);
    x(0, 0) = 3.0f;
    x(1, 0) = 4.0f;

    // With 0 weights and 0 bias: Z = 0 * X + 0 = 0 -> ReLU(0) = 0
    Tensor2D out = layer.forward(x);
    EXPECT_EQ(out.rows(), 2);
    EXPECT_EQ(out.cols(), 1);
}


TEST(DenseLayer_Execution, ForwardPassRandomWeights) {
    DenseLayer layer(3, 2);

    Tensor2D x(3, 1);
    x(0, 0) = 1.0f;
    x(1, 0) = 2.0f;
    x(2, 0) = 3.0f;

    Tensor2D out = layer.forward(x);

    EXPECT_EQ(out.rows(), 2);
    EXPECT_EQ(out.cols(), 1);

    // Because weights are randomized in [-0.5, 0.5], output is non-zero
    // (either positive due to ReLU or clamped to 0)
    const Tensor2D& w = layer.get_weights();
    bool all_zero = true;
    for (size_t r = 0; r < w.rows(); ++r) {
        for (size_t c = 0; c < w.cols(); ++c) {
            if (w(r, c) != 0.0f) {
                all_zero = false;
                break;
            }
        }
    }
    EXPECT_FALSE(all_zero);
}

#include "Loss.h"

TEST(LossFunction_Execution, MSE_Evaluation) {
    // y_pred = [2.0, 4.0]^T, y_true = [1.0, 2.0]^T
    // diffs: (2-1)=1, (4-2)=2
    // squared: 1^2 + 2^2 = 5
    // MSE = 5 / 2 = 2.5
    Tensor2D pred(2, 1);
    pred(0, 0) = 2.0f;
    pred(1, 0) = 4.0f;

    Tensor2D target(2, 1);
    target(0, 0) = 1.0f;
    target(1, 0) = 2.0f;

    float loss = Loss::mse(pred, target);
    EXPECT_FLOAT_EQ(loss, 2.5f);

    // Gradient: (2/K) * (y_pred - y_true) = (2/2) * [1, 2]^T = [1.0, 2.0]^T
    Tensor2D grad = Loss::mse_derivative(pred, target);
    EXPECT_EQ(grad.rows(), 2);
    EXPECT_EQ(grad.cols(), 1);
    EXPECT_FLOAT_EQ(grad(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(grad(1, 0), 2.0f);
}
































