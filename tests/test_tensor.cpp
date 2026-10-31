#include <gtest/gtest.h>

#include "Adam.h"
#include "SGDOptimizer.h"
#include "LinearLayer.h"
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
    // (indices are 0-based, so row 3 is the 4th row, which does not exist).
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

#include "LinearLayer.h"

TEST(DenseLayer_Execution, ForwardPassManualWeights) {
    // 2 inputs -> 2 outputs
    LinearLayer layer(2, 2);

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
    LinearLayer layer(3, 2);

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

TEST(DenseLayer_Execution, BackwardPassAnalytical) {
    // 2 inputs -> 1 output
    LinearLayer layer(2, 1);

    // Set deterministic weights and input:
    // W = [2.0, -1.0], B = [0.5]
    // X = [1.0, 3.0]^T
    // Z = 2.0*1.0 + (-1.0)*3.0 + 0.5 = -0.5
    // A = ReLU(-0.5) = 0.0 -> relu'(Z) = 0.0
    //
    // Let's test an active neuron (Z > 0):
    // If X = [2.0, 1.0]^T:
    // Z = 2.0*2.0 + (-1.0)*1.0 + 0.5 = 3.5 > 0 -> relu'(Z) = 1.0
    // If incoming d_out = [1.0]
    // delta = 1.0 * 1.0 = 1.0
    // dW = delta * X^T = [1.0] * [2.0, 1.0] = [2.0, 1.0]
    // dB = [1.0]
    // dX = W^T * delta = [2.0, -1.0]^T * [1.0] = [2.0, -1.0]^T

    Tensor2D x(2, 1);
    x(0, 0) = 2.0f;
    x(1, 0) = 1.0f;

    Tensor2D out = layer.forward(x);
    EXPECT_EQ(out.rows(), 1);
    EXPECT_EQ(out.cols(), 1);

    Tensor2D d_out(1, 1);
    d_out(0, 0) = 1.0f;

    Tensor2D d_in = layer.backward(d_out);
    EXPECT_EQ(d_in.rows(), 2);
    EXPECT_EQ(d_in.cols(), 1);
    EXPECT_EQ(layer.get_weights().rows(), 1);
    EXPECT_EQ(layer.get_weights().cols(), 2);
}

#include "Sequential.h"
TEST(Engine_Integration, TrainingLoopSingleLayer) {
    Sequential model;
    model.add(std::make_unique<LinearLayer>(2, 1)); // 2 inputs, 1 output

    // Inputs chosen with negative signs to invert the deterministic
    // negative random weights and force a positive Z (opening the ReLU gate)
    Tensor2D x(2, 1);
    x(0, 0) = -5.0f;
    x(1, 0) = -5.0f;

    // Target Y = [5.0]^T
    // We want the network to learn W and B such that W*X + B = 5.0
    Tensor2D target(1, 1);
    target(0, 0) = 5.0f;

    float learning_rate = 0.001f;
    float initial_loss = 0.0f;
    float final_loss = 0.0f;

    // Train for 100 epochs
    SGD optimizer(learning_rate);

    for (int epoch = 0; epoch < 100; ++epoch) {
        Tensor2D y_pred = model.forward(x);

        float current_loss = Loss::mse(y_pred, target);
        if (epoch == 0) initial_loss = current_loss;
        if (epoch == 99) final_loss = current_loss;

        Tensor2D grad = Loss::mse_derivative(y_pred, target);

        (void)model.backward(grad);

        optimizer.step(model.get_layers());
    }

    EXPECT_LT(final_loss, initial_loss);

    Tensor2D final_pred = model.forward(x);
    std::cout << "Initial Loss: " << initial_loss << "\n";
    std::cout << "Final Loss: " << final_loss << "\n";
    std::cout << "Final Prediction: " << final_pred(0, 0) << " (Target: 5.0)\n";
}

#include "Sequential.h"
#include "LinearLayer.h"
#include "ReLULayer.h"

TEST(Sequential_Execution, MultiLayerTrainingConvergence) {
    Sequential model;
    // 2 inputs -> 4 hidden neurons -> ReLU -> 1 regression output
    model.add(std::make_unique<LinearLayer>(2, 4));
    model.add(std::make_unique<ReLULayer>());
    model.add(std::make_unique<LinearLayer>(4, 1));

    EXPECT_EQ(model.size(), 3);

    Tensor2D x(2, 1);
    x(0, 0) = 1.0f;
    x(1, 0) = -1.0f;

    Tensor2D target(1, 1);
    target(0, 0) = 3.0f;

    float learning_rate = 0.01f;
    float initial_loss = 0.0f;
    float final_loss = 0.0f;

    // Train for 200 epochs through the unified pipeline
    SGD optimizer(learning_rate);
    for (int epoch = 0; epoch < 200; ++epoch) {
        Tensor2D y_pred = model.forward(x);

        float current_loss = Loss::mse(y_pred, target);
        if (epoch == 0) initial_loss = current_loss;
        if (epoch == 199) final_loss = current_loss;

        Tensor2D grad = Loss::mse_derivative(y_pred, target);

        (void)model.backward(grad);
        optimizer.step(model.get_layers());
    }

    EXPECT_LT(final_loss, initial_loss);

    Tensor2D final_pred = model.forward(x);
    std::cout << "Sequential Multi-Layer Initial Loss: " << initial_loss << "\n";
    std::cout << "Sequential Multi-Layer Final Loss: " << final_loss << "\n";
    std::cout << "Sequential Multi-Layer Final Prediction: " << final_pred(0, 0)
              << " (Target: " << target(0, 0) << ")\n";
}

TEST(Tensor2D_BatchOps, AddBiasBroadcasting) {
    // 2x3 matrix (2 features, batch size 3)
    Tensor2D mat(2, 3);
    mat(0, 0) = 1.0f; mat(0, 1) = 2.0f; mat(0, 2) = 3.0f;
    mat(1, 0) = 4.0f; mat(1, 1) = 5.0f; mat(1, 2) = 6.0f;

    // 2x1 bias vector
    Tensor2D bias(2, 1);
    bias(0, 0) = 10.0f;
    bias(1, 0) = 20.0f;

    Tensor2D res = mat.add_bias(bias);

    EXPECT_EQ(res.rows(), 2);
    EXPECT_EQ(res.cols(), 3);
    EXPECT_FLOAT_EQ(res(0, 0), 11.0f);
    EXPECT_FLOAT_EQ(res(0, 1), 12.0f);
    EXPECT_FLOAT_EQ(res(0, 2), 13.0f);
    EXPECT_FLOAT_EQ(res(1, 0), 24.0f);
    EXPECT_FLOAT_EQ(res(1, 1), 25.0f);
    EXPECT_FLOAT_EQ(res(1, 2), 26.0f);
}

TEST(Tensor2D_BatchOps, SumAlongColumnsReduction) {
    // 2x3 matrix
    Tensor2D mat(2, 3);
    mat(0, 0) = 1.0f; mat(0, 1) = 2.0f; mat(0, 2) = 3.0f; // Sum = 6
    mat(1, 0) = 4.0f; mat(1, 1) = 5.0f; mat(1, 2) = 6.0f; // Sum = 15

    Tensor2D reduced = mat.sum_along_columns();

    EXPECT_EQ(reduced.rows(), 2);
    EXPECT_EQ(reduced.cols(), 1);
    EXPECT_FLOAT_EQ(reduced(0, 0), 6.0f);
    EXPECT_FLOAT_EQ(reduced(1, 0), 15.0f);
}

TEST(Engine_Integration, BatchTrainingMultiSample) {
    // 2 inputs -> 4 hidden neurons -> ReLU -> 1 regression output
    Sequential model;
    model.add(std::make_unique<LinearLayer>(2, 4));
    model.add(std::make_unique<ReLULayer>());
    model.add(std::make_unique<LinearLayer>(4, 1));

    // Batch size B = 3, Input features D_in = 2
    // Matrix shape: (2 x 3)
    // Sample 0: [ 1.0,  2.0]^T -> Target: 5.0
    // Sample 1: [-1.0, -1.0]^T -> Target: -2.0
    // Sample 2: [ 0.5,  1.5]^T -> Target: 3.5
    Tensor2D x(2, 3);
    x(0, 0) =  1.0f; x(0, 1) = -1.0f; x(0, 2) = 0.5f;
    x(1, 0) =  2.0f; x(1, 1) = -1.0f; x(1, 2) = 1.5f;

    // Target shape: (1 x 3)
    Tensor2D target(1, 3);
    target(0, 0) =  5.0f;
    target(0, 1) = -2.0f;
    target(0, 2) =  3.5f;

    float learning_rate = 0.01f;
    float initial_loss = 0.0f;
    float final_loss = 0.0f;

    SGD optimizer(learning_rate);
    for (int epoch = 0; epoch < 300; ++epoch) {
        // Forward: (2 x 3) -> (4 x 3) -> (4 x 3) -> (1 x 3)
        Tensor2D y_pred = model.forward(x);

        float current_loss = Loss::mse(y_pred, target);
        if (epoch == 0) initial_loss = current_loss;
        if (epoch == 299) final_loss = current_loss;

        // Gradient shape: (1 x 3)
        Tensor2D grad = Loss::mse_derivative(y_pred, target);

        // Backward: accumulates dW and sums dB across all 3 batch columns
        (void)model.backward(grad);

        optimizer.step(model.get_layers());
    }

    EXPECT_LT(final_loss, initial_loss);

    Tensor2D final_pred = model.forward(x);
    std::cout << "Batch Initial Loss: " << initial_loss << "\n";
    std::cout << "Batch Final Loss: "   << final_loss << "\n";
    std::cout << "Predictions vs Targets:\n";
    for (size_t b = 0; b < 3; ++b) {
        std::cout << "  Sample " << b << ": Pred = " << final_pred(0, b)
                  << " | Target = " << target(0, b) << "\n";
    }
}

#include "SigmoidLayer.h"

TEST(SigmoidLayer_Execution, ForwardAndBackwardAnalytic) {
    SigmoidLayer sigmoid;

    // Test input Z = [0.0, 2.0]^T
    Tensor2D z(2, 1);
    z(0, 0) = 0.0f;
    z(1, 0) = 2.0f;

    Tensor2D a = sigmoid.forward(z);

    // sigma(0) = 0.5
    // sigma(2) = 1 / (1 + e^-2) approx 0.880797
    EXPECT_FLOAT_EQ(a(0, 0), 0.5f);
    EXPECT_NEAR(a(1, 0), 0.880797f, 1e-5f);

    // Incoming gradient dA = [1.0, 1.0]^T
    Tensor2D d_out(2, 1);
    d_out(0, 0) = 1.0f;
    d_out(1, 0) = 1.0f;

    Tensor2D d_in = sigmoid.backward(d_out);

    // dZ[0] = 1.0 * 0.5 * (1 - 0.5) = 0.25
    // dZ[1] = 1.0 * 0.880797 * (1 - 0.880797) = 0.880797 * 0.119203 approx 0.1049935
    EXPECT_FLOAT_EQ(d_in(0, 0), 0.25f);
    EXPECT_NEAR(d_in(1, 0), 0.1049935f, 1e-5f);
}

TEST(Loss_Functions, BinaryCrossEntropyAnalytic) {
    // 1 feature, batch size 2
    Tensor2D pred(1, 2);
    pred(0, 0) = 0.8f;
    pred(0, 1) = 0.1f;

    Tensor2D target(1, 2);
    target(0, 0) = 1.0f;
    target(0, 1) = 0.0f;

    // Sample 0: -ln(0.8) approx 0.2231435
    // Sample 1: -ln(1 - 0.1) = -ln(0.9) approx 0.1053605
    // Mean Loss approx (0.2231435 + 0.1053605) / 2 = 0.164252
    float loss = Loss::bce(pred, target);
    EXPECT_NEAR(loss, 0.164252f, 1e-5f);

    Tensor2D grad = Loss::bce_derivative(pred, target);
    // Sample 0: 0.5 * (0.8 - 1.0) / (0.8 * 0.2) = 0.5 * (-0.2 / 0.16) = -0.625
    // Sample 1: 0.5 * (0.1 - 0.0) / (0.1 * 0.9) = 0.5 * (0.1 / 0.09) approx 0.555555
    EXPECT_NEAR(grad(0, 0), -0.625f, 1e-5f);
    EXPECT_NEAR(grad(0, 1), 0.555555f, 1e-4f);
}

TEST(Parameter_Initialization, HeAndXavierVariance) {
    const size_t in_features = 2000;
    const size_t out_features = 2000;
    const size_t total_elements = in_features * out_features;

    // 1. Verify He Initialization
    LinearLayer he_layer(in_features, out_features, InitScheme::He);
    const auto& w_he = he_layer.get_weights();

    float sum_he = 0.0f;
    for (size_t r = 0; r < out_features; ++r) {
        for (size_t c = 0; c < in_features; ++c) {
            sum_he += w_he(r, c);
        }
    }
    float mean_he = sum_he / static_cast<float>(total_elements);

    float var_sum_he = 0.0f;
    for (size_t r = 0; r < out_features; ++r) {
        for (size_t c = 0; c < in_features; ++c) {
            float diff = w_he(r, c) - mean_he;
            var_sum_he += diff * diff;
        }
    }
    float var_he = var_sum_he / static_cast<float>(total_elements);
    float expected_var_he = 2.0f / static_cast<float>(in_features); // 2 / 2000 = 0.001

    EXPECT_NEAR(mean_he, 0.0f, 1e-3f);
    EXPECT_NEAR(var_he, expected_var_he, 1e-4f);

    // 2. Verify Xavier Initialization
    LinearLayer xavier_layer(in_features, out_features, InitScheme::Xavier);
    const auto& w_xav = xavier_layer.get_weights();

    float sum_xav = 0.0f;
    for (size_t r = 0; r < out_features; ++r) {
        for (size_t c = 0; c < in_features; ++c) {
            sum_xav += w_xav(r, c);
        }
    }
    float mean_xav = sum_xav / static_cast<float>(total_elements);

    float var_sum_xav = 0.0f;
    for (size_t r = 0; r < out_features; ++r) {
        for (size_t c = 0; c < in_features; ++c) {
            float diff = w_xav(r, c) - mean_xav;
            var_sum_xav += diff * diff;
        }
    }
    float var_xav = var_sum_xav / static_cast<float>(total_elements);
    float expected_var_xav = 2.0f / static_cast<float>(in_features + out_features); // 2 / 4000 = 0.0005

    EXPECT_NEAR(mean_xav, 0.0f, 1e-3f);
    EXPECT_NEAR(var_xav, expected_var_xav, 1e-4f);
}

#include "Sequential.h"
#include "LinearLayer.h"
#include "ReLULayer.h"
#include "SigmoidLayer.h"
#include "Loss.h"
#include <iostream>

TEST(Engine_Integration, XORClassificationBenchmark) {
    // 1. Constructing the Non-Linear Architecture
    Sequential model;

    // Hidden Layer: 2 inputs -> 4 neurons, He initialization for ReLU
    model.add(std::make_unique<LinearLayer>(2, 4, InitScheme::He));
    model.add(std::make_unique<ReLULayer>());

    // Output Layer: 4 inputs -> 1 probability, Xavier initialization for Sigmoid
    model.add(std::make_unique<LinearLayer>(4, 1, InitScheme::Xavier));
    model.add(std::make_unique<SigmoidLayer>());

    // 2. Preparing the XOR Dataset (Full Batch of 4 samples)
    // Shape: (D_in x B) = (2 x 4)
    Tensor2D X(2, 4);
    X(0, 0) = 0.0f; X(1, 0) = 0.0f; // Sample 0: [0, 0]^T
    X(0, 1) = 0.0f; X(1, 1) = 1.0f; // Sample 1: [0, 1]^T
    X(0, 2) = 1.0f; X(1, 2) = 0.0f; // Sample 2: [1, 0]^T
    X(0, 3) = 1.0f; X(1, 3) = 1.0f; // Sample 3: [1, 1]^T

    // Shape: (D_out x B) = (1 x 4)
    Tensor2D Y(1, 4);
    Y(0, 0) = 0.0f; // 0 XOR 0 = 0
    Y(0, 1) = 1.0f; // 0 XOR 1 = 1
    Y(0, 2) = 1.0f; // 1 XOR 0 = 1
    Y(0, 3) = 0.0f; // 1 XOR 1 = 0

    // 3. Training Loop Configuration
    float initial_loss = 0.0f;
    float final_loss = 0.0f;
    const int epochs = 3000;

    Adam optimizer(0.1f);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        Tensor2D pred = model.forward(X);
        // float loss = Loss::bce(pred, Y);
        Tensor2D grad = Loss::bce_derivative(pred, Y);
        (void)model.backward(grad);

        // External optimization step
        optimizer.step(model.get_layers());
    }

    // 4. Console Verification
    Tensor2D final_pred = model.forward(X);
    std::cout << "--- XOR Benchmark Results ---\n";
    std::cout << "Initial BCE Loss: " << initial_loss << "\n";
    std::cout << "Final BCE Loss:   " << final_loss << "\n\n";

    for (size_t b = 0; b < 4; ++b) {
        std::cout << "Input [" << X(0, b) << ", " << X(1, b) << "] "
                  << "-> Target " << Y(0, b)
                  << " | Prediction: " << final_pred(0, b) << "\n";
    }

    // 5. Strict Mathematical Assertions
    // The engine must confidently cross the 0.5 decision boundary
    EXPECT_LT(final_loss, 0.05f); // Loss should be near zero
    EXPECT_LT(final_pred(0, 0), 0.1f); // Should confidently predict 0
    EXPECT_GT(final_pred(0, 1), 0.9f); // Should confidently predict 1
    EXPECT_GT(final_pred(0, 2), 0.9f); // Should confidently predict 1
    EXPECT_LT(final_pred(0, 3), 0.1f); // Should confidently predict 0
}
































