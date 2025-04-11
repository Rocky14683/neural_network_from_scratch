#include <gtest/gtest.h>
#include "activation_function.hpp"

TEST(HelloWorldTest, BasicAssertions) {
    EXPECT_EQ(1, 1);
    EXPECT_TRUE(true);
}

TEST(sigmoid, PositiveInput) {
    EXPECT_NEAR(act_func::sigmoid(1.0), 0.7310585786300049, 1e-9);
}

TEST(sigmoid, NegativeInput) {
    EXPECT_NEAR(act_func::sigmoid(-1.0), 0.2689414213699951, 1e-9);
}

TEST(sigmoid, ZeroInput) {
    EXPECT_NEAR(act_func::sigmoid(0.0), 0.5, 1e-9);
}

TEST(tanh, PositiveInput) {
    EXPECT_NEAR(act_func::tanh(1.0), 0.7615941559557649, 1e-9);
}

TEST(tanh, NegativeInput) {
    EXPECT_NEAR(act_func::tanh(-1.0), -0.7615941559557649, 1e-9);
}

TEST(tanh, ZeroInput) {
    EXPECT_NEAR(act_func::tanh(0.0), 0.0, 1e-9);
}

TEST(relu, PositiveInput) {
    EXPECT_EQ(act_func::relu(1.0), 1.0);
}

TEST(relu, NegativeInput) {
    EXPECT_EQ(act_func::relu(-1.0), 0.0);
}

TEST(relu, ZeroInput) {
    EXPECT_EQ(act_func::relu(0.0), 0.0);
}

TEST(leaky_relu, PositiveInput) {
    EXPECT_EQ(act_func::leaky_relu(1.0), 1.0);
}

TEST(leaky_relu, NegativeInput) {
    EXPECT_NEAR(act_func::leaky_relu(-1.0), -0.01, 1e-9);
}

TEST(leaky_relu, ZeroInput) {
    EXPECT_EQ(act_func::leaky_relu(0.0), 0.0);
}

TEST(elu, PositiveInput) {
    EXPECT_EQ(act_func::elu(1.0), 1.0);
}

TEST(elu, NegativeInput) {
    EXPECT_NEAR(act_func::elu(-1.0), -0.6321205588285577, 1e-4);
}

TEST(elu, ZeroInput) {
    EXPECT_EQ(act_func::elu(0.0), 0.0);
}
