#pragma once
#include <cmath>
#include <Eigen/Core>

namespace act_func {

    template<class T> constexpr T sigmoid(T x) {
        return static_cast<T>(1) / (static_cast<T>(1) + std::exp(-x));
    }

    template <class T> constexpr T tanh(T x) {
        return std::tanh(x);
    }

    template <class T> constexpr T relu(T x) {
        return std::max(static_cast<T>(0), x);
    }

    template <class T> constexpr T leaky_relu(T x, T alpha = static_cast<T>(0.01)) {
        return std::max(alpha * x, x);
    }

    template <class T> constexpr T elu(T x, T alpha = static_cast<T>(1)) {
        return (x >= 0) ? x : alpha * (std::exp(x) - 1);
    }

};