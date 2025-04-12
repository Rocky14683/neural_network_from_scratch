#pragma once

#include <cmath>
#include <Eigen/Core>

namespace act_func {

    enum FunctionForm {
        BASE,
        DERIVATIVE,
        D = DERIVATIVE
    };


    template<class T = double>
    constexpr T sigmoid(T x, FunctionForm form = BASE) {
        if (form == DERIVATIVE) {
            return sigmoid(x) * (1 - sigmoid(x));
        } else {
            return static_cast<T>(1) / (static_cast<T>(1) + std::exp(-x));
        }
    }


    template<class T = double>
    constexpr T tanh(T x, FunctionForm form = BASE) {
        if (form == DERIVATIVE) {
            return 1 - std::pow(tanh(x), 2);
        } else {
            return std::tanh(x);
        }
    }


    template<class T = double>
    constexpr T relu(T x, FunctionForm form = BASE) {
        if (form == DERIVATIVE) {
            return (x > 0) ? static_cast<T>(1) : static_cast<T>(0);
        } else {
            return std::max(static_cast<T>(0), x);
        }
    }


    template<class T = double>
    constexpr T leaky_relu(T x, FunctionForm form = BASE) {
        if (form == DERIVATIVE) {
            return (x > 0) ? static_cast<T>(1) : static_cast<T>(0.01);
        } else {
            return std::max(static_cast<T>(0.01) * x, x);
        }
    }


    template<class T = double>
    constexpr T elu(T x, T alpha = static_cast<T>(1), FunctionForm form = BASE) {
        if (form == DERIVATIVE) {
            return (x >= 0) ? 1 : alpha * std::exp(x);
        } else {
            return (x >= 0) ? x : alpha * (std::exp(x) - 1);
        }
    }


};