#pragma once

#include <functional>


template<class T = double>
class NeuralNetwork {
public:
    template<class F>
    requires std::invocable<F, T, act_func::FunctionForm>
    NeuralNetwork(std::initializer_list<size_t> input_shape, F &&activation_function,
                  double learning_rate = 0.01) : shape(input_shape),
                                                 activation_function(activation_function),
                                                 learning_rate(learning_rate) {
        layers.reserve(shape.size());
        errors.reserve(shape.size());

        for (size_t i = 0; i < shape.size(); i++) {
            int layerSize = shape.at(i) + (i < shape.size() - 1 ? 1 : 0); // bias neuron
            layers.emplace_back(layerSize);
            errors.emplace_back(layerSize);
            if (i < shape.size() - 1) {
                layers.at(i)(layerSize - 1) = 1.0;
            }
        }

        for (size_t i = 0; i < shape.size() - 1; i++) {
            int rows = layers.at(i).size();
            int cols = layers.at(i + 1).size() - (i + 1 < shape.size() - 1 ? 1 : 0); // bias
            Eigen::MatrixXd weightMatrix = Eigen::MatrixXd::Random(rows, cols);
            weights.push_back(weightMatrix);
        }
    }


    /**
     * a^l = f(a^(l - 1) * W^(l - 1))
     *
     * a^l : activation of layer l
     * W^(l - 1) : weight matrix between layer l - 1 and layer l
     * f : activation function
     */
    void forward_propagation(const Eigen::RowVectorXd &input) {
        layers.at(0).head(shape.at(0)) = input;

        for (size_t i = 1; i < shape.size(); i++) {
            Eigen::VectorXd netInput = weights.at(i - 1).transpose() * layers.at(i - 1);

            for (int j = 0; j < netInput.size(); j++) {
                layers.at(i)(j) = this->activation_function(netInput(j), act_func::FunctionForm::BASE);
            }

            if (i < shape.size() - 1) {
                layers.at(i)(layers.at(i).size() - 1) = 1.0;
            }
        }
    }


    /**
     * ∂^l = (a^l - y) * f'(z^l)
     *
     * ∂^l : error of layer l
     * a^l : activation of layer l
     * y : target output
     * z^l : weighted input of layer l
     * f' : derivative of activation function
     *
     */
    void back_propagation(const Eigen::RowVectorXd &target) {
        Eigen::VectorXd new_input = weights.back().transpose() * layers.at(layers.size() - 2);
        for (int i = 0; i < shape.back(); i++) {
            double error = layers.back()(i) - target(i);
            errors.back()(i) = error * this->activation_function(new_input(i), act_func::FunctionForm::DERIVATIVE);
        }

        Eigen::VectorXd delta;
        for (int i = shape.size() - 2; i > 0; i--) {
            new_input = weights.at(i - 1).transpose() * layers.at(i - 1);
            delta = weights.at(i) * errors.at(i + 1).head(shape.at(i + 1));
            for (int j = 0; j < shape[i]; ++j) {
                errors.at(i)(j) = delta(j) * this->activation_function(new_input(j), act_func::FunctionForm::DERIVATIVE);
            }
        }

        for (size_t i = 0; i < weights.size(); i++) {
            weights.at(i) -= learning_rate * (layers.at(i) * errors.at(i + 1).head(shape.at(i + 1)).transpose());
        }
    }

    void print_dimensions() const {
        std::cout << "Network Dimensions:\n";
        for (size_t i = 0; i < shape.size(); i++) {
            std::cout << "Layer " << i << ": " << layers[i].size()
                      << " (active: " << shape[i] << ")\n";
            if (i > 0) {
                std::cout << "Weight " << i - 1 << ": " << weights[i - 1].rows()
                          << "x" << weights[i - 1].cols() << "\n";
            }
        }
        std::flush(std::cout);
    }


    Eigen::VectorXd train(const Eigen::RowVectorXd &input, const Eigen::RowVectorXd &target) {
//        print_dimensions();
        forward_propagation(input);
        auto out = this->get_output();
        back_propagation(target);
        return out;
    }

    Eigen::VectorXd predict(const Eigen::RowVectorXd &input) {
        forward_propagation(input);
        return layers.back().head(shape.back());
    }

    [[nodiscard]] Eigen::VectorXd get_output() const {
        return layers.back().head(shape.back());
    }


private:
    std::function<T(T, act_func::FunctionForm)> activation_function;
    std::vector<size_t> shape;
    double learning_rate = 0.01;
    std::vector<Eigen::VectorXd> layers;
    std::vector<Eigen::VectorXd> errors;
    std::vector<Eigen::MatrixXd> weights;
};
