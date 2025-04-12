#include <iostream>
#include <Eigen/Core>
#include "activation_function.hpp"
#include <ranges>
#include <neural_network.hpp>
#include <sciplot/sciplot.hpp>
#include "data_loader.hpp"

MNISTData mnist_train;
MNISTData mnist_test;
constexpr size_t epoch_size = 1;


Eigen::RowVectorXd img_to_row_vector(const float *image_data) {
    Eigen::Map<const Eigen::RowVectorXf> float_map(image_data, 28 * 28);

    return float_map.cast<double>();
}

int main() {

    NeuralNetwork nn({784, 30, 10}, act_func::relu<>, 0.1);

    if (!mnist_train.Load(true) || !mnist_test.Load(false)) {
        std::cerr << "Failed to load MNIST data." << std::endl;
        return -1;
    }

    std::vector<double> losses;
    losses.reserve(epoch_size);

    Eigen::RowVectorXd input(784);



    int g = 0;
    for (int epoch = 0; epoch < epoch_size; epoch++) {
        double epoch_loss = 0.0;
        for (size_t i = 0, c = mnist_train.NumImages(); i < c; ++i) {
            g++;
            uint8_t label;
            const float* pixels = mnist_train.GetImage(i, label);
            input = img_to_row_vector(pixels);

            Eigen::RowVectorXd target = Eigen::RowVectorXd::Zero(10);
            target(label) = 1.0;
            Eigen::RowVectorXd output = nn.train(input, target);
//            losses.push_back((output - target).squaredNorm());
            std::cout << "epoch: " << epoch << ", times: " << g << std::endl;
            double error = (output - target).squaredNorm();
            epoch_loss += error;
        }
//        losses.push_back(epoch_loss / mnist_train.NumImages());
    }


    size_t correct = 0;
    std::vector<double> accuracies;
    for (size_t i = 0, c = mnist_test.NumImages(); i < c; ++i) {
        g++;
        uint8_t label;
        const float* pixels = mnist_test.GetImage(i, label);
        input = img_to_row_vector(pixels);

        Eigen::RowVectorXd target = Eigen::RowVectorXd::Zero(10);
        target(label) = 1.0;
        Eigen::RowVectorXd output = nn.predict(input);
//            losses.push_back((output - target).squaredNorm());
        Eigen::Index ans;
        double confidence = output.maxCoeff(&ans);
        if(label == ans) {
            correct++;
        }
        double error = (output - target).squaredNorm();
        losses.push_back(error);
    }

    std::cout << "Accuracy: " << (double)correct / mnist_test.NumImages() * 100 << "%" << std::endl;




//    // Train the network
//    for (int epoch = 0; epoch < 10000; epoch++) {
//        double epoch_loss = 0.0;
//        for (size_t i = 0; i < inputs.size(); i++) {
//            auto output = nn.train(inputs.at(i), targets.at(i));
//            double error = (output - targets[i]).squaredNorm();
//            epoch_loss += error;
//        }
//
//        losses.push_back(epoch_loss / inputs.size());
//
//    }

    sciplot::Plot2D plot;
    plot.xlabel("epoch");
    plot.ylabel("loss");
    plot.legend()
            .atOutsideBottom()
            .displayHorizontal()
            .displayExpandWidthBy(2);

    plot.drawCurve(
            sciplot::linspace(0, losses.size(), losses.size()),
            losses
    ).label("Training Loss").lineWidth(2);
    plot.xtics().at({0, static_cast<double>(mnist_test.NumImages() / 2), static_cast<double>(mnist_test.NumImages())});

    sciplot::Figure fig = {{plot}};
    sciplot::Canvas canvas = {{fig}};
    canvas.show();

    // Test the network
//    for (size_t i = 0; i < inputs.size(); ++i) {
//        Eigen::RowVectorXd output = nn.predict(inputs[i]);
//        std::cout << "Input: " << inputs[i].transpose() << " Output: " << output(0) << std::endl;
//    }


    return 0;
}
