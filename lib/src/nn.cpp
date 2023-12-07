#include "../include/nn.h"
#include <cstddef>

namespace nn {

NN::NN(Region &r, const std::vector<size_t> &architecture) {
  initalize_weights();
  set_activation(Activation::Sigmoid);
}

void NN::initalize_weights() {
  for (size_t i = 0; i < architecture.size() - 1; i++) {
    weights.push_back(Matrix(architecture[i], architecture[i + 1]));
    biases.push_back(Matrix(1, architecture[i + 1]));
  }
}

void NN::set_activation(Activation act) {
  switch (act) {
  case Activation::Sigmoid:
    activationf = [](float x) { return 1.0f / (1.0f + std::exp(-x)); };
    activationdf = [](float x) { return x * (1 - x); };
    break;
  case Activation::Relu:
    activationf = [](float x) { return x > 0 ? x : 0; };
    activationdf = [](float x) { return x > 0 ? 1 : 0; };
    break;
  case Activation::Sin:
    activationf = [](float x) { return x > 0 ? x : 0.01f * x; };
    activationdf = [](float x) { return x > 0 ? 1 : 0.01f; };
    break;
  case Activation::Tanh:
    activationf = [](float x) { return std::tanh(x); };
    activationdf = [](float x) { return 1 - x * x; };
    break;
  }
}
} // namespace nn
