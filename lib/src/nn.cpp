#include "../include/nn.h"

#include <cassert>
#include <cstddef>

namespace nn {

NN::NN(Region &r, const std::vector<size_t> &architecture) {}

void NN::initalize_weights() {}

float activationf(float x, Activation act) {
  switch (act) {
    case Activation::Sigmoid:
      return sigmoidf(x);
    case Activation::Relu:
      return reluf(x);
    case Activation::Tanh:
      return tanhf(x);
    case Activation::Sin:
      return sinf(x);
    default:
      assert(0 && "Invalid activation function");
      return 0;
  }
}

float activationdf(float x, Activation act) {
  switch (act) {
    case Activation::Sigmoid:
      return x * (1 - x);
    case Activation::Relu:
      return x >= 0 ? 1 : NN_RELU_PARAM;
    case Activation::Tanh:
      return 1 - x * x;
    case Activation::Sin:
      return cosf(asinf(x));
    default:
      assert(0 && "Invalid activation function");
      return 0;
  }
}

void NN::zero() {}

void NN::randomize(float low, float high) {}

void NN::forward() {}

float NN::cost(const Matrix &t) {}

NN NN::backprop(Region &r, const Matrix &t) {}

void NN::learn(const NN &gradient, float rate) {}

}  // namespace nn
