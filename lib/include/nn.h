#pragma once

// C++ compatible headers
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <vector>

#include "matrix.h"
#include "region.h"

namespace nn {

constexpr float NN_RELU_PARAM = 0.01f;

enum class Activation {
  Sigmoid,
  Relu,
  Tanh,
  Sin,
};

// utility functions
inline float rand_float() {
  return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
};

inline float sigmoidf(float x) { return 1.0f / (1.0f + std::exp(-x)); };

inline float reluf(float x) { return x > 0.0f ? x : NN_RELU_PARAM * x; };

inline float tanhf(float x) { return std::tanh(x); };

float activationf(float x, Activation act);
float activationdf(float x, Activation act);

class NN {
 public:
  NN(Region &r, const std::vector<size_t> &architecture);
  float activationf(float x, Activation act);
  float activationdf(float x, Activation act);
  static double random_double(double min, double max);

  // architecture
  std::vector<size_t> architecture_;
  // weights
  std::vector<Matrix> weights_;
  // biases
  std::vector<Matrix> biases_;
  // activations
  std::vector<Matrix> activations_;
  // gradients
  std::vector<Matrix> gradients_;
  // cost
  float cost_;
  // learning rate
  float rate_;
  // momentum
  float momentum_;
  // weight decay
  float decay_;
  // dropout
  float dropout_;
  // dropout mask
  std::vector<Matrix> dropout_mask_;
  // activation function
  Activation activation_;
  // output activation function
  Activation output_activation_;

 private:
};

}  // namespace nn
