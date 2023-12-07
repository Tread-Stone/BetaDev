#pragma once

// C++ compatible headers
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

class Row {
public:
  explicit Row(size_t cols);

  void randomize(float low, float high);
  void fill(float x);
  void print(const char *name, size_t padding) const;
  Row slice(size_t i, size_t cols) const;
  Matrix as_matrix() const;
  void copy(const Row &src);

  float &at(size_t col);
  const float &at(size_t col) const;

private:
  size_t cols;
  std::vector<float> &elements;
};

class NN {
public:
  NN(Region &r, const std::vector<size_t> &arch);

  void zero();
  void print(const char *name) const;
  void randomize(float low, float high);
  void forward();
  float cost(const Matrix &t);
  NN backprop(Region &r, const Matrix &t);
  void learn(const NN &gradient, float rate);

  Matrix &input();
  const Matrix &input() const;
  Matrix &output();
  const Matrix &output() const;

  // getters and setters
  std::vector<Matrix> get_weights() const { return weights; }
  std::vector<Matrix> get_biases() const { return biases; }
  std::vector<Matrix> get_activations() const { return activations; }
  std::vector<size_t> get_arch() const { return arch; }

private:
  std::vector<size_t> arch;
  std::vector<Matrix> activations;
  std::vector<Matrix> weights;
  std::vector<Matrix> biases;
};

} // namespace nn
