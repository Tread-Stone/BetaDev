#pragma once

// C++ compatible headers
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

namespace neuralnet {

#ifndef NN_ACT
#define NN_ACT ACTIVATION_SIGMOID
#endif  // NN_ACT

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

class Region {
 public:
  Region(size_t capacity_bytes);

  void* alloc(size_t size_bytes);
  void reset();
  size_t occupied_bytes() const;
  void save();
  void rewind(size_t s);

 private:
  size_t capacity;
  size_t size;
  std::unique_ptr<uintptr_t> elements;
};

class Matrix {
 public:
  Matrix(size_t rows, size_t cols);
  void fill(float num);
  void multiplication(Matrix dest, Matrix a, Matrix b);
  void randomize(float low, float high);
  void activation();
  void print(const char* name, size_t padding) const;
  void shuffle_rows(Matrix m);
  void sigmoid(Matrix m);
  void copy(Matrix dst, Matrix src);
  void sum(const Matrix& a);

  float& at(size_t i, size_t j);
  const float& at(size_t i, size_t j) const;

 private:
  size_t rows, cols;
  std::vector<float> elements;
};

class Row {
 public:
  explicit Row(size_t cols);

  void randomize(float low, float high);
  void fill(float x);
  void print(const char* name, size_t padding) const;
  Row slice(size_t i, size_t cols) const;
  Matrix asMatrix() const;
  void copy(const Row& src);

  float& at(size_t col);
  const float& at(size_t col) const;

 private:
  size_t cols;
  std::vector<float> elements;
};

class NN {
 public:
  NN(Region& r, const std::vector<size_t>& arch);

  void zero();
  void print(const char* name) const;
  void randomize(float low, float high);
  void forward();
  float cost(const Matrix& t);
  NN backprop(Region& r, const Matrix& t);
  void learn(const NN& gradient, float rate);

  Matrix& input();
  const Matrix& input() const;
  Matrix& output();
  const Matrix& output() const;

 private:
  std::vector<size_t> arch;
  std::vector<Matrix> activations;
  std::vector<Matrix> weights;
  std::vector<Matrix> biases;
};

class Batch {
 public:
  Batch();

  void process(Region& r, size_t batch_size, NN& nn, const Matrix& t,
               float rate);
  bool isFinished() const;

 private:
  size_t begin;
  float cost;
  bool finished;
};

}  // namespace neuralnet
