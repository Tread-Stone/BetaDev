#pragma once

// C++ compatible headers
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>

#ifdef __cplusplus
namespace neuralnet {

#ifndef NN_ACT
#define NN_ACT ACTIVATION_SIGMOID
#endif  // NN_ACT

#ifndef NN_RELU_PARAM
#define NN_RELU_PARAM 0.01f
#endif  // NN_RELU_PARAM

#ifndef NN_MALLOC
#include <stdlib.h>
#define NN_MALLOC malloc
#endif  // NN_MALLOC

#ifndef NN_ASSERT
#include <assert.h>
#define NN_ASSERT assert
#endif  // NN_ASSERT

#define ARRAY_LEN(a) (sizeof(a) / sizeof(a[0]))

typedef enum {
  ACTIVATION_SIGMOID,
  ACTIVATION_RELU,
  ACTIVATION_TANH,
  ACTIVATION_SIN,
} ACTIVATION;

float rand_float(void);
float sigmoidf(float x);
float reluf(float x);
float tanhf(float x);
float activationf(float x, ACTIVATION act);
float activationdf(float x, ACTIVATION act);

class Region {
 public:
  Region(size_t capacity_bytes);
  void *alloc(size_t size_bytes);
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
  Matrix(size_t rows, size_t cols, float *elements);
  void fill(float num);
  void matrix_multi(Matrix dest, Matrix a, Matrix b);
  void randomize(float low, float high);
  void activation();
  void print(const char *name, size_t padding) const;
  void mat_shuffle_rows(Matrix m);
  void matrix_sig(Matrix m);
  void matrix_copy(Matrix dst, Matrix src);

  const float &at(size_t i, size_t j) const {
    assert(i < rows && j < cols);
    return elements[i * cols + j];
  }

 private:
  size_t rows;
  size_t cols;
  std::unique_ptr<float[]> elements;
};

class Row {
 public:
  Row(size_t cols, float *elements);
  float &at(size_t col);
  void randomize(float low, float high);
  void fill(float x);
  void print(const char *name, size_t padding) const;
  Row slice(size_t i, size_t cols);
  Matrix as_matrix(Row row);

 private:
  size_t cols;
  float *elements;
};

class NN {
 public:
  NN(Region &r, size_t *arch, size_t arch_count);
  void zero();
  void print(const char *name) const;
  void randomize(float low, float high);
  void forward();
  float cost(Matrix &t);
  NN backprop(Region &r, Matrix t);
  void learn(NN &gradient, float rate);

 private:
  size_t arch_count;
  size_t *arch;
  Matrix *activations;
  Matrix *weights;
  Matrix *biases;
};

class Batch {
 public:
  Batch();
  void process(Region *r, size_t batch_size, NN nn, Matrix t, float rate);
  bool is_finished();

 private:
  size_t begin;
  float cost;
  bool finished;
};

}  // namespace neuralnet
#endif  // __cplusplus
