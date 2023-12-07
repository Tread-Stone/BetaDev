#pragma once

#include "nn.h"
#include <cstddef>

class Matrix {
public:
  Matrix(size_t rows, size_t cols);
  void fill(float num);
  void multiplication(Matrix dest, Matrix a, Matrix b);
  void randomize(float low, float high);
  void activation();
  void print(const char *name, size_t padding) const;
  void shuffle_rows(Matrix m);
  void sigmoid(Matrix m);
  void copy(Matrix dst, Matrix src);
  void sum(const Matrix &a);

  float &at(size_t i, size_t j);
  const float &at(size_t i, size_t j) const;

private:
  size_t rows, cols;
  std::vector<float> *elements;
};
