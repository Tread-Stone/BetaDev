#pragma once

#include <cstddef>
#include <vector>

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

  // Getters and setters
  size_t get_rows() const { return rows; }
  size_t get_cols() const { return cols; }
  float *get_elements() { return elements->data(); }
  const float *get_elements() const { return elements->data(); }

  float &at(size_t i, size_t j);
  const float &at(size_t i, size_t j) const;

 private:
  size_t rows, cols;
  std::vector<float> *elements;
};
