#pragma once

#include <cassert>
#include <cstddef>
#include <vector>

#include "region.h"

namespace nn {

class Matrix {
#define MAT_AT(m, i, j) (m).elements[(i) * (m).cols + (j)]

 public:
  Matrix(Region* region, size_t rows, size_t cols)
      : rows_(rows), cols_(cols), elements_(rows * cols) {}
  void fill(Matrix mat, double val);
  void randomize(Matrix mat, double min, double max);
  void add(Matrix dst, Matrix x);
  void multiply(Matrix dst, Matrix x, Matrix y);

  double& at(Matrix mat, size_t row, size_t col) {
    assert(row < mat.rows_);
    assert(col < mat.cols_);
    return (mat).elements_[(row) * (mat).cols_ + (col)];
  }

 private:
  // rows cols and elements
  size_t rows_;
  size_t cols_;
  std::vector<double> elements_;
};

// Row is a view of a row of a matrix.
class Row {
 public:
 private:
  // cols and elements
};
}  // namespace nn
