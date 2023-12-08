#include <cassert>
#include <cstddef>

#include "../include/nn.h"

namespace nn {

void Matrix::fill(Matrix mat, double val) {
  for (size_t i = 0; i < mat.rows_; ++i) {
    for (size_t j = 0; j < mat.cols_; ++j) {
      Matrix::at(mat, mat.rows_, mat.cols_) = val;
    }
  }
}

void Matrix::randomize(Matrix mat, double min, double max) {
  for (size_t i = 0; i < mat.rows_; ++i) {
    for (size_t j = 0; j < mat.cols_; ++j) {
      Matrix::at(mat, mat.rows_, mat.cols_) = NN::random_double(min, max);
    }
  }
}

void Matrix::multiply(Matrix dst, Matrix x, Matrix y) {
  assert(x.cols_ == y.rows_);
  assert(dst.rows_ == x.rows_);
  assert(dst.cols_ == y.cols_);
  for (size_t i = 0; i < dst.rows_; ++i) {
    for (size_t j = 0; j < dst.cols_; ++j) {
      double sum = 0;
      for (size_t k = 0; k < x.cols_; ++k) {
        sum += Matrix::at(x, i, k) * Matrix::at(y, k, j);
      }
      Matrix::at(dst, i, j) = sum;
    }
  }
}

void Matrix::add(Matrix dst, Matrix x) {
  assert(dst.rows_ == x.rows_);
  assert(dst.cols_ == x.cols_);
  for (size_t i = 0; i < dst.rows_; ++i) {
    for (size_t j = 0; j < dst.cols_; ++j) {
      Matrix::at(dst, i, j) += Matrix::at(x, i, j);
    }
  }
}

}  // namespace nn
