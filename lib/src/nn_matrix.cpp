#include <cassert>

#include "../include/nn.h"

namespace nn {

Matrix(ssize_t rows, ssize_t cols) : rows(rows), cols(cols) {
  this->elements = std::make_unique<float[]>(rows * cols);
}

Matrix matrix_alloc(Region *r, size_t rows, size_t cols) {
  Matrix m = {
      .rows = rows,
      .cols = cols,
      .elements = region_alloc(r, rows * cols * sizeof(*m.get_elements())),
  };
  assert(m.get_elements() != NULL);

  return m;
}

void matrix_multi(Matrix dst, Matrix a, Matrix b) {
  assert(a.get_cols() == b.get_rows());
  size_t n = a.get_cols();
  assert(dst.get_rows() == a.get_rows());
  assert(dst.get_cols() == b.get_cols());

  for (size_t i = 0; i < dst.get_rows(); ++i) {
    for (size_t j = 0; j < dst.get_cols(); ++j) {
      MAT_AT(dst, i, j) = 0;
      for (size_t k = 0; k < n; ++k) {
        MAT_AT(dst, i, j) += MAT_AT(a, i, k) * MAT_AT(b, k, j);
      }
    }
  }
}

void matrix_sum(Matrix dest, Matrix a) {
  assert(dest.get_rows() == a.get_rows());
  assert(dest.get_cols() == a.get_cols());

  for (size_t i = 0; i < dest.get_rows(); ++i) {
    for (size_t j = 0; j < dest.get_cols(); ++j) {
      MAT_AT(dest, i, j) += MAT_AT(a, i, j);
    }
  }
}

void matrix_activation(Matrix m) {
  for (size_t i = 0; i < m.get_rows(); ++i) {
    for (size_t j = 0; j < m.get_cols(); ++j) {
      MAT_AT(m, i, j) = activationf(MAT_AT(m, i, j), NN_ACT);
    }
  }
}

void matrix_print(Matrix m, const char *name, size_t padding) {
  printf("%*s%s = [\n", (int)padding, "", name);
  for (size_t i = 0; i < m.rows; ++i) {
    printf("%*s", (int)padding, "");
    for (size_t j = 0; j < m.get_cols(); ++j) {
      printf("%f ", MAT_AT(m, i, j));
    }
    printf("\n");
  }
  printf("%*s]\n", (int)padding, "");
}

void matrix_fill(Matrix m, float num) {
  for (size_t i = 0; i < m.get_rows(); ++i) {
    for (size_t j = 0; j < m.get_cols(); ++j) {
      MAT_AT(m, i, j) = num;
    }
  }
}

void matrix_randomize(Matrix m, float low, float high) {
  for (size_t i = 0; i < m.get_rows(); ++i) {
    for (size_t j = 0; j < m.get_cols(); ++j) {
      MAT_AT(m, i, j) = rand_float() * (high - low) + low;
    }
  }
}

void matrix_sig(Matrix m) {
  for (size_t i = 0; i < m.rows; ++i) {
    for (size_t j = 0; j < m.get_cols(); ++j) {
      MAT_AT(m, i, j) = sigmoidf(MAT_AT(m, i, j));
    }
  }
}

Row matrix_row(Matrix m, size_t row) {
  return (Row){
      .get_cols() = m.get_cols(),
      .elements = &MAT_AT(m, row, 0),
  };
}

void matrix_copy(Matrix dst, Matrix src) {
  assert(dst.rows == src.rows);
  assert(dst.get_cols() == src.get_cols());

  for (size_t i = 0; i < dst.rows; ++i) {
    for (size_t j = 0; j < dst.get_cols(); ++j) {
      MAT_AT(dst, i, j) = MAT_AT(src, i, j);
    }
  }
}

void mat_shuffle_rows(Matrix m) {
  for (size_t i = 0; i < m.rows; ++i) {
    size_t j = i + rand() % (m.rows - i);
    if (i != j) {
      for (size_t k = 0; k < m.get_cols(); ++k) {
        float t = MAT_AT(m, i, k);
        MAT_AT(m, i, k) = MAT_AT(m, j, k);
        MAT_AT(m, j, k) = t;
      }
    }
  }
}

Matrix row_as_matrix(Row row) {
  return (Matrix){
      .rows = 1,
      .cols() = row.get_cols(),
      .elements = row.elements,
  };
}

Row row_slice(Row row, size_t i, size_t cols) {
  assert(i < row.get_cols());
  assert(i + cols <= row.get_cols());
  return (Row){
      .get_cols() = cols,
      .elements = &ROW_AT(row, i),
  };
}
}  // namespace nn
