#include "neuralnettmp.h"

namespace neuralnet {

Matrix::Matrix(size_t rows, size_t cols) : rows(rows), cols(cols) {
  elements = std::make_unique<float[]>(rows * cols);
}

Matrix matrix_alloc(Region *r, size_t rows, size_t cols) {
  Matrix m;
  m.rows = rows;
  m.cols = cols;
  m.elements = region_alloc(r, rows * cols * sizeof(*m.elements));
  NN_ASSERT(m.elements != NULL);

  return m;
}

void matrix_multi(Matrix dst, Matrix a, Matrix b) {
  NN_ASSERT(a.cols == b.rows);
  size_t n = a.cols;
  NN_ASSERT(dst.rows == a.rows);
  NN_ASSERT(dst.cols == b.cols);

  for (size_t i = 0; i < dst.rows; ++i) {
    for (size_t j = 0; j < dst.cols; ++j) {
      MAT_AT(dst, i, j) = 0;
      for (size_t k = 0; k < n; ++k) {
        MAT_AT(dst, i, j) += MAT_AT(a, i, k) * MAT_AT(b, k, j);
      }
    }
  }
}

void matrix_sum(Matrix dest, Matrix a) {
  NN_ASSERT(dest.rows == a.rows);
  NN_ASSERT(dest.cols == a.cols);

  for (size_t i = 0; i < dest.rows; ++i) {
    for (size_t j = 0; j < dest.cols; ++j) {
      MAT_AT(dest, i, j) += MAT_AT(a, i, j);
    }
  }
}

void matrix_activation(Matrix m) {
  for (size_t i = 0; i < m.rows; ++i) {
    for (size_t j = 0; j < m.cols; ++j) {
      MAT_AT(m, i, j) = activationf(MAT_AT(m, i, j), NN_ACT);
    }
  }
}

void matrix_print(Matrix m, const char *name, size_t padding) {
  printf("%*s%s = [\n", (int)padding, "", name);
  for (size_t i = 0; i < m.rows; ++i) {
    printf("%*s", (int)padding, "");
    for (size_t j = 0; j < m.cols; ++j) {
      printf("%f ", MAT_AT(m, i, j));
    }
    printf("\n");
  }
  printf("%*s]\n", (int)padding, "");
}

void matrix_fill(Matrix m, float num) {
  for (size_t i = 0; i < m.rows; ++i) {
    for (size_t j = 0; j < m.cols; ++j) {
      MAT_AT(m, i, j) = num;
    }
  }
}

void matrix_randomize(Matrix m, float low, float high) {
  for (size_t i = 0; i < m.rows; ++i) {
    for (size_t j = 0; j < m.cols; ++j) {
      MAT_AT(m, i, j) = rand_float() * (high - low) + low;
    }
  }
}

void matrix_sig(Matrix m) {
  for (size_t i = 0; i < m.rows; ++i) {
    for (size_t j = 0; j < m.cols; ++j) {
      MAT_AT(m, i, j) = sigmoidf(MAT_AT(m, i, j));
    }
  }
}

Row matrix_row(Matrix m, size_t row) {
  return (Row){
      .cols = m.cols,
      .elements = &MAT_AT(m, row, 0),
  };
}

void matrix_copy(Matrix dst, Matrix src) {
  NN_ASSERT(dst.rows == src.rows);
  NN_ASSERT(dst.cols == src.cols);

  for (size_t i = 0; i < dst.rows; ++i) {
    for (size_t j = 0; j < dst.cols; ++j) {
      MAT_AT(dst, i, j) = MAT_AT(src, i, j);
    }
  }
}

void mat_shuffle_rows(Matrix m) {
  for (size_t i = 0; i < m.rows; ++i) {
    size_t j = i + rand() % (m.rows - i);
    if (i != j) {
      for (size_t k = 0; k < m.cols; ++k) {
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
      .cols = row.cols,
      .elements = row.elements,
  };
}

Row row_slice(Row row, size_t i, size_t cols) {
  NN_ASSERT(i < row.cols);
  NN_ASSERT(i + cols <= row.cols);
  return (Row){
      .cols = cols,
      .elements = &ROW_AT(row, i),
  };
}
} // namespace neuralnet
