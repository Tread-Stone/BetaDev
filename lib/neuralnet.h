#ifndef NEURALNET_H_
#define NEURALNET_H_

#include <math.h>
#include <stddef.h>
#include <stdio.h>

#ifndef NN_MALLOC
#include <stdlib.h>
#define NN_MALLOC malloc
#endif  // NN_MALLOC

#ifndef NN_ASSERT
#include <assert.h>
#define NN_ASSERT assert
#endif  // NN_ASSERT

#define ARRAY_LEN(a) (sizeof(a) / sizeof(a[0]))

float rand_float(void);
float sigmoidf(float x);

typedef struct {
  size_t rows;
  size_t cols;
  size_t stride;
  float *elements;
} Matrix;

#define MAT_AT(m, i, j) (m).elements[(i) * (m).stride + (j)]

/**
 * Allocates memory for our matrix
 * @param size_t rows
 * @param size_t cols
 */
Matrix matrix_alloc(size_t rows, size_t cols);

/**
 * Fills a matrix with a number
 * @param Matrix m
 * @param float num
 */
void matrix_fill(Matrix m, float num);

/**
 * Randomize our matrix
 * @param Matrix m
 */
void matrix_randomize(Matrix m, float low, float high);

/**
 * Matrix Multiplication
 * @param size_t rows
 * @param size_t cols
 */
void matrix_multi(Matrix dest, Matrix a, Matrix b);

/**
 * Matrix summation
 * @param Matrix dest
 * @param Matrix a
 * @param Matrix b
 */
void matrix_sum(Matrix dest, Matrix a);

/**
 * Returns a single row of a matrix
 */
Matrix matrix_row(Matrix m, size_t row);

/**
 * memcopy a matrix
 */
void matrix_copy(Matrix dst, Matrix src);

/**
 * Applies sigmoid to our matrix
 * @param Matrix m
 */
void matrix_sig(Matrix m);

/**
 * Prints our matrix, and the name
 * @param Matrix m
 * @param char name
 */
void matrix_print(Matrix m, const char *name, size_t padding);
#define MAT_PRINT(m) matrix_print(m, #m);

typedef struct {
  size_t count;
  Matrix *weights;
  Matrix *biases;

  Matrix *activations;  // the amount of activations is count + 1
} NN;

#define NN_INPUT(nn) (nn).activations[0]
#define NN_OUTPUT(nn) (nn).activations[(nn).count]

/**
 * Allocates memory for a neural network
 * @param size_t arch_count
 * @param size_t *arch
 */
NN nn_alloc(size_t *arch, size_t arch_count);
void nn_zero(NN nn);

/**
 * Prints a neural network
 * @param NN nn
 * @param char name
 */
void nn_print(NN nn, const char *name);
#define NN_PRINT(nn) nn_print(nn, #nn);

/**
 * Randomizes a neural network
 * @param NN nn
 * @param float low
 * @param float high
 */
void nn_randomize(NN nn, float low, float high);

/**
 * Forward propagation
 * @param NN nn
 */
void nn_forward(NN nn);

/**
 * Cost function
 * @param NN nn
 * @param Matrix ti
 * @param Matrix to
 */
float nn_cost(NN nn, Matrix ti, Matrix to);

/**
 * Finite difference TODO make this a derivative
 * @param NN nn
 */
void nn_finite_diff(NN nn, NN g, float eps, Matrix ti, Matrix to);

/**
 * Backpropagation
 * @param NN nn
 * @param Matrix ti
 * @param Matrix to
 */
void nn_backprop(NN nn, NN g, Matrix ti, Matrix to);

void nn_learn(NN nn, NN g, float rate);

#endif  // NEURALNET_H_

#ifdef NN_IMPLEMENTATION

float sigmoidf(float x) { return 1.f / (1.f + expf(-x)); }

float rand_float(void) { return (float)rand() / (float)RAND_MAX; }

Matrix matrix_alloc(size_t rows, size_t cols) {
  Matrix m;
  m.rows = rows;
  m.cols = cols;
  m.stride = cols;
  m.elements = NN_MALLOC(sizeof(*m.elements) * rows * cols);
  NN_ASSERT(m.elements != NULL);

  return m;
}

void matrix_multi(Matrix dest, Matrix a, Matrix b) {
  NN_ASSERT(a.cols == b.rows);
  size_t n = a.cols;
  NN_ASSERT(dest.rows == a.rows);
  NN_ASSERT(dest.cols == b.cols);

  for (size_t i = 0; i < dest.rows; ++i) {
    for (size_t j = 0; j < dest.cols; ++j) {
      MAT_AT(dest, i, j) = 0;
      for (size_t k = 0; k < n; ++k) {
        MAT_AT(dest, i, j) += MAT_AT(a, i, k) * MAT_AT(b, k, j);
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

Matrix matrix_row(Matrix m, size_t row) {
  return (Matrix){
      .rows = 1,
      .cols = m.cols,
      .stride = m.stride,
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

NN nn_alloc(size_t *arch, size_t arch_count) {
  NN_ASSERT(arch_count > 0);

  NN nn;
  nn.count = arch_count - 1;

  nn.weights = NN_MALLOC(sizeof(*nn.weights) * nn.count);
  NN_ASSERT(nn.weights != NULL);
  nn.biases = NN_MALLOC(sizeof(*nn.biases) * nn.count);
  NN_ASSERT(nn.biases != NULL);
  nn.activations = NN_MALLOC(sizeof(*nn.activations) * (nn.count + 1));
  NN_ASSERT(nn.activations != NULL);

  nn.activations[0] = matrix_alloc(1, arch[0]);
  for (size_t i = 1; i < arch_count; ++i) {
    nn.weights[i - 1] = matrix_alloc(nn.activations[i - 1].cols, arch[i]);
    nn.biases[i - 1] = matrix_alloc(1, arch[i]);
    nn.activations[i] = matrix_alloc(1, arch[i]);
  }

  return nn;
}

void nn_zero(NN nn) {
  for (size_t i = 0; i < nn.count; ++i) {
    matrix_fill(nn.weights[i], 0);
    matrix_fill(nn.biases[i], 0);
    matrix_fill(nn.activations[i], 0);
  }
  matrix_fill(nn.activations[nn.count], 0);
}

void nn_print(NN nn, const char *name) {
  char buf[256];
  printf("%s = [\n", name);
  for (size_t i = 0; i < nn.count; ++i) {
    snprintf(buf, sizeof(buf), "weights%zu", i);
    matrix_print(nn.weights[i], buf, 4);
    snprintf(buf, sizeof(buf), "biases%zu", i);
    matrix_print(nn.biases[i], buf, 4);
  }
  printf("]\n");
}

void nn_rand(NN nn, float low, float high) {
  for (size_t i = 0; i < nn.count; ++i) {
    matrix_randomize(nn.weights[i], low, high);
    matrix_randomize(nn.biases[i], low, high);
  }
}

void nn_forward(NN nn) {
  for (size_t i = 0; i < nn.count; ++i) {
    matrix_multi(nn.activations[i + 1], nn.activations[i], nn.weights[i]);
    matrix_sum(nn.activations[i + 1], nn.biases[i]);
    matrix_sig(nn.activations[i + 1]);
  }
}

float nn_cost(NN nn, Matrix ti, Matrix to) {
  NN_ASSERT(ti.rows == to.rows);
  NN_ASSERT(to.cols == NN_OUTPUT(nn).cols);
  size_t n = ti.rows;

  float c = 0;
  for (size_t i = 0; i < n; ++i) {
    Matrix x = matrix_row(ti, i);
    Matrix y = matrix_row(to, i);

    matrix_copy(NN_INPUT(nn), x);
    nn_forward(nn);
    size_t q = to.cols;
    for (size_t j = 0; j < q; ++j) {
      float d = MAT_AT(NN_OUTPUT(nn), 0, j) - MAT_AT(y, 0, j);
      c += d * d;
    }
  }

  return c / n;
}

void nn_backprop(NN nn, NN g, Matrix ti, Matrix to) {
  NN_ASSERT(ti.rows == to.rows);
  size_t n = ti.rows;
  NN_ASSERT(NN_OUTPUT(nn).cols == to.cols);

  nn_zero(g);

  // i - current sample
  // l - current layer
  // j - current activations
  // k - previous activations

  for (size_t i = 0; i < n; ++i) {
    matrix_copy(NN_INPUT(nn), matrix_row(ti, i));
    nn_forward(nn);

    for (size_t j = 0; j <= nn.count; ++j) {
      matrix_fill(g.activations[j], 0);
    }

    for (size_t j = 0; j < to.cols; ++j) {
      MAT_AT(NN_OUTPUT(g), 0, j) =
          MAT_AT(NN_OUTPUT(nn), 0, j) - MAT_AT(to, i, j);
    }

    for (size_t l = nn.count; l > 0; --l) {
      for (size_t j = 0; j < nn.activations[l].cols; ++j) {
        float a = MAT_AT(nn.activations[l], 0, j);
        float da = MAT_AT(g.activations[l], 0, j);
        MAT_AT(g.biases[l - 1], 0, j) += 2 * da * a * (1 - a);
        for (size_t k = 0; k < nn.activations[l - 1].cols; ++k) {
          // j - weight matrix col
          // k - weight matrix row
          float pa = MAT_AT(nn.activations[l - 1], 0, k);
          float w = MAT_AT(nn.weights[l - 1], k, j);
          MAT_AT(g.weights[l - 1], k, j) += 2 * da * a * (1 - a) * pa;
          MAT_AT(g.activations[l - 1], 0, k) += 2 * da * a * (1 - a) * w;
        }
      }
    }
  }

  for (size_t i = 0; i < g.count; ++i) {
    for (size_t j = 0; j < g.weights[i].rows; ++j) {
      for (size_t k = 0; k < g.weights[i].cols; ++k) {
        MAT_AT(g.weights[i], j, k) /= n;
      }
    }
    for (size_t j = 0; j < g.biases[i].rows; ++j) {
      for (size_t k = 0; k < g.biases[i].cols; ++k) {
        MAT_AT(g.biases[i], j, k) /= n;
      }
    }
  }
}

void nn_finite_diff(NN nn, NN g, float eps, Matrix ti, Matrix to) {
  float saved;
  float c = nn_cost(nn, ti, to);

  for (size_t i = 0; i < nn.count; ++i) {
    for (size_t j = 0; j < nn.weights[i].rows; ++j) {
      for (size_t k = 0; k < nn.weights[i].cols; ++k) {
        saved = MAT_AT(nn.weights[i], j, k);
        MAT_AT(nn.weights[i], j, k) += eps;
        MAT_AT(g.weights[i], j, k) = (nn_cost(nn, ti, to) - c) / eps;
        MAT_AT(nn.weights[i], j, k) = saved;
      }
    }

    for (size_t j = 0; j < nn.biases[i].rows; ++j) {
      for (size_t k = 0; k < nn.biases[i].cols; ++k) {
        saved = MAT_AT(nn.biases[i], j, k);
        MAT_AT(nn.biases[i], j, k) += eps;
        MAT_AT(g.biases[i], j, k) = (nn_cost(nn, ti, to) - c) / eps;
        MAT_AT(nn.biases[i], j, k) = saved;
      }
    }
  }
}

void nn_learn(NN nn, NN g, float rate) {
  for (size_t i = 0; i < nn.count; ++i) {
    for (size_t j = 0; j < nn.weights[i].rows; ++j) {
      for (size_t k = 0; k < nn.weights[i].cols; ++k) {
        MAT_AT(nn.weights[i], j, k) -= rate * MAT_AT(g.weights[i], j, k);
      }
    }

    for (size_t j = 0; j < nn.biases[i].rows; ++j) {
      for (size_t k = 0; k < nn.biases[i].cols; ++k) {
        MAT_AT(nn.biases[i], j, k) -= rate * MAT_AT(g.biases[i], j, k);
      }
    }
  }
}

#endif  // NN_IMPLEMENTATION
