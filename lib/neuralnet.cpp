#include "./include/nn.h"
#include <cassert>

namespace nn {

float activationf(float x, Activation act) {
  switch (act) {
  case Activation::Sigmoid:
    return sigmoidf(x);
  case Activation::Relu:
    return reluf(x);
  case Activation::Tanh:
    return tanhf(x);
  case Activation::Sin:
    return sinf(x);
  default:
    assert(0 && "Invalid activation function");
    return 0;
  }
}

float activationdf(float x, Activation act) {
  switch (act) {
  case Activation::Sigmoid:
    return x * (1 - x);
  case Activation::Relu:
    return x >= 0 ? 1 : NN_RELU_PARAM;
  case Activation::Tanh:
    return 1 - x * x;
  case Activation::Sin:
    return cosf(asinf(x));
  default:
    assert(0 && "Invalid activation function");
    return 0;
  }
}

NN nn_alloc(Region *r, size_t *arch, size_t arch_count) {
  assert(arch_count > 0);

  NN nn;
  nn.get_arch() = arch;
  nn.arch_count = arch_count;

  nn.weights = region_alloc(r, sizeof(*nn.weights) * nn.arch_count - 1);
  NN_ASSERT(nn.weights != NULL);
  nn.biases = region_alloc(r, sizeof(*nn.biases) * nn.arch_count - 1);
  NN_ASSERT(nn.biases != NULL);
  nn.activations = region_alloc(r, sizeof(*nn.activations) * (nn.arch_count));
  NN_ASSERT(nn.activations != NULL);

  nn.activations[0] = row_alloc(r, arch[0]);

  for (size_t i = 1; i < arch_count; ++i) {
    nn.weights[i - 1] = matrix_alloc(r, nn.activations[i - 1].cols, arch[i]);
    nn.biases[i - 1] = row_alloc(r, arch[i]);
    nn.activations[i] = row_alloc(r, arch[i]);
  }

  return nn;
}

void nn_zero(NN nn) {
  for (size_t i = 0; i < nn.arch_count; ++i) {
    matrix_fill(nn.weights[i], 0);
    row_fill(nn.biases[i], 0);
    row_fill(nn.activations[i], 0);
  }
  row_fill(nn.activations[nn.arch_count], 0);
}

void nn_print(NN nn, const char *name) {
  char buf[256];
  printf("%s = [\n", name);
  for (size_t i = 0; i < nn.arch_count; ++i) {
    snprintf(buf, sizeof(buf), "weights%zu", i);
    matrix_print(nn.weights[i], buf, 4);
    snprintf(buf, sizeof(buf), "biases%zu", i);
    row_print(nn.biases[i], buf, 4);
  }
  printf("]\n");
}

void nn_randomize(NN nn, float low, float high) {
  for (size_t i = 0; i < nn.arch_count; ++i) {
    matrix_randomize(nn.weights[i], low, high);
    row_rand(nn.biases[i], low, high);
  }
}

void nn_forward(NN nn) {
  for (size_t i = 0; i < nn.arch_count - 1; ++i) {
    matrix_multi(row_as_matrix(nn.activations[i + 1]),
                 row_as_matrix(nn.activations[i]), nn.weights[i]);
    matrix_sum(row_as_matrix(nn.activations[i + 1]),
               row_as_matrix(nn.biases[i]));
    matrix_activation(row_as_matrix(nn.activations[i + 1]));
  }
}

float nn_cost(NN nn, Matrix t) {
  NN_ASSERT(NN_INPUT(nn).cols + NN_OUTPUT(nn).cols == t.cols);
  size_t n = t.rows;

  float c = 0;
  for (size_t i = 0; i < n; ++i) {
    Row row = matrix_row(t, i);
    Row x = row_slice(row, 0, NN_INPUT(nn).cols);
    Row y = row_slice(row, NN_INPUT(nn).cols, NN_OUTPUT(nn).cols);

    row_copy(NN_INPUT(nn), x);
    nn_forward(nn);
    size_t q = y.cols;
    for (size_t j = 0; j < q; ++j) {
      float d = ROW_AT(NN_OUTPUT(nn), j) - ROW_AT(y, j);
      c += d * d;
    }
  }

  return c / n;
}

/**
 * TODO HAYDEN: Refactor this
 * integrate dericvative activation function activationdf()
 * Better variable names
 *
 */
NN nn_backprop(Region *r, NN nn, Matrix t) {
  size_t n = t.rows;
  NN_ASSERT(NN_INPUT(nn).cols + NN_OUTPUT(nn).cols == t.cols);

  NN gradient = nn_alloc(r, nn.arch, nn.arch_count);
  nn_zero(gradient);

  // i - current sample
  // l - current layer
  // j - current activation
  // k - previous activation

  for (size_t i = 0; i < n; ++i) {
    Row row = matrix_row(t, i);
    Row in = row_slice(row, 0, NN_INPUT(nn).cols);
    Row out = row_slice(row, NN_INPUT(nn).cols, NN_OUTPUT(nn).cols);

    row_copy(NN_INPUT(nn), in);
    nn_forward(nn);

    for (size_t j = 0; j < nn.arch_count; ++j) {
      row_fill(gradient.activations[j], 0);
    }

    for (size_t j = 0; j < out.cols; ++j) {
      ROW_AT(NN_OUTPUT(gradient), j) =
          ROW_AT(NN_OUTPUT(nn), j) - ROW_AT(out, j);
    }

    float s = 2;

    for (size_t l = nn.arch_count - 1; l > 0; --l) {
      for (size_t j = 0; j < nn.activations[l].cols; ++j) {
        float a = ROW_AT(nn.activations[l], j);
        float da = ROW_AT(gradient.activations[l], j);
        float qa = activationdf(a, NN_ACT);
        ROW_AT(gradient.biases[l - 1], j) += s * da * qa;
        for (size_t k = 0; k < nn.activations[l - 1].cols; ++k) {
          // j - weigradientht matrix col
          // k - weigradientht matrix row
          float pa = ROW_AT(nn.activations[l - 1], k);
          float w = MAT_AT(nn.weights[l - 1], k, j);
          MAT_AT(gradient.weights[l - 1], k, j) += s * da * qa * pa;
          ROW_AT(gradient.activations[l - 1], k) += s * da * qa * w;
        }
      }
    }
  }

  for (size_t i = 0; i < gradient.arch_count - 1; ++i) {
    for (size_t j = 0; j < gradient.weights[i].rows; ++j) {
      for (size_t k = 0; k < gradient.weights[i].cols; ++k) {
        MAT_AT(gradient.weights[i], j, k) /= n;
      }
    }
    for (size_t k = 0; k < gradient.biases[i].cols; ++k) {
      ROW_AT(gradient.biases[i], k) /= n;
    }
  }

  return gradient;
}

void nn_learn(NN nn, NN gradient, float rate) {
  for (size_t i = 0; i < nn.arch_count; ++i) {
    for (size_t j = 0; j < nn.weights[i].rows; ++j) {
      for (size_t k = 0; k < nn.weights[i].cols; ++k) {
        MAT_AT(nn.weights[i], j, k) -= rate * MAT_AT(gradient.weights[i], j, k);
      }
    }

    for (size_t k = 0; k < nn.biases[i].cols; ++k) {
      ROW_AT(nn.biases[i], k) -= rate * ROW_AT(gradient.biases[i], k);
    }
  }
}

/**
 * Actual alloc
 */
Region region_alloc_alloc(size_t capacity_bytes) {
  Region r = {0};

  size_t elements_size = sizeof(*r.elements);
  size_t capacity = (capacity_bytes + elements_size - 1) / elements_size;

  void *elements = NN_MALLOC(capacity * elements_size);
  NN_ASSERT(elements != NULL);
  r.elements = elements;
  r.capacity = capacity;

  return r;
}

void *region_alloc(Region *r, size_t size) {
  if (r == NULL)
    return NN_MALLOC(size);
  size_t word_size = sizeof(*r->elements);
  size_t count = (size + word_size - 1) / word_size;

  NN_ASSERT(r->size + count <= r->capacity);
  if (r->size + count > r->capacity)
    return NULL;
  void *result = &r->elements[r->size];
  r->size += count;
  return result;
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

void batch_process(Region *r, Batch *b, size_t batch_size, NN nn, Matrix t,
                   float rate) {
  if (b->finished) {
    b->finished = false;
    b->begin = 0;
    b->cost = 0;
  }

  size_t size = batch_size;
  if (b->begin + batch_size >= t.rows) {
    size = t.rows - b->begin;
  }

  // TODO: introduce similar to row_slice operation but for Mat that will give
  // you subsequence of rows
  Matrix batch_t = {
      .rows = size,
      .cols = t.cols,
      .elements = &MAT_AT(t, b->begin, 0),
  };

  NN g = nn_backprop(r, nn, batch_t);
  nn_learn(nn, g, rate);
  b->cost += nn_cost(nn, batch_t);
  b->begin += batch_size;

  if (b->begin >= t.rows) {
    size_t batch_count = (t.rows + batch_size - 1) / batch_size;
    b->cost /= batch_count;
    b->finished = true;
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
} // namespace nn
