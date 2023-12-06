#pragma once
#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

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

/**
 * Takes values from -oo to +oo and maps them to 0 to 1
 * @param float x
 */
float sigmoidf(float x);

/**
 * F(x) = x if x > 0 else x * NN_RELU_PARAM
 * @param float x
 */
float reluf(float x);

/**
 * Hyperbolic time chamber tanh
 * @param float x
 * @return float
 */
float tanhf(float x);

/**
 * Dispatch to the corresponding activation function
 * @param float x
 * @param ACTIVATION act
 * @return float
 */
float activationf(float x, ACTIVATION act);

/**
 * Derivative of the activation function
 * @param float x
 * @param ACTIVATION activations
 * @return float
 */
float activationdf(float x, ACTIVATION act);

/**
 * A region is a block of memory that can be allocated from
 * @param Region *r
 * @param size_t size
 * @param uintptr_t elements
 * @return void *
 */
typedef struct {
  size_t capacity;
  size_t size;
  uintptr_t *elements;
} Region;

// capacity is in bytes
Region region_alloc_alloc(size_t capacity_bytes);
void *region_alloc(Region *r, size_t size_bytes);
#define region_reset(r) (NN_ASSERT((r) != NULL), (r)->size = 0)
#define region_occupied_bytes(r) \
  (NN_ASSERT((r) != NULL), (r)->size * sizeof(*(r)->elements))
#define region_save(r) (NN_ASSERT((r) != NULL), (r)->size)
#define region_rewind(r, s) (NN_ASSERT((r) != NULL), (r)->size = s)

typedef struct {
  size_t rows;
  size_t cols;
  float *elements;
} Matrix;

typedef struct {
  size_t cols;
  float *elements;
} Row;

#define ROW_AT(row, col) (row).elements[(col)]

Matrix row_as_matrix(Row row);
#define row_alloc(r, cols) matrix_row(matrix_alloc(r, 1, cols), 0)
Row row_slice(Row row, size_t i, size_t cols);
#define row_rand(row, low, high) matrix_randomize(row_as_matrix(row), low, high)
#define row_fill(row, x) matrix_fill(row_as_matrix(row), x);
#define row_print(row, name, padding) \
  matrix_print(row_as_matrix(row), name, padding)
#define row_copy(dst, src) matrix_copy(row_as_matrix(dst), row_as_matrix(src))

#define MAT_AT(m, i, j) (m).elements[(i) * (m).cols + (j)]

/**
 * Allocates memory for our matrix
 * @param size_t rows
 * @param size_t cols
 */
Matrix matrix_alloc(Region *r, size_t rows, size_t cols);

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
 * Matrix activation, applies the activation function to our matrix
 * @param Matrix m
 */
void matrix_activation(Matrix m);

/**
 * Returns a single row of a matrix
 */
Row matrix_row(Matrix m, size_t row);

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

/**
 * Shuffles the rows of a matrix
 * @param Matrix m
 */
void mat_shuffle_rows(Matrix m);

#define MAT_PRINT(m) matrix_print(m, #m);

typedef struct {
  size_t *arch;
  size_t arch_count;

  Matrix *weights;
  Row *biases;

  Row *activations;  // the amount of activations is count + 1
} NN;

#define NN_INPUT(nn) (NN_ASSERT((nn).arch_count > 0), (nn).activations[0])
#define NN_OUTPUT(nn) \
  (NN_ASSERT((nn).arch_count > 0), (nn).activations[(nn).arch_count - 1])

/**
 * Allocates memory for a neural network
 * @param size_t arch_count
 * @param size_t *arch
 */
NN nn_alloc(Region *r, size_t *arch, size_t arch_count);
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
 * @return float
 */
float nn_cost(NN nn, Matrix t);

/**
 * Backpropagation
 * @param NN nn
 * @param Matrix ti
 * @param Matrix to
 */
NN nn_backprop(Region *r, NN nn, Matrix t);

void nn_learn(NN nn, NN g, float rate);

typedef struct {
  size_t begin;
  float cost;
  bool finished;
} Batch;

void batch_process(Region *r, Batch *b, size_t batch_size, NN nn, Matrix t,
                   float rate);

#ifdef __cplusplus
}
#endif  // NEURALNET_H_
