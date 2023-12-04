#include <time.h>

#define NN_IMPLEMENTATION
#include "./lib/neuralnet.h"

size_t arch[] = {2, 2, 1};
size_t max_epoch = 100 * 1000;
float rate = 1.0f;

void verify_nn_gate(NN nn) {
  printf("Verifying XOR Gate:\n");
  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 2; ++j) {
      ROW_AT(NN_INPUT(nn), 0) = (float)i;
      ROW_AT(NN_INPUT(nn), 1) = (float)j;
      nn_forward(nn);
      printf("%zu XOR %zu = %f\n", i, j, ROW_AT(NN_OUTPUT(nn), 0));
    }
  }
  printf("\n");
}

int main(void) {
  Region temp = region_alloc_alloc(256 * 1024 * 1024);

  Mat t = mat_alloc(NULL, 4, 3);
  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 2; ++j) {
      size_t row = i * 2 + j;
      MAT_AT(t, row, 0) = (float)i;
      MAT_AT(t, row, 1) = (float)j;
      MAT_AT(t, row, 2) = (float)(i ^ j);
    }
  }

  NN nn = nn_alloc(NULL, arch, ARRAY_LEN(arch));
  nn_rand(nn, -1, 1);

  size_t epoch = 0;
  while (epoch < max_epoch) {
    NN g = nn_backprop(&temp, nn, t);
    nn_learn(nn, g, rate);
    epoch += 1;
  }

  verify_nn_gate(nn);

  return 0;
}
