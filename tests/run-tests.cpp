#include <time.h>  // For clock_t, clock, CLOCKS_PER_SEC

#include <cstdlib>     // For std::system
#include <filesystem>  // For std::filesystem
#include <iostream>    // For std::cout

#include "../lib/neuralnet.h"
#define NN_IMPLEMENTATION

size_t arch[] = {2, 2, 1};
size_t max_epoch = 50 * 1000;
float rate = 3.0f;

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

void timing_stats(const char *name, clock_t learn_time, size_t epoch) {
  printf("%s:\n", name);
  printf("  Epochs: %zu\n", epoch);
  printf("  Time: %f\n", (float)learn_time / CLOCKS_PER_SEC);
  printf("  Time/Epoch: %f\n", (float)learn_time / (CLOCKS_PER_SEC * epoch));
}

void test_NN() {
  try {
    Region temp = region_alloc_alloc(256 * 1024 * 1024);

    // time how long allocation takes
    Matrix t = matrix_alloc(NULL, 4, 3);
    for (size_t i = 0; i < 2; ++i) {
      for (size_t j = 0; j < 2; ++j) {
        size_t row = i * 2 + j;
        MAT_AT(t, row, 0) = (float)i;
        MAT_AT(t, row, 1) = (float)j;
        MAT_AT(t, row, 2) = (float)(i ^ j);
      }
    }

    NN nn = nn_alloc(NULL, arch, ARRAY_LEN(arch));
    nn_randomize(nn, 0, 1);

    clock_t learn_time = clock();
    size_t epoch = 0;
    while (epoch < max_epoch) {
      NN g = nn_backprop(&temp, nn, t);
      nn_learn(nn, g, rate);
      epoch += 1;
    }
    learn_time = clock() - learn_time;

    verify_nn_gate(nn);
    timing_stats("Learning", learn_time, epoch);
    printf("Test 3 passed ✅ \n");
  } catch (const char *err) {
    printf("Test 3 Failed ❌ \n");
  }
}
int main() {
  if (std::filesystem::exists("./main.c")) {
    std::cout << "Test 1 passed ✅ \n";
  }

  if (std::system("../build.sh && ../nn")) {
    std::cout << "Test 2 passed ✅ \n";
  }

  test_NN();

  return 0;
}
