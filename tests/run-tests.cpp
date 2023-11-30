#include <filesystem>

void test_init() {
  // Test 1: ...
  if (std::filesystem::exists("./main.cpp")) {
    printf("Test 1 passed ✅ \n");
  }
}

int main() {
  test_init();
  return 0;
}
