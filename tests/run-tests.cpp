void test_init() {
  if (std::filesystem::exists("./main.cpp")) {
    printf("Test 1 passed ✅ \n");
  }
}

void test_xor() {
  if (std::system("../build.sh && ../nn")) {
    printf("Test 2 passed ✅ \n");
  };
}

void test_pipeline() {
  if (std::system("python3 ../mnist.py")) {
    printf("Test 3 passed ✅ \n");
  };
}

int main() {
  test_init();
  test_xor();
  return 0;
}
