#include <cstdlib>     // For std::system
#include <filesystem>  // For std::filesystem
#include <iostream>    // For std::cout

int main() {
  if (std::filesystem::exists("./main.c")) {
    std::cout << "Test 1 passed ✅ \n";
  }

  if (std::system("../build.sh && ../nn")) {
    std::cout << "Test 2 passed ✅ \n";
  }

  if (std::system("python3 ../big_data/mnist.py")) {
    std::cout << "Test 3 passed ✅ \n";
  }

  return 0;
}
