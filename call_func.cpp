#include "func.h"

int main() {
  try {
    // Load the shared library
    SharedLibrary lib("/home/clement/projects/libtorch_example/build/"
                      "libexample.so"); // Use "example.dll" on Windows

    // Get a type-erased function
    Function myFunc = lib.getFunction("my_function");

    // Call-site determines the correct function signature automatically
    // (void(int, float))
    myFunc(42, 3.14f);

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
  }

  return 0;
}
