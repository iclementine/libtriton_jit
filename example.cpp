#include <iostream>

extern "C" void my_function(int a, float b) {
    std::cout << "my_function called with a=" << a << ", b=" << b << std::endl;
}