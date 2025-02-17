#include "jit_utils.h"
#include <array>
#include <memory>
#include <stdexcept>
#include <cstdio>


std::string strip(const std::string &str) {
  // Find the first non-whitespace character
  size_t start = str.find_first_not_of(" \t\n\r");
  if (start == std::string::npos) {
    return ""; // String is all whitespace
  }

  // Find the last non-whitespace character
  size_t end = str.find_last_not_of(" \t\n\r");

  // Return the substring without leading/trailing whitespace
  return str.substr(start, end - start + 1);
}

std::string executePythonScript(std::string_view command) {
  std::array<char, 128> buffer;
  std::string result;

  // Open the process and read its output
  std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(command.data(), "r"),
                                                pclose);
  if (!pipe) {
    throw std::runtime_error("popen() failed!");
  }

  while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
    result += buffer.data();
  }
  return strip(result);
}
