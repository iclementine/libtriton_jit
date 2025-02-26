#include <dlfcn.h> // Use <windows.h> and GetProcAddress for Windows
#include <iostream>
#include <stdexcept>

/*
This is a example in In c++, how to call a opaque function  loaded from some binary stub, without
knowning the signature of the function at compile time.(I mean the caller code)
Since the function is define in DSL, or some other languages without a signature that is seen by the
c++ compiler.
*/
class Function {
public:
  Function(void *handle, const std::string &functionName) : handle_(handle) {
    funcPtr_ = dlsym(handle_, functionName.c_str());
    if (!funcPtr_) {
      throw std::runtime_error("Failed to load function: " +
                               std::string(dlerror()));
    }
  }

  /* This the key part, the signature(args types only, return type cannot be infered)
  can be infered from input arguments, so we can get a signature from call-site that
  is used to reinterpre_cast the function pointer to, so we can call the function with
  that signature. Since the call-site is known at compiled-time, */
  template <typename... Args> void operator()(Args... args) {
    // const, volatile, reference decay?
    using FuncType = void (*)(Args...);
    FuncType func = reinterpret_cast<FuncType>(funcPtr_);
    func(args...);
  }

private:
  void *handle_ = nullptr;
  void *funcPtr_ = nullptr;
};

// Wrapper for loading shared libraries
class SharedLibrary {
public:
  explicit SharedLibrary(const std::string &path) {
    handle_ = dlopen(path.c_str(), RTLD_LAZY);
    if (!handle_) {
      throw std::runtime_error("Failed to open library: " +
                               std::string(dlerror()));
    }
  }

  ~SharedLibrary() {
    if (handle_) {
      dlclose(handle_);
    }
  }

  Function getFunction(const std::string &name) {
    return Function(handle_, name);
  }

private:
  void *handle_ = nullptr;
};