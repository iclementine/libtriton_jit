#pragma once

#include <cstdlib>
#include <filesystem>
#include <string>
#include "torch/torch.h"

namespace triton_jit {
std::string execute_command(std::string_view command);

constexpr const char *to_triton_typename(c10::ScalarType t) {
  switch (t) {
    case c10::ScalarType::Float:
      return "fp32";
    case c10::ScalarType::Double:
      return "fp64";
    case c10::ScalarType::Half:
      return "fp16";
    case c10::ScalarType::BFloat16:
      return "bf16";
    case c10::ScalarType::Int:
      return "i32";
    case c10::ScalarType::Long:
      return "i64";
    case c10::ScalarType::Bool:
      return "i1";
    default:
      return "<unsupported_type>";
  }
}

template <typename T>
constexpr const char *spec(T v) {
  return v % 16 == 0 ? ":16" : v == 1 ? ":1" : "";
}

template <typename T, typename = void>
struct has_data_ptr : std::false_type {};

template <typename T>
struct has_data_ptr<
    T,
    std::enable_if_t<std::conjunction_v<
        std::is_same<decltype(std::declval<std::remove_reference_t<T>>().data_ptr()), void *>,
        std::is_same<decltype(std::declval<std::remove_reference_t<T>>().scalar_type()), c10::ScalarType>>>>
    : std::true_type {};

template <typename T>
struct triton_type_helper;

#define DEFINE_TRITON_TYPE(T, Name)           \
  template <>                                 \
  struct triton_type_helper<T> {              \
    static constexpr const char *name = Name; \
  }

DEFINE_TRITON_TYPE(bool, "i1");
DEFINE_TRITON_TYPE(int, "i32");
DEFINE_TRITON_TYPE(int64_t, "i64");
DEFINE_TRITON_TYPE(float, "f32");
DEFINE_TRITON_TYPE(double, "f64");
DEFINE_TRITON_TYPE(std::nullptr_t, "*i8");

#undef DEFINE_TRITON_TYPE

template <typename T>
struct triton_type : triton_type_helper<std::remove_cv_t<std::remove_reference_t<T>>> {};

// path of python executable
const char *get_python_executable();
const char *get_gen_static_sig_script();
const char *get_standalone_compile_script();
std::filesystem::path get_cache_path();
}  // namespace triton_jit
