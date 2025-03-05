#pragma once

#include "torch/torch.h"
#include <cstdlib>
#include <filesystem>
#include <string>

std::string execute_command(std::string_view command);

inline const char *to_triton_typename(c10::ScalarType t) {
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

template <typename T> inline const char *spec(T v) {
  return v % 16 == 0 ? ":16" : v == 1 ? ":1" : "";
}

template <typename T, typename = void> struct has_data_ptr : std::false_type {};

template <typename T>
struct has_data_ptr<
    T, std::enable_if_t<std::conjunction_v<
           std::is_same<
               decltype(std::declval<std::remove_reference_t<T>>().data_ptr()),
               void *>,
           std::is_same<decltype(std::declval<std::remove_reference_t<T>>()
                                     .scalar_type()),
                        c10::ScalarType>>>> : std::true_type {};

template <typename T> struct triton_type_helper {};

template <> struct triton_type_helper<bool> {
  static constexpr const char *name = "i1";
};
template <> struct triton_type_helper<int> {
  static constexpr const char *name = "i32";
};
template <> struct triton_type_helper<int64_t> {
  static constexpr const char *name = "i64";
};

template <> struct triton_type_helper<float> {
  static constexpr const char *name = "f32";
};
template <> struct triton_type_helper<double> {
  static constexpr const char *name = "f64";
};
template <> struct triton_type_helper<std::nullptr_t> {
  static constexpr const char *name = "*i8";
};

template <typename T>
struct triton_type
    : triton_type_helper<std::remove_cv_t<std::remove_reference_t<T>>> {};

// path of python executable
const char *get_python_executable();
const char *get_gen_static_sig_script();
const char *get_standalone_compile_script();
std::filesystem::path get_cache_path();
std::filesystem::path get_triton_src_path();