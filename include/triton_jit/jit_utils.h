#pragma once

#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <optional>
#include <string>

#include "c10/util/Logging.h"  // use torch's logging
#include "torch/torch.h"

namespace triton_jit {

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
    case c10::ScalarType::Short:
      return "i16";
    case c10::ScalarType::Char:
      return "i8";
    case c10::ScalarType::Byte:
      return "u8";
    case c10::ScalarType::Bool:
      return "i1";
    default:
      throw std::runtime_error("<unsupported_type>");
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
struct is_optional_helper : public std::false_type {};

template <typename T>
struct is_optional_helper<std::optional<T>> : public std::true_type {};

template <typename T>
struct is_optional : public is_optional_helper<std::remove_const_t<std::remove_reference_t<T>>> {};

template <typename T>
struct is_scalar_helper : public std::false_type {};

template <>
struct is_scalar_helper<c10::Scalar> : public std::true_type {};

template <typename T>
struct is_scalar : public is_scalar_helper<std::remove_const_t<std::remove_reference_t<T>>> {};

template <typename T>
struct triton_type_helper;

template <typename T, typename U>
struct is_same_ignore_cvref : public std::is_same<std::remove_reference_t<std::remove_cv_t<T>>,
                                                  std::remove_reference_t<std::remove_cv_t<U>>> {};

#define DEFINE_TRITON_TYPE(T, Name)           \
  template <>                                 \
  struct triton_type_helper<T> {              \
    static constexpr const char *name = Name; \
  }

DEFINE_TRITON_TYPE(bool, "i1");
DEFINE_TRITON_TYPE(int, "i32");
DEFINE_TRITON_TYPE(uint, "u32");
DEFINE_TRITON_TYPE(int64_t, "i64");
DEFINE_TRITON_TYPE(uint64_t, "u64");
DEFINE_TRITON_TYPE(float, "fp32");
DEFINE_TRITON_TYPE(double, "fp64");
DEFINE_TRITON_TYPE(std::nullptr_t, "*i8");

#undef DEFINE_TRITON_TYPE

template <typename T>
struct triton_type : triton_type_helper<std::remove_cv_t<std::remove_reference_t<T>>> {};

// path of python executable
std::filesystem::path get_script_dir();
const char *get_gen_static_sig_script();
const char *get_standalone_compile_script();
}  // namespace triton_jit
