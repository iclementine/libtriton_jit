#include "torch/torch.h"
#include <string>



std::string strip(const std::string &str);
std::string executePythonScript(std::string_view command);

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


// template <typename T> struct triton_type {};

// template <> struct triton_type<bool> {
//   static constexpr const char *name = "i1";
// };
// template <> struct triton_type<int> {
//   static constexpr const char *name = "i32";
// };
// template <> struct triton_type<int64_t> {
//   static constexpr const char *name = "i64";
// };
// template <> struct triton_type<float> {
//   static constexpr const char *name = "f32";
// };
// template <> struct triton_type<double> {
//   static constexpr const char *name = "f64";
// };
// template <> struct triton_type<std::nullptr_t> {
//   static constexpr const char *name = "*i8";
// };


