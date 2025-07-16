
#include <gtest/gtest.h>
#include "axpy_op.h"
#include "c10/cuda/CUDAFunctions.h"
#include "torch/torch.h"

TEST(arg_handle_test, scalar_int) {
  at::Tensor a = at::rand({128 * 1024}, at::kCUDA);
  at::Tensor b = at::rand({128 * 1024}, at::kCUDA);
  // warm up
  at::Tensor result1 = my_ops::axpy(a, b, c10::Scalar(10));
  at::Tensor result2 = at::add(c10::Scalar(10) * a, b);
  EXPECT_TRUE(torch::allclose(result1, result2));
}

TEST(arg_handle_test, scalar_double) {
  at::Tensor a = at::rand({128 * 1024}, at::kCUDA);
  at::Tensor b = at::rand({128 * 1024}, at::kCUDA);
  // warm up
  at::Tensor result1 = my_ops::axpy(a, b, c10::Scalar(3.14));
  at::Tensor result2 = at::add(c10::Scalar(3.14) * a, b);
  EXPECT_TRUE(torch::allclose(result1, result2));
}

TEST(arg_handle_test, optional_scalar_null_opt) {
  at::Tensor a = at::rand({128 * 1024}, at::kCUDA);
  at::Tensor b = at::rand({128 * 1024}, at::kCUDA);

  std::optional<c10::Scalar> alpha = std::nullopt;
  at::Tensor result1 = my_ops::axpy2(a, b, alpha);
  at::Tensor result2 = at::add(a, b);
  EXPECT_TRUE(torch::allclose(result1, result2));
}

TEST(arg_handle_test, optional_scalar_has_value) {
  at::Tensor a = at::rand({128 * 1024}, at::kCUDA);
  at::Tensor b = at::rand({128 * 1024}, at::kCUDA);

  std::optional<c10::Scalar> alpha = c10::Scalar(3.14);
  at::Tensor result1 = my_ops::axpy2(a, b, alpha);
  at::Tensor result2 = at::add(alpha.value() * a, b);
  EXPECT_TRUE(torch::allclose(result1, result2));
}

TEST(arg_handle_test, optional_scalar_bare_nullopt) {
  at::Tensor a = at::rand({128 * 1024}, at::kCUDA);
  at::Tensor b = at::rand({128 * 1024}, at::kCUDA);

  at::Tensor result1 = my_ops::axpy2(a, b, std::nullopt);
  at::Tensor result2 = at::add(a, b);
  EXPECT_TRUE(torch::allclose(result1, result2));
}

TEST(arg_handle_test, optional_tensor_has_value) {
  at::Tensor a = at::rand({128 * 1024}, at::kCUDA);
  std::optional<at::Tensor> b = at::rand({128 * 1024}, at::kCUDA);

  c10::Scalar alpha(3.14);
  at::Tensor result1 = my_ops::axpy3(a, b, alpha);
  at::Tensor result2 = at::add(alpha * a, b.value());
  EXPECT_TRUE(torch::allclose(result1, result2));
}

TEST(arg_handle_test, optional_tensor_nullopt) {
  at::Tensor a = at::rand({128 * 1024}, at::kCUDA);
  std::optional<at::Tensor> b = std::nullopt;

  c10::Scalar alpha(3.14);
  at::Tensor result1 = my_ops::axpy3(a, b, alpha);
  at::Tensor result2 = alpha * a;
  EXPECT_TRUE(torch::allclose(result1, result2));
}
