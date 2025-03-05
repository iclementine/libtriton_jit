# targets
add_library(triton_kernel SHARED triton_kernel.cpp)
target_link_libraries(triton_kernel
  PUBLIC CUDA::cuda_driver
  PRIVATE nlohmann_json::nlohmann_json fmt::fmt)


add_library(jit_utils SHARED jit_utils.cpp)
target_link_libraries(jit_utils
  PUBLIC "${TORCH_LIBRARIES}")

add_library(triton_jit_function SHARED triton_jit_function.cpp)
target_link_libraries(triton_jit_function
  PUBLIC CUDA::cuda_driver jit_utils triton_kernel nlohmann_json::nlohmann_json fmt::fmt)

# add_executable(test_triton_add test_triton_add.cpp)
# target_link_libraries(test_triton_add
#   PUBLIC triton_jit_function)

# add_executable(test_triton_sum test_triton_sum.cpp)
# target_link_libraries(test_triton_sum
#   PUBLIC triton_jit_function)


