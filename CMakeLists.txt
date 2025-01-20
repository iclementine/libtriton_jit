cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
include(FetchContent)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
project(tensor_add LANGUAGES CUDA CXX)

find_package(CUDAToolkit REQUIRED)
find_package(Torch REQUIRED)

FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
FetchContent_MakeAvailable(json)

FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt
  GIT_TAG        e69e5f977d458f2650bb346dadf2ad30c5320281) # 10.2.1
FetchContent_MakeAvailable(fmt)


add_executable(tensor_add tensor_add.cu)
target_link_libraries(tensor_add "${TORCH_LIBRARIES}" CUDA::cuda_driver)
set_property(TARGET tensor_add PROPERTY CXX_STANDARD 17)

add_executable(triton_example triton_example.cpp)
target_link_libraries(triton_example "${TORCH_LIBRARIES}" CUDA::cuda_driver)
set_property(TARGET triton_example PROPERTY CXX_STANDARD 17)


add_executable(triton_example_jit triton_example_jit.cpp)
target_link_libraries(triton_example_jit PRIVATE "${TORCH_LIBRARIES}" CUDA::cuda_driver nlohmann_json::nlohmann_json fmt::fmt)
set_property(TARGET triton_example_jit PROPERTY CXX_STANDARD 17)