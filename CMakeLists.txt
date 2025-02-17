cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
include(FetchContent)

project(tensor_add LANGUAGES CUDA CXX)

# project-wide settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) # Ensures only standard-compliant C++ is used

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# dependencies
find_package(CUDAToolkit REQUIRED)
find_package(Torch REQUIRED)

FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
FetchContent_MakeAvailable(json)

FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt
  GIT_TAG e69e5f977d458f2650bb346dadf2ad30c5320281) # 10.2.1
# build fmt shared library so it can be linked against https://github.com/fmtlib/fmt/issues/833
set(BUILD_SHARED_LIBS ON CACHE BOOL "Build fmt as shared library" FORCE)
FetchContent_MakeAvailable(fmt)

# targets
add_executable(tensor_add tensor_add.cu)
target_link_libraries(tensor_add "${TORCH_LIBRARIES}" CUDA::cuda_driver)

add_executable(triton_example_jit triton_example_jit.cpp)
target_link_libraries(triton_example_jit
  PRIVATE "${TORCH_LIBRARIES}" CUDA::cuda_driver nlohmann_json::nlohmann_json fmt::fmt)

add_library(triton_kernel SHARED triton_kernel.cpp)
target_include_directories(triton_kernel PUBLIC "${PROJECT_SOURCE_DIR}")
target_link_libraries(triton_kernel
  PUBLIC CUDA::cuda_driver
  PRIVATE nlohmann_json::nlohmann_json fmt::fmt)

add_library(jit_utils SHARED jit_utils.cpp)
target_include_directories(jit_utils PUBLIC "${PROJECT_SOURCE_DIR}")
target_link_libraries(jit_utils
  PUBLIC "${TORCH_LIBRARIES}")