cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
include(FetchContent)

project(flaggems LANGUAGES CUDA CXX)

# project-wide settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) # Ensures only standard-compliant C++ is used

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# dependencies
find_package(CUDAToolkit REQUIRED)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS})

FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
FetchContent_MakeAvailable(json)

FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt
  GIT_TAG e69e5f977d458f2650bb346dadf2ad30c5320281) # 10.2.1
# build fmt shared library so it can be linked against https://github.com/fmtlib/fmt/issues/833
set(BUILD_SHARED_LIBS ON CACHE BOOL "Build fmt as shared library" FORCE)
FetchContent_MakeAvailable(fmt)


include_directories("${PROJECT_SOURCE_DIR}/include")
add_subdirectory(lib)
add_subdirectory(ctests)
