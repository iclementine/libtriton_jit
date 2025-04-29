include(FetchContent)

# dependencies: cuda toolkit
find_package(CUDAToolkit REQUIRED COMPONENTS cuda_driver)

# dependencies: python
find_package(Python REQUIRED COMPONENTS Interpreter Development)

# dependencies: torch
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}")
find_package(Torch MODULE REQUIRED) # This is the FindTorch.cmake

# dependencies: json
if (TRITON_JIT_USE_EXTERNAL_JSON)
  find_package(nlohmann_json VERSION 3.11.3 REQUIRED)
else()
  FetchContent_Declare(json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
    DOWNLOAD_EXTRACT_TIMESTAMP ON
  )
  FetchContent_MakeAvailable(json)
endif()

# dependencies: fmtlib
if (TRITON_JIT_USE_EXTERNAL_FMTLIB)
  find_package(fmt VERSION 10.2.1 REQUIRED)
else()
  FetchContent_Declare(fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt
    GIT_TAG e69e5f977d458f2650bb346dadf2ad30c5320281 # 10.2.1
    DOWNLOAD_EXTRACT_TIMESTAMP ON)
  # build fmt shared library so it can be linked against https://github.com/fmtlib/fmt/issues/833
  # set(BUILD_SHARED_LIBS ON CACHE BOOL "Build fmt as shared library" FORCE)
  FetchContent_MakeAvailable(fmt)
  # so as to build a shared library that links libfmt.a
  set_target_properties(fmt PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()


if (USE_EXTERNAL_PYBIND11)
  execute_process(COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_ROOT
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  find_package(pybind11 CONFIG REQUIRED)
else()
  FetchContent_Declare(pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11)
  FetchContent_MakeAvailable(pybind11)
endif()
