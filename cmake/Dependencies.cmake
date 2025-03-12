include(FetchContent)

# dependencies
find_package(CUDAToolkit REQUIRED)
find_package(Python REQUIRED COMPONENTS Interpreter Development)
execute_process(COMMAND ${Python_EXECUTABLE} "-c" "import torch;print(torch.utils.cmake_prefix_path)"
                OUTPUT_VARIABLE Torch_ROOT_RAW
                COMMAND_ECHO STDOUT)
string(STRIP ${Torch_ROOT_RAW} Torch_ROOT)
find_package(Torch REQUIRED)
message(WARNING "${TORCH_INCLUDE_DIRS}")

execute_process(COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
  OUTPUT_VARIABLE pybind11_ROOT_RAW
  COMMAND_ECHO STDOUT)
string(STRIP ${pybind11_ROOT_RAW} pybind11_ROOT)
find_package(pybind11)


# set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS})
# pybind
# if (FLAGGEMS_USE_EXTERNAL_PYBIND11)
#   find_package(pybind11 VERSION 2.13 REQUIRED)
# else()
#   FetchContent_Declare(
#     pybind11
#     GIT_REPOSITORY https://github.com/pybind/pybind11
#     GIT_TAG        v2.13.0
#   )
#   FetchContent_MakeAvailable(pybind11)
# endif()


## json
if (FLAGGEMS_USE_EXTERNAL_JSON)
  find_package(nlohmann_json VERSION 3.11.3 REQUIRED)
else()
  FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
  FetchContent_MakeAvailable(json)
endif()

## fmt
if (FLAGGEMS_USE_EXTERNAL_FMTLIB)
  find_package(fmt VERSION 10.2.1 REQUIRED)
else()
  FetchContent_Declare(fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt
    GIT_TAG e69e5f977d458f2650bb346dadf2ad30c5320281) # 10.2.1
  # build fmt shared library so it can be linked against https://github.com/fmtlib/fmt/issues/833
  set(BUILD_SHARED_LIBS ON CACHE BOOL "Build fmt as shared library" FORCE)
  FetchContent_MakeAvailable(fmt)
endif()
