include(FetchContent)

# dependencies
find_package(CUDAToolkit REQUIRED)
find_package(Torch REQUIRED)
# set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS})

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
