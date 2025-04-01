cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
project(TritonJIT
  LANGUAGES CUDA CXX
  VERSION 0.1.0)

# project-wide settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) # Ensures only standard-compliant C++ is used
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --------------------------- RPATH settings ---------------------------
# https://gitlab.kitware.com/cmake/community/-/wikis/doc/cmake/RPATH-handling
if(APPLE)
  set(CMAKE_MACOSX_RPATH ON)
  set(_rpath_portable_origin "@loader_path")
else()
  set(_rpath_portable_origin $ORIGIN)
endif(APPLE)

# default Use separate rpaths during build and install phases
set(CMAKE_SKIP_BUILD_RPATH  FALSE)
# Don't use the install-rpath during the build phase
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH "${_rpath_portable_origin}")
# Automatically add all linked folders that are NOT in the build directory to
# the rpath (per library?)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

include(cmake/Options.cmake)
include(cmake/Dependencies.cmake)

add_subdirectory(src)

# --------------------------- installtion of cmake package ---------------------------
include(CMakePackageConfigHelpers)
# generate the config file that includes the exports: TritonJITConfig.cmake
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/TritonJITConfig.cmake"
  INSTALL_DESTINATION "lib/cmake/TritonJIT"
  # NO_SET_AND_CHECK_MACRO
  # NO_CHECK_REQUIRED_COMPONENTS_MACRO
)
# generate the config file that includes the version: TritonJITConfigVersion.cmake
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/TritonJITConfigVersion.cmake"
  VERSION "${TritonJIT_VERSION_MAJOR}.${TritonJIT_VERSION_MINOR}.${TritonJIT_VERSION_PATCH}"
  COMPATIBILITY SameMajorVersion
)
# install the generated project cmake config & version file
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/TritonJITConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/TritonJITConfigVersion.cmake
  DESTINATION lib/cmake/TritonJIT
)
