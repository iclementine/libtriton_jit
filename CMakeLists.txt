cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
project(TritonJIT
  LANGUAGES CUDA CXX
  VERSION 0.1.0)


set(TRITON_JIT_MASTER_PROJECT OFF)
if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  set(TRITON_JIT_MASTER_PROJECT ON)
  message(STATUS "CMake version: ${CMAKE_VERSION}")
endif ()

include(cmake/Settings.cmake)
include(cmake/Options.cmake)
include(cmake/Dependencies.cmake)

add_subdirectory(src)

# --------------------------- installtion of cmake package ---------------------------
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
# generate the config file that includes the exports: TritonJITConfig.cmake
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/TritonJITConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TritonJIT
)
# generate the config file that includes the version: TritonJITConfigVersion.cmake
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/TritonJITConfigVersion.cmake"
  VERSION "${TritonJIT_VERSION_MAJOR}.${TritonJIT_VERSION_MINOR}.${TritonJIT_VERSION_PATCH}"
  COMPATIBILITY SameMajorVersion
)
# install the generated project cmake config & version file
install(
  FILES ${CMAKE_CURRENT_BINARY_DIR}/TritonJITConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/TritonJITConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TritonJIT
)

install(
  FILES ${CMAKE_CURRENT_BINARY_DIR}/TritonJITConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/TritonJITConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TritonJIT
)

# install the FindTorch
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindTorch.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TritonJIT)
