cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
project(flaggems
  LANGUAGES
  CUDA CXX
  VERSION
  0.0.1)

# project-wide settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) # Ensures only standard-compliant C++ is used
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# RPATH settings
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

include(GNUInstallDirs)
include(cmake/Dependencies.cmake)

add_subdirectory(lib)
add_subdirectory(ctests)


if(DEFINED SKBUILD)
  pybind11_add_module(c_operators "${PROJECT_SOURCE_DIR}/python/flaggems/csrc/cstub.cpp")
  target_link_libraries(c_operators PRIVATE operators)
  # Installing the extension module to the root of the package
  install(TARGETS c_operators
          LIBRARY DESTINATION ".")

  # configure_file("${PROJECT_SOURCE_DIR}/src/hello/__main__.py.in"
  #                "${PROJECT_BINARY_DIR}/src/hello/__main__.py")

  # install(FILES "${PROJECT_BINARY_DIR}/src/hello/__main__.py" DESTINATION .)

  # The extension module must load the hello library as a dependency when the
  # extension module is loaded. The easiest way to locate the hello library is
  # via RPATH. Absolute RPATHs are possible, but they make the resulting
  # binaries not redistributable to other Python installations (conda is broke,
  # wheel reuse is broke, and more!).
  #
  # Placing the hello library in the package and using relative RPATHs that
  # doesn't point outside of the package means that the built package is
  # relocatable. This allows for safe binary redistribution.
  if(APPLE)
    set_target_properties(
      c_operators PROPERTIES INSTALL_RPATH "@loader_path/${CMAKE_INSTALL_LIBDIR}")
  else()
    set_target_properties(c_operators PROPERTIES INSTALL_RPATH
                                            "$ORIGIN/${CMAKE_INSTALL_LIBDIR}")
  endif()

endif()
