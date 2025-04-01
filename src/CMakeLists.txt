# targets
## --------------------------- triton_kernel ---------------------------
add_library(triton_kernel SHARED triton_kernel.cpp)
target_include_directories(triton_kernel
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_link_libraries(triton_kernel
  PUBLIC CUDA::cuda_driver
  PRIVATE nlohmann_json::nlohmann_json fmt::fmt-header-only)

## --------------------------- jit utils ---------------------------
add_library(jit_utils SHARED jit_utils.cpp)
target_include_directories(jit_utils
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_link_libraries(jit_utils PUBLIC Torch::Torch)

# --------------------------- triton jit function ---------------------------
add_library(triton_jit SHARED triton_jit_function.cpp)
target_link_libraries(triton_jit
  PUBLIC CUDA::cuda_driver jit_utils triton_kernel fmt::fmt-header-only
  PRIVATE nlohmann_json::nlohmann_json)

# --------------------------- alias targets ---------------------------
# This is the target used in FetchContent, since FetchContent use add_subdirectory (a sub project build)
# So to use consistent target name, add namespace here
add_library(TritonJIT::triton_jit ALIAS triton_jit)
add_library(TritonJIT::jit_utils ALIAS jit_utils)
add_library(TritonJIT::triton_kernel ALIAS triton_kernel)

# --------------------------- install ---------------------------
include(GNUInstallDirs)
install(DIRECTORY "${PROJECT_SOURCE_DIR}/include/triton_jit" DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY "${PROJECT_SOURCE_DIR}/scripts"
  DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/triton_jit/" FILES_MATCHING PATTERN "*.py")
install(
  TARGETS triton_kernel jit_utils triton_jit
  EXPORT TritonJITTargets
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(
  EXPORT TritonJITTargets
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TritonJIT
  NAMESPACE TritonJIT::
  FILE TritonJITTargets.cmake
)

# This is for find_package to work with the build tree, instead of the install tree
export(EXPORT TritonJITTargets FILE "${PROJECT_BINARY_DIR}/TritonJITTargets.cmake")
