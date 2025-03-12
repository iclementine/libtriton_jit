# targets
## triton_kernel
add_library(triton_kernel SHARED triton_kernel.cpp)
target_include_directories(triton_kernel
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include>)
target_link_libraries(triton_kernel
  PUBLIC CUDA::cuda_driver
  PRIVATE nlohmann_json::nlohmann_json fmt::fmt)

## jit utils
add_library(jit_utils SHARED jit_utils.cpp)
target_include_directories(jit_utils
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include>)
target_link_libraries(jit_utils PUBLIC ${TORCH_LIBRARIES})

# triton jit function
add_library(triton_jit_function SHARED triton_jit_function.cpp)
target_link_libraries(triton_jit_function
  PUBLIC CUDA::cuda_driver jit_utils triton_kernel fmt::fmt
  PRIVATE nlohmann_json::nlohmann_json)

install(DIRECTORY "${CMAKE_SOURCE_DIR}/include/jit" DESTINATION "include")
install(DIRECTORY "${CMAKE_SOURCE_DIR}/scripts" DESTINATION "share/flaggems/" FILES_MATCHING PATTERN "*.py")
install(TARGETS triton_kernel jit_utils triton_jit_function)
