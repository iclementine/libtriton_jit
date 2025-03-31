add_library(operators SHARED add.cpp sum.cpp)
target_include_directories(operators
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include>)
target_link_libraries(operators
  PUBLIC ${TORCH_LIBRARIES}
  PRIVATE triton_jit_function)

install(
  DIRECTORY "${CMAKE_SOURCE_DIR}/include/operators"
  DESTINATION "include")
install(DIRECTORY "${CMAKE_SOURCE_DIR}/triton_src" DESTINATION "share/flaggems" FILES_MATCHING PATTERN "*.py")
install(TARGETS operators)
