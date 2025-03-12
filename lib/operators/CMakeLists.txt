add_library(operators SHARED add.cpp sum.cpp)
target_include_directories(operators
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include>)
target_link_libraries(operators
  PUBLIC ${TORCH_LIBRARIES}
  PRIVATE triton_jit_function)

# add_library(operator_sum SHARED sum.cpp)
# target_include_directories(operator_sum
#   PUBLIC
#     $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
#     $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include>)
# target_link_libraries(operator_sum
#   PUBLIC ${TORCH_LIBRARIES}
#   PRIVATE triton_jit_function)

# add_library(operators SHARED operators.cpp)
# target_include_directories(operators
#   PUBLIC
#     $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
#     $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include>)
# target_link_libraries(operators
#   PUBLIC ${TORCH_LIBRARIES}
#   PRIVATE operator_add operator_sum triton_jit_function)

install(
  DIRECTORY "${CMAKE_SOURCE_DIR}/include/operators"
  DESTINATION "include")
install(DIRECTORY "${CMAKE_SOURCE_DIR}/triton_src" DESTINATION "share/flaggems" FILES_MATCHING PATTERN "*.py")
install(TARGETS operators)
