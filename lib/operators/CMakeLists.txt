add_library(operator_add SHARED add.cpp)
target_link_libraries(operator_add
    PUBLIC triton_jit_function)

add_library(operator_sum SHARED sum.cpp)
target_link_libraries(operator_sum
    PUBLIC triton_jit_function)

add_library(operators SHARED operators.cpp)
target_link_libraries(operators
    PRIVATE operator_add operator_sum)

 install(
  DIRECTORY "${CMAKE_SOURCE_DIR}/include/operators"
  DESTINATION "include")
install(DIRECTORY "${CMAKE_SOURCE_DIR}/triton_src" DESTINATION "share/flaggems/" FILES_MATCHING PATTERN "*.py")
install(TARGETS operators operator_sum operator_add)
