add_library(operator_add add.cpp)
target_link_libraries(operator_add
    PUBLIC triton_jit_function)

add_library(operator_sum sum.cpp)
target_link_libraries(operator_sum
    PUBLIC triton_jit_function)
