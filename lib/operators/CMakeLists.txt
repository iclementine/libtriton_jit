add_library(operator_add SHARED add.cpp)
target_link_libraries(operator_add
    PUBLIC triton_jit_function)

add_library(operator_sum SHARED sum.cpp)
target_link_libraries(operator_sum
    PUBLIC triton_jit_function)

add_library(operators SHARED operators.cpp)
target_link_libraries(operators
    PRIVATE operator_add operator_sum )
