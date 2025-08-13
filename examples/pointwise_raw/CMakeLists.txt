add_custom_target(
    copy_triton_pointwise_src_unique
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/add.py
            ${CMAKE_CURRENT_BINARY_DIR}/add.py
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/add_triton_cpp_rt.py
            ${CMAKE_CURRENT_BINARY_DIR}/add_triton_cpp_rt.py
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/add.py
            ${CMAKE_CURRENT_SOURCE_DIR}/add_triton_cpp_rt.py
)

add_library(add_op_raw SHARED add_op.cpp)
target_include_directories(add_op_raw
    PRIVATE ${PROJECT_SOURCE_DIR}/include
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(add_op_raw
    PUBLIC Torch::Torch
    PRIVATE TritonJIT::triton_jit
)
add_dependencies(add_op_raw copy_triton_pointwise_src_unique)

add_executable(test_add_raw test_add.cpp)
target_link_libraries(test_add_raw
    PRIVATE add_op_raw Torch::Torch)
add_dependencies(test_add_raw copy_triton_pointwise_src_unique)