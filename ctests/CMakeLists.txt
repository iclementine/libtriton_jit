add_executable(test_triton_add test_triton_add.cpp)
target_link_libraries(test_triton_add PRIVATE operators)
set_target_properties(test_triton_add PROPERTIES
  INSTALL_RPATH "$ORIGIN/../lib"  # Better to use ${CMAKE_INSTALLPREFIX}/lin
  BUILD_WITH_INSTALL_RPATH FALSE  # Differentiate build vs install RPATH
)

add_executable(test_triton_sum test_triton_sum.cpp)
target_link_libraries(test_triton_sum PRIVATE operators)
set_target_properties(test_triton_sum PROPERTIES
  INSTALL_RPATH "$ORIGIN/../lib"  # Better to use ${CMAKE_INSTALLPREFIX}/lib
  BUILD_WITH_INSTALL_RPATH FALSE  # Differentiate build vs install RPATH
)

install(TARGETS test_triton_add test_triton_sum)
