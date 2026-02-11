set(EXAMPLES_COMMON_LIBS
    ${MKL_COMMON_LIBS}
    ${BLAS_LINK_LIBRARIES}
)

if (FFTW_FOUND)
    list(APPEND EXAMPLES_COMMON_LIBS ${FFTW_FLOAT_LIB} ${FFTW_FLOAT_OPENMP_LIB})
endif()

add_executable(simple_clustering.out simple_clustering.cpp)
target_link_libraries(simple_clustering.out PRIVATE ${EXAMPLES_COMMON_LIBS})

add_executable(hierarchical_clustering.out hierarchical_clustering.cpp)
target_link_libraries(hierarchical_clustering.out PRIVATE ${EXAMPLES_COMMON_LIBS})

# Custom target to build all examples with 'make examples'
add_custom_target(examples
    DEPENDS
        simple_clustering.out
        hierarchical_clustering.out
)
