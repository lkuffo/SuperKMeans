# Test suite for SuperKMeans

# Fetch GoogleTest
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Enable testing
enable_testing()

# Common test dependencies
set(TEST_COMMON_LIBS
    gtest
    gtest_main
    ${MKL_COMMON_LIBS}
    ${BLAS_LINK_LIBRARIES}
)

if (FFTW_FOUND)
    list(APPEND TEST_COMMON_LIBS ${FFTW_FLOAT_LIB} ${FFTW_FLOAT_OPENMP_LIB})
endif()

# Test: Rotation functionality (Rotate/Unrotate, norm/inner-product preservation)
add_executable(test_rotation.out test_rotation.cpp)
target_link_libraries(test_rotation.out PRIVATE ${TEST_COMMON_LIBS})

# Test: Assign functionality
add_executable(test_assign.out test_assign.cpp)
target_link_libraries(test_assign.out PRIVATE ${TEST_COMMON_LIBS})

# Test: SuperKMeans core functionality
add_executable(test_superkmeans.out test_superkmeans.cpp)
target_link_libraries(test_superkmeans.out PRIVATE ${TEST_COMMON_LIBS})

# Test: WCSS correctness (monotonic decrease and expected range)
add_executable(test_wcss.out test_wcss.cpp)
target_link_libraries(test_wcss.out PRIVATE ${TEST_COMMON_LIBS})

# Utility: Generate ground truth WCSS values for test_wcss.cpp
add_executable(generate_wcss_ground_truth.out generate_wcss_ground_truth.cpp)
target_link_libraries(generate_wcss_ground_truth.out PRIVATE ${TEST_COMMON_LIBS})

# Test: Distance computers (SIMD vs scalar, BatchComputer correctness)
add_executable(test_distance_computers.out test_distance_computers.cpp)
target_link_libraries(test_distance_computers.out PRIVATE ${TEST_COMMON_LIBS})

# Test: PDX layout dimension splitting
add_executable(test_pdx_layout.out test_pdx_layout.cpp)
target_link_libraries(test_pdx_layout.out PRIVATE ${TEST_COMMON_LIBS})

# Automatically discover all GoogleTest tests
include(GoogleTest)
gtest_discover_tests(test_rotation.out)
gtest_discover_tests(test_assign.out)
gtest_discover_tests(test_superkmeans.out)
gtest_discover_tests(test_wcss.out)
gtest_discover_tests(test_distance_computers.out)
gtest_discover_tests(test_pdx_layout.out)
