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
    ${BLAS_LINK_LIBRARIES}
)

if (FFTW_FOUND)
    list(APPEND TEST_COMMON_LIBS ${FFTW_FLOAT_LIB} ${FFTW_FLOAT_THREADS_LIB})
endif()

# Test: Unrotate functionality
add_executable(test_unrotate.out test_unrotate.cpp)
target_link_libraries(test_unrotate.out PRIVATE ${TEST_COMMON_LIBS})

# Test: Assign functionality
add_executable(test_assign.out test_assign.cpp)
target_link_libraries(test_assign.out PRIVATE ${TEST_COMMON_LIBS})

# Test: SuperKMeans core functionality
add_executable(test_superkmeans.out test_superkmeans.cpp)
target_link_libraries(test_superkmeans.out PRIVATE ${TEST_COMMON_LIBS})

# Automatically discover all GoogleTest tests
include(GoogleTest)
gtest_discover_tests(test_unrotate.out)
gtest_discover_tests(test_assign.out)
gtest_discover_tests(test_superkmeans.out)
