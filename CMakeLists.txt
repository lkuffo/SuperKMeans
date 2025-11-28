cmake_minimum_required(VERSION 3.26)
set(Python3_FIND_STRATEGY NEVER)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS -Wall)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/extern/findFFTW")
project(SuperKMeans)

add_compile_options(-fPIC)
add_compile_options(-fopenmp)
include(FetchContent)
include(CheckCXXCompilerFlag)
include(CMakePrintHelpers)
include(CTest)

# Find BLAS library (optional but recommended)
# Automatically finds: Accelerate on macOS, MKL/OpenBLAS on Linux, etc.
find_package(BLAS QUIET)
if (BLAS_FOUND)
    message(STATUS "BLAS library found: ${BLAS_LIBRARIES}")
    add_definitions(-DEIGEN_USE_BLAS)
    set(BLAS_LINK_LIBRARIES ${BLAS_LIBRARIES})
else()
    message(WARNING "BLAS library not found. Performance will be degraded. Consider installing: Accelerate (macOS), MKL, or OpenBLAS")
    set(BLAS_LINK_LIBRARIES "")
endif()

find_package(FFTW QUIET)
if (FFTW_FLOAT_LIB_FOUND)
    message(STATUS "FFTW (+float capabilities) found " ${FFTW_INCLUDE_DIR})
    add_definitions(-DHAS_FFTW)
    include_directories(${FFTW_INCLUDE_DIRS})
else()
    message(WARNING "FFTW (+float capabilities) not found, we recommend you install it")
endif()
set(FAISS_ENABLE_PYTHON OFF CACHE BOOL "disable python" FORCE)
set(FAISS_ENABLE_GPU OFF CACHE BOOL "disable gpu" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "disable faiss tests" FORCE)
set(BUILD_SHARED_LIBS ON CACHE BOOL "shared libs" FORCE)
# set(FAISS_OPT_LEVEL "generic" CACHE STRING "CPU optimization level" FORCE)
FetchContent_Declare(
        faiss
        GIT_REPOSITORY https://github.com/facebookresearch/faiss.git
        GIT_TAG        v1.11.0   # pick a known tag/commit
)
FetchContent_MakeAvailable(faiss)

# CMAKE_SOURCE_DIR: ----------------------------------------------------------------------------------------------------
add_compile_definitions(CMAKE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

include_directories(include extern extern/lorann/lorann)

# Legacy benchmarks and tests (kept for compatibility)
add_executable(bench_kmeans_t.out bench_kmeans.cpp)
add_executable(bench_kmeans_faiss.out bench_kmeans_faiss.cpp)

# Link BLAS to all targets (if found)
target_link_libraries(bench_kmeans_faiss.out PRIVATE faiss ${BLAS_LINK_LIBRARIES})
target_link_libraries(bench_kmeans_t.out PRIVATE ${BLAS_LINK_LIBRARIES})

# Link FFTW if available
if (FFTW_FOUND)
    message(STATUS "Linking FFTW: ${FFTW_FLOAT_THREADS_LIB}")
    target_link_libraries(bench_kmeans_t.out PRIVATE ${FFTW_FLOAT_LIB} ${FFTW_FLOAT_THREADS_LIB})
endif()

# Add test and benchmark subdirectories
add_subdirectory(tests)
add_subdirectory(benchmarks)